"""
Chroma + BM25 Hybrid Retriever — Retriever Protocol 구현체.

기존 `app.graph.nodes.knowledge_search`의 `_search_hybrid`를 Protocol에 wrap.
v1과 동일한 검색 로직 + score 정규화 + Document 변환.
"""
from __future__ import annotations

import asyncio
import os
import re
import threading
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument

from app.core.config import (
    get_embeddings,
    CHROMA_DB_PATH, CHROMA_COLLECTION,
    RETRIEVAL_TOP_K,
)
from app.graph_v2.retrievers.base import Document, Retriever


_HYBRID_FETCH_MULTIPLIER = 4
# RRF 상수.
#
# 원 논문의 60 은 다수 시스템을 융합하는 TREC 환경 기준이다. 여기는 벡터와
# BM25 둘뿐이고, **한쪽이 통째로 실패할 수 있다** — 한국어 질의와 영문 표가
# 섞인 코퍼스에서 BM25 는 어휘가 겹치지 않아 정답을 아예 못 찾는다.
#
# k 가 크면 역순위 분포가 평평해져 "양쪽에 등장했는가" 가 "얼마나 잘 맞는가" 를
# 압도한다. k=60 에서는 이런 역전이 일어난다:
#     벡터 9위 + BM25 없음      = 1/69          = 0.0145
#     벡터 40위 + BM25 40위     = 1/100 x 2     = 0.0200   <- 이쪽이 이긴다
#
# 【실측 — 283문항 평가셋, 정답 chunk 회수율 recall@20】
# 평가셋은 코퍼스 142 chunk 에서 생성했고 두 유형으로 나눈다.
#   direct  코퍼스 어휘를 그대로 쓰는 질문        175문항
#   gap     같은 사실을 다른 표현으로 묻는 질문    108문항
#           (Self-Managed -> "자체 호스팅", GPU -> "그래픽카드")
#
#   단독 검색기 기준선
#     벡터 단독            전체 98.6%   gap 97.2%
#     BM25 단독            전체 92.6%   gap 82.4%   <- 어휘격차에서 구조적 실패
#
#   RRF 상수별
#     k=60  전체 98.2%   gap 95.4%   <- 융합이 벡터 단독보다 나빴다
#     k=30  전체 98.6%   gap 96.3%
#     k=15  전체 99.3%   gap 98.1%
#     k=10  전체 99.6%   gap 99.1%   r@5 94.7%  MRR 0.781   <- 채택
#     k= 5  전체 99.3%   gap 98.1%
#     k= 2  전체 99.3%   gap 98.1%
#
# k=60 에서 융합 결과(95.4%)가 벡터 단독(97.2%)보다 나빴다는 점이 핵심이다.
# 융합이 정보를 더한 게 아니라 깎고 있었다.
#
# 대안 융합 전략도 비교했다. "두 랭킹 중 좋은 쪽만 취하기"(min-rank) 는
# gap 97.2% 로 RRF k=10 보다 나빴다 — 합의 신호를 완전히 버리면 손해다.
# 벡터/BM25 가중치 조정도 개선이 없었다. 적당한 k 의 RRF 가 최적이다.
#
# 【주의】 이전에 28문항으로 스윕했을 때는 k=5 가 최적으로 나왔다. 283문항에서는
# k=10 이 더 낫다. 작은 평가셋의 순위 차이는 잡음이었다.
_RRF_K = int(os.getenv("RRF_K", "10"))


# 같은 페이지 형제 chunk 동반 회수 방식. expand_with_same_page 주석 참조.
#   off     가져오지 않음 (기본). Chroma 조회 생략.
#   append  뒤에 붙임 — FINAL_K 컷에서 전부 잘려 사실상 무동작이던 기존 동작
#   inline  부모 바로 뒤 삽입 — 의도대로 작동시킨 버전
#
# 【기본값을 off 로 정한 근거 — 실측, 120문항 답변 정확도】
#     CO=append (기존)   전체 90.0%   direct 91.9%   gap 87.0%
#     CO=off             전체 89.2%   direct 89.2%   gap 89.1%
#     CO=inline (수정)   전체 88.3%   direct 90.5%   gap 84.8%
#
# inline 은 형제를 부모 뒤에 끼워 넣어 "본문과 표를 함께 보기" 를 의도대로
# 작동시킨 버전인데, 오히려 정확도가 떨어졌다. 더 잘 맞는 문서를 컨텍스트에서
# 밀어내기 때문이다. 청킹 단계에서 이미 맥락 헤더(_context_header)와 표 행 단위
# 분할을 적용해 chunk 가 자체 완결적이므로, 형제를 끌어오는 이득이 없다.
#
# off 와 append 는 통계적으로 동률이다 (n=120 에서 1문항 = 0.83%p). 다만 append
# 는 가져온 형제가 FINAL_K 컷에서 전부 잘리는 것이 증명된 동작이면서 페이지당
# Chroma 조회 2회는 계속 발생시킨다. 같은 정확도라면 조회를 생략하는 쪽이 맞다.
#
# 청킹 방식이 달라져 chunk 가 자체 완결적이지 않게 되면 inline 을 재검토할 것.
_CO_RETRIEVAL_MODE = (os.getenv("CO_RETRIEVAL_MODE", "off") or "off").strip().lower()


class ChromaHybridRetriever:
    """Chroma 시맨틱 + BM25 RRF hybrid retriever.

    - name: "chroma_hybrid"
    - description: 사내 문서 검색 (사내 PPT, PDF, DOCX, TXT)
    """

    name: str = "chroma_hybrid"
    description: str = (
        "사내 지식 베이스 검색. PPT 슬라이드, PDF 페이지, DOCX/TXT 문서를 "
        "시맨틱 + 키워드 hybrid (RRF)로 검색. 사내 정책·매뉴얼·회의록·기획 산출물·강사 자료 등."
    )

    def __init__(self) -> None:
        self._chroma: Optional[Chroma] = None
        self._chroma_lock = threading.Lock()
        self._bm25 = None
        self._bm25_docs: list[LCDocument] = []
        self._bm25_lock = threading.Lock()

    # ─── public Retriever Protocol ───────────────────────

    def retrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[Document]:
        fetch_k = top_k * _HYBRID_FETCH_MULTIPLIER
        sem = self._semantic_search(query, fetch_k)
        bm = self._bm25_search(query, fetch_k)
        merged = self._rrf(sem, bm, top_k)
        return [self._to_document(d, score=s) for d, s in merged]

    async def aretrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[Document]:
        # Chroma·BM25 모두 sync 라이브러리 — thread pool로 비동기화
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.retrieve, query, top_k)

    # ─── 내부 ──────────────────────────────────────────

    def _get_chroma(self) -> Chroma:
        if self._chroma is None:
            with self._chroma_lock:
                if self._chroma is None:
                    self._chroma = Chroma(
                        persist_directory=CHROMA_DB_PATH,
                        embedding_function=get_embeddings(),
                        collection_name=CHROMA_COLLECTION,
                    )
        return self._chroma

    def _get_bm25(self):
        if self._bm25 is not None:
            return self._bm25, self._bm25_docs
        with self._bm25_lock:
            if self._bm25 is not None:
                return self._bm25, self._bm25_docs
            from rank_bm25 import BM25Okapi  # type: ignore
            collection = self._get_chroma()._collection
            res = collection.get(include=["documents", "metadatas"])
            raw_docs = res.get("documents") or []
            raw_metas = res.get("metadatas") or []
            docs = [
                LCDocument(page_content=t, metadata=m)
                for t, m in zip(raw_docs, raw_metas)
                if t
            ]
            if not docs:
                return None, []
            tokenized = [self._tokenize(d.page_content) for d in docs]
            self._bm25 = BM25Okapi(tokenized)
            self._bm25_docs = docs
        return self._bm25, self._bm25_docs

    def invalidate_bm25_cache(self) -> None:
        """문서 재인제스트 후 BM25 인덱스 초기화."""
        with self._bm25_lock:
            self._bm25 = None
            self._bm25_docs = []

    # 형태소 분석기 — lazy singleton (Kiwi 초기화 비용 ~1s)
    _kiwi = None

    @classmethod
    def _get_kiwi(cls):
        if cls._kiwi is None:
            try:
                from kiwipiepy import Kiwi  # type: ignore
                cls._kiwi = Kiwi()
            except ImportError:
                # fallback — kiwipiepy 미설치 시 정규식 토크나이저
                cls._kiwi = False
        return cls._kiwi

    # BM25 인덱싱·검색에 의미있는 품사만 유지
    # NN* 명사, NP 대명사, SL/SH 외국어/한자, SN 숫자, VV/VA 동사/형용사 어간, XR 어근
    _BM25_KEEP_TAGS = ("NN", "NP", "SL", "SH", "SN", "VV", "VA", "XR")

    def _tokenize(self, text: str) -> list[str]:
        """한국어 형태소 분석 + 영숫자 추출.

        조사·어미·문장부호 제거하여 BM25 매칭 정확도 향상:
          "그레이트프로의 슬로건" → ["그레이트프로", "슬로건"] (조사 분리)
          "교통·식비" → ["교통", "식비"] (구분자 분리)
        """
        if not text:
            return []
        kiwi = self._get_kiwi()
        if kiwi is False:  # kiwipiepy 미설치 시 fallback
            return re.findall(r"[가-힣a-zA-Z0-9]+", text.lower())
        tokens = kiwi.tokenize(text)
        out: list[str] = []
        for t in tokens:
            tag = t.tag or ""
            if not any(tag.startswith(prefix) for prefix in self._BM25_KEEP_TAGS):
                continue
            form = (t.form or "").strip().lower()
            if len(form) < 1:
                continue
            out.append(form)
        return out

    def _semantic_search(self, query: str, k: int) -> list[tuple[LCDocument, float]]:
        # Chroma의 similarity_search_with_score는 distance 반환 (작을수록 유사)
        # → 1 / (1 + distance) 로 0~1 정규화
        try:
            results = self._get_chroma().similarity_search_with_score(query, k=k)
            return [(d, 1.0 / (1.0 + dist)) for d, dist in results]
        except Exception:
            # Fallback — score 정보 없으면 균일 점수
            docs = self._get_chroma().similarity_search(query, k=k)
            return [(d, 0.5) for d in docs]

    def _bm25_search(self, query: str, k: int) -> list[tuple[LCDocument, float]]:
        bm25, docs = self._get_bm25()
        if bm25 is None:
            return []
        scores = bm25.get_scores(self._tokenize(query))
        if not len(scores):
            return []
        # BM25 score 정규화 (max 기준)
        max_s = max(scores) if max(scores) > 0 else 1.0
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(docs[i], scores[i] / max_s) for i in ranked if scores[i] > 0]

    def _rrf(
        self,
        sem: list[tuple[LCDocument, float]],
        bm: list[tuple[LCDocument, float]],
        k: int,
    ) -> list[tuple[LCDocument, float]]:
        """Reciprocal Rank Fusion. 두 랭킹의 rank로 결합. 최종 score는 정규화된 normalized score 평균."""
        scores: dict[str, float] = {}
        norm_scores: dict[str, list[float]] = {}
        doc_map: dict[str, LCDocument] = {}

        for rank, (doc, ns) in enumerate(sem):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
            norm_scores.setdefault(key, []).append(ns)
            doc_map[key] = doc
        for rank, (doc, ns) in enumerate(bm):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
            norm_scores.setdefault(key, []).append(ns)
            doc_map.setdefault(key, doc)

        sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)[:k]
        return [
            (doc_map[k_], sum(norm_scores[k_]) / len(norm_scores[k_]))
            for k_ in sorted_keys
        ]

    def _to_document(self, lc: LCDocument, score: float) -> Document:
        md = dict(lc.metadata or {})
        page_unit_index = md.get("page_unit_index")
        page_unit_title = md.get("page_unit_title", "")
        location = ""
        if page_unit_index is not None:
            location = f"{md.get('page_unit_type', 'page').title()} {page_unit_index}"
            if page_unit_title:
                location += f" — {page_unit_title}"
        elif md.get("page_number"):
            location = f"Page {md['page_number']}"
        else:
            location = f"Chunk {md.get('chunk_index', 0)}"

        return Document(
            content=lc.page_content or "",
            source=md.get("doc_id") or md.get("display_source") or md.get("title") or "",
            score=max(0.0, min(1.0, score)),
            metadata={
                "title": md.get("title", ""),
                "doc_id": md.get("doc_id") or md.get("display_source", md.get("title", "")),
                "doc_format": md.get("doc_format", ""),
                "doc_topic": md.get("doc_topic", "general"),
                "page_unit_index": page_unit_index,
                "page_unit_type": md.get("page_unit_type", ""),
                "page_unit_title": page_unit_title,
                "section_path": md.get("section_path", ""),
                "is_table": md.get("is_table", False),
                "table_index": md.get("table_index"),
                "parent_page_index": md.get("parent_page_index"),
                "entities": md.get("entities", ""),
                "location": location,
            },
        )

    # ────────────────────────────────────────────────────────
    # Phase G: Page-level co-retrieval
    # ────────────────────────────────────────────────────────

    def expand_with_same_page(self, retrieved: list[Document]) -> list[Document]:
        """같은 페이지의 형제 chunk 를 함께 가져온다 (본문이 걸리면 그 페이지의 표도).

        【이 함수가 프로덕션에서 무동작이던 경위 — 실측】
        이전 구현은 형제 chunk 를 결과 리스트 **맨 뒤에 append** 하고 고정 score 0.5
        를 부여했다. 그런데 retrieve_node 의 호출 순서는 이렇다:

            pool -> 상위 CANDIDATE_TOP_K(20)개
                 -> expand_with_same_page  (뒤에 append, 예: 35개가 됨)
                 -> boost_by_query_context (인자 없이 호출돼 즉시 return, 무동작)
                 -> docs[:RETRIEVAL_FINAL_K]  (=20)

        CANDIDATE_TOP_K 와 FINAL_K 가 모두 20 이므로 **append 된 형제는 전부 잘렸다**.
        의도했던 "본문과 표를 함께 보기" 는 한 번도 작동하지 않았고, 그러면서
        페이지당 Chroma 조회 2회는 계속 발생해 지연만 늘었다.

        【수정 방식】
        형제를 **부모 바로 뒤에 끼워 넣는다**. 그래야 상위권 문서의 표가 하위권
        문서를 밀어내고 컨텍스트에 들어간다. score 는 고정값 대신 부모 점수에
        0.9 를 곱해 물려받는다 — 형제의 가치는 부모의 관련성에 종속되므로
        모든 형제에게 같은 0.5 를 주면 상위권 부모의 표와 하위권 부모의 표가
        구분되지 않는다.

        모드는 환경변수로 전환한다 (기본 inline):
            inline  부모 뒤 삽입 (수정된 동작)
            append  기존 동작 (뒤에 붙임 — 사실상 무동작)
            off     형제를 가져오지 않음 (Chroma 조회 생략, 가장 빠름)
        """
        mode = _CO_RETRIEVAL_MODE
        if mode == "off" or not retrieved:
            return retrieved

        existing_contents = {d.content for d in retrieved}
        page_keys: set[tuple[str, int]] = set()
        for d in retrieved:
            doc_id = d.metadata.get("doc_id")
            page = d.metadata.get("page_unit_index")
            parent = d.metadata.get("parent_page_index")
            if doc_id and page is not None:
                page_keys.add((doc_id, page))
            if doc_id and parent is not None:
                page_keys.add((doc_id, parent))
        if not page_keys:
            return retrieved

        # (doc_id, page) -> 형제 Document 목록
        siblings: dict[tuple[str, int], list[LCDocument]] = {}
        try:
            collection = self._get_chroma()._collection
            for doc_id, page in page_keys:
                found: list[LCDocument] = []
                for where in (
                    {"$and": [{"doc_id": doc_id}, {"page_unit_index": page}]},
                    {"$and": [{"doc_id": doc_id}, {"parent_page_index": page}]},
                ):
                    res = collection.get(where=where, include=["documents", "metadatas"])
                    for txt, meta in zip(res.get("documents") or [], res.get("metadatas") or []):
                        if txt and txt not in existing_contents:
                            found.append(LCDocument(page_content=txt, metadata=meta or {}))
                if found:
                    siblings[(doc_id, page)] = found
        except Exception as e:
            print(f"[CO_RETRIEVAL] failed (non-fatal): {e}")
            return retrieved

        if not siblings:
            return retrieved

        if mode == "append":
            out = list(retrieved)
            for group in siblings.values():
                for lc in group:
                    if lc.page_content not in existing_contents:
                        existing_contents.add(lc.page_content)
                        out.append(self._to_document(lc, score=0.5))
            return out

        # inline — 부모 바로 뒤에 삽입
        out: list[Document] = []
        for d in retrieved:
            out.append(d)
            doc_id = d.metadata.get("doc_id")
            for page in (d.metadata.get("page_unit_index"), d.metadata.get("parent_page_index")):
                if doc_id is None or page is None:
                    continue
                for lc in siblings.get((doc_id, page), []):
                    if lc.page_content in existing_contents:
                        continue
                    existing_contents.add(lc.page_content)
                    out.append(self._to_document(lc, score=d.score * 0.9))
        return out

    # ────────────────────────────────────────────────────────
    # Phase G: doc_topic / qtype boost
    # ────────────────────────────────────────────────────────

    def boost_by_query_context(
        self,
        docs: list[Document],
        query_doc_topic: str | None = None,
    ) -> list[Document]:
        """query의 doc_topic과 일치하는 chunks score boost.

        chunk_question_types boost는 폐기 — Q&A cache retrieval로 대체.
        """
        if not docs:
            return docs
        if not query_doc_topic:
            return docs
        boosted: list[Document] = []
        for d in docs:
            score = d.score
            if d.metadata.get("doc_topic") == query_doc_topic:
                score = min(1.0, score + 0.15)
            boosted.append(Document(
                content=d.content,
                source=d.source,
                score=score,
                metadata=d.metadata,
            ))
        boosted.sort(key=lambda x: x.score, reverse=True)
        return boosted
