"""
Chroma + BM25 Hybrid Retriever — Retriever Protocol 구현체.

기존 `app.graph.nodes.knowledge_search`의 `_search_hybrid`를 Protocol에 wrap.
v1과 동일한 검색 로직 + score 정규화 + Document 변환.
"""
from __future__ import annotations

import asyncio
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
_RRF_K = 60


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
        """Retrieve된 chunks의 (doc_id, page_unit_index)와 같은 모든 chunks를 추가 fetch.

        예: chunk_19a (본문)이 retrieve되면 → chunk_19b (표)도 함께.
        같은 페이지의 본문·표를 함께 보면 retrieval 정밀도 + 답변 컨텍스트 풍부.
        """
        if not retrieved:
            return retrieved

        # 이미 가진 chunks의 (doc_id, page) 키 집합
        page_keys: set[tuple[str, int]] = set()
        existing_contents = {d.content for d in retrieved}
        for d in retrieved:
            doc_id = d.metadata.get("doc_id")
            page = d.metadata.get("page_unit_index")
            parent = d.metadata.get("parent_page_index")
            if doc_id and page is not None:
                page_keys.add((doc_id, page))
            # 표면 본문 페이지도 함께 fetch
            if doc_id and parent is not None:
                page_keys.add((doc_id, parent))

        if not page_keys:
            return retrieved

        # ChromaDB에서 같은 page_keys의 모든 chunks fetch
        expanded = list(retrieved)
        try:
            collection = self._get_chroma()._collection
            # ChromaDB get은 단일 where만 — page_keys별로 반복
            for doc_id, page in page_keys:
                # 1) page_unit_index가 page인 chunks
                res = collection.get(
                    where={"$and": [{"doc_id": doc_id}, {"page_unit_index": page}]},
                    include=["documents", "metadatas"],
                )
                # 2) parent_page_index가 page인 chunks (같은 페이지의 표)
                res2 = collection.get(
                    where={"$and": [{"doc_id": doc_id}, {"parent_page_index": page}]},
                    include=["documents", "metadatas"],
                )
                for r in (res, res2):
                    docs_t = r.get("documents") or []
                    metas = r.get("metadatas") or []
                    for txt, meta in zip(docs_t, metas):
                        if txt and txt not in existing_contents:
                            existing_contents.add(txt)
                            expanded.append(self._to_document(
                                LCDocument(page_content=txt, metadata=meta or {}),
                                score=0.5,  # co-retrieval은 중간 score 부여
                            ))
        except Exception as e:
            print(f"[CO_RETRIEVAL] failed (non-fatal): {e}")

        return expanded

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
