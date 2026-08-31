from __future__ import annotations

import re
import threading
from typing import Any, Dict, List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.config import (
    get_embeddings, get_llm,
    CHROMA_DB_PATH, CHROMA_COLLECTION,
    RETRIEVAL_MIN_RELEVANCE, RETRIEVAL_TOP_K,
)
from app.core.history_utils import (
    extract_text_content,
)
from app.graph.states.state import GraphState
from app.security.content_sanitizer import sanitize_docs
from app.security.output_validator import validate as validate_output

# ── Chroma 싱글톤 ────────────────────────────────────────────
_chroma_instance: Chroma | None = None
_chroma_lock = threading.Lock()

# ── BM25 싱글톤 ─────────────────────────────────────────────
_bm25_index = None
_bm25_docs: List[Document] = []
_bm25_lock = threading.Lock()

# quality_check 기준
_QUALITY_MIN_DOCS = 1
_QUALITY_MIN_SCORE = float(RETRIEVAL_MIN_RELEVANCE)
_MAX_RETRY = 2

# 하이브리드 검색에서 시맨틱 후보 배수 (RRF 풀 크기)
_HYBRID_FETCH_MULTIPLIER = 4


def invalidate_bm25_cache() -> None:
    """문서 재인제스트 후 BM25 인덱스를 초기화합니다."""
    global _bm25_index, _bm25_docs
    with _bm25_lock:
        _bm25_index = None
        _bm25_docs = []


def _get_chroma() -> Chroma:
    global _chroma_instance
    if _chroma_instance is None:
        with _chroma_lock:
            if _chroma_instance is None:
                _chroma_instance = Chroma(
                    persist_directory=CHROMA_DB_PATH,
                    embedding_function=get_embeddings(),
                    collection_name=CHROMA_COLLECTION,
                )
    return _chroma_instance


def _tokenize(text: str) -> List[str]:
    """공백·특수문자 기준 토큰 분리 (한국어 포함)."""
    return re.findall(r"[가-힣a-zA-Z0-9]+", text.lower())


def _get_bm25():
    """BM25 인덱스 싱글톤. ChromaDB 전체 문서로 구축. 빈 corpus면 (None, []) 반환 후 캐시 안 함."""
    global _bm25_index, _bm25_docs
    if _bm25_index is not None:
        return _bm25_index, _bm25_docs

    with _bm25_lock:
        if _bm25_index is not None:
            return _bm25_index, _bm25_docs

        from rank_bm25 import BM25Okapi  # type: ignore

        collection = _get_chroma()._collection
        result = collection.get(include=["documents", "metadatas"])
        raw_docs = result.get("documents") or []
        raw_metas = result.get("metadatas") or []

        bm25_docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(raw_docs, raw_metas)
            if text
        ]

        # ChromaDB가 비어있으면 인덱스를 캐시하지 않고 (None, []) 반환.
        # 다음 호출 시 다시 시도하여 인제스트 완료 후 자동 복구.
        if not bm25_docs:
            return None, []

        tokenized = [_tokenize(d.page_content) for d in bm25_docs]
        _bm25_index = BM25Okapi(tokenized)
        _bm25_docs = bm25_docs

    return _bm25_index, _bm25_docs


def _search_bm25(query: str, k: int) -> List[Document]:
    bm25, docs = _get_bm25()
    if bm25 is None:
        return []
    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [docs[i] for i in top_indices if scores[i] > 0]


def _reciprocal_rank_fusion(
    semantic_docs: List[Document],
    bm25_docs: List[Document],
    k: int,
    rrf_k: int = 60,
) -> List[Document]:
    """두 랭킹 결과를 RRF로 병합. 문서 식별은 page_content 기준."""
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for rank, doc in enumerate(semantic_docs):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
        doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys[:k]]


def _search_hybrid(query: str, k: int = RETRIEVAL_TOP_K) -> List[Document]:
    """시맨틱 + BM25 하이브리드 검색 (RRF 병합).

    시맨틱 결과 양과 무관하게 항상 BM25 결과와 RRF로 병합한다.
    키워드 정확 매칭(고유명사·약어)과 의미적 유사도를 동시에 활용해
    검색 정밀도를 높이는 것이 RRF 하이브리드의 본 의도이며,
    이전 'semantic 결과가 k개 이상이면 BM25 생략' 최적화는 README 명세
    ('하이브리드+RRF')와 어긋나 BM25를 사실상 무력화시켰다.
    """
    fetch_k = k * _HYBRID_FETCH_MULTIPLIER
    vectorstore = _get_chroma()
    semantic_docs = vectorstore.similarity_search(query, k=fetch_k)
    bm25_docs = _search_bm25(query, k=fetch_k)
    return _reciprocal_rank_fusion(semantic_docs, bm25_docs, k=k)


def _format_docs(docs: List[Document]) -> str:
    if not docs:
        return ""
    blocks: List[str] = []
    for i, d in enumerate(docs, start=1):
        text = (d.page_content or "").strip()
        md = getattr(d, "metadata", {}) or {}
        title = md.get("title") or md.get("file_name") or "문서"
        page_num = md.get("page_number")
        location = f" (p.{page_num})" if page_num else ""
        blocks.append(f"[{i}] {title}{location}\n{text[:1200]}")
    return "\n\n".join(blocks)


# ── 노드 1: search_node ──────────────────────────────────────

def search_node(state: GraphState) -> Dict[str, Any]:
    """ChromaDB 벡터 검색."""
    print("--- [NODE] Search ---")

    user_input = (state.get("input_data") or "").strip()

    docs = _search_hybrid(user_input, k=RETRIEVAL_TOP_K)
    docs = sanitize_docs(docs, source="rag")

    return {"task_args": {**(state.get("task_args") or {}), "search_docs": docs, "search_query": user_input}}


# ── 노드 2: quality_check_node ───────────────────────────────

def quality_check_node(state: GraphState) -> Dict[str, Any]:
    """
    검색 결과 품질 판단 — 규칙 기반 (LLM 없음).
    - 결과 없거나 최고 score < threshold → retry
    - retry_count >= _MAX_RETRY → 강제 ok (무한루프 방지)
    """
    print("--- [NODE] Quality Check ---")

    task_args = state.get("task_args") or {}
    docs: List[Document] = task_args.get("search_docs") or []
    retry_count = state.get("retry_count") or 0

    if retry_count >= _MAX_RETRY:
        ok = True   # 한계 도달 → 무한루프 방지, 강제 통과
    else:
        ok = bool(docs)  # docs 없으면 rewrite, 있으면 통과

    return {"task_args": {**task_args, "quality_ok": ok}}


def route_after_quality(state: GraphState) -> str:
    task_args = state.get("task_args") or {}
    if task_args.get("quality_ok"):
        return "answer"
    retry_count = state.get("retry_count") or 0
    if retry_count >= _MAX_RETRY:
        return "answer"
    return "rewrite"


# ── 노드 3: rewrite_node ─────────────────────────────────────

def rewrite_node(state: GraphState) -> Dict[str, Any]:
    """LLM으로 검색 쿼리를 재작성하고 retry_count를 증가."""
    print("--- [NODE] Rewrite ---")

    user_input = (state.get("input_data") or "").strip()
    retry_count = (state.get("retry_count") or 0) + 1

    try:
        response = get_llm().invoke([
            SystemMessage(content=(
                "사용자의 검색 쿼리를 사내 문서 검색에 더 적합하게 재작성하세요.\n"
                "더 일반적인 용어를 사용하고 핵심 키워드만 남기세요.\n"
                "재작성된 쿼리만 출력하세요."
            )),
            HumanMessage(content=user_input),
        ])
        rewritten = extract_text_content(response.content)
        if rewritten and rewritten != user_input:
            new_input = rewritten
        else:
            new_input = user_input
    except Exception:
        new_input = user_input

    return {"input_data": new_input, "retry_count": retry_count}


# ── 노드 4: answer_node ──────────────────────────────────────

def answer_node(state: GraphState) -> Dict[str, Any]:
    """검색 결과를 바탕으로 LLM 답변 생성."""
    print("--- [NODE] Answer ---")

    user_input = (state.get("input_data") or "").strip()
    task_args = state.get("task_args") or {}
    docs: List[Document] = task_args.get("search_docs") or []

    if not docs:
        final_message = AIMessage(
            content="관련 사내 문서를 찾을 수 없습니다. 다른 키워드로 검색해 보시거나 담당 부서에 문의해 주세요."
        )
    else:
        search_result = _format_docs(docs)
        system_content = (
            "당신은 Kaiper AI입니다. 아래 사내 문서 검색 결과를 바탕으로 사용자 질문에 답하세요.\n\n"
            "【응답 원칙】\n"
            "• 검색 결과 중 사용자 질문과 관련 있는 내용만 선별하여 답하세요.\n"
            "• 관련 있는 내용이 단 하나의 문서에만 있더라도 그것을 바탕으로 답하세요.\n"
            "• 검색 결과 전체에 관련 내용이 전혀 없을 때만 '관련 사내 문서를 찾을 수 없습니다'라고 답하세요.\n"
            "• 사용자 질문 중심으로 핵심만 요약하세요 (원문 나열 금지)\n"
            "• 수치·날짜·고유명사는 원문 그대로, 출처는 [1][2] 형식으로 문장 끝에 표시\n"
            "• 정중한 비즈니스 어투, 불렛(•) 활용\n\n"
            f"【검색 결과】\n{search_result}"
        )
        messages = (
            [SystemMessage(content=system_content)]
            + [HumanMessage(content=user_input)]
        )
        response = get_llm().invoke(messages)
        text_for_validate = extract_text_content(response.content)
        if not text_for_validate.strip():
            final_message = AIMessage(
                content="관련 사내 문서를 찾을 수 없습니다. 다른 키워드로 검색해 보시거나 담당 부서에 문의해 주세요."
            )
        else:
            is_valid, safe_content = validate_output(text_for_validate)
            final_message = response if is_valid else AIMessage(content=safe_content)

    citations_used = []
    for i, doc in enumerate(docs, start=1):
        md = getattr(doc, "metadata", {}) or {}
        raw_path = md.get("display_source") or md.get("path") or ""
        citations_used.append({
            "id": i,
            "title": md.get("title") or md.get("file_name") or f"문서 {i}",
            "snippet": (doc.page_content or "")[:200],
            "path": raw_path,
        })

    return {
        "messages": [HumanMessage(content=user_input), final_message],
        "citations_used": citations_used,
        "retry_count": 0,
        "clarification_count": 0,
    }
