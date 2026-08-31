from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.config import get_llm, RETRIEVAL_TOP_K
from app.core.history_utils import extract_text_content
from app.graph.nodes.knowledge_search import _get_chroma, _search_hybrid
from app.graph.states.state import GraphState
from app.security.content_sanitizer import sanitize_docs

# 상세 검색 시 확장 k
_DETAIL_TOP_K = RETRIEVAL_TOP_K * 3


def _reconstruct_query(prev_human: str, prev_ai: str, current_input: str) -> tuple[str, str]:
    """
    직전 Q&A와 현재 입력을 바탕으로 검색 쿼리와 참조 문서명을 재구성.
    반환: (재구성된 쿼리, 참조 문서명 또는 "")
    """
    try:
        response = get_llm().invoke([
            SystemMessage(content=(
                "이전 대화와 현재 요청을 분석하여 아래 두 가지를 추출하세요.\n"
                "반드시 다음 형식으로만 출력하세요 (다른 텍스트 금지):\n"
                "QUERY: <검색에 사용할 구체적인 쿼리>\n"
                "SOURCE: <이전 AI 답변에서 언급된 문서명, 없으면 빈 값>\n\n"
                "규칙:\n"
                "• QUERY는 현재 요청('자세히', '더 알려줘' 등)을 이전 주제에 결합한 구체적 질문으로 작성\n"
                "• SOURCE는 이전 AI 답변의 출처 표기([1][2] 등)에서 가장 관련 높은 문서명만 추출\n"
                "• 문서명을 특정할 수 없으면 SOURCE는 비워두세요"
            )),
            HumanMessage(content=(
                f"[이전 사용자 질문]\n{prev_human}\n\n"
                f"[이전 AI 답변]\n{prev_ai[:1000]}\n\n"
                f"[현재 요청]\n{current_input}"
            )),
        ])
        content = extract_text_content(response.content)
        query = current_input
        source = ""
        for line in content.splitlines():
            if line.startswith("QUERY:"):
                q = line[len("QUERY:"):].strip()
                if q:
                    query = q
            elif line.startswith("SOURCE:"):
                source = line[len("SOURCE:"):].strip()
        return query, source
    except Exception:
        return current_input, ""


def _search_with_filter(query: str, source_name: str) -> List[Document]:
    """
    source_name이 있으면 해당 문서 내에서 확장 k로 검색.
    없으면 일반 하이브리드 검색.
    """
    if not source_name:
        return _search_hybrid(query, k=_DETAIL_TOP_K)

    vectorstore = _get_chroma()
    try:
        results = vectorstore.similarity_search(
            query,
            k=_DETAIL_TOP_K,
            filter={"display_source": source_name},
        )
        # 필터 결과가 없으면 필터 없이 재검색
        if not results:
            return _search_hybrid(query, k=_DETAIL_TOP_K)
        return results
    except Exception:
        return _search_hybrid(query, k=_DETAIL_TOP_K)


def detail_search_node(state: GraphState) -> Dict[str, Any]:
    """직전 Q&A 컨텍스트 기반 상세 검색 노드."""
    print("--- [NODE] Detail Search ---")

    user_input = (state.get("input_data") or "").strip()
    task_args = state.get("task_args") or {}

    # 직전 Human/AI 메시지 쌍 추출
    messages = list(state.get("messages") or [])
    prev_human = ""
    prev_ai = ""
    for msg in reversed(messages):
        if not prev_ai and isinstance(msg, AIMessage):
            prev_ai = str(msg.content)
        elif prev_ai and not prev_human and isinstance(msg, HumanMessage):
            prev_human = str(msg.content)
            break

    # 직전 컨텍스트가 없으면 일반 하이브리드 검색으로 fallback
    if not prev_human or not prev_ai:
        docs = _search_hybrid(user_input, k=_DETAIL_TOP_K)
        docs = sanitize_docs(docs, source="rag")
        return {
            "task_args": {**task_args, "search_docs": docs, "search_query": user_input},
        }

    # LLM으로 쿼리 + 참조 문서명 재구성
    reconstructed_query, source_name = _reconstruct_query(prev_human, prev_ai, user_input)

    # 참조 문서명이 있으면 해당 문서 범위 내 확장 검색, 없으면 일반 확장 검색
    docs = _search_with_filter(reconstructed_query, source_name)
    docs = sanitize_docs(docs, source="rag")

    return {
        "task_args": {
            **task_args,
            "search_docs": docs,
            "search_query": reconstructed_query,
        },
    }
