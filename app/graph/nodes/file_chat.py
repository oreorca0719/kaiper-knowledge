from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.config import get_llm
from app.core.history_utils import (
    HISTORY_MAX_MESSAGES,
    filter_history_by_relevance as _filter_history_by_relevance,
)
from app.graph.states.state import GraphState
from app.security.content_sanitizer import sanitize


def file_chat_node(state: GraphState) -> Dict[str, Any]:
    """
    첨부 파일 Q&A 전용 노드 (file_chat).
    state.file_context를 시스템 프롬프트에 포함하여 LLM 직접 호출. 사내 RAG 검색 없음.
    """
    print("--- [NODE] File Chat ---")

    user_input = (state.get("input_data") or "").strip()
    raw_history = list(state.get("messages") or [])[-HISTORY_MAX_MESSAGES:]
    chat_history = _filter_history_by_relevance(raw_history, user_input)
    file_context = (state.get("file_context") or "").strip()
    file_context_name = (state.get("file_context_name") or "첨부 파일").strip()

    if not file_context:
        return {
            "messages": [
                HumanMessage(content=user_input),
                AIMessage(content="첨부된 파일을 찾지 못했습니다. 파일을 먼저 업로드해 주세요."),
            ],
            "citations_used": [],
            "clarification_count": 0,
        }

    safe_context = sanitize(file_context, source=f"file_chat:{file_context_name}")

    system_content = (
        f"당신은 사용자가 첨부한 파일 '{file_context_name}'을(를) 분석하는 AI 어시스턴트입니다.\n"
        "아래 [파일 내용]만을 근거로 사용자의 질문에 답하세요.\n"
        "파일에 없는 내용은 '파일에서 확인할 수 없습니다'라고 명시하세요.\n"
        "정중한 비즈니스 어투로 답하고, 필요 시 불릿(•)을 활용하세요.\n\n"
        f"[파일 내용]\n{safe_context}"
    )

    messages = (
        [SystemMessage(content=system_content)]
        + chat_history
        + [HumanMessage(content=user_input)]
    )

    try:
        response = get_llm().invoke(messages)
        finish_reason = (
            (getattr(response, "response_metadata", None) or {}).get("finish_reason", "") or ""
        ).upper()
        if finish_reason == "MAX_TOKENS":
            response = AIMessage(
                content="요청하신 내용이 처리 가능한 텍스트 범위를 초과하였습니다. 질문을 더 구체적으로 나누어 입력해 주세요."
            )
    except Exception as e:
        print(f"[File Chat] LLM 호출 실패: {type(e).__name__}: {e}")
        response = AIMessage(content="파일 분석 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.")

    return {
        "messages": [HumanMessage(content=user_input), response],
        "citations_used": [],
        "clarification_count": 0,
    }
