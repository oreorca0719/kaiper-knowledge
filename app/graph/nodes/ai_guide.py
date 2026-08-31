from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.config import get_llm
from app.core.history_utils import (
    HISTORY_MAX_MESSAGES,
    extract_text_content,
    filter_history_by_relevance as _filter_history_by_relevance,
)
from app.graph.states.state import GraphState


_AI_GUIDE_SYSTEM_PROMPT = (
    "당신은 Kaiper AI 사내 어시스턴트입니다. 정중한 비즈니스 어투로 응답하세요.\n\n"
    "본 어시스턴트는 다음 3가지 기능을 제공합니다.\n"
    "1. 사내 문서 검색: 사내 정책·매뉴얼·회의록·기획 산출물 등을 자연어로 검색하고 출처와 함께 답합니다.\n"
    "2. 심화 검색: 직전 검색 결과에 대한 후속 질의를 받아 더 구체적인 정보를 제공합니다.\n"
    "3. 파일 분석: 업로드한 파일(PDF, DOCX, PPTX, XLSX, TXT, MD)의 내용을 분석해 답합니다.\n\n"
    "사용자가 인사·기능 안내·자기소개를 요청하면 위 3가지 기능을 간결하게 안내하고, "
    "구체적인 사내 정보 질의는 검색 기능으로 처리됨을 알려주세요. 불릿(•) 기호를 활용해 "
    "가독성 있게 정리하세요. 사내 업무와 무관한 일반 잡담에는 응답하지 마시고 사내 업무 "
    "관련 질문을 유도하세요."
)


def ai_guide_node(state: GraphState) -> Dict[str, Any]:
    """AI 자기소개 / 기능 안내 전용 노드. 도구 없이 LLM 직접 호출."""
    print("--- [NODE] AI Guide ---")

    user_input = (state.get("input_data") or "").strip()
    raw_history = list(state.get("messages") or [])[-HISTORY_MAX_MESSAGES:]
    chat_history = _filter_history_by_relevance(raw_history, user_input)

    messages = (
        [SystemMessage(content=_AI_GUIDE_SYSTEM_PROMPT)]
        + chat_history
        + [HumanMessage(content=user_input)]
    )

    try:
        response = get_llm().invoke(messages)
        response_text = extract_text_content(response.content)
        if not response_text.strip():
            response = AIMessage(content="기능 안내를 생성하지 못했습니다. 잠시 후 다시 시도해 주세요.")
    except Exception as e:
        print(f"[AI Guide] LLM 호출 실패: {type(e).__name__}: {e}")
        response = AIMessage(content="기능 안내를 생성하지 못했습니다. 잠시 후 다시 시도해 주세요.")

    return {
        "messages": [HumanMessage(content=user_input), response],
        "citations_used": [],
        "clarification_count": 0,
    }
