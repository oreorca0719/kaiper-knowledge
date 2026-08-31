"""
LLM 2차 의도 분류기.

semantic router가 'unknown'을 반환한 경우에만 실행됩니다.
LLM_FALLBACK_ENABLED=0 으로 비활성화 가능.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, Tuple

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import get_llm, LLM_FALLBACK_ENABLED, LLM_FALLBACK_TIMEOUT_SEC
from app.core.history_utils import extract_text_content

_ALLOWED = {"knowledge_search", "ai_guide", "file_chat", "detail_search"}
_CONFIDENCE_MIN = 0.7  # 이 값 미만이면 unknown으로 처리

_PROMPT = ChatPromptTemplate.from_template(
    "당신은 사용자 메시지의 의도를 분류하는 분류기입니다.\n"
    "아래 태스크 중 하나를 선택하고, 확신도(0.0~1.0)를 함께 반환하세요.\n\n"
    "태스크 종류:\n"
    "- knowledge_search: 사내 문서·정보 검색, 부서 문의처 조회, 사내 자료 탐색\n"
    "- detail_search: 직전 검색 결과에 대한 후속 심화 질의\n"
    "- ai_guide: 인사, AI 자기소개, 기능 안내, 도움말 요청\n"
    "- file_chat: 첨부 파일 분석, 업로드한 파일 내용 질문\n"
    "- unknown: 위 태스크와 무관하거나 의도를 파악할 수 없는 경우\n\n"
    "규칙:\n"
    "- 확신도가 0.7 미만이면 반드시 unknown을 반환하세요.\n"
    "- 짧거나 대명사만 있는 입력('이거', '저거', '그거', '해줘')은 unknown으로 처리하세요.\n\n"
    "반드시 아래 형식으로만 출력:\n"
    "{{\"task\": \"<태스크>\", \"confidence\": <0.0~1.0>, \"reason\": \"<한 줄 이유>\"}}\n\n"
    "사용자 메시지: {user_input}"
)


def llm_intent_fallback(user_input: str) -> Tuple[str, Dict[str, Any]]:
    """
    LLM으로 의도를 재분류합니다.

    Returns:
        (task, debug_dict) — task는 _ALLOWED 중 하나 또는 'unknown'
    """
    if LLM_FALLBACK_ENABLED.strip() in ("0", "false", "False", "NO", "no"):
        return "unknown", {"invoked": False, "reason": "disabled"}

    timeout_sec = LLM_FALLBACK_TIMEOUT_SEC

    def _call() -> Dict[str, Any]:
        response = (_PROMPT | get_llm()).invoke({"user_input": user_input})
        content = extract_text_content(response.content)
        return JsonOutputParser().parse(content) or {}

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call)
            result = future.result(timeout=timeout_sec)

        raw_task = (result.get("task") or "").strip().lower()
        confidence = float(result.get("confidence", 0.0))

        if raw_task not in _ALLOWED or confidence < _CONFIDENCE_MIN:
            task = "unknown"
        else:
            task = raw_task

        debug: Dict[str, Any] = {
            "invoked": True,
            "source": "llm_fallback",
            "raw_task": raw_task,
            "confidence": confidence,
            "final_task": task,
            "reason": result.get("reason", ""),
        }
        return task, debug

    except FuturesTimeoutError:
        return "unknown", {"invoked": True, "source": "llm_fallback", "error": "timeout"}
    except Exception as e:
        return "unknown", {"invoked": True, "source": "llm_fallback", "error": str(e)}