from __future__ import annotations

from typing import Any, Dict, List

from langgraph.types import interrupt

from app.graph.states.state import GraphState


_SLOT_QUESTIONS: Dict[str, str] = {
    "file_context": "첨부 파일이 필요합니다. 먼저 파일을 업로드해 주세요.",
}


_NEGATIVE_HINTS = ["아니요", "아니오", "no", "취소", "관둬", "그만", "안할래", "안 할래", "안할게", "안 할게"]
_POSITIVE_HINTS = ["예", "네", "yes", "ok", "okay", "확인", "진행", "좋아", "응", "맞아"]


def _is_negative(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    return any(t == h or t.startswith(h) for h in _NEGATIVE_HINTS)


def clarification_slot_node(state: GraphState) -> Dict[str, Any]:
    """
    누락된 슬롯이 있을 때 사용자에게 질문 → interrupt → 응답으로 슬롯을 채움.

    흐름:
    1. clarification_count >= 2 → 무한 루프 방어, knowledge_search 강제 fallback
    2. missing_slots 비어 있으면 confirm 단계로 전이
    3. 첫 번째 slot에 대해 interrupt 질문 송출
    4. 응답이 음성/공백이면 종료, 정상이면 task_args에 슬롯 채움 후 task_router 재진입
    """
    print("--- [NODE] Clarification Slot ---")

    task_args = state.get("task_args") or {}
    missing_slots: List[str] = list(task_args.get("missing_slots") or [])
    clarification_count = (state.get("clarification_count") or 0)

    # 루프 방어
    if clarification_count >= 2:
        return {
            "task_type": "knowledge_search",
            "task_args": {**task_args, "missing_slots": []},
            "clarification_count": 0,
        }

    # 슬롯이 이미 채워졌으면 confirm 단계로
    if not missing_slots:
        return {
            "task_type": "clarification_confirm",
            "pending_confirm_msg": "입력하신 내용으로 진행할까요? '예' 또는 '아니요'로 답변해 주세요.",
        }

    slot = missing_slots[0]
    question = _SLOT_QUESTIONS.get(slot, f"'{slot}' 정보를 알려주세요.")

    raw = interrupt({"type": "clarification", "message": question})
    answer = (raw or "").strip() if isinstance(raw, str) else ""

    if not answer or _is_negative(answer):
        return {
            "task_type": "knowledge_search",
            "task_args": {**task_args, "missing_slots": []},
            "clarification_count": 0,
        }

    new_task_args = dict(task_args)
    new_task_args[slot] = answer
    new_task_args["missing_slots"] = [s for s in missing_slots if s != slot]

    combined_input = f"{state.get('input_data') or ''} {answer}".strip()

    return {
        "input_data": combined_input,
        "task_args": new_task_args,
        "clarification_count": clarification_count + 1,
    }


def clarification_confirm_node(state: GraphState) -> Dict[str, Any]:
    """
    슬롯이 모두 채워진 후 사용자에게 진행 여부 확인 → interrupt → 응답 처리.
    부정 응답이면 knowledge_search로 fallback. 그 외에는 task_router 재진입.
    """
    print("--- [NODE] Clarification Confirm ---")

    msg = (state.get("pending_confirm_msg") or "진행할까요?").strip()

    raw = interrupt({"type": "clarification", "message": msg})
    answer = (raw or "").strip() if isinstance(raw, str) else ""

    if _is_negative(answer):
        return {
            "task_type": "knowledge_search",
            "pending_confirm_msg": "",
            "clarification_count": 0,
        }

    return {
        "pending_confirm_msg": "",
        "clarification_count": 0,
    }


def route_after_clarification(state: GraphState) -> str:
    """clarification_slot_node 실행 후 다음 분기."""
    task_type = (state.get("task_type") or "").strip()

    if task_type == "knowledge_search":
        return "knowledge_search"
    if task_type == "clarification_confirm":
        return "clarification_confirm"
    if task_type == "rejection":
        return "rejection"
    return "task_router"
