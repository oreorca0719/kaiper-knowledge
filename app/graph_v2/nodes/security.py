"""
Security gate node — v1의 4계층(Layer 1~4)을 입력 단계 통합 구현.

기존 (v1):
  Layer 1: input_guard (임베딩 + 슬라이딩 윈도우)
  Layer 2: task_router → rejection (실제 미구현)
  Layer 3: content_sanitizer (Retriever 내부에서 호출 — graph_v2에서도 동일)
  Layer 4: output_validator (응답 sensitive 패턴)

신규 (v2):
  pre_input_check: Layer 1 통합 — 임베딩 1회 + 슬라이딩 윈도우
  output_validate: Layer 4를 별도 함수로 (generator/reflection 후 호출 가능)

이 파일은 pre_input_check만 담당. content_sanitize는 retrievers/에서.
"""
from __future__ import annotations

from typing import List

from langchain_core.messages import HumanMessage

from app.core.config import get_embeddings
from app.graph_v2.states.state import GraphState
from app.security.injection_detector import (
    InjectionCheckUnavailable,
    check_strict as injection_check,
)


def security_gate_node(state: GraphState) -> dict:
    """입력 단계 보안 검사. v1 Layer 1과 동일 메커니즘. **fail-closed**.

    - 임베딩 1회 계산 → state.input_embedding 캐시
    - 임베딩 기반 injection 패턴 매칭 + 슬라이딩 윈도우
    - 차단 시 security_blocked=True

    세 가지 결과를 구분한다:
      security:pass                    정상 통과
      security:blocked(injection)      인젝션 판정 → 차단
      security:blocked(detector_down)  판정 불가 → 차단 (통과 아님)

    마지막 항목이 핵심이다. LLM(Anthropic)과 임베딩(Gemini)이 서로 다른
    프로바이더로 분리된 뒤로는 임베딩만 단독 장애가 가능하다. 이때 통과시키면
    인젝션 방어가 꺼진 채 Claude 가 정상 응답을 만들어내므로 장애가
    관측되지 않는다. 판정 불가는 차단으로 처리한다.
    """
    user_input = (state.input_data or "").strip()
    if not user_input:
        return {
            "security_blocked": False,
            "decision_path": ["security:empty_input"],
        }

    # 임베딩 1회 (downstream 재사용). 실패 시 빈 값으로 두고 판정 단계에서 처리.
    input_embedding: list[float] = []
    embed_error: str = ""
    try:
        input_embedding = get_embeddings().embed_query(user_input)
    except Exception as e:
        embed_error = f"{type(e).__name__}: {e}"
        print(f"[SECURITY] embedding failed: {embed_error}")
        input_embedding = []

    # 이전 HumanMessage 턴 (슬라이딩 윈도우용)
    recent_turns: List[str] = [
        msg.content for msg in (state.messages or [])
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str)
    ][-3:]

    try:
        blocked = injection_check(user_input, recent_turns, input_embedding=input_embedding)
    except InjectionCheckUnavailable as e:
        reason = embed_error or str(e)
        print(f"[SECURITY] detector unavailable → 요청 차단: {reason}")
        return {
            "security_blocked": True,
            "security_reason": "detector_unavailable",
            "input_embedding": [],
            "decision_path": [f"security:blocked(detector_down:{reason[:60]})"],
        }

    if blocked:
        return {
            "security_blocked": True,
            "security_reason": "injection_detected",
            "input_embedding": input_embedding,
            "decision_path": ["security:blocked(injection)"],
        }

    return {
        "security_blocked": False,
        "input_embedding": input_embedding,
        "decision_path": ["security:pass"],
    }


def route_after_security(state: GraphState) -> str:
    return "rejected" if state.security_blocked else "router"
