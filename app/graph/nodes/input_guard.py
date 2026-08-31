from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import HumanMessage

from app.core.config import get_embeddings
from app.graph.states.state import GraphState
from app.security.injection_detector import check as injection_check


def input_guard_node(state: GraphState) -> Dict[str, Any]:
    """
    그래프 진입점 보안 필터.
    - 사용자 입력을 임베딩 (1회만)
    - 임베딩 기반 injection 감지
    - 감지 시 task_type = 'injection' 설정 → rejection_node로 라우팅
    - input_embedding을 State에 저장 → 이후 노드에서 재사용
    """
    print("--- [NODE] Input Guard ---")

    user_input = (state.get("input_data") or "").strip()

    if not user_input:
        return {"task_type": "knowledge_search"}

    # ✅ 임베딩 1회 계산
    input_embedding: List[float] = []
    try:
        embeddings = get_embeddings()
        input_embedding = embeddings.embed_query(user_input)
    except Exception as e:
        print(f"[INPUT_GUARD] embedding failed: {e}")
        input_embedding = []

    # 이전 HumanMessage 턴 수집 (슬라이딩 윈도우용)
    recent_turns = [
        msg.content for msg in (state.get("messages") or [])
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str)
    ][-3:]

    # ✅ input_embedding 전달
    if injection_check(user_input, recent_turns, input_embedding=input_embedding):
        return {"task_type": "injection"}

    # ✅ State에 임베딩 저장
    return {"input_embedding": input_embedding}
