from __future__ import annotations

from typing import TypedDict, Annotated, Sequence, List, Optional, Dict, Any
from operator import add

from langchain_core.messages import BaseMessage


class GraphState(TypedDict, total=False):
    # ── 기본 입출력 ──────────────────────────────────────────
    input_data: str
    input_embedding: List[float]       # ← 임베딩 캐시: 사용자 입력 벡터화 결과
    task_type: str
    task_args: Dict[str, Any]
    messages: Annotated[Sequence[BaseMessage], add]
    citations_used: List[Dict[str, Any]]

    # ── 파일 ────────────────────────────────────────────────
    file_context: Optional[str]
    file_context_name: Optional[str]

    # ── 순환 / 상태 제어 ─────────────────────────────────────
    retry_count: int            # knowledge search 재시도 횟수
    clarification_count: int    # clarification 발동 횟수 (루프 방어용)
    interrupt_type: str         # "clarification"
    pending_confirm_msg: str    # 슬롯 수집 후 확인 메시지 (clarification_confirm_node 용)

    # ── 트레이스 ─────────────────────────────────────────────
    trace_id: str
