"""
v2 GraphState — Pydantic v2 BaseModel + LangGraph Reducer.

설계 원칙:
1. **타입 안전** — 필드별 명시적 타입. dict 결합 (`task_args`) 없음.
2. **Reducer 명시** — Annotated로 누적 동작 명확화 (messages 추가, decision_path 누적 등).
3. **노드 격리** — 각 노드는 자기 필드만 업데이트. 다른 노드 필드는 read-only.
4. **Subgraph state 분리** — RetrievalState, GenerationState는 별도 파일에 정의.

LangGraph 0.2+ Pydantic 지원 패턴.
"""
from __future__ import annotations

from typing import Annotated, Any, Optional, Sequence
from operator import add

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# ────────────────────────────────────────────────────────────
# Question type — 답변 형식 분기에 사용 (Lever 5와 통합)
# ────────────────────────────────────────────────────────────

QuestionType = str  # "exact_phrase" | "numerical" | "list_n" | "fill_blank" | "reasoning" | "comparison"


# ────────────────────────────────────────────────────────────
# Routing decision — Planner 출력
# ────────────────────────────────────────────────────────────

RoutingDecision = str  # "no_retrieval" | "single_retrieval" | "multi_hop_retrieval" | "ai_guide" | "file_chat" | "rejected"


# ────────────────────────────────────────────────────────────
# Citation — 답변에 부착되는 출처 식별자
# ────────────────────────────────────────────────────────────

class Citation(BaseModel):
    id: int
    doc_id: str             # 출처 문서 ID (Retriever에서 부여)
    snippet: str            # 답변 근거가 된 chunk 전체 내용
    score: float = 0.0      # 0~1 정규화 (FR-204)
    location: str = ""      # "Slide 3", "Page 2" 등


# ────────────────────────────────────────────────────────────
# Verification result — Reflection 출력
# ────────────────────────────────────────────────────────────

class VerificationResult(BaseModel):
    groundedness: float = 1.0       # 답변이 검색 근거에서 도출됐는가 (0~1)
    relevance: float = 1.0          # 답변이 원 질문에 답하는가
    hallucination_risk: float = 0.0 # 환각 의심 여부 (0~1, 높을수록 위험)
    fact_issues: list[str] = Field(default_factory=list)
    no_info_misclaim: bool = False  # "찾을 수 없다" 답변인데 실제로 chunks에 답이 있는 경우
    passed: bool = True             # 모든 검증 통과 여부

    def needs_replan(self) -> bool:
        return not self.passed or self.hallucination_risk > 0.5


# ────────────────────────────────────────────────────────────
# Main GraphState
# ────────────────────────────────────────────────────────────

class GraphState(BaseModel):
    """v2 main graph의 공유 상태.

    Reducer 가이드:
    - `messages`: add_messages (LangGraph 표준 — append + dedupe)
    - `decision_path`: add (노드 진입 trace 누적)
    - 그 외: last-write-wins (한 노드가 update하면 그 값 유지)
    """

    # ─── 입력 ───────────────────────────────────────────────
    input_data: str = ""
    original_input: str = ""        # rewrite 시에도 불변 (drift 방지 anchor, router에서 1회 초기화)
    input_embedding: Optional[list[float]] = None  # 한 번 계산 후 재사용
    trace_id: str = ""

    # ─── 라우팅 ─────────────────────────────────────────────
    routing_decision: RoutingDecision = ""
    question_type: QuestionType = "reasoning"
    sub_questions: list[str] = Field(default_factory=list)  # multi_hop 분해 결과 (FR-103, 최대 5)

    # ─── 검색 결과 ──────────────────────────────────────────
    retrieved_docs: list[Any] = Field(default_factory=list)   # list[Document], 모듈 import 순환 회피
    citations: list[Citation] = Field(default_factory=list)   # generator가 부착

    # ─── 답변 ───────────────────────────────────────────────
    answer: str = ""
    verification: Optional[VerificationResult] = None

    # ─── 제어 ───────────────────────────────────────────────
    retrieval_iterations: int = 0     # FR-303 (기본 3, 절대 상한 5)
    replan_iterations: int = 0        # FR-404 (최대 1)
    llm_call_count: int = 0           # NFR-010 (기본 10 상한)

    # ─── 트레이스 (NFR-020) ─────────────────────────────────
    decision_path: Annotated[list[str], add] = Field(default_factory=list)
    # 예: ["security:pass", "router:single_retrieval", "retrieve:hybrid",
    #     "grade:partial", "rewrite:1", "retrieve:hybrid", "grade:pass",
    #     "generate", "reflect:pass"]

    # ─── 메시지 (LangGraph checkpointer 호환) ──────────────
    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(default_factory=list)

    # ─── 첨부 파일 (file_chat 경로) ────────────────────────
    file_context: Optional[str] = None
    file_context_name: Optional[str] = None

    # ─── 보안 ───────────────────────────────────────────────
    security_blocked: bool = False
    security_reason: str = ""

    class Config:
        arbitrary_types_allowed = True   # BaseMessage 등 LangChain 타입 허용
