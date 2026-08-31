"""
Router (Planner) node — 질문 분류 + 검색 도구 선택 (FR-101, FR-102, FR-103, FR-104).

분류 (LLM structured output, FR-104):
  - "no_retrieval": 인사·기능 안내·메타 질의 (검색 불필요)
  - "single_retrieval": 단일 검색으로 충분
  - "multi_hop_retrieval": sub-question 분해 후 다단계
  - "ai_guide": 시스템 안내 요청
  - "file_chat": file_context 있을 때
  - "rejected": 명백히 범위 밖

추가 (Lever 5 학습 통합):
  - question_type: exact_phrase / numerical / list_n / fill_blank / reasoning / comparison

Phase A 학습 반영:
  - file_chat 분류 시 명시적 file 키워드 검증 (false-positive 차단)
  - 짧은 사실 질의가 ai_guide로 가지 않음
"""
from __future__ import annotations

import json
import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import get_llm
from app.core.history_utils import extract_text_content
from app.graph_v2.states.state import GraphState


_FILE_MENTION_PATTERNS = [
    "파일", "첨부", "업로드", "올린", "올려놓은", "업로드한", "첨부한",
    "file", "attach", "upload", "pdf", "엑셀", "워드", "ppt", "pptx", "docx", "xlsx",
]


def _has_file_mention(text: str) -> bool:
    t = (text or "").lower()
    return any(p.lower() in t for p in _FILE_MENTION_PATTERNS)


_ROUTER_SYSTEM_PROMPT = """당신은 사내 AI 어시스턴트의 라우터입니다. 사용자 질문을 분석해 다음을 결정합니다:

1. routing_decision (검색 필요성):
   - "no_retrieval": 인사·메타 질의 (예: "안녕하세요", "오늘 뭘 할 수 있어요?")
   - "single_retrieval": 단일 검색으로 충분한 사실 질의 (대부분의 사실 회수)
   - "multi_hop_retrieval": 여러 단계 검색·추론 필요 (예: "A 부서와 B 부서의 차이", "X와 Y의 관계")
   - "ai_guide": 시스템 기능 안내 요청 (예: "어떤 기능이 있어?", "사용법 알려줘")
   - "file_chat": 첨부 파일 분석 (질문에 "파일", "첨부" 같은 명시적 단어 있을 때만)

2. question_type (답변 형식):
   - "exact_phrase": 정확 문구 회수 (예: "슬로건이 뭐야?")
   - "numerical": 수치·시간 회수 (예: "회원 수는?", "12:30 같은 시간")
   - "list_n": N개 항목 (예: "3가지", "4개", "각 단계", "5요소")
   - "fill_blank": 빈칸 채우기 (예: "___에 들어갈 단어")
   - "reasoning": 종합·추론 (예: "왜", "어떤 의미", "이유")
   - "comparison": 비교 (예: "차이", "vs", "대비")

3. sub_questions (multi_hop_retrieval일 때만, 최대 5개):
   - 원 질문을 더 작은 검색 가능 단위로 분해

【중요】
- 짧은 사실 질의 ("C안의 리스크는?", "회원수는?")는 ai_guide가 아니라 single_retrieval
- 명시적 파일 언급("파일", "첨부", "업로드") 없으면 file_chat 금지
- 대부분의 사내 질의는 single_retrieval

JSON으로만 출력. 다른 텍스트 금지:
{
  "routing_decision": "...",
  "question_type": "...",
  "sub_questions": [],
  "reason": "한 줄 이유"
}
"""


def _call_router_llm(user_input: str) -> dict:
    """LLM 호출 → structured JSON. 실패 시 default."""
    try:
        resp = get_llm().invoke([
            SystemMessage(content=_ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=f"사용자 질문: {user_input}"),
        ])
        raw = extract_text_content(resp.content)
        # Fenced code block 제거
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip().rstrip("`").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[ROUTER] LLM call failed (non-fatal): {e}")
        return {}


def router_node(state: GraphState) -> dict:
    """Planner: routing_decision + question_type + sub_questions 결정."""
    user_input = (state.input_data or "").strip()
    if not user_input:
        return {
            "routing_decision": "no_retrieval",
            "question_type": "reasoning",
            "decision_path": ["router:empty_input"],
        }

    # original_input anchor 초기화 (첫 진입 시 1회. replan으로 회귀해도 보존됨)
    original_init: dict = {}
    if not (state.original_input or "").strip():
        original_init["original_input"] = user_input

    parsed = _call_router_llm(user_input)
    decision = (parsed.get("routing_decision") or "single_retrieval").strip()
    qtype = (parsed.get("question_type") or "reasoning").strip()
    subs = parsed.get("sub_questions") or []
    if not isinstance(subs, list):
        subs = []
    subs = [str(s).strip() for s in subs if str(s).strip()][:5]  # FR-103 max 5

    # ─── Phase A-1 학습 반영: file_chat 분류 검증 ───
    # 명시적 file 키워드 없는데 file_chat으로 분류 → single_retrieval로 회귀
    file_present = bool((state.file_context or "").strip())
    if decision == "file_chat":
        if not file_present and not _has_file_mention(user_input):
            decision = "single_retrieval"

    # ─── Phase A-2 학습 반영: 짧은 사실 질의가 ai_guide로 가지 않음 ───
    # ai_guide 분류 케이스 중 명시적 안내 키워드 없으면 single_retrieval 회귀
    _GUIDE_TRIGGERS = ["안녕", "반가워", "도움말", "help", "소개", "기능 안내", "사용법", "어떤 기능", "뭘 할 수 있"]
    if decision == "ai_guide":
        if not any(g in user_input for g in _GUIDE_TRIGGERS):
            decision = "single_retrieval"

    # ─── Multi-hop이지만 sub_questions 없으면 single로 강등 ───
    if decision == "multi_hop_retrieval" and not subs:
        decision = "single_retrieval"

    # ─── 알려지지 않은 decision은 single로 ───
    valid = {"no_retrieval", "single_retrieval", "multi_hop_retrieval", "ai_guide", "file_chat", "rejected"}
    if decision not in valid:
        decision = "single_retrieval"

    valid_qtypes = {"exact_phrase", "numerical", "list_n", "fill_blank", "reasoning", "comparison"}
    if qtype not in valid_qtypes:
        qtype = "reasoning"

    return {
        "routing_decision": decision,
        "question_type": qtype,
        "sub_questions": subs,
        "llm_call_count": state.llm_call_count + 1,
        "decision_path": [f"router:{decision}/{qtype}"],
        **original_init,
    }


def route_by_decision(state: GraphState) -> str:
    d = state.routing_decision
    if d == "rejected":
        return "rejected"
    if d == "no_retrieval":
        return "no_retrieval_answer"
    if d == "ai_guide":
        return "ai_guide"
    if d == "file_chat":
        return "file_chat"
    return "retrieve"
