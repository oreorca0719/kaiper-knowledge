"""
Reflection node — 답변 자기검증 (FR-403, FR-404, FR-405).

검증 항목:
  - groundedness: 답변이 검색 근거에서 도출됐는가
  - relevance: 답변이 원 질문에 답하는가
  - hallucination_risk: 환각 의심 여부
  - 추가 (Phase D 학습 — 명확한 trigger만):
    * 수치/entity in chunks 검증 (regex 기반, 빠름)
    * no_info_misclaim 검출

Reflection 실패 + replan_iterations < 1 → router로 회귀.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import get_llm
from app.core.history_utils import extract_text_content
from app.graph_v2.states.state import GraphState, VerificationResult
from app.graph_v2.retrievers.base import Document


# ────────────────────────────────────────────────────────────
# Regex 기반 가벼운 검증 (LLM 호출 없음)
# ────────────────────────────────────────────────────────────

_NUMBER_PATTERN = re.compile(
    r"(\d{1,2}:\d{2}|\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s*(?:명|억|만원|만\s*원|억\s*원|시간|분|일|차|곳|개|단계|자|회|건|개월|년|％|%|원))?)"
)


def _extract_numbers(text: str) -> set[str]:
    return {re.sub(r"\s+", "", m.group(1)) for m in _NUMBER_PATTERN.finditer(text or "")}


def _check_facts_in_chunks(answer: str, docs: list[Document]) -> list[str]:
    """답변의 수치가 chunks에 존재하는지 정규식 검증. 부재 시 issue."""
    if not docs or not answer:
        return []
    chunks_norm = re.sub(r"\s+", "", "\n".join(d.content or "" for d in docs))
    issues = []
    for num in _extract_numbers(answer):
        if num not in chunks_norm:
            issues.append(f"수치 '{num}' 검색 결과에 없음")
            if len(issues) >= 5:
                break
    return issues


_NO_INFO_PHRASES = [
    "찾을 수 없습니다", "확인할 수 없습니다", "확인되지 않", "포함되어 있지 않",
    "정보가 없", "찾을 수 없다",
]


def _check_no_info_misclaim(answer: str, docs: list[Document]) -> bool:
    """답변이 '정보 없음'인데 chunks가 충분히 있으면 의심."""
    if len(docs) < 3:
        return False
    short = len(answer) < 200
    has_no_info = any(p in answer for p in _NO_INFO_PHRASES)
    return short and has_no_info


# ────────────────────────────────────────────────────────────
# list_n 전용 검증 (regex/substring 기반, LLM 호출 없음)
# ────────────────────────────────────────────────────────────

# 질문에서 N 명시 추출: "3가지", "4개", "5단계", "두 가지", "세 곳" 등
_N_HANGUL_MAP = {"한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5, "여섯": 6, "일곱": 7,
                 "여덟": 8, "아홉": 9, "열": 10}
_N_PATTERNS = [
    re.compile(r"(\d{1,2})\s*(?:가지|개|단계|곳|명|항목|요소|축|영역)"),
    re.compile(r"(한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*(?:가지|개|단계|곳|명|항목|요소|축|영역)"),
]


def _extract_expected_n(question: str) -> int | None:
    """질문에서 명시된 N 추출. 없으면 None."""
    if not question:
        return None
    for pat in _N_PATTERNS:
        m = pat.search(question)
        if m:
            v = m.group(1)
            if v.isdigit():
                return int(v)
            return _N_HANGUL_MAP.get(v)
    return None


def _extract_list_items(answer: str) -> list[str]:
    """답변에서 list 항목 추출 (불릿/줄바꿈 단위)."""
    if not answer:
        return []
    items = []
    for line in answer.split("\n"):
        line = line.strip()
        if not line:
            continue
        # 불릿/번호 prefix 제거
        line = re.sub(r"^[\s•·\-\*\d.)\]\[]+", "", line).strip()
        # citation 제거
        line = re.sub(r"\s*\[\d{1,3}\]\s*", " ", line).strip()
        if 2 <= len(line) <= 200:
            items.append(line)
    return items


def _check_list_n_consistency(question: str, answer: str, docs: list[Document]) -> list[str]:
    """list_n 답변 검증.

    1. expected_n 명시되어 있으면 항목 수 일치 확인
    2. 각 항목이 chunks 어딘가에 verbatim 또는 substantial substring 매칭되는지 확인 (환각 방지)
    """
    if not docs or not answer:
        return []
    issues: list[str] = []
    items = _extract_list_items(answer)
    if not items:
        return []

    # 1. expected_n 검증
    n_required = _extract_expected_n(question)
    if n_required and abs(len(items) - n_required) > 0:
        issues.append(f"list_n 개수 불일치: 질문 {n_required}개 요구, 답변 {len(items)}개")

    # 2. 각 항목이 chunks 에 등장하는지 (공백 무시 substring 매칭)
    chunks_norm = re.sub(r"\s+", "", "\n".join(d.content or "" for d in docs)).lower()
    for item in items[:15]:  # 너무 많으면 처음 15개만
        item_norm = re.sub(r"\s+", "", item).lower()
        # 너무 짧은 항목 (≤2자) 은 스킵 (false positive 위험)
        if len(item_norm) < 3:
            continue
        if item_norm not in chunks_norm:
            # 핵심 단어 (공백/특수문자 제외 6자 이상) 라도 매칭되는지 확인
            core = re.sub(r"[^가-힣a-zA-Z0-9]", "", item)[:30]
            if len(core) >= 4 and core.lower() not in chunks_norm:
                issues.append(f"list_n 항목 '{item[:30]}' 검색 결과에 없음 (환각 의심)")
                if len(issues) >= 5:
                    break
    return issues


# ────────────────────────────────────────────────────────────
# LLM-as-judge 검증 (groundedness, relevance, hallucination)
# ────────────────────────────────────────────────────────────

_REFLECTION_PROMPT = """당신은 사내 RAG 시스템의 답변 자기검증 평가자입니다.

【검증 절차 — 반드시 순서대로】
1. 답변에서 핵심 사실(fact) 또는 entity를 모두 추출하시오 (3~5개).
2. 각 fact에 대해 chunks 어느 부분에 해당 의미가 등장하는지 짚으시오.
   - 의역·요약·축약·바꿔쓰기도 의미가 보존되면 supported로 간주.
   - 답변의 인용 [N]은 chunks의 [N]번을 가리킴 (1-based).
   - 답변의 인용 인덱스가 잘못됐어도 다른 chunk에 해당 fact가 있으면 supported.
3. 위 분석을 토대로 점수 산출.

【평가 항목】 (각 0.0~1.0)
1. groundedness: 답변의 fact 중 chunks에서 매칭된 비율
   - 100% 매칭 → 1.0
   - 75% 매칭 → 0.75
   - 일부만 → 0.3~0.6
   - 전혀 매칭 안 됨 → 0.0
2. relevance: 답변이 원 질문에 답하는가 (0=무관, 1=정확히 답)
3. hallucination_risk: chunks에 명백히 없고 사실로 검증할 수 없는 새 fact 도입 여부
   - 단, supportive sub-fact (질문 외 추가 맥락)는 hallucination이 아님

【출력】 JSON만:
{
  "matched_facts": [{"fact": "...", "chunk_idx": 1, "evidence": "..."}],
  "unmatched_facts": ["chunks에 없는 fact"],
  "groundedness": 0.0~1.0,
  "relevance": 0.0~1.0,
  "hallucination_risk": 0.0~1.0,
  "passed": true/false,
  "reason": "한 줄 요약"
}

passed 기준: groundedness >= 0.6 AND relevance >= 0.7 AND hallucination_risk <= 0.4
"""


def _call_reflection_llm(question: str, answer: str, docs: list[Document]) -> dict:
    if not answer or not docs:
        return {}
    # 1-based 인덱싱 (generator citation [1], [2]…와 정렬)
    # chunks 절단 400→1500자 (Gemini Flash 3 context 충분)
    chunks_text = "\n".join(
        f"[{i}] {(d.content or '')[:1500]}"
        for i, d in enumerate(docs[:5], start=1)
    )
    user_content = (
        f"[질문]\n{question}\n\n"
        f"[검색 결과 chunks]\n{chunks_text}\n\n"
        f"[시스템 답변]\n{answer[:1500]}"
    )
    try:
        resp = get_llm().invoke([
            SystemMessage(content=_REFLECTION_PROMPT),
            HumanMessage(content=user_content),
        ])
        raw = extract_text_content(resp.content)
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip().rstrip("`").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[REFLECTION] LLM call failed (non-fatal): {e}")
        return {}


def reflection_node(state: GraphState) -> dict:
    """답변 검증."""
    answer = state.answer or ""
    docs: list[Document] = state.retrieved_docs or []

    # Regex-based 빠른 검증
    fact_issues = _check_facts_in_chunks(answer, docs)
    no_info_misclaim = _check_no_info_misclaim(answer, docs)

    # list_n 전용 검증 (해당 qtype 만)
    list_n_issues: list[str] = []
    if (state.question_type or "").lower() == "list_n":
        list_n_issues = _check_list_n_consistency(state.input_data, answer, docs)

    # LLM-as-judge (chunks가 있을 때만)
    if docs and answer and not answer.startswith("관련 사내 문서를 찾을 수 없"):
        judge = _call_reflection_llm(state.input_data, answer, docs)
        groundedness = float(judge.get("groundedness", 1.0))
        relevance = float(judge.get("relevance", 1.0))
        hallucination_risk = float(judge.get("hallucination_risk", 0.0))
        llm_passed = bool(judge.get("passed", True))
        llm_calls_added = 1
    else:
        # docs 없거나 명시적 "정보 없음" 응답이면 LLM 호출 생략
        groundedness, relevance, hallucination_risk, llm_passed = 1.0, 1.0, 0.0, True
        llm_calls_added = 0

    # 종합 passed (list_n 환각·개수 불일치는 strict 처리)
    passed = (
        llm_passed
        and not no_info_misclaim
        and len(fact_issues) <= 2  # 수치 issue 최대 2개까지 관용
        and len(list_n_issues) == 0  # list_n issue 는 0 만 허용
    )

    fact_issues = list(fact_issues) + list(list_n_issues)
    verif = VerificationResult(
        groundedness=groundedness,
        relevance=relevance,
        hallucination_risk=hallucination_risk,
        fact_issues=fact_issues,
        no_info_misclaim=no_info_misclaim,
        passed=passed,
    )

    label = (
        f"reflect:pass(g={groundedness:.2f},r={relevance:.2f},h={hallucination_risk:.2f})"
        if passed else
        f"reflect:fail(facts={len(fact_issues)},miscalim={no_info_misclaim},g={groundedness:.2f})"
    )

    return {
        "verification": verif,
        "llm_call_count": state.llm_call_count + llm_calls_added,
        "decision_path": [label],
    }


def route_after_reflection(state: GraphState) -> str:
    """fail 시 replan 결정.

    효율화: groundedness가 합리적 수준(>=0.5)인데 fail이면 같은 chunks로
    재 retrieval해도 결과 거의 동일. 무용한 replan 비용 차단 → end.
    완전히 grounded되지 않은(g<0.5) 케이스만 replan으로.
    """
    if not state.verification:
        return "end"
    v = state.verification
    if not v.passed and state.replan_iterations < 1:
        # g >= 0.5: 답이 어느 정도 chunks 기반. replan 무의미 → end
        if v.groundedness >= 0.5:
            return "end"
        return "replan"
    return "end"
