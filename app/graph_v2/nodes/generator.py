"""
Generator node — 검색 결과 기반 답변 생성 + Citation 부착 (FR-401, FR-402).

핵심 설계 (Phase B 학습 회복):
  - question_type별 프롬프트 분기 — list_n 누락 방지, verbatim 강제 분기 적용
  - 모든 사실 진술에 [N] citation 강제 (FR-402)
  - relevant + partial 문서만 컨텍스트로 사용 (FR-401)
"""
from __future__ import annotations

import re
import urllib.parse
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def _decode_urls_in_answer(text: str) -> str:
    """답변 안의 URL-encoded segment(@%XX… 또는 %XX…)를 한글로 디코드.

    chunk 원문에 인코딩된 URL이 있을 때 사용자 가독성 향상.
    """
    if not text or "%" not in text:
        return text

    def _decode(m: re.Match) -> str:
        seg = m.group(0)
        try:
            decoded = urllib.parse.unquote(seg)
            # 디코딩 후 사람이 읽을 수 있는 한글/문자가 나타나면 사용
            if decoded != seg and re.search(r"[가-힣a-zA-Z]", decoded):
                return decoded
        except Exception:
            pass
        return seg

    # @로 시작하는 채널/사용자명 또는 일반 %xx 시퀀스 (3개 이상 연속)
    return re.sub(r"@?(?:%[0-9A-Fa-f]{2}){3,}", _decode, text)

from app.core.config import get_llm
from app.core.history_utils import extract_text_content
from app.graph_v2.states.state import Citation, GraphState
from app.graph_v2.retrievers.base import Document


# ────────────────────────────────────────────────────────────
# question_type별 프롬프트
# ────────────────────────────────────────────────────────────

_COMMON_PRINCIPLES = """【핵심 원칙】
1. 수치·고유명사·인용 문구는 검색 결과 그대로 (verbatim, paraphrase 금지).
2. 질문이 묻는 것만 답하세요. 관련 배경·추가 설명은 절대 추가 금지.
3. 모든 사실 진술 뒤에 [N] citation 부착. 부착할 수 없는 진술은 작성 금지.
4. 검색 결과에 답이 없으면 "관련 사내 문서를 찾을 수 없습니다."
5. 답변 시작에 '~는 다음과 같습니다:' 같은 서두 금지 — 바로 답.

【답변 작성 전 4단계 자기검증】 (반드시 수행)
1. **질문 entity 식별**: 질문이 묻는 정확한 대상(예: "체크리스트 8번 항목" / "Day 3 산출물 중 발표" / "프로젝트 수행 단계의 평가 주체")을 한 줄로 명시.
2. **chunk entity 매칭**: 검색 결과 chunks 중 질문 entity와 정확히 매칭되는 chunk만 골라내시오.
   - 비슷하지만 다른 entity의 chunk는 사용 금지 (예: 질문이 "체크리스트 8번"인데 "체크리스트 슬라이드 활용법"을 답하지 말 것).
   - 같은 단어가 등장해도 다른 맥락이면 무관.
3. **답변 facet 일치**: 답변이 질문의 정확한 facet을 답하는가?
   - 질문이 "구분"을 물으면 "구분"만, "의미"를 물으면 "의미"만, "표기"를 물으면 "표기"만 답.
   - 다른 facet(특성, 이유, 배경)을 답하면 오답으로 간주됨.
4. **불필요 정보 제거**: 답변 작성 후 정답에 직접 기여 안 하는 모든 문장·항목을 제거.
   - 예: 정답이 "성과 기반 보증"이면 "프로젝트 성공에 연동된 운영 수익 모델"은 paraphrase로 부족, verbatim 필수.
   - "추가 정보를 친절히 알려주는 것"이 답을 망친다 — **간결성 > 풍부함**.

【extra info 금지 사례】
- 질문 "12시간 내 빠른 매칭 강조 플랫폼은?" / 정답 "원티드긱스" / X 답변: "원티드긱스 [1]. 그 외 AI 자동 매칭 시스템 [1], 1:1 매니징 [2]…" → AI 매칭/매니징 부분이 extra info.
- 질문 "구분과 의미는?" / 정답 "구분: 협의, 의미: 강사 더 효과적 방식" / X 답변: 정답 + "협의는 강사와 조율 사항을 뜻합니다" → 마지막 문장 불필요.
"""

_PROMPT_BY_TYPE = {
    "exact_phrase": _COMMON_PRINCIPLES + """
【exact_phrase 응답 규칙】
- 검색 결과의 정확한 문구만. 한 줄로.
- 예: "대한민국 최초 금융 IT 매칭 플랫폼 [1]"
""",

    "numerical": _COMMON_PRINCIPLES + """
【numerical 응답 규칙】
- 수치·시간만 답. 단위 포함 (예: 41만+, 09:00~12:30).
- 한 줄로. 부수 설명 금지.
- 예: "09:00~12:30 [1]"
""",

    "list_n": _COMMON_PRINCIPLES + """
【list_n 응답 규칙】 (★ 가장 주의 ★)
- 질문이 'N가지/N개/N단계'로 명시한 경우 정확히 N개 항목.
- 각 항목을 새 줄로 분리, 검색 결과 표현 그대로.
- 누락 절대 금지 — 답하기 전 모든 항목을 검색 결과에서 찾았는지 확인.
- **다른 슬라이드/페이지의 다른 주제 항목 절대 추가 금지.** 같은 chunk 안에 있는 동일 주제 항목만 사용.
- 예: "• 차별성 [1]\\n• 베네핏 [1]\\n• 행동 유발 [1]"
- N이 명시되지 않은 경우 검색 결과에 있는 모든 항목을 나열.

【list_n soft completion — N개 다 못 찾은 경우】
- 질문이 N개 요구하지만 검색 결과에서 명백히 N개 모두 찾을 수 없으면:
  1. 찾은 M개만 verbatim으로 답.
  2. 답변 마지막에 한 줄로 명시: "(검색 결과에서 {N-M}개 항목은 확인되지 않음)"
- 절대 hallucination으로 빈 자리를 채우지 말 것.
- 예: 정답 5개 요구, chunk에 1개만 있을 때
  → "• 최저의 진입장벽 [1]\\n(검색 결과에서 4개 항목은 확인되지 않음)"

【list_n 자기검증 4단계】 (답변 작성 직전 반드시 수행)
1. 질문에서 N 추출 (예: "3가지", "4개", "각 단계" → 모든 항목)
2. 검색 결과에서 답이 될 N 개 항목 후보를 verbatim 으로 나열
3. 각 항목이 검색 결과에 그대로 등장하는가? (paraphrase·번역 금지)
4. 다른 슬라이드의 무관한 항목 (배경 설명, 다른 주제) 이 섞이지 않았는가?
   ↳ 의심되면 제거. 정답 N개 보다 적게 답하는 것이 잘못된 항목 추가보다 낫다.
""",

    "fill_blank": _COMMON_PRINCIPLES + """
【fill_blank 응답 규칙】
- 빈칸에 들어갈 단어/구만 답. 전체 문장 절대 금지.
- 예: "충격 [1]" (질문이 'Day 1 = 첫인상 + ___ 충격'일 때 → 답: "이게 되네!")
""",

    "comparison": _COMMON_PRINCIPLES + """
【comparison 응답 규칙】
- 비교 대상 entity별로 명시.
- 검색 결과에 있는 차이만. 추측 금지.
- 예: "• 일반 매칭: 단발성 연결, 자원 제공 [1]\\n• 그레이트프로: 지속 가능한 동반, 가치 존중 [1]"
""",

    "reasoning": _COMMON_PRINCIPLES + """
【reasoning 응답 규칙】
- 검색 결과에 있는 사실로만 추론.
- 짧게 답하되 핵심 근거에 [N] 부착.
- 결론 → 근거 순서.
""",
}


def _format_docs_for_context(docs: list[Document]) -> str:
    if not docs:
        return ""
    blocks = []
    for i, d in enumerate(docs, start=1):
        title = d.metadata.get("title", d.source)
        loc = d.metadata.get("location", "")
        blocks.append(f"[{i}] {title}{(' ' + loc) if loc else ''}\n{(d.content or '')[:1500]}")
    return "\n\n".join(blocks)


def _extract_cited_ids(text: str) -> set[int]:
    return {int(x) for x in re.findall(r"\[(\d{1,3})\]", text or "")}


def _build_citations(docs: list[Document], cited_ids: set[int]) -> list[Citation]:
    out = []
    for i, d in enumerate(docs, start=1):
        if i not in cited_ids:
            continue
        out.append(Citation(
            id=i,
            doc_id=d.metadata.get("doc_id", d.source),
            snippet=(d.content or ""),
            score=d.score,
            location=d.metadata.get("location", ""),
        ))
    return out


def generator_node(state: GraphState) -> dict:
    """답변 생성 + citations 부착."""
    docs: list[Document] = state.retrieved_docs or []
    user_input = (state.input_data or "").strip()

    # docs 없으면 명시적 거절. 시스템 로그는 두 케이스 분리:
    #  (a) 처음부터 retrieve가 결과를 못 가져옴 — 사내 문서에 정말 없음 가능성
    #  (b) rewrite 3회 후에도 못 찾음 — 시스템 회복 실패
    # 사용자 응답은 동일.
    if not docs:
        msg = "관련 사내 문서를 찾을 수 없습니다. 다른 키워드로 검색해 보시거나 담당 부서에 문의해 주세요."
        path = state.decision_path or []
        exhausted = any("exhausted_after_" in p for p in path)
        if exhausted:
            print("[GENERATOR] no_docs (retrieve exhausted after rewrite limit)")
            log_label = "generate:no_docs(exhausted)"
        else:
            print("[GENERATOR] no_docs (initial empty retrieve)")
            log_label = "generate:no_docs(initial)"
        return {
            "answer": msg,
            "citations": [],
            "messages": [HumanMessage(content=user_input), AIMessage(content=msg)],
            "decision_path": [log_label],
        }

    qtype = state.question_type or "reasoning"
    sys_prompt = _PROMPT_BY_TYPE.get(qtype, _PROMPT_BY_TYPE["reasoning"])
    sys_content = f"{sys_prompt}\n\n【검색 결과】\n{_format_docs_for_context(docs)}"

    try:
        resp = get_llm().invoke([
            SystemMessage(content=sys_content),
            HumanMessage(content=user_input),
        ])
        answer = extract_text_content(resp.content)
    except Exception as e:
        print(f"[GENERATOR] LLM call failed: {e}")
        answer = "응답 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."

    if not answer:
        answer = "관련 사내 문서를 찾을 수 없습니다. 다른 키워드로 검색해 보시거나 담당 부서에 문의해 주세요."

    # URL-encoded segment 디코딩 (가독성)
    answer = _decode_urls_in_answer(answer)

    cited_ids = _extract_cited_ids(answer)
    citations = _build_citations(docs, cited_ids)

    return {
        "answer": answer,
        "citations": citations,
        "messages": [HumanMessage(content=user_input), AIMessage(content=answer)],
        "llm_call_count": state.llm_call_count + 1,
        "decision_path": [f"generate:{qtype}({len(citations)}cit)"],
    }
