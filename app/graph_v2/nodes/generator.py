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

# 【이 프롬프트가 축소된 경위】
#
# 원래는 "답변 작성 전 4단계 자기검증 (반드시 수행)" 이라는 명시적 절차와
# 오답 사례 나열이 붙어 있었다. gemini-3-flash-preview 시절, 모델이 스스로
# 검증하지 못하던 때 이를 프롬프트로 보상한 것이다.
#
# claude-sonnet-5 는 그 절차를 **답변에 그대로 출력했다** (실측):
#     **질문 entity**: GitLab DAP(인터넷 연결 가능시)에서 지원 가능한 모델 목록
#     **해당 모델들**:
#     - Claude 4 Sonnet [14] ...
# "한 줄로 명시" 라는 지시를 문자 그대로 따른 결과다. 내부 절차가 사용자에게
# 노출되는 것은 결함이며, 신형 모델은 이런 검증을 지시 없이도 수행한다.
#
# 규칙은 **결과물의 성질**로만 남기고, 사고 절차 지시는 제거한다.
#
# 【규칙 2 를 완화한 경위 — 283문항 실측】
# 원래 규칙 2 는 "질문이 묻는 것만 답한다. 배경 설명·추가 정보는 넣지 않는다" 였다.
# 장황한 답변을 막으려는 의도였는데, 수치를 해석하는 데 필요한 기준까지 잘라냈다:
#
#   Q "파이프라인 실패 자동 수정으로 건당 얼마나 절약?"
#     전: "1.5시간 이상 절감"      정답: "시간당 $75 기준 약 1.5시간"
#   Q "정보보호 통제 이행 여부를 얼마나 자주 점검?"
#     전: "반기 1회"               정답: "반기 1회 ... 사내 정보보호위원회에 보고"
#
# 283문항 판정 분포 (짝 비교):
#     기존   correct 259  partial 11  wrong 10  refused 3
#     완화   correct 264  partial 10  wrong  6  refused 3
#
# 정답률 차이는 통계적으로 유의하지 않다 (불일치 4 대 9, p=0.267).
# 채택 근거는 두 가지다:
#   (1) **위험한 실패인 wrong 이 10 -> 6 으로 줄었다.** 다른 사실을 단정하는 실패는
#       지식 어시스턴트에서 가장 문제되는 유형이다.
#   (2) 퇴행이 없다. 잃은 4건은 전부 partial 이고 wrong 으로 떨어진 건은 없다.
# question_type 별로는 list_n 이 78.0% -> 85.4% (불일치 0 대 3, 잃은 문항 없음).
#
# "정확도가 향상됐다" 고 단정할 근거는 아니다. 방향이 일관되고 손해가 없어서 택했다.
_COMMON_PRINCIPLES = """【답변 규칙】
1. 수치·고유명사·인용 문구는 검색 결과 그대로 (verbatim, paraphrase 금지).
2. 질문이 묻는 것에 답한다. 다만 그 답을 해석하는 데 필요한 조건·단위·
   기준·전제는 검색 결과에 있는 대로 함께 밝힌다
   (예: "1.5시간" 이 아니라 "시간당 $75 기준 1.5시간").
   질문과 무관한 배경 설명은 넣지 않는다.
3. 모든 사실 진술 뒤에 [N] citation 을 붙인다. 붙일 수 없는 진술은 쓰지 않는다.
4. 검색 결과에 답이 없으면 "관련 사내 문서를 찾을 수 없습니다."
   비슷하지만 다른 대상의 chunk 로 대신 답하지 않는다.
5. 답변 시작에 '~는 다음과 같습니다:' 같은 서두를 쓰지 않는다 — 바로 답한다.

【출력 형식】
- 답변 본문만 출력한다.
- 판단 근거·검토 과정·"질문 entity" 같은 내부 절차는 출력하지 않는다.
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
【list_n 응답 규칙】
- 질문이 'N가지/N개/N단계'로 명시하면 정확히 N개 항목.
  N이 명시되지 않았으면 검색 결과에 있는 항목을 모두 나열한다.
- 각 항목은 새 줄로 분리하고 검색 결과 표현 그대로 쓴다 (paraphrase·번역 금지).
- 같은 주제의 항목만 모은다. 다른 슬라이드의 다른 주제 항목을 섞지 않는다.
- 예: "• 차별성 [1]\\n• 베네핏 [1]\\n• 행동 유발 [1]"

【N개를 다 찾지 못한 경우】
- 빈 자리를 추측으로 채우지 않는다.
- 찾은 M개만 쓰고 마지막 줄에 "(검색 결과에서 {N-M}개 항목은 확인되지 않음)" 을 붙인다.
- 예: "• 최저의 진입장벽 [1]\\n(검색 결과에서 4개 항목은 확인되지 않음)"
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


# ── 과잉 거부 보정 ────────────────────────────────────────
#
# 【현상】 문서를 충분히 확보하고도 "찾을 수 없습니다" 로 끝내는 경우가 있다.
#   실측 (질의 3회 반복, 모두 거부):
#     Q "GitLab Duo Agent Platform은 어떤 팀들을 통합하나요?"
#       retrieve:3q->20docs
#       grade:pass(20rel/20kept/20total, max=0.95, cov=full)   <- 검색은 완벽
#       generate:list_n(0cit)                                  <- 그런데 거부
#       reflect:fail(miscalim=True)                            <- reflection 은 감지함
#     같은 내용을 길게 물으면("...통합하는 하나의 워크플로우를 제공하나요?") 정답.
#
# 【원인】 답변 규칙 4번("검색 결과에 답이 없으면 찾을 수 없다고 하라. 비슷하지만
#   다른 대상의 chunk 로 대신 답하지 말라")이 짧고 모호한 질문에서 과하게 작동한다.
#   질문이 짧으면 모델이 "이 chunk 가 정확히 그것인가"를 확신하지 못하고 거부한다.
#
# 【규칙을 풀지 않는 이유】 이 규칙은 '오선택'(비슷한 다른 표의 행을 답하는 실패)을
#   막고 있고 실제로 작동한다 — 날조 0%, 오답 1.3%. 규칙을 풀면 거부는 줄지만
#   오답·날조가 늘어난다. 트레이드오프를 맞바꾸는 것일 뿐 개선이 아니다.
#
# 【대신 하는 것】 거부의 **형태**만 바꾼다. 사실을 단정하지 않으므로 날조 위험을
#   다시 들이지 않으면서, 막다른 길을 출발점으로 만든다. 사용자는 관련 문서를
#   직접 열어보거나 다른 표현으로 재질의할 수 있다 (긴 질문은 정답이었다).
#
# 프롬프트가 아니라 후처리인 이유: 모델에게 판단을 더 시키는 변경은 이 프로젝트에서
# 한 번 실패했다(표 레이블 규칙 - 얻은 것 없이 정확도만 하락). 출력 형식만 바꾸는
# 결정적 후처리가 훨씬 안전하다.

_REFUSAL_MARKERS = (
    "찾을 수 없", "확인되지 않", "확인할 수 없", "포함되어 있지 않",
    "정보가 없", "제공되지 않", "명시되어 있지 않",
)


def _is_refusal(text: str) -> bool:
    """답변이 실질적으로 '모른다' 인가. 부연이 길면 이미 정보를 준 것이므로 제외."""
    t = (text or "").strip()
    if not t or len(t) > 400:
        return False
    return any(m in t for m in _REFUSAL_MARKERS)


def _augment_refusal(answer: str, docs: list[Document], top: int = 3) -> str:
    """거부 답변에 관련 자료 목록과 재질의 안내를 붙인다.

    [N] 번호는 generator 가 컨텍스트에 넣은 순서와 같으므로, 붙이는 순간
    _extract_cited_ids 가 인식해 출처가 자동으로 응답에 포함된다.
    """
    lines = []
    for i, d in enumerate(docs[:top], start=1):
        title = (d.metadata.get("title") or d.source or "").strip()
        loc = (d.metadata.get("location") or "").strip()
        if not title:
            continue
        lines.append(f"- {title}{(' ' + loc) if loc else ''} [{i}]")
    if not lines:
        return answer

    return (
        "질문하신 내용에 대한 직접적인 답은 검색 결과에서 확인되지 않았습니다.\n\n"
        "관련이 있어 보이는 자료는 다음과 같습니다:\n"
        + "\n".join(lines)
        + "\n\n좀 더 구체적인 표현으로 다시 질문해 주시면 더 정확히 찾아드릴 수 있습니다."
    )


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

    # 과잉 거부 보정 — 문서가 있는데 거부했고 인용도 없으면 관련 자료를 안내한다.
    # (문서가 아예 없는 경우는 위쪽 no_docs 분기에서 이미 처리됨)
    refusal_augmented = False
    if docs and _is_refusal(answer) and not _extract_cited_ids(answer):
        augmented = _augment_refusal(answer, docs)
        if augmented != answer:
            answer = augmented
            refusal_augmented = True

    cited_ids = _extract_cited_ids(answer)
    citations = _build_citations(docs, cited_ids)

    return {
        "answer": answer,
        "citations": citations,
        "messages": [HumanMessage(content=user_input), AIMessage(content=answer)],
        "llm_call_count": state.llm_call_count + 1,
        "decision_path": [
            f"generate:{qtype}({len(citations)}cit)"
            + ("+refusal_augmented" if refusal_augmented else "")
        ],
    }
