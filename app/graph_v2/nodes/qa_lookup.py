"""
QA Lookup Node — Q&A 캐시 유사도 검색 (chunk_question_types 대체).

흐름:
  router → (single/multi_retrieval인 경우만) qa_lookup → 분기
    ├─ bypass hit (≥0.92): answer/citation 직접 작성, retrieve 스킵
    ├─ hint hit (0.78~0.92): sub_questions에 캐시된 질문 추가 (재현율 향상),
    │                       generator는 retrieve 결과만 사용 (cache는 query 보강용)
    └─ miss: 그대로 retrieve로
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from app.graph_v2.states.state import Citation, GraphState
from app.knowledge.qa_cache import get_qa_cache


def qa_lookup_node(state: GraphState) -> dict:
    """Q&A 캐시 조회. hit 강도에 따라 분기 정보를 state에 기록.

    상태 변경:
      - bypass hit: answer/citations/messages 직접 작성 (route_after_qa_lookup → end)
      - hint hit: sub_questions에 캐시 질문 prepend (retrieve가 multi-query로 사용)
      - miss: 변경 없음
    """
    user_input = (state.input_data or "").strip()
    if not user_input:
        return {"decision_path": ["qa_lookup:empty_query"]}

    cache = get_qa_cache()
    if cache.count() == 0:
        return {"decision_path": ["qa_lookup:empty_cache"]}

    hit = cache.lookup(user_input)
    if hit is None:
        return {"decision_path": ["qa_lookup:miss"]}

    # ─── Bypass: 사실상 동일 질문 — 캐시된 답변 그대로 반환 ───
    if hit.is_bypass:
        citation = Citation(
            id=1,
            doc_id=hit.doc_id or "qa_cache",
            snippet=hit.snippet,
            score=hit.score,
            location=hit.location,
        )
        # citation 표기는 [1]로 부착
        answer_with_cite = f"{hit.answer} [1]" if "[1]" not in hit.answer else hit.answer
        return {
            "answer": answer_with_cite,
            "citations": [citation],
            # question_type을 캐시 metadata 기준으로 덮어쓰기 (downstream 안전)
            "question_type": hit.answer_type or state.question_type,
            "messages": [HumanMessage(content=user_input), AIMessage(content=answer_with_cite)],
            "decision_path": [f"qa_lookup:bypass({hit.score:.2f})"],
        }

    # ─── Hint: 유사 질문 — 캐시된 질문을 sub_question으로 추가 ───
    # retrieve_node는 sub_questions가 있으면 multi-query로 검색.
    # 캐시 질문이 더 잘 정제된 검색어이므로 재현율이 올라감.
    new_subs = list(state.sub_questions or [])
    if hit.question and hit.question not in new_subs and hit.question != user_input:
        new_subs.insert(0, hit.question)

    return {
        "sub_questions": new_subs,
        # answer_type도 hint로 사용 (router가 잘못 분류했을 수 있음)
        "question_type": hit.answer_type or state.question_type,
        "decision_path": [f"qa_lookup:hint({hit.score:.2f})"],
    }


def route_after_qa_lookup(state: GraphState) -> str:
    """이번 턴의 qa_lookup 결과로만 분기.

    decision_path 의 마지막 항목이 'qa_lookup:bypass(...)' 인 경우만 end.
    state.answer / state.citations 를 직접 검사하면 LangGraph 체크포인터가
    이전 턴 값을 그대로 로드한 상태이므로 신구 턴 구분 불가 → bug.
    """
    last_path = state.decision_path[-1] if state.decision_path else ""
    if last_path.startswith("qa_lookup:bypass"):
        return "end"
    return "retrieve"
