from __future__ import annotations

import threading
from typing import Any, Dict, List, Tuple

from app.core.config import get_embeddings, ROUTER_TOP1_MIN, ROUTER_MARGIN_MIN
from app.core.history_utils import cosine as _cosine
from app.graph.states.state import GraphState
from app.graph.nodes.llm_intent_fallback import llm_intent_fallback
from app.auth.intent_samples import load_all_samples, add_sample

_ALLOWED = {"knowledge_search", "ai_guide", "file_chat", "detail_search"}

_DETAIL_HINTS = [
    "자세히", "자세하게", "더 알려줘", "좀 더", "구체적", "상세히", "상세하게",
    "더 설명", "추가로 설명", "자세한 내용", "더 자세한",
]

_SAMPLE_VECTORS: Dict[str, List[List[float]]] | None = None
_SAMPLE_LOCK = threading.Lock()


def _contains_any(text: str, keywords: List[str]) -> bool:
    t = (text or "").lower()
    return any(k.lower() in t for k in keywords)


def invalidate_sample_cache() -> None:
    global _SAMPLE_VECTORS
    with _SAMPLE_LOCK:
        _SAMPLE_VECTORS = None


def _load_sample_vectors() -> Dict[str, List[List[float]]]:
    global _SAMPLE_VECTORS
    if _SAMPLE_VECTORS is not None:
        return _SAMPLE_VECTORS
    with _SAMPLE_LOCK:
        if _SAMPLE_VECTORS is not None:
            return _SAMPLE_VECTORS
        samples = load_all_samples()
        embeddings = get_embeddings()
        vectors: Dict[str, List[List[float]]] = {}
        for task, texts in samples.items():
            if task not in _ALLOWED:
                continue
            vectors[task] = embeddings.embed_documents(texts)
        _SAMPLE_VECTORS = vectors
        return _SAMPLE_VECTORS


def _semantic_route(
    user_input: str,
    input_embedding: List[float] | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """Semantic 임베딩 기반 라우팅. file_extract 라우팅 제거 후 1차 라우터로 사용."""
    vectors = _load_sample_vectors()
    embeddings = get_embeddings()

    query_vec = input_embedding if input_embedding else embeddings.embed_query(user_input)

    ranked: List[Tuple[str, float]] = []
    for task, task_vectors in vectors.items():
        if not task_vectors:
            continue
        score = max(_cosine(query_vec, svec) for svec in task_vectors)
        ranked.append((task, score))

    if not ranked:
        return "unknown", {"mode": "semantic", "reason": "no_samples", "decision": "unknown"}

    ranked.sort(key=lambda x: x[1], reverse=True)
    top1_task, top1_score = ranked[0]
    top2_score = ranked[1][1] if len(ranked) > 1 else -1.0
    margin = top1_score - top2_score

    decision = top1_task
    if top1_score < ROUTER_TOP1_MIN or margin < ROUTER_MARGIN_MIN:
        decision = "unknown"

    debug = {
        "mode": "semantic",
        "top1_task": top1_task,
        "top1_score": round(top1_score, 4),
        "top2_score": round(top2_score, 4),
        "margin": round(margin, 4),
        "decision": decision,
        "threshold_top1": ROUTER_TOP1_MIN,
        "threshold_margin": ROUTER_MARGIN_MIN,
        "ranked": [{"task": t, "score": round(s, 4)} for t, s in ranked[:4]],
    }
    return decision, debug


def task_router_node(state: GraphState) -> GraphState:
    print("--- [NODE] Task Router ---")

    user_input = (state.get("input_data") or "").strip()
    task_args = state.get("task_args") or {}
    input_embedding = state.get("input_embedding")

    if not user_input:
        return {"task_type": "knowledge_search", "task_args": task_args}

    try:
        # Step 1: Semantic 라우팅
        routed, debug = _semantic_route(user_input, input_embedding)
        debug["final_source"] = "semantic"

        # Step 2: LLM Fallback (semantic 결과가 unknown이면)
        if routed == "unknown":
            llm_task, llm_debug = llm_intent_fallback(user_input)
            debug["llm_fallback"] = llm_debug
            if llm_task != "unknown":
                routed = llm_task
                debug["decision"] = llm_task
                debug["final_source"] = "llm_fallback"
                added = add_sample(llm_task, user_input, source="llm_fallback")
                if added:
                    invalidate_sample_cache()
            else:
                debug["final_source"] = "unknown"

        # 상세 검색 감지: 후속 심화 패턴 + 직전 task가 knowledge_search/detail_search + 메시지 존재
        if routed in ("unknown", "knowledge_search"):
            prev_task = (state.get("task_type") or "").strip()
            prev_messages = state.get("messages") or []
            if (
                prev_task in ("knowledge_search", "detail_search")
                and prev_messages
                and _contains_any(user_input, _DETAIL_HINTS)
            ):
                routed = "detail_search"
                debug["decision"] = "detail_search"
                debug["final_source"] = "detail_search_detected"

        # file_context 있는데 unknown이면 file_chat으로 fallback
        file_context_present = bool((state.get("file_context") or "").strip())
        pending_slots = [slot for slot in (task_args.get("missing_slots") or []) if slot == "file_context"]

        if routed == "unknown" and file_context_present:
            routed = "file_chat"
            debug["decision"] = "file_chat"
            debug["final_source"] = "file_context_fallback"

        if routed == "unknown" and pending_slots and file_context_present:
            routed = "file_chat"
            debug["decision"] = "file_chat"
            debug["final_source"] = "pending_file_slot_resolved"

        # 슬롯 누락 감지: file_chat에 file_context 없을 때만
        missing_slots: list[str] = []
        if routed == "file_chat" and not file_context_present:
            missing_slots = ["file_context"]

        if missing_slots:
            routed = "clarification"
            debug["decision"] = "clarification"
            debug["final_source"] = "slot_missing"
            debug["missing_slots"] = missing_slots

        merged_args = {**task_args, "routing_debug": debug}
        if missing_slots:
            merged_args["missing_slots"] = missing_slots

        return {"task_type": routed, "task_args": merged_args}

    except Exception as e:
        merged_args = {
            **task_args,
            "routing_debug": {
                "mode": "rule_fallback", "decision": "knowledge_search",
                "final_source": "rule_fallback", "reason": str(e),
            },
        }
        return {"task_type": "knowledge_search", "task_args": merged_args}


def rejection_node(state: GraphState) -> Dict[str, Any]:
    """
    인젝션 시도 등으로 거부된 turn은 thread history에 추가하지 않는다.
    거부 응답 텍스트는 main.py /chat 엔드포인트가 task_type=injection을
    감지해 직접 반환한다.
    """
    print("--- [NODE] Rejection ---")
    return {"citations_used": []}


def _unknown_fallback_route(state: GraphState) -> str:
    """Unknown task를 AI 안내 또는 지식 검색으로 우선 실행."""
    user_input = (state.get("input_data") or "").strip()
    _AI_GUIDE_TRIGGERS = [
        "안녕", "반가워", "뭐 할 수 있어", "무슨 기능", "도움말", "help",
        "소개해", "뭐야", "누구야", "어떤 ai", "기능 안내", "사용법",
    ]
    if len(user_input) <= 15 or _contains_any(user_input, _AI_GUIDE_TRIGGERS):
        return "ai_guide"
    return "knowledge_search"


def route_by_task(state: GraphState) -> str:
    task = (state.get("task_type") or "knowledge_search").strip()
    if task == "chat":
        return "knowledge_search"
    if task in ("unknown", ""):
        return _unknown_fallback_route(state)
    if task == "injection":
        return "rejection"
    if task == "clarification":
        return "clarification"
    if task == "detail_search":
        return "detail_search"
    if task in _ALLOWED:
        return task
    return "knowledge_search"


def route_after_input_guard(state: GraphState) -> str:
    task = (state.get("task_type") or "").strip()
    if task == "injection":
        return "rejection"
    return "task_router"
