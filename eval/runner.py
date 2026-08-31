"""
평가 runner — 250문항을 현재 시스템에 던지고 응답을 수집한다.

직접 graph_app.invoke를 호출 (HTTP /chat 거치지 않음). 각 질문은 독립 thread_id로
실행하여 대화 누적의 영향을 차단한다.

출력: eval/results/<run_name>.json
각 결과 객체:
{
  "id": int,
  "question": str,
  "answer": str,           # 시스템 응답
  "task_type": str,        # 라우팅 결정
  "task_args": dict,       # 라우팅 디버그 + 검색 docs 메타
  "citations": [...],      # citations_used
  "elapsed_sec": float,
  "error": str | null
}

Usage:
    GEMINI_API_KEY=... python eval/runner.py --run-name baseline
    GEMINI_API_KEY=... python eval/runner.py --run-name baseline --limit 10  # smoke test
    GEMINI_API_KEY=... python eval/runner.py --run-name lever3_prompt --categories A,B
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

EVAL_DIR = Path(__file__).parent
DATA_DIR = EVAL_DIR / "data"
RESULTS_DIR = EVAL_DIR / "results"
QUESTIONS_FILE = DATA_DIR / "questions.json"


def build_graph_app(version: str = "v1"):
    """그래프 빌더. v1 또는 v2 선택 가능.

    v1: 기존 13-노드 플랫 그래프 (main.py에서 재구성)
    v2: 새 아키텍처 (app/graph_v2/builder.py)
    """
    from dotenv import load_dotenv
    load_dotenv()

    # DynamoDB 의존성 회피: import 전 환경변수 강제
    os.environ.setdefault("CREATE_USERS_TABLE", "0")
    os.environ.setdefault("CREATE_INTENT_SAMPLES_TABLE", "0")
    os.environ.setdefault("CREATE_ROUTING_LOG_TABLE", "0")

    if version == "v2":
        from langgraph.checkpoint.memory import InMemorySaver
        from app.graph_v2.builder import build_main_graph
        return build_main_graph(checkpointer=InMemorySaver())

    # v1 (기존)

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, StateGraph
    from app.graph.states.state import GraphState
    from app.graph.nodes.input_guard import input_guard_node
    from app.graph.nodes.task_router import (
        task_router_node, route_by_task, rejection_node, route_after_input_guard,
    )
    from app.graph.nodes.clarification import (
        clarification_slot_node, clarification_confirm_node, route_after_clarification,
    )
    from app.graph.nodes.knowledge_search import (
        search_node, quality_check_node, route_after_quality, rewrite_node, answer_node,
    )
    from app.graph.nodes.detail_search import detail_search_node
    from app.graph.nodes.ai_guide import ai_guide_node
    from app.graph.nodes.file_chat import file_chat_node

    workflow = StateGraph(GraphState)
    workflow.add_node("input_guard",          input_guard_node)
    workflow.add_node("task_router",          task_router_node)
    workflow.add_node("clarification",         clarification_slot_node)
    workflow.add_node("clarification_confirm", clarification_confirm_node)
    workflow.add_node("rejection",             rejection_node)
    workflow.add_node("search",                search_node)
    workflow.add_node("quality_check",         quality_check_node)
    workflow.add_node("rewrite",               rewrite_node)
    workflow.add_node("answer",                answer_node)
    workflow.add_node("ai_guide",              ai_guide_node)
    workflow.add_node("file_chat",             file_chat_node)
    workflow.add_node("detail_search",         detail_search_node)

    workflow.set_entry_point("input_guard")
    workflow.add_conditional_edges(
        "input_guard", route_after_input_guard,
        {"rejection": "rejection", "task_router": "task_router"},
    )
    workflow.add_conditional_edges(
        "task_router", route_by_task,
        {"knowledge_search": "search", "detail_search": "detail_search",
         "ai_guide": "ai_guide", "file_chat": "file_chat",
         "rejection": "rejection", "clarification": "clarification"},
    )
    workflow.add_conditional_edges(
        "clarification", route_after_clarification,
        {"rejection": "rejection", "task_router": "task_router",
         "knowledge_search": "search", "clarification_confirm": "clarification_confirm"},
    )
    workflow.add_edge("clarification_confirm", "task_router")
    workflow.add_edge("search", "quality_check")
    workflow.add_conditional_edges(
        "quality_check", route_after_quality,
        {"answer": "answer", "rewrite": "rewrite"},
    )
    workflow.add_edge("rewrite", "search")
    workflow.add_edge("detail_search", "answer")
    workflow.add_edge("answer", END)
    workflow.add_edge("ai_guide", END)
    workflow.add_edge("file_chat", END)
    workflow.add_edge("rejection", END)

    return workflow.compile(checkpointer=InMemorySaver())


def extract_text(content) -> str:
    """LangChain message.content가 list / str / dict 형태일 때 안전 추출."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(p.get("text", ""))
            else:
                parts.append(str(p))
        return "".join(parts)
    return str(content)


def run_one(graph_app, q: dict, version: str = "v1") -> dict:
    """단일 질문 실행. 신규 thread_id로 매번 격리."""
    thread_id = f"eval-{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
    if version == "v2":
        inputs = {
            "trace_id": thread_id,
            "input_data": q["question"],
        }
    else:
        inputs = {
            "trace_id": thread_id,
            "input_data": q["question"],
            "task_args": {},
            "retry_count": 0,
            "pending_confirm_msg": "",
        }

    t0 = time.perf_counter()
    try:
        result = graph_app.invoke(inputs, config=config)
        elapsed = time.perf_counter() - t0

        # v2와 v1 결과 구조 차이 흡수
        if version == "v2":
            # v2: result는 dict (Pydantic을 dump한 형태)
            answer_text = result.get("answer", "")
            if not answer_text:
                msgs = result.get("messages") or []
                answer_text = extract_text(msgs[-1].content) if msgs else ""
            task_type = result.get("routing_decision", "")
            decision_path = result.get("decision_path", [])
            citations = result.get("citations") or []
            # Citation 객체를 dict로
            citations = [
                (c.dict() if hasattr(c, "dict") else c)
                for c in citations
            ]
            return {
                "id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "answer": answer_text,
                "task_type": task_type,
                "question_type": result.get("question_type", ""),
                "decision_path": decision_path,
                "citations": citations,
                "retrieval_iterations": result.get("retrieval_iterations", 0),
                "replan_iterations": result.get("replan_iterations", 0),
                "llm_call_count": result.get("llm_call_count", 0),
                "elapsed_sec": round(elapsed, 3),
                "error": None,
            }
        else:
            msgs = result.get("messages") or []
            answer_text = extract_text(msgs[-1].content) if msgs else ""
            return {
                "id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "answer": answer_text,
                "task_type": result.get("task_type", ""),
                "routing_debug": (result.get("task_args") or {}).get("routing_debug", {}),
                "retry_count": result.get("retry_count", 0),
                "citations": result.get("citations_used") or [],
                "elapsed_sec": round(elapsed, 3),
                "error": None,
            }
    except Exception as e:
        return {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "answer": "",
            "task_type": "",
            "routing_debug": {},
            "retry_count": 0,
            "citations": [],
            "elapsed_sec": round(time.perf_counter() - t0, 3),
            "error": f"{type(e).__name__}: {e}",
        }


def _report_measurement_integrity(version: str) -> None:
    """실행 전 측정 조건을 로그에 남긴다. 결과 해석에 필요한 정보.

    특히 Q&A 캐시: `eval/build_qa_cache.py` 가 `eval/data/labels.json`(= 정답 라벨)로
    캐시를 채우고, v2 의 `qa_lookup` 노드는 유사도 0.92 이상이면 그 정답을 그대로
    반환한다(retrieve 스킵). 캐시가 채워진 상태로 평가를 돌리면 정확도가
    자기충족적이 되므로, 캐시 항목 수를 반드시 로그에 남겨야 한다.
    """
    from app.core.config import LLM_PROVIDER, get_llm

    try:
        model = getattr(get_llm(), "model", None) or getattr(get_llm(), "model_name", "?")
    except Exception as e:
        model = f"<init failed: {e}>"

    print("─" * 68, flush=True)
    print(f"[MEASURE] graph_version = {version}", flush=True)
    print(f"[MEASURE] llm_provider  = {LLM_PROVIDER}", flush=True)
    print(f"[MEASURE] llm_model     = {model}", flush=True)
    print("[MEASURE] embedding     = google / gemini-embedding-001 (프로바이더 고정)", flush=True)

    if version == "v2":
        try:
            from app.knowledge.qa_cache import get_qa_cache
            n = get_qa_cache().count()
        except Exception as e:
            n = -1
            print(f"[MEASURE] qa_cache 조회 실패: {e}", flush=True)
        if n > 0:
            print(
                f"[MEASURE] ⚠️  qa_cache = {n}개 — 정답 라벨 기반 캐시가 활성 상태입니다.\n"
                f"[MEASURE] ⚠️  이 상태의 정확도는 자기충족적일 수 있어 회귀 측정에 쓸 수 없습니다.\n"
                f"[MEASURE] ⚠️  깨끗한 측정을 원하면: python -c \""
                f"from app.knowledge.qa_cache import get_qa_cache; get_qa_cache().clear()\"",
                flush=True,
            )
        elif n == 0:
            print("[MEASURE] qa_cache     = 0개 (비어 있음 — 측정 조건 정상)", flush=True)
    print("─" * 68, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True, help="결과 파일명 (예: baseline)")
    parser.add_argument("--categories", help="콤마구분 (A,B,C)")
    parser.add_argument("--ids", help="콤마구분 question id (예: 5,26,41,60). categories와 함께면 OR")
    parser.add_argument("--ids-file", help="줄 또는 콤마 구분 id 목록 파일")
    parser.add_argument("--limit", type=int, help="앞에서 N개만 (smoke test)")
    parser.add_argument("--sleep", type=float, default=0.0, help="질의 간 sleep (rate limit)")
    parser.add_argument("--version", default="v1", choices=["v1", "v2"], help="그래프 버전")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    target_ids: set[int] | None = None
    if args.ids:
        target_ids = {int(x) for x in args.ids.split(",") if x.strip()}
    if args.ids_file:
        from pathlib import Path as _P
        text = _P(args.ids_file).read_text(encoding="utf-8")
        ids_from_file = {int(x) for x in text.replace(",", "\n").split() if x.strip().isdigit()}
        target_ids = (target_ids or set()) | ids_from_file
    if target_ids:
        questions = [q for q in questions if q["id"] in target_ids]
    elif args.categories:
        cats = set(args.categories.split(","))
        questions = [q for q in questions if q["category"] in cats]
    if args.limit:
        questions = questions[: args.limit]

    print(f"Building graph app (version={args.version})...", flush=True)
    graph_app = build_graph_app(version=args.version)

    _report_measurement_integrity(args.version)

    print(f"Running {len(questions)} questions...", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.run_name}.json"

    results: list[dict] = []
    for idx, q in enumerate(questions, 1):
        r = run_one(graph_app, q, version=args.version)
        results.append(r)
        status = "ERR" if r["error"] else "OK"
        print(f"[{idx}/{len(questions)}] Q{q['id']} ({q['category']}) "
              f"{r['elapsed_sec']:.2f}s {r['task_type']} {status}", flush=True)
        if args.sleep:
            time.sleep(args.sleep)

        # 진행 상황 incremental 저장 (장시간 실행 중 중단 대비)
        if idx % 10 == 0 or idx == len(questions):
            out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    err_count = sum(1 for r in results if r["error"])
    print(f"\nDone. {len(results)} results saved to {out_path}")
    print(f"  - Errors: {err_count}")
    print(f"  - Avg elapsed: {sum(r['elapsed_sec'] for r in results) / max(len(results), 1):.2f}s")


if __name__ == "__main__":
    main()
