"""Grader 임계 ablation — 12개 대표 질문 × 3개 임계 조합.

방법: grader 모듈의 globals(RERANK_RELEVANT_THRESHOLD, RERANK_PARTIAL_THRESHOLD)을
runtime에 patch. reranker 모델은 1회 로드 후 재사용.

대상 12개:
  의심(max<0.6 + 오답): 16, 74, 93, 121, 161, 165
  잃을위험(max<0.6 + 정답): 20, 115, 119, 198, 4, 56
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("CREATE_USERS_TABLE", "0")
os.environ.setdefault("CREATE_INTENT_SAMPLES_TABLE", "0")
os.environ.setdefault("CREATE_ROUTING_LOG_TABLE", "0")

from langgraph.checkpoint.memory import InMemorySaver
from app.graph_v2.builder import build_main_graph
from app.graph_v2.nodes import grader as grader_mod

EVAL_DIR = Path(__file__).parent
QUESTIONS = json.loads((EVAL_DIR / "data" / "questions.json").read_text(encoding="utf-8"))
LABELS = json.loads((EVAL_DIR / "data" / "labels.json").read_text(encoding="utf-8"))
QMAP = {q["id"]: q for q in QUESTIONS}
LMAP = {l["id"]: l for l in LABELS} if isinstance(LABELS, list) else LABELS

TARGET_IDS = [
    # 의심 (현 grader가 약하게 통과시킨 것 추정)
    16, 74, 93, 121, 161, 165,
    # 잃을 위험 (max<0.6 인데 정답)
    20, 115, 119, 198, 4, 56,
]
THRESHOLDS = [
    ("baseline", 0.50, 0.30),
    ("medium",   0.55, 0.35),
    ("strict",   0.60, 0.40),
]


def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def get_gt(qid: int) -> str:
    l = LMAP.get(qid, {})
    return l.get("ground_truth") or l.get("answer") or l.get("label") or "?"


def run_one(graph_app, q: dict) -> dict:
    thread_id = f"abl-{uuid.uuid4().hex[:10]}"
    inputs = {"trace_id": thread_id, "input_data": q["question"]}
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}
    t0 = time.perf_counter()
    try:
        result = graph_app.invoke(inputs, config=config)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "elapsed": time.perf_counter() - t0}
    elapsed = time.perf_counter() - t0
    answer = result.get("answer", "")
    if not answer:
        msgs = result.get("messages") or []
        answer = extract_text(msgs[-1].content) if msgs else ""
    cits = []
    for c in result.get("citations") or []:
        if hasattr(c, "model_dump"):
            c = c.model_dump()
        elif hasattr(c, "dict"):
            c = c.dict()
        cits.append({"doc_id": c.get("doc_id"), "loc": c.get("location") or c.get("page")})
    # decision_path에서 첫 grade max만 추출
    dp = result.get("decision_path") or []
    import re
    grade_re = re.compile(
        r"grade:(?P<status>\w+)(?:\((?P<rel>\d+)rel/(?:(?P<kept>\d+)kept/)?(?P<total>\d+)total,"
        r"max=(?P<max>[\d.]+),cov=(?P<cov>\w+)\))?"
    )
    first_grade = None
    for s in dp:
        m = grade_re.search(s)
        if m:
            first_grade = m.groupdict()
            break
    return {
        "answer": answer,
        "task_type": result.get("routing_decision", ""),
        "qtype": result.get("question_type", ""),
        "first_grade": first_grade,
        "n_grades": sum(1 for s in dp if s.startswith("grade:")),
        "n_replan": result.get("replan_iterations", 0),
        "n_retrieve": result.get("retrieval_iterations", 0),
        "n_llm": result.get("llm_call_count", 0),
        "elapsed": round(elapsed, 2),
        "cits": cits[:3],
        "error": None,
    }


def main() -> None:
    print(f"[BUILD] graph (1회)...", flush=True)
    graph_app = build_main_graph(checkpointer=InMemorySaver())
    print("[OK] graph ready", flush=True)

    all_results: dict = {}  # {threshold_name: {qid: run_result}}
    for thr_name, rel_thr, par_thr in THRESHOLDS:
        # grader 모듈 globals 직접 패치 (import-from으로 복사된 변수를 갱신)
        grader_mod.RERANK_RELEVANT_THRESHOLD = rel_thr
        grader_mod.RERANK_PARTIAL_THRESHOLD = par_thr
        print(f"\n========== [{thr_name}] rel={rel_thr} par={par_thr} ==========", flush=True)
        thr_results: dict = {}
        for qid in TARGET_IDS:
            q = QMAP[qid]
            r = run_one(graph_app, q)
            r["question"] = q["question"]
            r["category"] = q["category"]
            r["gt"] = get_gt(qid)
            thr_results[qid] = r
            fg = r.get("first_grade")
            fg_max = fg.get("max") if fg else "?"
            ans = (r.get("answer") or "")[:80]
            print(f"  Q{qid} max={fg_max} t={r['elapsed']}s replan={r['n_replan']} | {ans}", flush=True)
        all_results[thr_name] = thr_results

    out = EVAL_DIR / "results" / "ablation_grader_threshold.json"
    out.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[SAVED] {out}")

    # 사람이 보기 쉬운 비교 리포트
    print("\n\n========== 비교 리포트 ==========")
    for qid in TARGET_IDS:
        q = QMAP[qid]
        gt = get_gt(qid)
        print(f"\n--- Q{qid} [{q['category']}] {q['question']}")
        print(f"GT: {gt}")
        for thr_name, _, _ in THRESHOLDS:
            r = all_results[thr_name][qid]
            ans = (r.get("answer") or "").replace("\n", " ")[:140]
            fg = r.get("first_grade")
            fg_max = fg.get("max") if fg else "?"
            print(f"  [{thr_name:<8}] max={fg_max} replan={r['n_replan']} llm={r['n_llm']} | {ans}")


if __name__ == "__main__":
    main()
