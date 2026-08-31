"""Reflection 수정 검증 — 5건.

Q229 두 케이스 + Q161 + Q165 + Q121.
변경 전 g=0.0 fail 케이스가 변경 후 pass 또는 합리적 g로 나오는지 확인.
"""
from __future__ import annotations
import json, os, sys, time, uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault("CREATE_USERS_TABLE", "0")
os.environ.setdefault("CREATE_INTENT_SAMPLES_TABLE", "0")
os.environ.setdefault("CREATE_ROUTING_LOG_TABLE", "0")

from langgraph.checkpoint.memory import InMemorySaver
from app.graph_v2.builder import build_main_graph

CASES = [
    ("Q229 원본",  "시장 관심이 어디로 이동했다고 메모되어 있나요?"),
    ("Q229 수정",  "자동화 플랫폼의 시장 관심이 어디로 이동했다고 메모되어 있나요?"),
    ("Q161",       "보조 강사 인력은 어떻게 운영하나요?"),
    ("Q165",       "강사 교통·식비는 어떻게 처리되나요?"),
    ("Q121",       "체크리스트 10번 항목 구분과 그 의미는?"),
]


def extract_text(content):
    if isinstance(content, str): return content
    if isinstance(content, list):
        return "".join(p.get("text","") if isinstance(p,dict) else str(p) for p in content)
    return str(content)


def main():
    graph_app = build_main_graph(checkpointer=InMemorySaver())
    print("=== Reflection 수정 검증 ===\n")
    for label, q in CASES:
        thread = f"reftest-{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread}, "recursion_limit": 30}
        t0 = time.perf_counter()
        try:
            result = graph_app.invoke({"trace_id": thread, "input_data": q}, config=config)
        except Exception as e:
            print(f"[{label}] ERROR: {e}\n")
            continue
        elapsed = time.perf_counter() - t0
        ans = result.get("answer","")
        if not ans:
            msgs = result.get("messages") or []
            ans = extract_text(msgs[-1].content) if msgs else ""
        dp = result.get("decision_path", [])
        # reflect 결과만 추출
        reflect_steps = [s for s in dp if s.startswith("reflect:")]
        replan_steps = [s for s in dp if s.startswith("replan:")]
        print(f"--- {label}")
        print(f"Q: {q}")
        print(f"A: {ans[:200]}")
        print(f"reflect: {reflect_steps}")
        print(f"replan: {replan_steps}")
        print(f"replan_iter: {result.get('replan_iterations',0)}, llm_calls: {result.get('llm_call_count',0)}, elapsed: {elapsed:.1f}s")
        print()

if __name__ == "__main__":
    main()
