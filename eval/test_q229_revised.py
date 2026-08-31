"""Q229 수정 문구 단발 테스트.

원: '시장 관심이 어디로 이동했다고 메모되어 있나요?'
수정: '자동화 플랫폼의 시장 관심이 어디로 이동했다고 메모되어 있나요?'
기대 정답: AI 에이전트형 자동화, 개발자 친화형 워크플로
출처: Ai_교육과정_정리_초안.txt
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


def extract_text(content) -> str:
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


def run(question: str, label: str) -> None:
    graph_app = build_main_graph(checkpointer=InMemorySaver())
    thread_id = f"q229test-{uuid.uuid4().hex[:10]}"
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
    inputs = {"trace_id": thread_id, "input_data": question}

    print(f"\n========== {label} ==========")
    print(f"Q: {question}")
    t0 = time.perf_counter()
    try:
        result = graph_app.invoke(inputs, config=config)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        return
    elapsed = time.perf_counter() - t0

    answer = result.get("answer", "")
    if not answer:
        msgs = result.get("messages") or []
        answer = extract_text(msgs[-1].content) if msgs else ""

    print(f"\nA: {answer}")
    print(f"\nrouting_decision: {result.get('routing_decision', '')}")
    print(f"question_type: {result.get('question_type', '')}")
    print(f"decision_path: {result.get('decision_path', [])}")
    print(f"retrieval_iterations: {result.get('retrieval_iterations', 0)}")
    print(f"replan_iterations: {result.get('replan_iterations', 0)}")
    print(f"llm_call_count: {result.get('llm_call_count', 0)}")
    print(f"elapsed: {elapsed:.2f}s")

    citations = result.get("citations") or []
    print(f"\ncitations ({len(citations)}):")
    for c in citations:
        if hasattr(c, "dict"):
            c = c.dict()
        doc_id = c.get("doc_id") or c.get("source_file") or ""
        loc = c.get("location") or c.get("page") or ""
        print(f"  - {doc_id} {loc}")


if __name__ == "__main__":
    run("시장 관심이 어디로 이동했다고 메모되어 있나요?", "원본 (Q229)")
    run("자동화 플랫폼의 시장 관심이 어디로 이동했다고 메모되어 있나요?", "수정 (자동화 플랫폼의 추가)")
