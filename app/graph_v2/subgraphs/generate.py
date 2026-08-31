"""
Generation subgraph — generator → reflection → (replan 시 외부로 신호).

흐름:
  generate
    → reflect
    → (passed) END
    → (fail + replan_iterations < 1) → replan signal (main graph에서 처리)
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from app.graph_v2.states.state import GraphState
from app.graph_v2.nodes.generator import generator_node
from app.graph_v2.nodes.reflection import reflection_node, route_after_reflection


def build_generate_subgraph():
    g = StateGraph(GraphState)
    g.add_node("generate", generator_node)
    g.add_node("reflect", reflection_node)

    g.set_entry_point("generate")
    g.add_edge("generate", "reflect")

    # subgraph 내부에서는 항상 END. replan 결정은 main graph가.
    # (LangGraph에서 subgraph 종료 후 main이 verification.passed 검사 → replan 분기)
    g.add_edge("reflect", END)
    return g.compile()
