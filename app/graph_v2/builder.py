"""
Main graph builder — v2 그래프 조립.

토폴로지:
  ┌─────────┐
  │  START  │
  └─────────┘
       ↓
  [security_gate]
       ├─(blocked)→ END
       └─(pass)
       ↓
  [router]
       ├─(no_retrieval)→ [no_retrieval_answer] → END
       ├─(ai_guide)→ [ai_guide] → END
       ├─(file_chat)→ [file_chat] → END
       └─(single/multi)
       ↓
  [retrieve_subgraph]
       ↓
  [generate_subgraph]
       ↓
  (verification.passed?)
       ├─(pass)→ END
       └─(fail + replan < 1)→ [router] (재계획)
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from app.core.config import get_llm
from app.core.history_utils import extract_text_content
from app.graph_v2.states.state import GraphState
from app.graph_v2.nodes.security import security_gate_node, route_after_security
from app.graph_v2.nodes.router import router_node, route_by_decision
from app.graph_v2.nodes.qa_lookup import qa_lookup_node, route_after_qa_lookup
from app.graph_v2.subgraphs.retrieve import build_retrieve_subgraph
from app.graph_v2.subgraphs.generate import build_generate_subgraph


# ─── Leaf nodes (검색 없는 분기) ──────────────────────────────

def _no_retrieval_answer_node(state: GraphState) -> dict:
    """검색 없이 일반 응답 (인사·메타 질의). 짧은 정중 답변."""
    user_input = state.input_data or ""
    sys = (
        "당신은 Kaiper AI 사내 어시스턴트입니다. 인사·간단한 메타 질의에 짧고 정중하게 답하세요. "
        "사내 문서 검색·파일 분석·기능 안내 등을 도울 수 있다고 안내하세요. "
        "구체적인 사실 회수가 필요한 질의는 검색 기능 사용을 권유하세요."
    )
    try:
        resp = get_llm().invoke([SystemMessage(content=sys), HumanMessage(content=user_input)])
        ans = extract_text_content(resp.content) or "안녕하세요. Kaiper AI 사내 어시스턴트입니다."
    except Exception:
        ans = "안녕하세요. Kaiper AI 사내 어시스턴트입니다."
    return {
        "answer": ans,
        "messages": [HumanMessage(content=user_input), AIMessage(content=ans)],
        "decision_path": ["no_retrieval_answer"],
    }


def _ai_guide_node(state: GraphState) -> dict:
    """기능 안내 응답."""
    user_input = state.input_data or ""
    ans = (
        "Kaiper AI 사내 어시스턴트는 다음 기능을 제공합니다.\n"
        "• **사내 문서 검색**: 사내 정책·매뉴얼·기획 산출물·회의록 등을 자연어로 검색하고 출처와 함께 답변\n"
        "• **심화 검색**: 직전 검색 결과에 대한 후속 질의 처리\n"
        "• **파일 분석**: PDF·DOCX·XLSX·PPTX·TXT 첨부파일의 내용 요약·Q&A\n\n"
        "구체적인 질문(예: '체크리스트 9번 항목', '2일차 오후 시간')은 그대로 입력하시면 됩니다."
    )
    return {
        "answer": ans,
        "messages": [HumanMessage(content=user_input), AIMessage(content=ans)],
        "decision_path": ["ai_guide"],
    }


def _file_chat_node(state: GraphState) -> dict:
    """file_context 있을 때만 사용 — 파일 내용 기반 답변."""
    user_input = state.input_data or ""
    file_text = (state.file_context or "").strip()
    if not file_text:
        ans = "첨부 파일이 필요합니다. 먼저 파일을 업로드해 주세요."
        return {
            "answer": ans,
            "messages": [HumanMessage(content=user_input), AIMessage(content=ans)],
            "decision_path": ["file_chat:no_file"],
        }
    sys = (
        "사용자가 첨부한 파일 내용을 분석하는 어시스턴트입니다. 아래 [파일 내용]만을 근거로 답하세요.\n"
        "파일에 없는 내용은 '파일에서 확인할 수 없습니다'로 답하세요.\n\n"
        f"[파일 내용]\n{file_text[:8000]}"
    )
    try:
        resp = get_llm().invoke([SystemMessage(content=sys), HumanMessage(content=user_input)])
        ans = extract_text_content(resp.content) or "응답을 생성하지 못했습니다."
    except Exception as e:
        ans = f"파일 분석 중 오류가 발생했습니다: {e}"
    return {
        "answer": ans,
        "messages": [HumanMessage(content=user_input), AIMessage(content=ans)],
        "llm_call_count": state.llm_call_count + 1,
        "decision_path": ["file_chat"],
    }


def _rejected_node(state: GraphState) -> dict:
    """Security 또는 router에서 차단된 케이스. messages에 추가 안 함 (history 오염 방지).

    차단 사유를 구분한다. detector_unavailable(임베딩 프로바이더 장애로 판정 불가)을
    "지원 범위 밖"으로 안내하면 사용자는 질문을 바꿔 재시도하게 되고, 운영자는
    장애를 인지하지 못한다. 이 경우 main.py 가 503 으로 승격시킨다.
    """
    if state.security_reason == "detector_unavailable":
        ans = (
            "보안 검사 모듈을 일시적으로 사용할 수 없어 요청을 처리하지 않았습니다. "
            "잠시 후 다시 시도해 주세요. 문제가 계속되면 관리자에게 문의해 주세요."
        )
        return {
            "answer": ans,
            "decision_path": ["rejected:detector_unavailable"],
        }

    ans = "해당 질문은 사내 AI 어시스턴트의 지원 범위에 포함되지 않아 답변을 제공하지 않습니다. 사내 업무 관련 질문을 입력해 주세요."
    return {
        "answer": ans,
        "decision_path": ["rejected"],
    }


# ─── Main graph builder ──────────────────────────────────
# replan 메커니즘 폐기: rewrite 1회 + doc-aware 카탈로그로 대체.
# 1차 시도가 빗나간 경우 사용자에게 빠른 fail 후 재질문을 유도하는 게
# 자동 추측 재시도 (~120초 대기) 보다 사용자 UX 가 더 낫다는 결정.
#
# 단, reflection 자체는 유지 (verification 결과를 답변 메타로 활용 가능).
# generator → reflection → END 경로는 항상 일관됨.

def _changed(before, after) -> bool:
    """두 값이 실질적으로 다른가. 비교 불가 타입은 '바뀜'으로 본다(안전 측)."""
    if before is after:
        return False
    try:
        return bool(before != after)
    except Exception:
        return True


def _subgraph_node(sub):
    """서브그래프를 노드로 감싸며 **실제로 바뀐 채널만** 반환하게 한다.

    두 가지 문제를 함께 막는다.

    (1) decision_path 중복
        서브그래프는 부모와 같은 GraphState 스키마를 공유하므로 부모의
        누적 리스트를 물려받아 자기 항목을 붙인 뒤 **전체**를 반환한다.
        부모의 reducer 가 그걸 다시 이어붙여 앞부분이 중복된다.
        → 진입 시점 길이 이후의 증분만 잘라 넘긴다.

    (2) 상위 노드 재실행  ← 이것이 더 심각했다
        전체 상태를 반환하면 서브그래프가 건드리지도 않은 채널
        (input_data / routing_decision / question_type ...)까지 "갱신됨"으로
        표시된다. LangGraph 는 채널 버전으로 노드 실행을 결정하므로
        security_gate / router 같은 상위 노드가 다시 트리거된다.

        이 현상은 체크포인터 구현에 따라 갈렸다 (실측, 검색 경로 1턴):
            InMemorySaver         →  8개  (정상)
            DynamoDBCheckpointer  → 18개  (security_gate 2회, router 4회)
        자체 구현 체크포인터는 thread 당 단일 슬롯에 checkpoint_id 를
        항상 "latest" 로 두기 때문에 버전 체인이 달라진다.
        → 바뀜 채널만 반환해 불필요한 버전 증가 자체를 없앨다.

    messages 는 add_messages 가 id 로 중복을 제거하므로 별도 처리하지 않는다.
    """
    def _node(state: GraphState) -> dict:
        before_len = len(state.decision_path or [])
        out = sub.invoke(state)
        if not isinstance(out, dict):
            return out

        result: dict = {}
        for key, value in out.items():
            if key == "decision_path":
                delta = (value or [])[before_len:]
                if delta:
                    result[key] = delta
                continue
            if _changed(getattr(state, key, None), value):
                result[key] = value
        return result

    _node.__name__ = "subgraph_node"
    return _node


def build_main_graph(checkpointer=None):
    g = StateGraph(GraphState)

    # Compile subgraphs
    retrieve_sub = build_retrieve_subgraph()
    generate_sub = build_generate_subgraph()

    # Nodes
    g.add_node("security_gate", security_gate_node)
    g.add_node("router", router_node)
    g.add_node("qa_lookup", qa_lookup_node)
    g.add_node("retrieve", _subgraph_node(retrieve_sub))
    g.add_node("generate", _subgraph_node(generate_sub))
    g.add_node("no_retrieval_answer", _no_retrieval_answer_node)
    g.add_node("ai_guide", _ai_guide_node)
    g.add_node("file_chat", _file_chat_node)
    g.add_node("rejected", _rejected_node)

    g.set_entry_point("security_gate")

    # Edges
    g.add_conditional_edges(
        "security_gate", route_after_security,
        {"rejected": "rejected", "router": "router"},
    )
    # router → retrieve 경로는 qa_lookup을 먼저 거침 (캐시 hit 시 retrieve 스킵)
    g.add_conditional_edges(
        "router", route_by_decision,
        {
            # "rejected" 없음 — 라우터는 범위 밖 판정을 하지 않는다.
            # 보안 차단은 security_gate 가, 자료 없음은 generator 가 담당한다.
            "no_retrieval_answer": "no_retrieval_answer",
            "ai_guide": "ai_guide",
            "file_chat": "file_chat",
            "retrieve": "qa_lookup",
        },
    )
    g.add_conditional_edges(
        "qa_lookup", route_after_qa_lookup,
        {"end": END, "retrieve": "retrieve"},
    )
    g.add_edge("retrieve", "generate")
    # generate → END (replan 분기 폐기). reflection 결과는 verification 으로
    # state 에 기록되며 답변 응답에 메타 정보로 포함 가능.
    g.add_edge("generate", END)
    g.add_edge("no_retrieval_answer", END)
    g.add_edge("ai_guide", END)
    g.add_edge("file_chat", END)
    g.add_edge("rejected", END)

    return g.compile(checkpointer=checkpointer)
