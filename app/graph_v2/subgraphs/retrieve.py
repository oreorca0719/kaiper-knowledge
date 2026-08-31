"""
Retrieval subgraph — query_planner → retrieve → grade → (rewrite if needed).

Phase C 학습 반영:
  - multi-query는 question_type 분기로만 (verbatim 질문엔 끔 — K 카테고리 -41pp 회귀 방지)
  - reasoning / comparison 질문에만 query 변형 생성

흐름:
  query_planner → retrieve → grade → END (또는 rewrite로 회귀)
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Optional

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from app.core.config import (
    RETRIEVAL_CANDIDATE_TOP_K,
    RETRIEVAL_FINAL_K,
    RETRIEVAL_MAX_REWRITES,
    get_llm,
)
from app.core.history_utils import extract_text_content
from app.graph_v2.states.state import GraphState
from app.graph_v2.retrievers.base import Document, RetrieverRegistry
from app.graph_v2.retrievers.chroma_hybrid import ChromaHybridRetriever
from app.graph_v2.nodes.grader import grader_node, route_after_grade
from app.knowledge.chunking.tagger import extract_entities
from app.security.content_sanitizer import sanitize as sanitize_content


# ────────────────────────────────────────────────────────────
# Doc 카탈로그 — rewrite_node 가 사내 코퍼스 어휘를 알 수 있게 함
# 모든 chunks 메타에 동일하게 상속된 doc_summary / key_terms 를
# unique doc 단위로 추출하여 prompt 주입 형식으로 포맷팅.
# ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _build_doc_catalog() -> str:
    """ChromaDB 의 모든 chunks 메타에서 unique doc 단위 카탈로그 빌드.

    재인제스트 시 _invalidate_doc_catalog() 로 캐시 무효화 필수.
    """
    try:
        retriever = ChromaHybridRetriever()
        chroma = retriever._get_chroma()
        res = chroma._collection.get(include=["metadatas"])
        metas = res.get("metadatas") or []

        seen: set[str] = set()
        catalog_lines: list[str] = []
        for m in metas:
            doc_id = (m or {}).get("doc_id")
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)

            summary = ((m or {}).get("doc_summary") or "").strip()
            key_terms = ((m or {}).get("key_terms") or "").strip()
            if not summary:
                continue   # 메타 미구축 doc 스킵 (graceful)

            catalog_lines.append(
                f"- [{doc_id}]\n"
                f"  요약: {summary}\n"
                f"  관련 어휘: {key_terms}"
            )

        if not catalog_lines:
            return "(카탈로그 미구축 — 재인제스트 필요)"
        return "\n".join(catalog_lines)
    except Exception as e:
        print(f"[DOC_CATALOG] build failed (non-fatal): {e}")
        return "(카탈로그 빌드 실패)"


def _invalidate_doc_catalog() -> None:
    """재인제스트 직후 호출. 캐시 무효화."""
    _build_doc_catalog.cache_clear()


# ────────────────────────────────────────────────────────────
# Retriever Registry — 현재는 chroma_hybrid 단일.
# 추후 sql_retriever / web_retriever 추가 시 여기 등록.
# ────────────────────────────────────────────────────────────

_registry: Optional[RetrieverRegistry] = None


def _get_registry() -> RetrieverRegistry:
    global _registry
    if _registry is None:
        r = RetrieverRegistry()
        r.register(ChromaHybridRetriever())
        _registry = r
    return _registry


# ────────────────────────────────────────────────────────────
# Query planner (Phase C 학습 — question_type 분기로 multi-query 적용)
# ────────────────────────────────────────────────────────────

# multi-query를 적용할 question_type (검색 표현 다양화가 도움되는 case)
_MULTI_QUERY_TYPES = {"reasoning", "comparison", "list_n"}

_VARIANT_PROMPT = """사용자 질문을 사내 문서 검색에 적합한 검색 쿼리 변형 2개로 만들어주세요.

원칙:
- 원본 질문의 핵심 단어(고유명사·수치·시간)는 변형에도 포함
- 같은 의미를 다른 표현·키워드 조합으로

출력: JSON 배열만, 다른 텍스트 금지
["변형1", "변형2"]

원본 질문: {q}"""


def _generate_variants(query: str, n: int = 2) -> list[str]:
    """LLM으로 query 변형 생성. 실패 시 원본만 반환."""
    try:
        resp = get_llm().invoke([
            HumanMessage(content=_VARIANT_PROMPT.format(q=query)),
        ])
        raw = extract_text_content(resp.content)
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip().rstrip("`").strip()
        variants = json.loads(raw)
        if isinstance(variants, list):
            out = [str(v).strip() for v in variants if str(v).strip()][:n]
            return [query] + out
    except Exception as e:
        print(f"[QUERY_PLANNER] failed (non-fatal): {e}")
    return [query]


def query_planner_node(state: GraphState) -> dict:
    """question_type 기반 multi-query 분기.

    - exact_phrase / numerical / fill_blank: 변형 안 함 (정확 매칭이 중요)
    - reasoning / comparison / list_n: 변형 2개 추가
    """
    qtype = state.question_type or "reasoning"
    user_input = (state.input_data or "").strip()

    if qtype not in _MULTI_QUERY_TYPES:
        # 단일 query (변형 없음)
        return {
            "decision_path": [f"query_plan:single({qtype})"],
        }

    variants = _generate_variants(user_input, n=2)
    # 변형은 sub_questions 필드에 임시 저장 (retrieve 노드가 사용)
    return {
        "sub_questions": variants,
        "llm_call_count": state.llm_call_count + 1,
        "decision_path": [f"query_plan:multi({len(variants)},{qtype})"],
    }


# ────────────────────────────────────────────────────────────
# Retrieve node
# ────────────────────────────────────────────────────────────

def _sanitize_docs(docs: list[Document]) -> tuple[list[Document], int]:
    """검색된 chunk 의 간접 인젝션 차단 (Layer 3).

    v2 의 Document 는 `.content` 를 쓰므로 LangChain Document(`.page_content`)를
    전제한 `content_sanitizer.sanitize_docs` 를 그대로 쓸 수 없다.
    스칼라 `sanitize()` 를 적용하고 Document 를 재구성한다.

    Returns:
        (sanitize 적용된 docs, 차단된 chunk 수)
    """
    out: list[Document] = []
    blocked = 0
    for d in docs:
        original = d.content or ""
        doc_id = (d.metadata or {}).get("doc_id") or d.source
        safe = sanitize_content(original, source=f"rag:{doc_id}")
        if safe != original:
            blocked += 1
            out.append(Document(content=safe, source=d.source, score=d.score, metadata=d.metadata))
        else:
            out.append(d)
    return out, blocked


def retrieve_node(state: GraphState) -> dict:
    """Retriever Protocol 통해 검색. 단일 또는 multi-query.

    Phase G 추가:
      1. 기본 hybrid 검색 후
      2. expand_with_same_page: 같은 page_unit의 다른 chunks 함께 fetch
      3. boost_by_query_context: query qtype 일치 chunks score boost
         (doc_topic boost는 router가 query_doc_topic 분류 시 활성화)
    """
    queries = state.sub_questions or [state.input_data]
    queries = [q for q in queries if q]

    retriever = _get_registry().get("chroma_hybrid")

    pool: dict[str, Document] = {}
    for q in queries:
        try:
            # 각 query 당 후보 chunk 수 — phrasing 격차로 정답 chunk 가
            # 5위 밖으로 밀리는 결함 해소 위해 top_k 5 → 20 확장.
            # reranker 가 받는 후보 풀 크기 격증 → 정확도 향상.
            for d in retriever.retrieve(q, top_k=RETRIEVAL_CANDIDATE_TOP_K):
                key = d.content
                if key in pool:
                    pool[key] = Document(
                        content=d.content,
                        source=d.source,
                        score=min(1.0, max(pool[key].score, d.score) + 0.05),
                        metadata=d.metadata,
                    )
                else:
                    pool[key] = d
        except Exception as e:
            print(f"[RETRIEVE] '{q[:40]}' failed (non-fatal): {e}")

    # 통합 풀 상위 20개 유지 (5 → 20). Generator 컨텍스트 폭주 방지는
    # downstream 의 grader (reranker threshold) 가 담당.
    docs = sorted(pool.values(), key=lambda d: d.score, reverse=True)[:RETRIEVAL_CANDIDATE_TOP_K]

    # Phase G: 같은 page의 다른 chunks 함께 fetch (표·본문 함께 보기)
    if hasattr(retriever, "expand_with_same_page"):
        try:
            docs = retriever.expand_with_same_page(docs)
        except Exception as e:
            print(f"[RETRIEVE] co-retrieval failed (non-fatal): {e}")

    # Phase G: doc_topic boost (router가 query_doc_topic 분류 시 활성화)
    if hasattr(retriever, "boost_by_query_context"):
        try:
            docs = retriever.boost_by_query_context(docs)
        except Exception as e:
            print(f"[RETRIEVE] boost failed (non-fatal): {e}")

    # 최종 top 7 (co-retrieval로 늘었으니 약간 더)
    docs = docs[:RETRIEVAL_FINAL_K]

    # Layer 3 — 간접 프롬프트 인젝션 방어 (사내 문서 경유).
    # v1 은 knowledge_search 에서 sanitize_docs 를 호출했으나 v2 전환 시 누락되어
    # 프로덕션에서 미작동 상태였다. chunk 원문이 generator 의 system prompt 에
    # 그대로 주입되므로, 문서에 심긴 지시문이 곧 모델 지시가 된다.
    #
    # grade 이전에 수행한다: 차단된 chunk 는 차단 메시지로 대체되어
    # reranker score 가 낮아지고 자연히 필터링된다.
    docs, blocked_n = _sanitize_docs(docs)

    path = f"retrieve:{len(queries)}q→{len(docs)}docs(co+boost)"
    if blocked_n:
        path += f"+sanitized({blocked_n})"

    return {
        "retrieved_docs": docs,
        "decision_path": [path],
    }


# ────────────────────────────────────────────────────────────
# Rewrite node (FR-302, FR-304) — Negative Feedback 주입
# ────────────────────────────────────────────────────────────
#
# 두 케이스 분기:
#  Case A — 직전 retrieve가 빈 결과 (또는 entity 추출 실패)
#           → 일반화 prompt
#  Case B — 직전 chunks 있었으나 grader가 모두 irrelevant 판정
#           → extraneous entities (chunk_entities − query_entities) 를 회피 키워드로 명시
#
# anchor: state.original_input (router에서 1회 초기화, 이후 불변)
# rewrite 결과: state.input_data 만 갱신, original_input은 보존

_PROMPT_DOC_AWARE = """당신은 사내 RAG 시스템의 query rewriter 입니다.
직전 검색이 정답을 못 찾았으니 사내 문서 카탈로그를 참고하여 재작성합니다.

[원본 사용자 질문]
{original}

[직전 시도 쿼리]
{current}

[직전 retrieve 결과 — grader 가 모두 무관 판정]
- 회피해야 할 키워드: {extraneous}

[사내 문서 카탈로그]
{catalog}

【재작성 원칙】
1. 카탈로그에서 사용자 의도와 가장 가까운 1~2개 문서를 식별
2. 그 문서의 "관련 어휘" 중 사용자 질의에 없던 것을 query 에 반영
   (예: 사용자 "재택근무" → 카탈로그에 "원격 근무" 발견 → query 에 추가)
3. 원본 질문의 핵심 entity 는 유지
4. 회피 키워드 방향으로 가는 표현 사용 금지
5. 군더더기 제거, 검색에 효과적인 핵심어만

【출력】 재작성된 query 만 한 줄. 다른 텍스트 금지.
"""


def _collect_chunk_entities(docs: list[Document]) -> set[str]:
    """retrieved_docs의 metadata.entities (공백 join 문자열) 합집합."""
    out: set[str] = set()
    for d in docs:
        ent_str = (d.metadata or {}).get("entities", "") or ""
        for e in ent_str.split():
            e = e.strip().lower()
            if e:
                out.add(e)
    return out


def _collect_chunk_topics(docs: list[Document]) -> set[str]:
    out: set[str] = set()
    for d in docs:
        t = (d.metadata or {}).get("doc_topic", "") or ""
        if t and t != "general":
            out.add(t)
    return out


def _parse_rewrite_response(raw, original_input: str) -> str:
    """LLM 응답에서 첫 줄 추출 + fallback 처리."""
    text = extract_text_content(raw)
    if not text:
        return original_input
    first_line = text.splitlines()[0].strip()
    if not first_line:
        return original_input
    # 따옴표 / 마크다운 fence 제거
    first_line = first_line.strip("`\"'“”").strip()
    if not first_line or first_line == original_input.strip():
        return original_input
    return first_line


def rewrite_node(state: GraphState) -> dict:
    """Grader 가 모두 irrelevant 판정 시 query 재작성.

    Doc-aware 방식:
      - 사내 문서 카탈로그(doc_summary + key_terms)를 LLM 에 주입
      - LLM 이 단순 일반 지식 추측이 아니라 실제 코퍼스 어휘를 참조하여 재작성
      - LLM 이 코퍼스를 모르는 본질적 한계 보완
    """
    original = (state.original_input or state.input_data or "").strip()
    current = (state.input_data or "").strip()
    iter_next = state.retrieval_iterations + 1
    docs: list[Document] = state.retrieved_docs or []

    # 회피 키워드 추출 (직전 retrieve 가 빗나간 방향)
    query_entities = set(extract_entities(original))
    chunk_entities = _collect_chunk_entities(docs)
    extraneous = sorted(chunk_entities - {e.lower() for e in query_entities})

    # 사내 문서 카탈로그 (메모리 캐시)
    catalog = _build_doc_catalog()

    prompt = _PROMPT_DOC_AWARE.format(
        original=original,
        current=current,
        extraneous=", ".join(extraneous[:8]) or "(없음)",
        catalog=catalog,
    )

    # LLM 호출
    fallback = False
    try:
        resp = get_llm().invoke([HumanMessage(content=prompt)])
        new_query = _parse_rewrite_response(resp.content, original)
        if new_query == original or new_query == current:
            fallback = True
            new_query = original
    except Exception as e:
        print(f"[REWRITE] failed (non-fatal): {e}")
        new_query = original
        fallback = True

    label_suffix = (
        f"rewrite:doc_aware({iter_next},avoid={len(extraneous)})"
        if not fallback else
        f"rewrite:fallback({iter_next})"
    )

    return {
        "input_data": new_query,
        "sub_questions": [],  # 변형 reset (다음 retrieve가 새 쿼리로 단일 검색)
        "retrieval_iterations": iter_next,
        "llm_call_count": state.llm_call_count + 1,
        "decision_path": [label_suffix],
    }


# ────────────────────────────────────────────────────────────
# Subgraph builder
# ────────────────────────────────────────────────────────────

def _route_after_grade_with_limit(state: GraphState) -> str:
    """rewrite 한도: RETRIEVAL_MAX_REWRITES (기본 1회, 이전 3회에서 축소).

    근거: 1회 doc-aware rewrite 로 코퍼스 어휘 격차 케이스 대부분 해소.
    1회로 못 찾는 케이스는 코퍼스에 없는 정보거나 query 가 너무 모호한 경우라
    추가 rewrite 로도 회복 안 됨. 빠른 fail 이 사용자 UX 에 더 적합.
    """
    # 한도 판정은 route_after_grade 가 RETRIEVAL_MAX_REWRITES 로 단독 수행한다.
    # 여기서는 서브그래프 종료 라벨로 매핑만 한다 (한도를 두 곳에서 세지 않는다).
    return "rewrite" if route_after_grade(state) == "rewrite" else "end"


def build_retrieve_subgraph():
    g = StateGraph(GraphState)
    g.add_node("query_planner", query_planner_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grader_node)
    g.add_node("rewrite", rewrite_node)

    g.set_entry_point("query_planner")
    g.add_edge("query_planner", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", _route_after_grade_with_limit, {"rewrite": "rewrite", "end": END})
    g.add_edge("rewrite", "retrieve")

    return g.compile()
