"""
Grader node — 검색 결과의 관련성 평가 (FR-301, FR-302, FR-303, FR-304).

판정 방식:
  Cross-encoder reranker (BAAI/bge-reranker-v2-m3) 가 (query, chunk) 페어마다
  0~1 relevance score 산출. threshold 기반 라벨링.

  - score >= RERANK_RELEVANT_THRESHOLD       → "relevant"
  - score >= RERANK_PARTIAL_THRESHOLD         → "partially_relevant"
  - else                                       → "irrelevant"

LLM 호출 0회 (LLM grader 폐기 — 비용·동질편향·비결정성 해소).

흐름:
  1. reranker로 모든 chunk score 산출 (1회 배치 추론)
  2. relevant + partially_relevant chunk만 retrieved_docs에 유지 (FR-401)
  3. kept 비면 → route_after_grade가 rewrite 트리거 (RETRIEVAL_MAX_REWRITES 한도까지)
"""
from __future__ import annotations

from app.core.config import RETRIEVAL_MAX_REWRITES
from app.graph_v2.states.state import GraphState
from app.graph_v2.retrievers.base import Document
from app.graph_v2.retrievers.reranker import (
    get_reranker,
    RERANK_RELEVANT_THRESHOLD,
    RERANK_PARTIAL_THRESHOLD,
)


def _coverage_from_scores(scores: list[float], relevant_count: int) -> str:
    """Reranker score 분포로 premise_coverage 휴리스틱 산출 (LLM premise_coverage 대체).

    - 모든 score가 PARTIAL 미만: "none" — retrieve가 완전히 빗나감
    - relevant chunk 2개 이상: "full"
    - 그 외: "partial"
    """
    if not scores or max(scores) < RERANK_PARTIAL_THRESHOLD:
        return "none"
    if relevant_count >= 2:
        return "full"
    return "partial"


def grader_node(state: GraphState) -> dict:
    """Cross-encoder로 chunks score → threshold 기반 필터링.

    Rescue 로직 없음 — 모두 irrelevant면 빈 kept 그대로 반환하여
    route_after_grade가 rewrite를 트리거하게 함 (RETRIEVAL_MAX_REWRITES 한도까지).
    """
    docs: list[Document] = state.retrieved_docs or []
    if not docs:
        return {
            "decision_path": ["grade:no_docs"],
        }

    reranker = get_reranker()
    scores = reranker.score(state.input_data, [d.content or "" for d in docs])

    # threshold 기반 라벨링 + kept 구성. doc.score를 reranker score로 갱신
    # (downstream generator/reflection이 더 정확한 relevance 신호 사용 가능).
    kept: list[Document] = []
    relevant_count = 0
    for d, s in zip(docs, scores):
        if s >= RERANK_RELEVANT_THRESHOLD:
            relevant_count += 1
            kept.append(Document(content=d.content, source=d.source, score=s, metadata=d.metadata))
        elif s >= RERANK_PARTIAL_THRESHOLD:
            kept.append(Document(content=d.content, source=d.source, score=s, metadata=d.metadata))

    # kept 정렬 (reranker score 내림차순 — generator가 상위부터 보게)
    kept.sort(key=lambda d: d.score, reverse=True)

    coverage = _coverage_from_scores(scores, relevant_count)
    max_s = max(scores) if scores else 0.0

    if not kept:
        if state.retrieval_iterations >= RETRIEVAL_MAX_REWRITES:
            path_label = (
                f"grade:exhausted_after_{RETRIEVAL_MAX_REWRITES}rewrites(0rel/{len(docs)}total,"
                f"max={max_s:.2f},cov={coverage})"
            )
        else:
            path_label = (
                f"grade:all_irrelevant(0rel/{len(docs)}total,"
                f"max={max_s:.2f},cov={coverage})"
            )
    else:
        path_label = (
            f"grade:pass({relevant_count}rel/{len(kept)}kept/{len(docs)}total,"
            f"max={max_s:.2f},cov={coverage})"
        )

    return {
        "retrieved_docs": kept,
        "decision_path": [path_label],
        # llm_call_count 증가 안 함 — reranker는 LLM이 아님
    }


def route_after_grade(state: GraphState) -> str:
    """grade 통과 여부 + retry 한도.

    - kept = []  && iters < RETRIEVAL_MAX_REWRITES  → rewrite
    - kept = []  && iters >= RETRIEVAL_MAX_REWRITES → generate (no_docs path, "찾을 수 없습니다")
    - kept = [.] → generate
    """
    docs = state.retrieved_docs or []
    iters = state.retrieval_iterations
    if not docs:
        if iters < RETRIEVAL_MAX_REWRITES:
            return "rewrite"
        return "generate"  # 한도 도달
    return "generate"
