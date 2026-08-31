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

import os

from app.core.config import RETRIEVAL_MAX_REWRITES
from app.graph_v2.states.state import GraphState
from app.graph_v2.retrievers.base import Document
from app.graph_v2.retrievers.reranker import (
    get_reranker,
    RERANK_RELEVANT_THRESHOLD,
    RERANK_PARTIAL_THRESHOLD,
)


# 리랭커 사용 방식.
#
#   replace  기본. 리랭커 점수로 순위를 완전히 대체한다.
#   fuse     RRF 순위와 리랭커 순위를 Reciprocal Rank Fusion 으로 합친다.
#   off      리랭커를 쓰지 않고 RRF 순위를 그대로 사용한다.
#
# 전환 가능하게 둔 이유 — 이 코퍼스에서 리랭커의 변별력이 낮다는 실측이 있다.
# 슬라이드 PDF 를 평탄화한 텍스트라 문장 구조가 깨져 있고, 한국어 질의와
# 영문 혼재 표가 섞여 리랭커가 신호를 잡지 못한다:
#
#   질의 "gitlab dap에 적용할 수 있는 모델들이 어떤 게 있나요?"
#     RRF 순위      4위  (정답 chunk 를 제대로 올림)
#     리랭커 순위   10위  score=0.524  -> 상위 7 컷오프에서 탈락
#     20개 중 17개가 0.50~0.54 구간 (sigmoid(0)=0.5 = "판단 불가")
#
# 【기본값을 off 로 정한 근거 — 실측】
#
# 1) recall 에 영향이 없다.
#    retrieve_node 가 RETRIEVAL_FINAL_K 로 자른 뒤 grader 에 넘기므로,
#    어떤 문서가 generator 에 도달하는지는 RRF + co-retrieval + boost 가
#    전적으로 결정한다. 리랭커는 그 안의 순서만 바꾼다.
#
# 2) 정확도를 떨어뜨린다.  질문 5개 x 키워드 적중 / 평균 응답시간
#      FINAL=7  리랭커 replace   10/24 (42%)   59.9s
#      FINAL=15 리랭커 off       13/24 (54%)   19.4s
#      FINAL=15 리랭커 fuse      12/24 (50%)  107.1s
#      FINAL=20 리랭커 off       17/24 (71%)   21.4s   <- 채택
#
# 3) 비용이 크다. 쌍당 4~6초 (2 vCPU ARM, max_length=512).
#    코드 주석의 '~50ms/pair' 는 x86 기준으로 80배 차이가 난다.
#
# 4) 이 코퍼스에서 변별력이 없다. 20개 chunk 중 17개가 0.50~0.54 구간
#    (sigmoid(0)=0.5 = 판단 불가) 에 몰렸다. 슬라이드 PDF 평탄화 텍스트라
#    문장 구조가 깨져 있고 한국어 질의와 영문 표가 섞이기 때문이다.
#
# off 이면 모델을 로드하지 않아 메모리 2.3GB 도 회수된다.
# 코퍼스 성격이 다른 환경(정제된 산문 등)에서는 replace 가 유리할 수 있어
# 코드는 남겨 두고 기본값만 off 로 둔다.
RERANK_MODE = (os.getenv("RERANK_MODE", "off") or "off").strip().lower()


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

    rrf_order = list(docs)   # retrieve 가 넘겨준 순서 = RRF 순위

    if RERANK_MODE == "off":
        # 리랭커를 쓰지 않고 RRF 순위를 그대로 신뢰한다.
        scores = [d.score for d in rrf_order]
        kept = list(rrf_order)
        relevant_count = len(kept)
    else:
        reranker = get_reranker()
        scores = reranker.score(state.input_data, [d.content or "" for d in docs])

        # threshold 기반 라벨링 + kept 구성. doc.score를 reranker score로 갱신
        # (downstream generator/reflection이 더 정확한 relevance 신호 사용 가능).
        # (원래 RRF 순위, 문서) 를 함께 들고 다닌다.
        # 새 Document 객체를 만들기 때문에 id() 로는 원 순위를 되찾을 수 없다.
        picked: list[tuple[int, Document]] = []
        relevant_count = 0
        for rrf_idx, (d, s) in enumerate(zip(docs, scores)):
            if s >= RERANK_RELEVANT_THRESHOLD:
                relevant_count += 1
            elif s < RERANK_PARTIAL_THRESHOLD:
                continue
            picked.append((rrf_idx,
                           Document(content=d.content, source=d.source, score=s, metadata=d.metadata)))

        if RERANK_MODE == "fuse":
            # RRF 순위와 리랭커 순위를 융합한다 (Reciprocal Rank Fusion).
            #
            # 근거: 리랭커가 이 코퍼스에서 변별력이 낮다. 실측 20개 chunk 중
            # 17개가 0.50~0.54 (sigmoid(0)=0.5, 즉 "판단 불가") 구간에 몰려
            # 정답을 RRF 4위 -> 10위로 강등시킨 사례가 있었다.
            # 한쪽 순위를 버리지 않고 두 신호를 합쳐 단일 실패를 완화한다.
            K = 60  # RRF 표준 상수
            # 리랭커 점수 기준 순위를 picked 내 위치별로 구한다
            by_rr = sorted(range(len(picked)), key=lambda i: picked[i][1].score, reverse=True)
            rr_rank = {pos: rank for rank, pos in enumerate(by_rr)}
            fused = [
                (1.0 / (K + rrf_idx) + 1.0 / (K + rr_rank[pos]), rrf_idx, doc)
                for pos, (rrf_idx, doc) in enumerate(picked)
            ]
            fused.sort(key=lambda t: t[0], reverse=True)
            picked = [(rrf_idx, doc) for _, rrf_idx, doc in fused]
        else:
            # "replace" (기본) — 리랭커 순위로 완전 대체
            picked.sort(key=lambda pair: pair[1].score, reverse=True)

        kept = [doc for _, doc in picked]

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
