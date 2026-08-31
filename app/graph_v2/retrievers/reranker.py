"""
Cross-encoder reranker — grader LLM 호출 대체.

설계:
- BAAI/bge-reranker-v2-m3 (다국어, 한국어 OK, 567M params)
- (query, chunk) 페어 → relevance logit → sigmoid → 0~1 score
- LLM 호출 없음. 결정적 (재현 가능). ~50ms/페어 (CPU).

사용:
    from app.graph_v2.retrievers.reranker import get_reranker
    scores = get_reranker().score(query, [d.content for d in docs])
    # → [0.87, 0.41, 0.05, ...]

첫 호출 시 모델 다운로드 (~600MB) + CPU 메모리 로드 (~1GB).
의존성: sentence-transformers, torch (lazy import).
"""
from __future__ import annotations

import math
import os
import threading
from typing import Optional


# 환경변수로 튜닝 가능
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_RELEVANT_THRESHOLD = float(os.getenv("RERANK_RELEVANT_THRESHOLD", "0.5"))
RERANK_PARTIAL_THRESHOLD = float(os.getenv("RERANK_PARTIAL_THRESHOLD", "0.3"))
RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "512"))


class Reranker:
    """Cross-encoder 모델 wrapper. Singleton 사용 권장 (모델 로드 비용 큼)."""

    def __init__(self, model_name: str = RERANK_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        """Lazy load — sentence-transformers, torch는 첫 score() 호출 시 import."""
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            print(f"[RERANK] loading {self.model_name} (first call — may take ~30s, downloads ~600MB on first run)")
            try:
                from sentence_transformers import CrossEncoder  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "sentence-transformers is required for reranker. "
                    "Install with: pip install sentence-transformers"
                ) from e
            self._model = CrossEncoder(self.model_name, max_length=RERANK_MAX_LENGTH)
            print(f"[RERANK] model ready (max_length={RERANK_MAX_LENGTH})")
        return self._model

    def warmup(self) -> None:
        """평가 시작 전 미리 모델 로드 (첫 질문 지연 방지)."""
        self._load()
        # 더미 추론 1회 — JIT/캐시 워밍업
        try:
            self._load().predict([("warmup query", "warmup text")])
        except Exception:
            pass

    def score(self, query: str, texts: list[str]) -> list[float]:
        """(query, text) 페어들에 대해 sigmoid 정규화된 0~1 relevance score.

        bge-reranker-v2-m3는 raw logit 출력 — 음수면 무관, 양수면 관련.
        Sigmoid 적용으로 0~1 정규화하여 threshold 비교 단순화.
        """
        if not texts:
            return []
        if not query or not query.strip():
            return [0.0] * len(texts)

        model = self._load()
        pairs = [(query, t or "") for t in texts]
        try:
            raw = model.predict(pairs)
        except Exception as e:
            print(f"[RERANK] predict failed (non-fatal): {e}")
            return [0.0] * len(texts)

        # numpy array → list. sigmoid: 1 / (1 + exp(-x))
        return [1.0 / (1.0 + math.exp(-float(s))) for s in raw]


# ─── Singleton ─────────────────────────────────────

_reranker: Optional[Reranker] = None
_reranker_lock = threading.Lock()


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = Reranker()
    return _reranker
