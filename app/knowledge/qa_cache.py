"""
Q&A Cache — 자주 들어오는 질문에 대한 정답 캐시 (별도 ChromaDB collection).

설계 원칙:
1. **질문 텍스트를 임베딩** — 질문 유사도 매칭으로 lookup.
2. **답변 + citation은 metadata로 저장** — Chroma document=question, metadata=answer/source.
3. **별도 collection** — 본 지식 베이스(`my_knowledge`)와 분리, 오염 방지.
4. **threshold 기반 hit/miss** — 임계값 미만은 cache miss로 처리, 일반 retrieve로 fallback.

동작:
  build_from_labels(path)  → labels.json의 (question, answer, citation) 적재
  lookup(query, threshold) → 가장 유사한 Q&A 반환 (없으면 None)
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma

from app.core.config import (
    CHROMA_DB_PATH,
    get_embeddings,
)

# 별도 collection — 본 지식 베이스와 분리
QA_CACHE_COLLECTION = "qa_cache"

# 매칭 임계값 (cosine similarity 0~1 정규화 기준)
# - 0.92 이상: 사실상 동일 질문 (fast-path bypass)
# - 0.78~0.92: 유사 질문 (hint로 generator에 주입)
# - 0.78 미만: cache miss
QA_BYPASS_THRESHOLD = 0.92
QA_HINT_THRESHOLD = 0.78


@dataclass
class QAHit:
    """Q&A cache lookup 결과."""
    question: str           # 캐시된 질문
    answer: str             # 정답
    doc_id: str             # 출처 doc_id
    snippet: str            # 발췌 (citation용)
    location: str           # "Slide 3", "Page 2" 등
    answer_type: str        # exact_phrase / numerical / list_n / ...
    score: float            # 0~1 유사도
    is_bypass: bool         # True = fast-path bypass, False = hint only

    @property
    def confidence_band(self) -> str:
        return "bypass" if self.is_bypass else "hint"


class QACache:
    """질문 임베딩 기반 Q&A 캐시.

    내부적으로 ChromaDB의 별도 collection 사용 (`qa_cache`).
    document = 질문 텍스트, metadata = answer/doc_id/snippet/location/answer_type.
    """

    def __init__(self) -> None:
        self._chroma: Optional[Chroma] = None
        self._lock = threading.Lock()

    def _get_chroma(self) -> Chroma:
        if self._chroma is None:
            with self._lock:
                if self._chroma is None:
                    self._chroma = Chroma(
                        persist_directory=CHROMA_DB_PATH,
                        embedding_function=get_embeddings(),
                        collection_name=QA_CACHE_COLLECTION,
                    )
        return self._chroma

    # ─── 적재 ────────────────────────────────────────────

    def upsert(self, entries: list[dict]) -> int:
        """Q&A 항목 일괄 적재. 같은 id는 덮어씀.

        entries 형식: [
          {
            "id": "qa_001",
            "question": "...",
            "answer": "...",
            "doc_id": "...",
            "snippet": "...",
            "location": "Slide 3",
            "answer_type": "exact_phrase",
          },
          ...
        ]
        """
        if not entries:
            return 0

        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict] = []
        for e in entries:
            qid = str(e.get("id") or "").strip()
            question = (e.get("question") or "").strip()
            answer = (e.get("answer") or "").strip()
            if not qid or not question or not answer:
                continue
            ids.append(qid)
            docs.append(question)
            metas.append({
                "answer": answer,
                "doc_id": str(e.get("doc_id") or ""),
                "snippet": str(e.get("snippet") or "")[:500],
                "location": str(e.get("location") or ""),
                "answer_type": str(e.get("answer_type") or "reasoning"),
            })

        if not ids:
            return 0

        chroma = self._get_chroma()
        # 기존 같은 id 삭제 후 추가 (langchain_chroma의 add_texts는 ids 충돌 시 덮어쓰지 않음)
        try:
            chroma._collection.delete(ids=ids)
        except Exception:
            pass
        chroma.add_texts(texts=docs, metadatas=metas, ids=ids)
        return len(ids)

    def build_from_labels(self, labels_path: Path) -> int:
        """eval/data/labels.json (또는 동일 schema) 파일에서 Q&A 적재.

        labels.json 항목 schema:
          {
            "id": int, "question": str, "answer": str, "answer_type": str,
            "citation": {"doc_id": str, "snippet": str, "location": str},
            "confidence": float, "needs_review": bool, ...
          }

        confidence < 0.7 또는 needs_review=True인 항목은 스킵 (오답 캐시 방지).
        """
        if not labels_path.exists():
            print(f"[QA_CACHE] labels file not found: {labels_path}")
            return 0
        raw = json.loads(labels_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            print(f"[QA_CACHE] labels file not a list: {labels_path}")
            return 0

        entries: list[dict] = []
        skipped_lowconf = 0
        skipped_review = 0
        for item in raw:
            if not isinstance(item, dict):
                continue
            if item.get("needs_review"):
                skipped_review += 1
                continue
            conf = item.get("confidence", 1.0)
            try:
                conf = float(conf)
            except Exception:
                conf = 1.0
            if conf < 0.7:
                skipped_lowconf += 1
                continue

            citation = item.get("citation") or {}
            entries.append({
                "id": f"qa_{item.get('id')}",
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "doc_id": citation.get("doc_id", ""),
                "snippet": citation.get("snippet", ""),
                "location": citation.get("location", ""),
                "answer_type": item.get("answer_type", "reasoning"),
            })

        n = self.upsert(entries)
        print(
            f"[QA_CACHE] build_from_labels: {n}개 적재 "
            f"(스킵 needs_review={skipped_review}, low_conf={skipped_lowconf})"
        )
        return n

    # ─── 조회 ────────────────────────────────────────────

    def lookup(
        self,
        query: str,
        bypass_threshold: float = QA_BYPASS_THRESHOLD,
        hint_threshold: float = QA_HINT_THRESHOLD,
    ) -> Optional[QAHit]:
        """가장 유사한 Q&A 반환. hint_threshold 미만은 None."""
        if not query or not query.strip():
            return None
        try:
            chroma = self._get_chroma()
            results = chroma.similarity_search_with_score(query, k=1)
        except Exception as e:
            print(f"[QA_CACHE] lookup failed (non-fatal): {e}")
            return None

        if not results:
            return None

        doc, dist = results[0]
        # Chroma cosine distance → 0~1 similarity
        sim = 1.0 / (1.0 + max(dist, 0.0))
        if sim < hint_threshold:
            return None

        md = doc.metadata or {}
        return QAHit(
            question=doc.page_content or "",
            answer=str(md.get("answer", "")),
            doc_id=str(md.get("doc_id", "")),
            snippet=str(md.get("snippet", "")),
            location=str(md.get("location", "")),
            answer_type=str(md.get("answer_type", "reasoning")),
            score=sim,
            is_bypass=sim >= bypass_threshold,
        )

    def count(self) -> int:
        try:
            return self._get_chroma()._collection.count()
        except Exception:
            return 0

    def clear(self) -> None:
        """전체 cache 삭제 (테스트·재빌드용)."""
        try:
            chroma = self._get_chroma()
            ids = chroma._collection.get(include=[]).get("ids", [])
            if ids:
                chroma._collection.delete(ids=ids)
            print(f"[QA_CACHE] cleared: {len(ids)}개 삭제")
        except Exception as e:
            print(f"[QA_CACHE] clear failed: {e}")


# ─── 싱글턴 ─────────────────────────────────────────────

_cache: Optional[QACache] = None
_cache_lock = threading.Lock()


def get_qa_cache() -> QACache:
    global _cache
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                _cache = QACache()
    return _cache
