"""
Retriever Protocol — 모든 retriever 구현체가 따라야 하는 인터페이스 (FR-202).

설계 원칙:
1. 통일된 인터페이스 — Vector, BM25, Hybrid, SQL, Web 모두 같은 시그니처
2. name + description — Router/Planner가 도구 선택 시 사용 (FR-102)
3. score 정규화 — 0~1 범위, 모든 구현체 (FR-204)
4. source 명시 — chunk가 어느 문서에서 왔는지 식별 가능
5. metadata 보존 — entity, page, timestamp 등 라우팅·재정렬용 정보
6. 동기·비동기 모두 (FR-203) — async가 future-proof
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable
from pydantic import BaseModel, Field


class Document(BaseModel):
    """검색 결과 단위. 모든 retriever가 이 형식으로 반환."""
    content: str
    source: str                     # 문서 식별자 (URL, 파일경로, 테이블명 등)
    score: float = 0.0              # 0~1 정규화 (FR-204)
    metadata: dict = Field(default_factory=dict)
    # 권장 metadata 키:
    # - "title": 표시용 제목
    # - "doc_id": 출처 doc 고유 ID (citation에 사용)
    # - "location": "Slide 3", "Page 2" 등
    # - "entities": list[str] — entity-aware 검색용
    # - "chunk_index", "total_chunks": chunk 위치 정보


@runtime_checkable
class Retriever(Protocol):
    """모든 retriever 구현체의 공통 인터페이스.

    Router/Planner는 `name`과 `description`을 보고 적합한 retriever를 선택한다 (FR-102).
    """
    name: str
    description: str

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """동기 검색."""
        ...

    async def aretrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """비동기 검색."""
        ...


class RetrieverRegistry:
    """이름으로 retriever를 조회하는 레지스트리.

    Router가 `route → registry.get("vector").retrieve(...)` 형태로 사용.
    Multi-retriever 환경에서 도구 카탈로그 역할.
    """

    def __init__(self) -> None:
        self._retrievers: dict[str, Retriever] = {}

    def register(self, retriever: Retriever) -> None:
        if not retriever.name:
            raise ValueError("Retriever must have a non-empty name.")
        self._retrievers[retriever.name] = retriever

    def get(self, name: str) -> Retriever:
        if name not in self._retrievers:
            raise KeyError(f"Retriever '{name}' not registered. Available: {list(self._retrievers)}")
        return self._retrievers[name]

    def names(self) -> list[str]:
        return list(self._retrievers.keys())

    def descriptions(self) -> list[dict]:
        """Router/Planner가 도구 카탈로그로 사용할 형식."""
        return [
            {"name": r.name, "description": r.description}
            for r in self._retrievers.values()
        ]
