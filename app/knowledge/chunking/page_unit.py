"""
PageUnit — 형식별 extractor가 반환하는 통일 단위.

각 형식의 "자연 단위"를 동일 schema로 표현:
  - PPTX → 슬라이드 1개 = PageUnit 1개 (표는 별도 PageUnit, unit_type="table")
  - PDF  → 페이지 1개 = PageUnit 1개
  - DOCX → 헤딩 섹션 1개 = PageUnit 1개 (큰 표는 별도)
  - TXT  → 헤딩·번호 섹션 1개 = PageUnit 1개
  - XLSX → 시트 1개 = PageUnit 1개

이 통일 schema로 size_analyzer + chunker가 형식 무관하게 작업 가능.
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class PageUnit(BaseModel):
    """문서의 자연 단위 (슬라이드/페이지/섹션). chunker의 입력."""

    # ─── 위치 ───────────────────────────────────────────────
    unit_index: int                       # 1-indexed (슬라이드 1, 페이지 1, ...)
    unit_type: str                        # "slide" | "page" | "section" | "table"
    title: str = ""                       # 슬라이드 제목 / 섹션 헤딩 / 페이지 첫 줄
    section_path: str = ""                # 헤딩 계층 (예: "PART 03 / 진입 검증")

    # ─── 내용 ───────────────────────────────────────────────
    text: str                             # 본문 텍스트

    # ─── 표 관련 (해당 시) ─────────────────────────────────
    is_table: bool = False
    table_index: Optional[int] = None     # 페이지 내 표 순서 (1, 2, ...)
    parent_page_index: Optional[int] = None  # 표의 경우, 본문 페이지의 unit_index

    # ─── 형식별 추가 (raw) ─────────────────────────────────
    raw_metadata: dict[str, Any] = Field(default_factory=dict)
    # 형식별 활용:
    #   PPTX: {"notes": "발표자 노트", "shape_count": 5}
    #   PDF:  {"page_size": (w, h)}
    #   DOCX: {"heading_level": 2}

    @property
    def char_count(self) -> int:
        return len(self.text)


class Chunk(BaseModel):
    """청킹 후 ChromaDB에 저장될 단위. chunker의 출력."""

    chunk_id: str                         # 예: "01.brand::page_19::chunk_0"
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
