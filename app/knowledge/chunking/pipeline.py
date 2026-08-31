"""
Chunking Pipeline — 5단계 통합.

run_pipeline(path) → list[Chunk]
  1. format_classifier로 형식 결정
  2. extractor로 PageUnit 추출
  3. size_analyzer로 적응적 청킹 (PageUnit → 합쳐지거나 분할된 PageUnit)
  4. tagger로 metadata 부착 (코드 + LLM)
  5. Chunk 객체 생성
"""
from __future__ import annotations

from pathlib import Path

from app.knowledge.chunking.format_classifier import classify_format, is_supported
from app.knowledge.chunking.page_unit import Chunk, PageUnit
from app.knowledge.chunking.size_analyzer import adaptive_chunk

# Extractor lazy import (각 라이브러리 의존성 회피)
def _get_extractor(format: str):
    if format == "pptx":
        from app.knowledge.chunking.extractors.pptx_extractor import extract_pptx
        return extract_pptx
    if format == "pdf":
        from app.knowledge.chunking.extractors.pdf_extractor import extract_pdf
        return extract_pdf
    if format == "docx":
        from app.knowledge.chunking.extractors.docx_extractor import extract_docx
        return extract_docx
    if format in ("txt", "md"):
        from app.knowledge.chunking.extractors.txt_extractor import extract_txt
        return extract_txt
    if format == "xlsx":
        from app.knowledge.chunking.extractors.xlsx_extractor import extract_xlsx
        return extract_xlsx
    raise ValueError(f"Unsupported format: {format}")


def extract_units(path: Path) -> tuple[str, list[PageUnit]]:
    """파일 → (format, list[PageUnit])."""
    if not is_supported(path):
        return ("unknown", [])
    fmt = classify_format(path)
    extractor = _get_extractor(fmt)
    units = extractor(path)
    return (fmt, units)


def run_chunking_only(path: Path) -> tuple[str, list[PageUnit]]:
    """1~3단계만 (extract + adaptive chunking). metadata 부착 없음.
    Tagger는 별도 호출 (LLM 호출 분리).
    """
    fmt, units = extract_units(path)
    if not units:
        return (fmt, [])
    chunked = adaptive_chunk(units, fmt)
    return (fmt, chunked)
