"""
Format Classifier — 파일 확장자 기반 형식 분류 (1단계).

확장자만 보고 결정. magic byte 검증은 file_extractor가 별도 수행.
"""
from __future__ import annotations

from pathlib import Path


SUPPORTED_FORMATS = {"pptx", "pdf", "docx", "txt", "md", "xlsx"}

_EXT_MAP = {
    ".pptx": "pptx", ".ppt": "pptx",
    ".pdf": "pdf",
    ".docx": "docx", ".doc": "docx",
    ".xlsx": "xlsx", ".xlsm": "xlsx", ".xls": "xlsx",
    ".txt": "txt",
    ".md": "md",
}


def classify_format(path: Path) -> str:
    """확장자 → format 문자열. 미지원 시 'unknown' 반환."""
    return _EXT_MAP.get(path.suffix.lower(), "unknown")


def is_supported(path: Path) -> bool:
    """청킹 파이프라인이 처리할 수 있는 형식인지 확인."""
    return classify_format(path) in SUPPORTED_FORMATS
