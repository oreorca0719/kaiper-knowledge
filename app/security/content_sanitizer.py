"""
3차 방어 — RAG 검색 결과 및 파일 첨부 내용 sanitize.

문서/파일 내에 인젝션 텍스트가 포함된 경우 해당 내용을 차단 메시지로 대체합니다.
정상 내용은 그대로 통과합니다.
"""

from __future__ import annotations

import re

# 문서·파일 내 인젝션 감지 패턴
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.I),
    re.compile(r"forget\s+your\s+(persona|instructions?|rules?)", re.I),
    re.compile(r"system\s*(prompt|override|instruction)", re.I),
    re.compile(r"이전\s*지침\s*(을|를)?\s*무시", re.I),
    re.compile(r"시스템\s*프롬프트\s*(를|을|출력|보여)", re.I),
    re.compile(r"모든\s*규칙\s*(을|를)?\s*무시", re.I),
    re.compile(r"(reveal|show|print|output)\s+(your\s+)?(system\s+prompt|instructions?|api\s*key)", re.I),
    re.compile(r"\[IGNORE\s*PREVIOUS\]", re.I),
    re.compile(r"<<\s*SYS\s*>>", re.I),
    re.compile(r"(DAN|do\s+anything\s+now)\s+mode", re.I),
    re.compile(r"you\s+are\s+now\s+(DAN|a\s+different|an?\s+unrestricted)", re.I),
    re.compile(r"제약\s*(없는|없이)\s*AI", re.I),
    re.compile(r"(api[\s_]?key|access[\s_]?key|secret[\s_]?key)\s*[=:]\s*\S{10,}", re.I),
]

_BLOCK_MESSAGE = "[보안 정책에 의해 해당 내용이 차단되었습니다.]"


def is_injection_content(text: str) -> bool:
    """텍스트에 인젝션 패턴이 포함되어 있으면 True 반환."""
    if not text:
        return False
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


def sanitize(text: str, source: str = "content") -> str:
    """
    인젝션 패턴이 감지되면 차단 메시지로 대체, 아니면 원본 반환.

    Args:
        text: 검사할 텍스트 (문서 내용 또는 파일 내용)
        source: 로그용 출처 식별자

    Returns:
        정상이면 원본 text, 감지되면 _BLOCK_MESSAGE
    """
    if not text:
        return text
    if is_injection_content(text):
        print(f"[SANITIZER] injection detected - blocked: source={source}")
        return _BLOCK_MESSAGE
    return text


def sanitize_docs(docs: list, source: str = "rag") -> list:
    """
    LangChain Document 리스트의 page_content를 sanitize합니다.
    감지된 문서는 page_content를 차단 메시지로 대체합니다.
    """
    from langchain_core.documents import Document
    result = []
    for doc in docs:
        content = getattr(doc, "page_content", "") or ""
        if is_injection_content(content):
            print(f"[SANITIZER] RAG doc injection detected - blocked: source={source}")
            doc = Document(
                page_content=_BLOCK_MESSAGE,
                metadata=getattr(doc, "metadata", {}),
            )
        result.append(doc)
    return result
