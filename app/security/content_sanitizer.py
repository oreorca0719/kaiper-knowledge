"""
3차 방어 — RAG 검색 결과 및 파일 첨부 내용 sanitize.

문서/파일 내에 인젝션 텍스트가 포함된 경우 해당 내용을 차단 메시지로 대체합니다.
정상 내용은 그대로 통과합니다.
"""

from __future__ import annotations

import re

# 문서·파일 내 인젝션 감지 패턴
#
# 【설계 원칙 — 명사구 단독으로 차단하지 않는다】
# 인젝션은 "무엇을(명사)"이 아니라 "무엇을 하라(명령)"로 성립한다.
# 명사구만 보면 정상 기술 문서를 대량 오탐한다.
#
# 실측 오탐 (수정 전, 패턴 `system\s*(prompt|override|instruction)`):
#     GitLab 제품 설명 chunk 2건이 차단됨
#     "Custom Agent 는 ... System Prompt 를 통해 동작을 정의하고,
#      접근할 수 있는 도구를 선택합니다."
#   → 사용자 질의 2건이 "관련 문서를 찾을 수 없습니다" 로 실패
#
# "system prompt" / "instruction" / "override" 는 AI·DevOps 문서의 일상 용어다.
# 코드·이슈를 색인하는 환경(PMS)에서는 오탐이 더 심해진다.
# 따라서 **명령 동사와 결합될 때만** 인젝션으로 본다.
_INJECTION_PATTERNS = [
    # ── 지침 무시 지시 (명령형) ──
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|rules?)", re.I),
    re.compile(r"forget\s+(your|all)\s+(persona|instructions?|rules?)", re.I),
    re.compile(r"이전\s*(지침|명령|규칙)\s*(을|를)?\s*무시", re.I),
    re.compile(r"모든\s*(규칙|지침|명령)\s*(을|를)?\s*무시", re.I),

    # ── 내부 정보 요구 (동사 + 목적어) ──
    # 명사구 단독("system prompt")은 매치하지 않는다.
    re.compile(
        r"(reveal|show|print|output|dump|expose|disclose|tell\s+me)\s+"
        r"(me\s+)?(your|the)?\s*"
        r"(system\s*prompt|system\s*instructions?|initial\s*instructions?|"
        r"api\s*key|secret\s*key|credentials?)",
        re.I,
    ),
    re.compile(
        r"(시스템\s*프롬프트|내부\s*지침|시스템\s*지침|프롬프트\s*내용)"
        r"\s*(을|를|이|가)?\s*[^.\n]{0,20}?"
        r"(출력|공개|보여|알려|보고|노출|말해|드러)",
        re.I,
    ),

    # ── 역할·모드 강제 전환 ──
    re.compile(r"(DAN|do\s+anything\s+now)\s+mode", re.I),
    re.compile(r"you\s+are\s+now\s+(DAN|a\s+different|an?\s+unrestricted)", re.I),
    re.compile(r"(제약|제한)\s*(없는|없이)\s*AI\s*(로|처럼|인\s*척)", re.I),
    re.compile(r"지금부터\s*너는\s*[^.\n]{0,20}(AI|봇|어시스턴트)", re.I),

    # ── 형식 우회 마커 (정상 문서에 나타나지 않는 토큰) ──
    re.compile(r"\[IGNORE\s*PREVIOUS\]", re.I),
    re.compile(r"<<\s*SYS\s*>>", re.I),
    re.compile(r"system\s*override\s*[:：]", re.I),

    # ── 실제 자격증명 노출 (값이 붙어 있을 때만) ──
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
