"""
4차 방어 — LLM 응답 출력 검증 (규칙 기반).

인젝션이 1·2·3차를 통과하더라도, 응답에 민감 정보가 노출되면 차단합니다.
정상 응답은 그대로 통과합니다.
"""

from __future__ import annotations

import re

# 응답에서 차단할 민감 패턴
_SENSITIVE_PATTERNS = [
    # 내부 툴 함수명 (시스템 내부 구현 노출)
    re.compile(r"\bsearch_knowledge_base\b", re.I),
    re.compile(r"\bget_attached_file\b", re.I),

    # 환경변수 값 노출 패턴 (변수명=값 형태)
    re.compile(r"(GOOGLE_API_KEY|GEMINI_API_KEY)\s*[=:]\s*\S+", re.I),
    re.compile(r"(AWS_SECRET_ACCESS_KEY|AWS_ACCESS_KEY_ID)\s*[=:]\s*\S+", re.I),
    re.compile(r"SESSION_SECRET\s*[=:]\s*\S+", re.I),

    # 내부 DB·경로 노출
    re.compile(r"\bchroma_db\b", re.I),
    re.compile(r"\bCHROMA_DB_PATH\b", re.I),
    re.compile(r"\bCHROMA_COLLECTION\b", re.I),
    re.compile(r"\blanggraph_checkpoints\b", re.I),

    # 시스템 프롬프트 내용 직접 인용 징후 (20자 이상 연속 노출)
    re.compile(r"당신의 이름은 Kaiper AI입니다.{20,}", re.I),
    re.compile(r"아래 지침을 따르세요.{10,}", re.I),
]

_SAFE_RESPONSE = (
    "요청하신 내용에 대한 답변을 제공하기 어렵습니다. "
    "사내 업무 관련 질문을 입력해 주세요."
)


def validate(response_text: str) -> tuple[bool, str]:
    """
    응답 텍스트에 민감 패턴이 있으면 차단 메시지로 대체합니다.

    Returns:
        (True, 원본 응답)   → 정상
        (False, 안전 메시지) → 차단
    """
    if not response_text:
        return True, response_text

    for pattern in _SENSITIVE_PATTERNS:
        if pattern.search(response_text):
            print(f"[OUTPUT_VALIDATOR] sensitive pattern detected - blocked: pattern={pattern.pattern[:50]}")
            return False, _SAFE_RESPONSE

    return True, response_text
