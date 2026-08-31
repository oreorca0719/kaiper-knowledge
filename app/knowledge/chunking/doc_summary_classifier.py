"""
doc_summary Classifier — 문서 전체를 LLM 이 1회 보고 요약 + 핵심 어휘 추출.

호출 빈도: 문서당 1회 (인제스트 시점)
용도: rewrite_node 가 사내 코퍼스 어휘 격차 케이스에서 query 재작성 시 참조.
     LLM 이 코퍼스를 모르는 본질적 한계를 보완하는 doc 카탈로그의 원천 데이터.
"""
from __future__ import annotations

import json
import re
from typing import TypedDict

from langchain_core.messages import HumanMessage

from app.core.config import get_llm
from app.core.history_utils import extract_text_content


class DocSummary(TypedDict):
    summary: str
    key_terms: list[str]


_DOC_SUMMARY_PROMPT = """다음 문서의 핵심을 요약하고 검색 인덱스 어휘를 추출하세요.

【출력 JSON 형식】
{{
  "summary": "150~250자 한국어 요약. 문서 주제·도메인·다루는 사실 요점.",
  "key_terms": ["용어1", "용어2", ...]
}}

【key_terms 작성 규칙】
- 5~10개
- 사용자가 같은 정보를 찾을 때 사용할 가능성 있는 어휘 모두 포함
- 동의어·약어 (예: "재택근무, 원격 근무, WFH")
- 사내 고유 용어 (회사명, 제품명, 약어)
- 한국어 우선, 필요 시 영문 병기

【출력】 JSON 만. 다른 텍스트 금지.

[문서 텍스트]
{text}
"""


def classify_doc_summary(doc_text: str, model_name: str | None = None) -> DocSummary:
    """문서 전체 텍스트를 받아 summary + key_terms 추출 (LLM 1회).

    실패 시 빈 값 반환 (graceful degradation — rewrite_node 가 카탈로그
    빌드 시 빈 entry 는 자동 제외).
    """
    fallback: DocSummary = {"summary": "", "key_terms": []}
    if not doc_text or not doc_text.strip():
        return fallback

    snippet = doc_text[:5000]
    try:
        resp = get_llm(model_name).invoke([
            HumanMessage(content=_DOC_SUMMARY_PROMPT.format(text=snippet))
        ])
        raw = extract_text_content(resp.content)
        # JSON 추출 (마크다운 fence 제거)
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)
        parsed = json.loads(cleaned)

        summary = str(parsed.get("summary", "")).strip()[:300]
        key_terms_raw = parsed.get("key_terms", [])
        if isinstance(key_terms_raw, list):
            key_terms = [str(t).strip() for t in key_terms_raw if t][:10]
        else:
            key_terms = []

        return {"summary": summary, "key_terms": key_terms}
    except Exception as e:
        print(f"[DOC_SUMMARY] classify failed (non-fatal): {e}")
        return fallback
