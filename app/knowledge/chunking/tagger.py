"""
Tagger — 청킹된 PageUnit에 metadata 부착 (코드 + LLM).

코드 metadata:
  - doc_id, doc_format, page_unit_index, page_unit_title, section_path
  - chunk_index_in_page, is_table, table_index, parent_page_index
  - char_count, merged_units, entities (정규식)

LLM metadata:
  - doc_topic (문서당 1회)
  (chunk_question_types는 폐기 — Q&A cache retrieval로 대체)
"""
from __future__ import annotations

import re
from pathlib import Path

from app.knowledge.chunking.page_unit import Chunk, PageUnit
from app.knowledge.chunking.doc_topic_classifier import classify_doc_topic
from app.knowledge.chunking.doc_summary_classifier import classify_doc_summary


# ────────────────────────────────────────────────────────────
# Entity 추출 (정규식 + 도메인 키워드 사전)
# ────────────────────────────────────────────────────────────

_ENTITY_PATTERNS = {
    "time": re.compile(r"\b(\d{1,2}:\d{2})\b"),
    "abbr": re.compile(r"\b([A-Z]{2,}[A-Z0-9]*)\b"),
    "structural": re.compile(r"\b(Day\s*[1-9]|Phase\s*[1-9]|[1-9]\s*단계|[A-C]안|Step\s*[1-9])\b", re.I),
    "number_unit": re.compile(
        r"\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(명|억|만원|만\s*원|억\s*원|시간|분|일|차|곳|개|단계|자|회|건|개월|년|％|%|원|건수|명\+|건\+)\b"
    ),
    "quoted": re.compile(r"['‘'\"\"「『]([^'’\"\"」』]{2,30})['’\"\"」』]"),
}

_DOMAIN_KEYWORDS = [
    "위시캣", "원티드긱스", "원티드", "점핏", "프리모아", "이랜서",
    "그레이트프로", "하이프로", "프로엔솔루션", "농협", "농협대학", "농협중앙회",
    "Genspark", "Make", "Make.com", "Vrew", "Canva", "Claude", "Gemini", "ChatGPT", "Cowork",
    "갓찌뇽", "최재완", "이준영", "안유진", "안충호", "김범준", "이현용",
    "FP", "OM", "CM", "Career Manager", "Operations Manager", "Franchise Partner",
    "Alignment", "Alliance", "Onboarding", "Engagement", "Optimization",
    "브릿지 프로토콜", "SSO", "Single Sign-On",
]


def extract_entities(text: str) -> list[str]:
    if not text:
        return []
    found: set[str] = set()
    t_lower = text.lower()
    for kw in _DOMAIN_KEYWORDS:
        if kw.lower() in t_lower:
            found.add(kw.lower())
    for pat in _ENTITY_PATTERNS.values():
        for m in pat.finditer(text):
            v = m.group(1).strip().lower()
            if v and len(v) <= 50:
                found.add(v)
    return sorted(found)[:30]


# ────────────────────────────────────────────────────────────
# Tagger main
# ────────────────────────────────────────────────────────────

def tag_chunks(
    units: list[PageUnit],
    doc_id: str,
    doc_format: str,
    enable_llm_doc_topic: bool = True,
) -> list[Chunk]:
    """
    PageUnit list → Chunk list (metadata 완전 부착).

    1. doc_topic 1회 분류 (LLM, enable_llm_doc_topic=True 시)
    2. 각 unit별:
        - 코드 metadata (위치/구조/entities)
       (chunk_question_types는 폐기 — Q&A cache retrieval로 대체)
    """
    if not units:
        return []

    # 1. doc_topic 1회 (전체 문서 텍스트 합쳐서)
    doc_topic = "general"
    doc_summary = ""
    key_terms_str = ""
    if enable_llm_doc_topic:
        full_text = "\n\n".join(u.text for u in units if u.text)[:5000]
        doc_topic = classify_doc_topic(full_text)

        # doc_summary, key_terms 분류 (LLM 1회 추가) — rewrite_node 의 doc 카탈로그 원천
        s = classify_doc_summary(full_text)
        doc_summary = s["summary"]
        key_terms_str = ",".join(s["key_terms"])

    # 2. unit별 chunk 생성
    chunks: list[Chunk] = []
    for u in units:
        # entities (코드)
        entities = extract_entities(u.text)

        chunk_id = f"file::{doc_id}::page_{u.unit_index}"
        if u.is_table:
            chunk_id = f"file::{doc_id}::page_{u.parent_page_index or u.unit_index}::table_{u.table_index}"
        elif u.raw_metadata.get("split_part"):
            # 큰 unit 분할된 경우 part로 disambiguate
            sp = str(u.raw_metadata["split_part"]).replace("/", "of")
            chunk_id = f"file::{doc_id}::page_{u.unit_index}::part_{sp}"

        metadata = {
            # 위치/구조 (코드)
            "doc_id": doc_id,
            "doc_format": doc_format,
            "page_unit_index": u.unit_index,
            "page_unit_type": u.unit_type,
            "page_unit_title": u.title,
            "section_path": u.section_path,
            "is_table": u.is_table,
            "char_count": u.char_count,
            # 표 관련
            "table_index": u.table_index if u.is_table else None,
            "parent_page_index": u.parent_page_index,
            "merged_units": u.raw_metadata.get("merged_units", []),
            "split_part": u.raw_metadata.get("split_part"),
            # entities (코드, 정규식)
            "entities": " ".join(entities),
            # LLM metadata
            "doc_topic": doc_topic,
            "doc_summary": doc_summary,    # 신규 — doc 카탈로그용
            "key_terms": key_terms_str,    # 신규 — comma-separated, ChromaDB 호환
            # 기존 v1 호환 필드
            "title": u.title or doc_id,
            "display_source": doc_id,
            "source": "file",
        }

        # ChromaDB는 None 값 metadata 거부 — None 키 제거
        metadata = {k: v for k, v in metadata.items() if v is not None and v != []}
        # list/dict 값은 string화 (ChromaDB metadata 제약)
        for k, v in list(metadata.items()):
            if isinstance(v, (list, dict)):
                metadata[k] = str(v)

        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=u.text,
            metadata=metadata,
        ))

    return chunks
