"""
Size Analyzer — 형식별 임계값 + 합침/분할 결정.

원칙:
  - 작은 unit (size < merge_below): 인접 unit과 합침 (단, 같은 section_path)
  - 적정 unit: 1 unit = 1 chunk
  - 큰 unit (size > split_above): 단락 분할 (단, 표는 절대 분할 금지)

형식별 임계값:
  PPTX: merge<200, split>800     (슬라이드 평균 280자 — 작은 슬라이드 합침)
  PDF:  merge<300, split>1500    (페이지 평균 1000자 — 1 페이지 = 1 chunk 자연스러움)
  DOCX: merge<300, split>1500
  TXT:  merge<200, split>1200
  MD:   merge<200, split>1200
  XLSX: merge=0,   split>2000    (시트는 합치지 않음)
"""
from __future__ import annotations

import re
from typing import Iterable

from app.knowledge.chunking.page_unit import PageUnit


# 형식별 임계값 (유저 확정안)
SIZE_THRESHOLDS: dict[str, dict[str, int]] = {
    "pptx": {"merge_below": 200, "split_above": 800},
    "pdf":  {"merge_below": 300, "split_above": 1500},
    "docx": {"merge_below": 300, "split_above": 1500},
    "txt":  {"merge_below": 200, "split_above": 1200},
    "md":   {"merge_below": 200, "split_above": 1200},
    "xlsx": {"merge_below": 0,   "split_above": 2000},
}


def _context_header(unit) -> str:
    """분할 조각에 붙일 맥락 헤더 (section_path / title)."""
    parts = [p.strip() for p in (getattr(unit, "section_path", ""), getattr(unit, "title", "")) if (p or "").strip()]
    # 중복 제거하되 순서 보존
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p); out.append(p)
    return " / ".join(out)


def _with_header(fragment: str, header: str) -> str:
    """분할 조각 앞에 맥락 헤더를 붙인다.

    【이유】
    긴 unit 을 자르면 2번째 이후 조각은 **제목을 잃는다**. 임베딩도 BM25 도
    그 조각을 원 주제와 연결하지 못해 검색에서 사라진다.

    실측 (KB국민은행_GitLab_Duo_설명회자료 p41, VLM 전사 2,279자):
        chunk#119  152자  "-41 지원 가능한 모델들 ... ## 지원 모델"   split=1/4  제목만
        chunk#120 1500자  "| Model family | Model | ..."          split=2/4  표만
        chunk#122  607자  "## 호환 가능 모델 | CodeGemma | ..."     split=4/4  표만

      질의 "지원 가능한 모델"은 제목만 있는 빈 껍데기(#119)를 찾고,
      정작 답이 있는 #120·#122 는 그 문구가 없어 검색되지 않았다.
      같은 슬라이드라도 493자로 안 쪼개진 GPU 표(#123)는 정상 검색됐다.

    원본 저장소에서 "PPT 비교 표 평탄화" 로 8건 미해결로 남아 있던 결함의
    실제 메커니즘이 이것이다. 조각마다 제목을 상속시켜 해소한다.
    (chunk 별 맥락 헤더를 붙이는 Contextual Retrieval 패턴의 구조 기반 구현)
    """
    if not header:
        return fragment
    # 이미 헤더로 시작하면 중복해서 붙이지 않는다
    head_line = fragment.lstrip().split("\n", 1)[0]
    if header in fragment[:len(header) + 120] or head_line.strip() == header:
        return fragment
    return f"{header}\n\n{fragment}"


def _split_text(text: str, max_size: int, overlap: int = 50) -> list[str]:
    """단락 경계 우선 분할. 단락이 더 크면 슬라이딩 윈도우."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        if len(para) > max_size:
            # 큰 단락 — 슬라이딩 윈도우
            if current:
                chunks.append("\n\n".join(current))
                current, current_len = [], 0
            step = max(max_size - overlap, 1)
            for i in range(0, len(para), step):
                chunks.append(para[i: i + max_size])
        elif current_len + len(para) + 2 > max_size and current:
            chunks.append("\n\n".join(current))
            current, current_len = [para], len(para)
        else:
            current.append(para)
            current_len += len(para) + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks or [text[:max_size]]


def adaptive_chunk(units: list[PageUnit], format: str) -> list[PageUnit]:
    """
    PageUnit list → chunk 단위 PageUnit list.

    핵심 동작:
      1. 큰 unit (split_above 초과) → 분할. 단, is_table=True면 분할 금지 (그대로 유지).
      2. 작은 unit (merge_below 미만) → pending에 누적, merge_below 도달 시 합침.
         합침 조건: 인접 + 같은 section_path + 둘 다 본문(is_table=False).
      3. 적정 unit → 그대로 1 chunk (현재 pending 먼저 flush).

    출력: chunk 단위로 변환된 PageUnit (각각 1 chunk = 1 PageUnit).
    표 unit은 변환 없이 그대로 유지 (parent_page_index 보존).
    """
    if not units:
        return []
    th = SIZE_THRESHOLDS.get(format, SIZE_THRESHOLDS["txt"])
    merge_below = th["merge_below"]
    split_above = th["split_above"]

    out: list[PageUnit] = []
    pending: list[PageUnit] = []
    pending_size = 0

    def _flush_pending():
        """pending 합쳐서 1개 PageUnit으로 out에 추가."""
        nonlocal pending, pending_size
        if not pending:
            return
        if len(pending) == 1:
            out.append(pending[0])
        else:
            merged_indices = [u.unit_index for u in pending]
            merged_titles = [u.title for u in pending if u.title]
            combined_text = "\n\n".join(u.text for u in pending if u.text)
            out.append(PageUnit(
                unit_index=pending[0].unit_index,
                unit_type=pending[0].unit_type,
                title=" + ".join(merged_titles) if merged_titles else "",
                section_path=pending[0].section_path,
                text=combined_text,
                is_table=False,
                raw_metadata={
                    **pending[0].raw_metadata,
                    "merged_units": merged_indices,
                },
            ))
        pending = []
        pending_size = 0

    for unit in units:
        # 표는 무조건 그대로 (분할·합침 금지)
        if unit.is_table:
            _flush_pending()
            out.append(unit)
            continue

        size = unit.char_count

        # Case 1: 너무 큼 → 분할
        if size > split_above:
            _flush_pending()
            split_texts = _split_text(unit.text, max_size=split_above)
            header = _context_header(unit)
            for i, t in enumerate(split_texts):
                out.append(PageUnit(
                    unit_index=unit.unit_index,
                    unit_type=unit.unit_type,
                    title=unit.title + (f" (part {i+1}/{len(split_texts)})" if len(split_texts) > 1 else ""),
                    section_path=unit.section_path,
                    text=_with_header(t, header),
                    is_table=False,
                    raw_metadata={**unit.raw_metadata, "split_part": f"{i+1}/{len(split_texts)}"},
                ))
            continue

        # Case 2: 너무 작음 → pending에 추가
        if size < merge_below and merge_below > 0:
            # 합칠 수 있는지 확인 (같은 section_path)
            if pending and pending[-1].section_path != unit.section_path:
                _flush_pending()
            pending.append(unit)
            pending_size += size
            if pending_size >= merge_below:
                _flush_pending()
            continue

        # Case 3: 적정 → pending flush 후 그대로
        _flush_pending()
        out.append(unit)

    _flush_pending()  # 남은 pending
    return out
