"""
PDF Extractor — 페이지 단위.

각 페이지 = PageUnit 1개. 표 분리는 PDF 라이브러리 한계로 본문에 포함된 채 둠.
(pypdf는 표 구조 정보 제공 안 함. pdfplumber로 옵션 확장 가능하나 기본은 pypdf.)
"""
from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from app.knowledge.chunking.page_unit import PageUnit
from app.knowledge.chunking.extractors.vlm_page import should_try_vlm, transcribe_page


# 헤딩 추정: 페이지 첫 줄 중 짧고 다음 줄과 빈 줄이 있는 경우
def _guess_title(page_text: str) -> str:
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    if not lines:
        return ""
    first = lines[0]
    # 30자 이내 + 첫 줄이 단순한 경우 제목으로 간주
    if len(first) <= 30 and not first.endswith(("다.", "요.", "음.")):
        return first
    return ""


# 섹션 패턴 (예: "1. 교육 대상", "PART 02", "3-1 후속 액션")
_SECTION_RE = re.compile(r"^(?:PART\s*\d+|\d+\.\s*\S+|\d+-\d+\s+\S+)", re.MULTILINE)


def extract_pdf(path: Path) -> list[PageUnit]:
    """PDF → 페이지별 PageUnit. section_path는 페이지 시작에서 발견되는 섹션 헤더로 추적."""
    reader = PdfReader(str(path))
    units: list[PageUnit] = []
    current_section = ""

    for page_idx, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""

        # 이미지 지배 페이지 복구.
        # PPT 를 PDF 로 내보낸 자료는 표·다이어그램이 이미지로 박혀 있어
        # 텍스트 레이어에 제목만 남는다. 이 자료의 경우 64페이지 중 27페이지가
        # 120자 미만이었고, "지원 가능한 모델" 표가 통째로 소실돼 사용자 질의에
        # 답할 수 없었다. VLM 으로 전사해 본문을 복구한다 (인제스트 시점 1회).
        vlm_text = ""
        if should_try_vlm(text, page):
            vlm_text = transcribe_page(path, page_idx - 1, existing_text=text)
        if vlm_text:
            text = "\n\n".join(p for p in (text, vlm_text) if p).strip()

        if not text:
            continue

        # section 갱신
        m = _SECTION_RE.search(text[:300])
        if m:
            current_section = m.group(0).strip()

        title = _guess_title(text)

        units.append(PageUnit(
            unit_index=page_idx,
            unit_type="page",
            title=title,
            section_path=current_section,
            text=text,
            is_table=False,
            raw_metadata={
                "format": "pdf",
                "page_size_chars": len(text),
                "vlm_transcribed": bool(vlm_text),
            },
        ))

    return units
