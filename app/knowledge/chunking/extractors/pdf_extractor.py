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


# 페이지 번호·머리말만 있는 줄 (제목으로 쓰면 안 됨)
#   슬라이드 PDF 는 "-41", "page_ 37", "© GitLab Inc." 같은 줄로 시작하는 일이 잦다.
#   이걸 제목으로 잡으면 분할 조각에 붙는 맥락 헤더가 무의미해진다.
_PAGE_MARKER_RE = re.compile(
    r"^(?:[-–—]?\s*\d{1,4}|page[_\s]*\d+|p\.?\s*\d+|©.*|slide\s*\d+)$", re.I
)


# 제목 최대 길이. 분할 조각마다 앞에 붙으므로 지나치게 길면 안 된다.
_TITLE_MAX = 120


def _title_candidates(page_text: str) -> list[str]:
    """제목 후보들을 모은다.

    두 갈래를 모두 본다:
      (a) pypdf 원문 첫 줄  — 슬라이드 제목이 한 줄에 온전히 들어 있는 경우가 많다
      (b) VLM 마크다운 헤딩 + 바로 뒤 부제 줄

    한쪽만 쓰면 구분 정보를 잃는다. 실측:
        pypdf   '지원 가능한 모델들 – Models and hardware requirements-Self-Managed환경 (폐쇄망)'
        VLM     '# 지원 가능한 모델들 – Models and hardware requirements'
                '- Self-Managed환경 (폐쇄망)'      <- 부제가 분리됨

    VLM 헤딩만 채택하면 'Self-Managed / 폐쇄망' 이 제목에서 빠지고, 같은 문서의
    다른 슬라이드('GitLab DAP (인터넷 연결 가능시)') 와 구별되지 않는다.
    실제로 "자체 호스팅 모델" 질의가 클라우드 모델 표만 찾고 실패했다.
    """
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    out: list[str] = []

    # (a) 페이지 마커가 아닌 첫 줄
    for ln in lines[:4]:
        if _PAGE_MARKER_RE.match(ln):
            continue
        if not ln.startswith("#") and not ln.endswith(("다.", "요.", "음.")):
            out.append(ln)
        break

    # (b) 마크다운 헤딩 + 뒤따르는 부제 줄
    for i, ln in enumerate(lines[:12]):
        if not ln.startswith("#"):
            continue
        parts = [ln.lstrip("#").strip()]
        for nxt in lines[i + 1: i + 3]:
            # 부제로 볼 수 있는 짧은 줄만 (표 행·본문 제외)
            if nxt.startswith("|") or len(nxt) > 60:
                break
            cleaned = nxt.lstrip("#-–—•* ").strip()
            if not cleaned or cleaned.startswith("|"):
                break
            parts.append(cleaned)
        out.append(" ".join(parts))
        break

    return [t for t in out if 2 <= len(t) <= _TITLE_MAX]


def _guess_title(page_text: str) -> str:
    """페이지 제목 추정 — 후보 중 정보량이 가장 많은 것을 고른다."""
    cands = _title_candidates(page_text)
    if not cands:
        return ""
    return max(cands, key=len)


# 섹션 패턴 (예: "1. 교육 대상", "PART 02", "3-1 후속 액션")
#
# 【주의】 번호+마침표는 슬라이드 본문의 열거에서도 흔하다.
#   실측 오탐: "1.중앙화된 AI 접근 지점 -GitLab 인스턴스와 AI 모델 사이의..."
#     -> section_path 가 '1.중앙화된' 으로 잡히고, 이후 페이지들에 계속 상속돼
#        분할 조각의 맥락 헤더가 '1.중앙화된 / 지원 가능한 모델들 – ...' 로 오염됐다.
#
# 마침표 뒤에 공백이 있는 형태만 섹션 헤더로 인정하고(열거는 보통 붙여 씀),
# 뒤에 오는 제목이 지나치게 길면 본문으로 본다.
_SECTION_RE = re.compile(
    r"^(?:PART\s*\d+|\d{1,2}\.\s+\S[^\n]{0,28}|\d+-\d+\s+\S[^\n]{0,28})$",
    re.MULTILINE,
)


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
