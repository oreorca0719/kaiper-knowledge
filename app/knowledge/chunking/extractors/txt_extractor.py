"""
TXT Extractor — 보고서 헤딩·번호 섹션 인식.

TXT 보고서의 흔한 구조 인식:
  ■ 핵심 이슈: ...           ← 마커: ■, ▶, ●
  [고객사 요청] ...          ← 대괄호 헤딩
  1. 원가 관련                ← 번호 섹션
  ## 헤딩                    ← Markdown 헤딩

규칙:
  - 위 패턴들을 섹션 경계로 사용
  - 섹션 간 PageUnit 1개씩
  - 첫 줄(제목 추정)은 섹션의 title로
  - 패턴 없는 단순 TXT는 paragraph 단위 (1 unit)
"""
from __future__ import annotations

import re
from pathlib import Path

from app.knowledge.chunking.page_unit import PageUnit


# 섹션 헤더 패턴 (우선순위 순)
_HEADER_PATTERNS = [
    re.compile(r"^#{1,3}\s+(.+)$"),                     # Markdown 헤딩
    re.compile(r"^[■▶●◆]\s*(.+)$"),                    # 마커 헤딩
    re.compile(r"^\[([^\]]+)\]\s*$"),                   # [대괄호 헤딩]
    re.compile(r"^(\d+)\.\s+(.+)$"),                    # 1. 숫자 섹션
    re.compile(r"^([가-힣A-Z][가-힣A-Z\s]{1,30})\n[-=]{3,}$", re.MULTILINE),  # 한글/대문자 + 밑줄
]


def _detect_section_marker(line: str) -> tuple[str, str] | None:
    """라인이 섹션 헤더면 (kind, title) 반환. 아니면 None."""
    line = line.rstrip()
    if not line:
        return None

    # Markdown
    m = re.match(r"^(#{1,3})\s+(.+)$", line)
    if m:
        return ("md_h" + str(len(m.group(1))), m.group(2).strip())
    # 마커
    m = re.match(r"^[■▶●◆]\s*(.+)$", line)
    if m:
        return ("marker", m.group(1).strip())
    # 대괄호
    m = re.match(r"^\[([^\]]+)\]\s*$", line)
    if m:
        return ("bracket", m.group(1).strip())
    # 숫자 섹션
    m = re.match(r"^(\d+)\.\s+(.+)$", line)
    if m:
        # 너무 긴 라인은 단순 본문일 수 있으니 제목으로는 사용 안 함
        title = m.group(2).strip()
        if len(title) <= 50:
            return ("num", title)
    return None


def extract_txt(path: Path) -> list[PageUnit]:
    """TXT → 섹션 단위 PageUnit. 첫 줄을 문서 제목 추정."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="cp949")

    lines = text.splitlines()
    if not lines:
        return []

    # 첫 줄을 문서 제목으로 추정 (헤딩 마커 있으면 그것 우선)
    doc_title = ""
    for ln in lines[:5]:
        s = ln.strip()
        if not s:
            continue
        marker = _detect_section_marker(s)
        if marker:
            doc_title = marker[1]
            break
        if len(s) <= 60:
            doc_title = s
            break

    # 섹션 분할
    units: list[PageUnit] = []
    section_idx = 0
    current_title = doc_title
    current_lines: list[str] = []
    pending_first_section_done = False

    def _flush(title: str, lines_acc: list[str]) -> None:
        nonlocal section_idx
        if not lines_acc:
            return
        text_str = "\n".join(lines_acc).strip()
        if not text_str:
            return
        section_idx += 1
        # 본문에 제목 포함 (검색 매칭 향상)
        full_text = f"# {title}\n\n{text_str}" if title else text_str
        units.append(PageUnit(
            unit_index=section_idx,
            unit_type="section",
            title=title,
            section_path=doc_title,
            text=full_text,
            is_table=False,
            raw_metadata={"format": "txt"},
        ))

    for ln in lines:
        marker = _detect_section_marker(ln.strip())
        if marker:
            # 직전 섹션 flush
            _flush(current_title, current_lines)
            current_title = marker[1]
            current_lines = []
            pending_first_section_done = True
        else:
            current_lines.append(ln)

    _flush(current_title, current_lines)

    # 섹션이 1개도 분할 안 됐으면(헤더 없는 짧은 TXT) 전체를 1 unit으로
    if not units and text.strip():
        units.append(PageUnit(
            unit_index=1,
            unit_type="section",
            title=doc_title,
            section_path=doc_title,
            text=text.strip(),
            is_table=False,
            raw_metadata={"format": "txt"},
        ))

    return units
