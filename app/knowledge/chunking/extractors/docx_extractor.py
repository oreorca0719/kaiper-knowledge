"""
DOCX Extractor — 헤딩 단위 섹션 분리.

Heading 1/2 스타일을 섹션 경계로 사용.
표는 별도 PageUnit으로 분리 (parent_page_index = 직전 섹션 unit_index).
"""
from __future__ import annotations

from pathlib import Path

import docx
from docx.text.paragraph import Paragraph
from docx.table import Table

from app.knowledge.chunking.page_unit import PageUnit


def _heading_level(p: Paragraph) -> int:
    """Heading 1/2/3 인식. 0이면 본문."""
    if not p.style or not p.style.name:
        return 0
    name = p.style.name.lower()
    if "heading 1" in name:
        return 1
    if "heading 2" in name:
        return 2
    if "heading 3" in name:
        return 3
    if name.startswith("heading"):
        return 9
    return 0


def _table_to_markdown(table: Table) -> str:
    rows: list[list[str]] = []
    for row in table.rows:
        cells = [(c.text or "").strip().replace("\n", " ") for c in row.cells]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    col_count = max(len(r) for r in rows)
    def _norm(c: list[str]) -> list[str]:
        padded = (c + [""] * col_count)[:col_count]
        return [x.replace("|", "\\|") for x in padded]
    out = ["| " + " | ".join(_norm(rows[0])) + " |",
           "|" + "|".join(["---"] * col_count) + "|"]
    for r in rows[1:]:
        out.append("| " + " | ".join(_norm(r)) + " |")
    return "\n".join(out)


def extract_docx(path: Path) -> list[PageUnit]:
    """DOCX → 헤딩 단위 섹션. 표는 별도 PageUnit."""
    d = docx.Document(str(path))
    units: list[PageUnit] = []

    current_h1 = ""
    current_h2 = ""
    section_paragraphs: list[str] = []
    section_index = 0  # 헤딩 단위 섹션 카운터

    def _flush_section():
        nonlocal section_paragraphs, section_index
        if not section_paragraphs:
            return
        section_index += 1
        title = current_h2 or current_h1 or f"섹션 {section_index}"
        section_path = " / ".join(filter(None, [current_h1, current_h2]))
        text_parts = []
        if title:
            text_parts.append(f"# {title}")
        text_parts.extend(section_paragraphs)
        units.append(PageUnit(
            unit_index=section_index,
            unit_type="section",
            title=title,
            section_path=section_path,
            text="\n\n".join(text_parts),
            is_table=False,
            raw_metadata={"format": "docx"},
        ))
        section_paragraphs = []

    table_counter = 0
    for el in d.element.body.iterchildren():
        tag = el.tag.split("}")[-1]
        if tag == "p":
            p = Paragraph(el, d)
            text = (p.text or "").strip()
            if not text:
                continue
            level = _heading_level(p)
            if level == 1:
                _flush_section()
                current_h1 = text
                current_h2 = ""
            elif level == 2:
                _flush_section()
                current_h2 = text
            elif level >= 3:
                section_paragraphs.append(f"### {text}")
            else:
                section_paragraphs.append(text)
        elif tag == "tbl":
            t = Table(el, d)
            md = _table_to_markdown(t)
            if md:
                table_counter += 1
                # 표는 현재 섹션의 별도 PageUnit으로
                parent_index = section_index + 1  # 현재 섹션의 다음 index (flush 후)
                section_path = " / ".join(filter(None, [current_h1, current_h2]))
                units.append(PageUnit(
                    unit_index=10000 + table_counter,
                    unit_type="table",
                    title=f"{current_h2 or current_h1 or '본문'} — 표 {table_counter}",
                    section_path=section_path,
                    text=md,
                    is_table=True,
                    table_index=table_counter,
                    parent_page_index=parent_index,
                    raw_metadata={"format": "docx"},
                ))

    _flush_section()  # 마지막 섹션
    return units
