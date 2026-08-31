"""
XLSX Extractor — 시트 단위.

각 시트 = PageUnit 1개 (markdown table).
큰 시트는 size_analyzer가 추후 분할 결정.
"""
from __future__ import annotations

from pathlib import Path

import openpyxl

from app.knowledge.chunking.page_unit import PageUnit


def extract_xlsx(path: Path) -> list[PageUnit]:
    """XLSX → 시트별 PageUnit (markdown table)."""
    wb = openpyxl.load_workbook(str(path), data_only=True)
    units: list[PageUnit] = []

    for sheet_idx, ws in enumerate(wb.worksheets, start=1):
        rows: list[list[str]] = []
        for row in ws.iter_rows(values_only=True):
            vals = ["" if v is None else str(v) for v in row]
            if any(v.strip() for v in vals):
                rows.append(vals)
        if not rows:
            continue

        # markdown table
        col_count = max(len(r) for r in rows)
        def _norm(r: list[str]) -> list[str]:
            padded = (r + [""] * col_count)[:col_count]
            return [c.replace("|", "\\|").replace("\n", " ").strip() for c in padded]
        lines = ["| " + " | ".join(_norm(rows[0])) + " |",
                 "|" + "|".join(["---"] * col_count) + "|"]
        for r in rows[1:]:
            lines.append("| " + " | ".join(_norm(r)) + " |")
        md = "\n".join(lines)

        units.append(PageUnit(
            unit_index=sheet_idx,
            unit_type="page",
            title=ws.title,
            section_path=ws.title,
            text=f"# 시트: {ws.title}\n\n{md}",
            is_table=True,                    # XLSX는 시트 자체가 표
            table_index=sheet_idx,
            parent_page_index=None,
            raw_metadata={
                "format": "xlsx",
                "row_count": len(rows),
                "col_count": col_count,
            },
        ))
    return units
