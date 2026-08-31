"""
출처 문서 13개를 텍스트로 추출하여 eval/data/sources/ 에 저장.

PPTX/DOCX/PDF는 구조 정보(슬라이드 번호, 페이지, 헤딩 등)를 보존하여 추출.
TXT는 그대로 복사.

이 스크립트는 1회 실행용. 출처 폴더(SOURCE_DIR)는 외부 디렉토리이므로 git에 포함되지 않는다.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from pypdf import PdfReader
from pptx import Presentation
from docx import Document


SOURCE_DIR = Path(r"C:\Users\User\Desktop\새 폴더")
OUTPUT_DIR = Path(__file__).parent / "data" / "sources"


def extract_pptx(path: Path) -> str:
    """PPTX → 슬라이드별 텍스트 (제목 + 본문 + 노트). 슬라이드 번호 명시."""
    prs = Presentation(str(path))
    blocks: list[str] = []
    for idx, slide in enumerate(prs.slides, start=1):
        parts: list[str] = [f"### Slide {idx}"]
        # 제목 우선
        title = ""
        for shape in slide.shapes:
            if shape.has_text_frame and shape == slide.shapes.title:
                title = (shape.text_frame.text or "").strip()
                break
        if title:
            parts.append(f"# {title}")
        # 본문
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            if shape == slide.shapes.title:
                continue
            text = (shape.text_frame.text or "").strip()
            if text:
                parts.append(text)
        # 표 셀 (행 단위 markdown)
        for shape in slide.shapes:
            if not shape.has_table:
                continue
            tbl = shape.table
            for row in tbl.rows:
                cells = [(c.text or "").strip().replace("\n", " ") for c in row.cells]
                parts.append("| " + " | ".join(cells) + " |")
        # 노트
        if slide.has_notes_slide:
            note = (slide.notes_slide.notes_text_frame.text or "").strip()
            if note:
                parts.append(f"[NOTES]\n{note}")
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


def extract_docx(path: Path) -> str:
    """DOCX → 단락 + 표 (markdown table). 헤딩 스타일은 # 으로 변환."""
    doc = Document(str(path))
    parts: list[str] = []
    for el in doc.element.body.iterchildren():
        tag = el.tag.split("}")[-1]
        if tag == "p":
            # 단락 — 스타일 확인
            from docx.text.paragraph import Paragraph
            p = Paragraph(el, doc)
            text = (p.text or "").strip()
            if not text:
                continue
            style = (p.style.name or "").lower() if p.style else ""
            if "heading 1" in style:
                parts.append(f"# {text}")
            elif "heading 2" in style:
                parts.append(f"## {text}")
            elif "heading" in style:
                parts.append(f"### {text}")
            else:
                parts.append(text)
        elif tag == "tbl":
            from docx.table import Table
            t = Table(el, doc)
            for row in t.rows:
                cells = [(c.text or "").strip().replace("\n", " ") for c in row.cells]
                parts.append("| " + " | ".join(cells) + " |")
    return "\n\n".join(parts)


def extract_pdf(path: Path) -> str:
    """PDF → 페이지별 텍스트. 페이지 번호 명시."""
    reader = PdfReader(str(path))
    blocks: list[str] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            blocks.append(f"### Page {idx}\n{text}")
    return "\n\n".join(blocks)


def extract_txt(path: Path) -> str:
    """TXT → 그대로 (UTF-8 가정, 실패 시 cp949 재시도)."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp949")


# 확장자별 dispatcher
EXTRACTORS = {
    ".pptx": extract_pptx,
    ".docx": extract_docx,
    ".pdf":  extract_pdf,
    ".txt":  extract_txt,
}


def slugify(name: str) -> str:
    """한글·공백·특수문자를 안전한 ASCII로 매핑. 충돌 방지를 위해 원본 stem 보존."""
    return name.replace(" ", "_").replace("[", "").replace("]", "")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    files = sorted(SOURCE_DIR.iterdir())
    for src in files:
        if not src.is_file():
            continue
        suffix = src.suffix.lower()
        extractor = EXTRACTORS.get(suffix)
        if not extractor:
            print(f"SKIP (unsupported): {src.name}")
            continue
        try:
            text = extractor(src)
        except Exception as e:
            print(f"FAIL: {src.name} — {e}")
            continue

        out_name = slugify(src.stem) + ".txt"
        out_path = OUTPUT_DIR / out_name
        out_path.write_text(text, encoding="utf-8")

        manifest.append({
            "doc_id": out_name.replace(".txt", ""),
            "original_name": src.name,
            "source_path": str(src),
            "extracted_path": str(out_path.relative_to(OUTPUT_DIR.parent.parent)),
            "format": suffix,
            "char_count": len(text),
        })
        print(f"OK: {src.name} → {out_name} ({len(text)} chars)")

    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n총 {len(manifest)}개 추출 완료 → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
