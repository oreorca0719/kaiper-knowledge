"""
VLM 페이지 전사 — 이미지가 지배적인 슬라이드/페이지에서 텍스트를 복구한다.

【존재 이유】
PPT 를 PDF 로 내보낸 사내 자료는 표·다이어그램이 **이미지로 박혀** 있는 경우가
많다. pypdf 는 텍스트 레이어만 읽으므로 제목만 남고 본문이 통째로 소실된다.

실측 (KB국민은행_GitLab_Duo_설명회자료, 64페이지):
    120자 미만 페이지        27개 / 64  (42%)
    p41 "지원 가능한 모델들"   71자 (제목만) + 이미지 3개
        → 실제 내용은 지원 모델 12종 x 4개 기능 등급의 표 + 호환 모델 13종

    사용자가 "GitLab DAP 에 적용할 수 있는 모델"을 물었을 때 정답 표가
    인덱스에 아예 없어 답할 수 없었다. 검색·리랭킹을 아무리 고쳐도
    없는 정보는 찾을 수 없다.

VLM 전사 후: 71자 → 2,326자, 표 구조가 마크다운으로 보존됨.

【비용·시점】
인제스트 시점에만 호출한다 (질의 시점 아님). 페이지당 ~11초, 문서당 이미지
지배 페이지 수만큼. 증분 인제스트이므로 최초 1회 이후에는 변경 문서에만 든다.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

# 활성화 여부 (기본 ON). 폐쇄망 등에서 LLM 사용 불가 시 0 으로 끈다.
VLM_ENABLED = (os.getenv("PDF_VLM_EXTRACT", "1") or "1").strip().lower() not in ("0", "false", "no")

# 이 글자 수 미만이면 "이미지 지배 페이지"로 간주하고 VLM 전사를 시도한다.
VLM_MIN_TEXT_CHARS = int(os.getenv("PDF_VLM_MIN_TEXT_CHARS", "200"))

# 렌더 해상도. 130 이면 A4 슬라이드가 대략 1500px 폭 — 표 글자가 읽히는 최소선.
VLM_DPI = int(os.getenv("PDF_VLM_DPI", "130"))

# 채택 조건 두 가지를 모두 만족해야 인덱스에 넣는다.
#   1) 원문 대비 배수     — 이득이 있어야 함
#   2) 절대 글자 수 하한  — 배수만으로는 부족하다. 섹션 구분 슬라이드는
#                          원문이 12~15자라 43자만 나와도 배수 조건을 통과한다.
#                          실질 내용이 없는 chunk 가 검색 후보를 잠식하므로 막는다.
VLM_MIN_GAIN = float(os.getenv("PDF_VLM_MIN_GAIN", "1.5"))
VLM_MIN_OUTPUT_CHARS = int(os.getenv("PDF_VLM_MIN_OUTPUT_CHARS", "150"))

_PROMPT = """이 슬라이드의 내용을 텍스트로 전사하세요.

규칙:
- 표는 마크다운 표로 변환. 행/열 관계를 정확히 보존.
- 다이어그램은 구성요소와 관계를 문장으로 서술.
- 슬라이드에 실제로 있는 내용만. 추측·보충 금지.
- 로고·페이지번호·장식 요소는 제외.
- 설명하지 말고 전사 결과만 출력."""


def _render_page_png(pdf_path: Path, page_index: int) -> Optional[bytes]:
    """페이지를 PNG 바이트로 렌더. pymupdf 미설치 시 None."""
    try:
        try:
            import pymupdf as fitz  # type: ignore
        except ImportError:
            import fitz  # type: ignore
    except ImportError:
        print("[VLM_PAGE] pymupdf 미설치 — 이미지 페이지 전사를 건너뜁니다")
        return None
    try:
        with fitz.open(str(pdf_path)) as doc:
            return doc[page_index].get_pixmap(dpi=VLM_DPI).tobytes("png")
    except Exception as e:
        print(f"[VLM_PAGE] 렌더 실패 p{page_index + 1} (non-fatal): {e}")
        return None


def page_has_images(page) -> bool:
    """pypdf page 에 이미지 객체가 있는가."""
    try:
        return len(page.images) > 0
    except Exception:
        # pypdf 버전에 따라 images 접근이 실패할 수 있다 — 판단 불가 시 시도해 본다
        return True


def should_try_vlm(text: str, page) -> bool:
    if not VLM_ENABLED:
        return False
    if len((text or "").strip()) >= VLM_MIN_TEXT_CHARS:
        return False
    return page_has_images(page)


def transcribe_page(pdf_path: Path, page_index: int, existing_text: str = "") -> str:
    """페이지를 VLM 으로 전사. 실패하거나 이득이 없으면 빈 문자열."""
    png = _render_page_png(pdf_path, page_index)
    if not png:
        return ""

    try:
        from langchain_core.messages import HumanMessage
        from app.core.config import get_llm
        from app.core.history_utils import extract_text_content

        resp = get_llm().invoke([HumanMessage(content=[
            {"type": "image",
             "source": {"type": "base64", "media_type": "image/png",
                        "data": base64.b64encode(png).decode("ascii")}},
            {"type": "text", "text": _PROMPT},
        ])])
        out = extract_text_content(resp.content).strip()
    except Exception as e:
        print(f"[VLM_PAGE] 전사 실패 p{page_index + 1} (non-fatal): {e}")
        return ""

    base_len = max(len((existing_text or "").strip()), 1)
    if len(out) < VLM_MIN_OUTPUT_CHARS or len(out) < base_len * VLM_MIN_GAIN:
        # 얻은 게 없다 — 섹션 구분 슬라이드 등
        return ""
    print(f"[VLM_PAGE] p{page_index + 1}: {base_len}자 -> {len(out)}자 전사")
    return out
