"""
VLM 페이지 전사 — 텍스트 레이어가 잃어버린 내용과 구조를 렌더 이미지에서 복구한다.

【존재 이유】
PPT 를 PDF 로 내보낸 사내 자료는 두 가지 방식으로 텍스트 레이어가 망가진다.

  (a) 표·다이어그램이 **이미지로 박힘** — pypdf 는 제목만 읽고 본문이 통째로 소실
  (b) 텍스트는 있으나 **구조와 글자가 깨짐** — 표가 PowerPoint 텍스트 상자로
      만들어져 격자가 없고, 폰트 매핑 정보가 없어 글리프가 깨진다

실측 (a): KB국민은행 자료 64페이지 중 27페이지가 120자 미만.
    p41 "지원 가능한 모델들" 71자(제목만) → 전사 후 2,326자, 표 구조 보존.

실측 (b): 아래 세 페이지는 텍스트가 충분히 길어 (a) 판정에 걸리지 않았고,
    그래서 전사되지 않은 채 색인됐다. 전부 오답을 냈다.

    SAST ROI 표 (GitLab_DAP_Evaluation p24)
      색인된 원문: "100x 19x / 16x 12x 30x건당 1.5시간 이상 이슈당 2시간 이상 …"
                   ROI 값 5개와 시간 값들이 뭉쳐 어느 행의 값인지 알 수 없다
      VLM 전사   : "## SAST 취약점 해결 / 절감 시간 / 건당 1.5시간 이상 / **30x**"
                   행-열 연관 완전 복원

    플래티어 연혁 (GitLab_제품소개자료 p5)
      원문: "201520092008 20162005 … ㎼코스닥 상장8월"  (연도-사건 연관 파괴)
      전사: "**2021** 8월 - 코스닥 상장"

【pdfplumber 를 쓰지 않는 이유 — 측정 완료】
좌표로 격자를 복원하면 원본 값을 그대로 쓸 수 있어 전사보다 낫다. 실제로
시도했으나 실패했다. 슬라이드 표는 PowerPoint 텍스트 상자라 **격자가 없고**,
폰트 매핑도 깨져 있다:
    SAST 표   -> 5행 x 14열, 대부분 빈 셀. 값이 셀 중간에서 끊김("16x 간 이상")
    플래티어  -> 30행 x 4열이지만 전부 "(cid:13244)" (폰트 매핑 실패)
텍스트 레이어가 이중으로 망가져 있으므로 **렌더 픽셀을 읽는 VLM 만이 경로다.**

【비용·시점】
인제스트 시점에만 호출한다 (질의 시점 아님). 페이지당 ~11초.
증분 인제스트이므로 최초 1회 이후에는 변경된 문서에만 든다. 런타임 비용 0.
"""
from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from typing import Optional

# 활성화 여부 (기본 ON). 폐쇄망 등에서 LLM 사용 불가 시 0 으로 끈다.
VLM_ENABLED = (os.getenv("PDF_VLM_EXTRACT", "1") or "1").strip().lower() not in ("0", "false", "no")

# 렌더 해상도. 130 이면 A4 슬라이드가 대략 1500px 폭 — 표 글자가 읽히는 최소선.
VLM_DPI = int(os.getenv("PDF_VLM_DPI", "130"))

# 실질 내용이 없는 전사를 걸러내는 하한. 섹션 구분 슬라이드 등은 43자 정도가
# 나오는데, 이런 chunk 가 검색 후보를 잠식하므로 막는다.
VLM_MIN_OUTPUT_CHARS = int(os.getenv("PDF_VLM_MIN_OUTPUT_CHARS", "150"))

# 손상 신호가 없을 때, 이 길이를 넘으면 산문 페이지로 보고 전사를 건너뛴다.
#
# 【값을 800 으로 정한 이유 — 색인 비대화와의 균형】
# 전사 결과는 원문에 덧붙으므로(pdf_extractor 참조) 전사 대상이 늘수록 색인이
# 커진다. 임계값 1200 이면 142 chunk 중 92% 가 대상이 되어 색인이 대략 두 배가
# 된다. 그러면 recall 이 떨어진다 — 20/142 와 20/280 은 다르고, RRF_K·FINAL_K
# 는 현재 규모에 맞춰 측정한 값이다.
#
# 800 이면 오답을 냈던 세 페이지를 모두 포함한다:
#     SAST ROI 표     267자  (손상 신호 없음 -> 길이로 포함)
#     이관 마일스톤    269자  (깨진 글리프로 포함)
#     플래티어 연혁    502자  (깨진 글리프로 포함)
# 손상 신호가 있으면 길이와 무관하게 전사하므로, 이 값은 "깨끗해 보이는 짧은
# 그래픽 슬라이드" 를 얼마나 포함할지만 결정한다.
VLM_SKIP_CLEAN_TEXT_CHARS = int(os.getenv("PDF_VLM_SKIP_CLEAN_CHARS", "800"))

_PROMPT = """이 슬라이드의 내용을 텍스트로 전사하세요.

【표】
- 표는 **반드시 마크다운 표**로 출력하세요. 헤딩·불릿 나열로 바꾸지 마세요.
- 첫 행은 열 이름입니다. 열 이름이 그림에 없으면 내용에 맞게 붙이세요.
- 각 행의 첫 열에는 그 행을 식별하는 이름(항목명·연도·구분)을 넣으세요.
- 셀이 비어 있으면 빈 칸으로 두고 행·열 위치를 유지하세요.
- 병합된 셀은 해당하는 모든 행에 값을 반복해 넣으세요.

예시:
| 항목 | 절감 시간 | ROI |
|---|---|---|
| 코드 리뷰 | 리뷰당 20분 이상 | 100x |
| SAST 취약점 해결 | 건당 1.5시간 이상 | 30x |

【그 외】
- 연표·타임라인도 표로 만드세요 (열: 시기 / 내용).
- 다이어그램은 구성요소와 관계를 문장으로 서술하세요.
- 슬라이드에 실제로 있는 내용만. 추측·보충 금지.
- 로고·페이지번호·장식 요소는 제외.
- 설명하지 말고 전사 결과만 출력."""


# ── 텍스트 레이어 손상 판정 ──────────────────────────────
#
# 폰트 매핑이 실패하면 한국어 문서에 나올 수 없는 문자가 나온다.
# 실측: ㎼ ㏖ ㏗ 㒬 㖅 㚅 (CJK 호환/확장A), pdfplumber 에서는 "(cid:13244)".
# 전체 142 chunk 중 16개(11%)에서 발견됐고, 오답 chunk 에서 3배 더 흔하다
# (실패 31% vs 정답 10%).
_CID_RE = re.compile(r"\(cid:\d+\)")


def _broken_glyph_count(text: str) -> int:
    n = len(_CID_RE.findall(text or ""))
    for ch in text or "":
        o = ord(ch)
        if (0xE000 <= o <= 0xF8FF          # 사용자 정의 영역
                or 0x3200 <= o <= 0x33FF   # CJK 호환 기호 (㎼ 류)
                or 0x3400 <= o <= 0x4DBF   # CJK 확장 A (㚅 류)
                or 0x2E80 <= o <= 0x2FDF): # 부수 보충
            n += 1
    return n


def _looks_flattened(text: str) -> bool:
    """표가 평탄화된 흔적 — 구분자 없이 서로 다른 종류의 토큰이 붙어 있다."""
    t = text or ""
    if re.search(r"\d{6,}", t):                       # 연도·수치 열이 붙음
        return True
    if re.search(r"[●○◆▪■□▶►]\S", t):                 # 불릿 뒤에 공백 없음
        return True
    if len(re.findall(r"[가-힣][A-Z][a-z]{2,}", t)) >= 3:   # 셀 경계 소실
        return True
    return False


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
    """전사를 시도할 것인가.

    【이전 조건이 틀렸던 이유 — 실측】
    이전에는 `len(text) >= 200 이면 건너뜀` 이었다. **길이를 "구조가 온전함" 의
    대리 지표로 쓴 것**인데, 평탄화된 표는 오히려 글자 수가 많다. 그래서
    구조가 파괴된 페이지가 정확히 걸러졌다. 오답을 낸 세 페이지 모두 이
    조건에서 차단됐다 (원문 502자 / 267자 / 269자).

    지금은 반대로 본다. **손상 신호가 있으면 길이와 무관하게 시도**하고,
    손상 신호가 없고 충분히 긴 산문 페이지만 건너뛴다.
    """
    if not VLM_ENABLED:
        return False
    t = (text or "").strip()

    if _broken_glyph_count(t) > 0:
        return True                       # 폰트 매핑 실패 — 텍스트 레이어를 믿을 수 없다
    if _looks_flattened(t):
        return True                       # 표 평탄화 흔적

    if not page_has_images(page):
        return False                      # 순수 텍스트 페이지 — 전사할 그림이 없다
    if len(t) >= VLM_SKIP_CLEAN_TEXT_CHARS:
        return False                      # 손상 신호 없는 긴 산문 — 이득이 적다
    return True


def transcribe_page(pdf_path: Path, page_index: int, existing_text: str = "") -> str:
    """페이지를 VLM 으로 전사. 실패하거나 내용이 없으면 빈 문자열."""
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

    # 【채택 조건에서 길이 배수를 뺀 이유 — 손익이 비대칭이다】
    #
    # 이전 조건은 `len(out) >= 원문길이 * 1.5` 였다. 길이 증가를 "품질 향상" 의
    # 대리 지표로 쓴 것인데, 표를 구조화하면 내용은 같고 구조만 생기므로 길이가
    # 늘지 않는다. 즉 **좋은 전사일수록 버려질 수 있었다.**
    #
    # 실측 (강제 전사 후 게이트만 적용해 본 결과):
    #     플래티어 연혁   원문 502자 -> 전사 604자 (1.20배)  -> 버려짐
    #     SAST ROI 표     원문 267자 -> 전사 242자 (0.91배)  -> 버려짐  ★
    #     이관 마일스톤    원문 269자 -> 전사 1483자 (5.51배) -> 채택
    # SAST 는 행-열 연관을 완전히 복원한 최상의 전사인데 짧다는 이유로 탈락했다.
    #
    # 그리고 전사 결과는 원문을 **대체하지 않고 덧붙는다** (pdf_extractor 참조):
    #     text = 원문 + "\n\n" + 전사
    # 따라서
    #     나쁜 전사를 채택 -> 원문은 그대로 남음. 손실은 색인 크기뿐.
    #     좋은 전사를 버림 -> 복구 불가.
    # 되돌릴 수 없는 쪽을 막아야 하는데 반대로 막고 있었다. 길이 조건을 제거하고
    # "실질 내용이 있는가"(MIN_OUTPUT_CHARS)만 남긴다.
    if len(out) < VLM_MIN_OUTPUT_CHARS:
        return ""

    base_len = max(len((existing_text or "").strip()), 1)
    tag = " [표]" if out.count("|") > 4 else ""
    print(f"[VLM_PAGE] p{page_index + 1}: {base_len}자 -> {len(out)}자 전사{tag}")
    return out
