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
    # xlsx 는 2000 이었다. generator 가 컨텍스트를 [:1500] 로 자르므로
    # 1500~2000 구간 chunk 는 뒷부분이 조용히 버려진다. 다른 형식은 모두
    # 1500 이하로 맞춰져 있었는데 xlsx 만 어긋나 있었다.
    "xlsx": {"merge_below": 0,   "split_above": 1500},
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


_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


def _is_markdown_table(block: str) -> bool:
    """마크다운 표 블록인가 (2행 이상이 파이프 행)."""
    lines = [ln for ln in block.splitlines() if ln.strip()]
    if len(lines) < 3:
        return False
    return sum(1 for ln in lines if _TABLE_ROW_RE.match(ln)) >= len(lines) * 0.8


def _split_markdown_table(block: str, max_size: int) -> list[str]:
    """마크다운 표를 **행 경계**에서 자르고 헤더 행을 각 조각에 반복한다.

    【이유】
    글자 수로 자르면 표가 행 중간에서 끊긴다. 실측:
        chunk#118 ... '| GPT | GPT-'      (여기서 끊김)
        chunk#119 'y |  | GPT | GPT-oss-120B | ...'
    끊긴 조각은 어느 열이 무엇인지 알 수 없어 검색으로도, 답변 근거로도
    쓸 수 없다. 표는 정보 밀도가 가장 높은 부분이라 손실이 크다.

    헤더 행(제목행 + 구분행)을 조각마다 반복해 각 조각이 독립적으로
    해석 가능하게 만든다.
    """
    lines = [ln for ln in block.splitlines() if ln.strip()]
    # 헤더 = 첫 파이프 행 + 그 다음 구분 행(|---|)
    head: list[str] = []
    body_start = 0
    for i, ln in enumerate(lines[:3]):
        if _TABLE_ROW_RE.match(ln):
            head.append(ln)
            body_start = i + 1
            if len(head) == 2:
                break
        else:
            head.append(ln)          # 표 앞 설명 줄
            body_start = i + 1
    head_text = "\n".join(head)
    head_len = len(head_text) + 1

    out: list[str] = []
    cur: list[str] = []
    cur_len = head_len
    for ln in lines[body_start:]:
        add = len(ln) + 1
        if cur and cur_len + add > max_size:
            out.append(head_text + "\n" + "\n".join(cur))
            cur, cur_len = [], head_len
        cur.append(ln)
        cur_len += add
    if cur:
        out.append(head_text + "\n" + "\n".join(cur))
    return out or [block]


def _split_text(text: str, max_size: int, overlap: int = 50) -> list[str]:
    """단락 경계 우선 분할. 마크다운 표는 행 경계로 자르고 헤더를 반복한다."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        if len(para) > max_size:
            if current:
                chunks.append("\n\n".join(current))
                current, current_len = [], 0
            if _is_markdown_table(para):
                chunks.extend(_split_markdown_table(para, max_size))
                continue
            # 큰 단락 — 슬라이딩 윈도우
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
            # 【표라고 무조건 통째로 두면 안 된다 — 실측】
            # XLSX 는 시트 하나를 is_table=True 로 만든다. 그런데 이 분기가
            # split_above 보다 먼저 걸려서 아무리 큰 시트도 한 chunk 가 됐다.
            # 파일럿 실측:
            #     24,944자 chunk 존재 — generator 는 [:1500] 로 자르므로 94% 손실
            #      5,216자 교육_커리큘럼_v0.91  → 71% 손실
            #
            # 표를 함부로 조각내지 않겠다는 의도는 유지하되, 컨텍스트 한도를
            # 넘는 것은 **행 경계에서** 자른다. _split_markdown_table 이
            # 헤더 행을 각 조각에 반복해 넣으므로 조각마다 독립 해석이 된다.
            if unit.char_count > split_above:
                parts = _split_text(unit.text, max_size=split_above)
                for i, t in enumerate(parts):
                    out.append(PageUnit(
                        unit_index=unit.unit_index,
                        unit_type=unit.unit_type,
                        title=unit.title + (f" (part {i+1}/{len(parts)})" if len(parts) > 1 else ""),
                        section_path=unit.section_path,
                        text=t,
                        is_table=True,
                        table_index=getattr(unit, "table_index", None),
                        parent_page_index=getattr(unit, "parent_page_index", None),
                        raw_metadata={**unit.raw_metadata, "split_part": f"{i+1}/{len(parts)}"},
                    ))
            else:
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

        # Case 3: 적정 크기
        # 【짧은 조각을 혼자 내보내지 않는다 — 실측】
        # 기존에는 여기서 무조건 _flush_pending() 을 불렀다. 그래서 짧은 조각이
        # **연속으로 이어질 때만** 합쳐지고, 보통 크기 사이에 낀 조각 하나는
        # 그대로 배출됐다. 파일럿 2,314 chunk 중 276개(12%)가 100자 미만이었고
        # 내용이 이랬다:
        #     "운영 리소스 | 도구 및 라이선스\n\n비용 확인 필요 항목"   (33자)
        #     "→ 이 4가지 조건을 충족시키는 방향으로 주제 배치·실습 설계"   (42자)
        # 이런 조각은 맥락이 없어 임베딩이 모호해지고 검색 후보만 잠식한다.
        #
        # 합쳐도 split_above 를 넘지 않으면 다음 unit 에 흡수시킨다.
        if pending:
            merged = "\n\n".join([u.text for u in pending if u.text] + [unit.text or ""])
            if len(merged) <= split_above:
                titles = [u.title for u in pending if u.title]
                unit = PageUnit(
                    unit_index=pending[0].unit_index,
                    unit_type=unit.unit_type,
                    title=unit.title or (titles[0] if titles else ""),
                    section_path=unit.section_path,
                    text=merged,
                    is_table=False,
                    raw_metadata={**unit.raw_metadata,
                                  "absorbed_units": [u.unit_index for u in pending]},
                )
                pending = []
                pending_size = 0
            else:
                _flush_pending()
        out.append(unit)

    _flush_pending()  # 남은 pending
    return out
