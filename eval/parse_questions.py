"""
KakaoTalk 평가셋 텍스트 파일을 questions.json으로 파싱.

입력 형식:
    A. 그레이트프로 브랜드 정체성·리브랜딩 (1~25)
    그레이트프로의 브랜드 슬로건 3요소는 무엇인가요?
    ...
    B. 그레이트프로 경쟁사 분석 (26~45)
    ...

출력: 250개 질문 객체 배열. 각 객체:
    {
      "id": 1,
      "category": "A",
      "category_name": "그레이트프로 브랜드 정체성·리브랜딩",
      "category_range": [1, 25],
      "question": "..."
    }
"""
from __future__ import annotations

import json
import re
from pathlib import Path


SOURCE_FILE = Path(r"C:\Users\User\Desktop\KakaoTalk_Longtxt_20260504_0030_38_676.txt")
OUTPUT = Path(__file__).parent / "data" / "questions.json"

# "A. <이름> (1~25)" — 카테고리 헤더
HEADER_RE = re.compile(r"^([A-P])\.\s+(.+?)\s*\((\d+)~(\d+)\)\s*$")


def main() -> None:
    text = SOURCE_FILE.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    questions: list[dict] = []
    cur_cat = None  # (letter, name, start, end)
    qid = 0

    for ln in lines:
        m = HEADER_RE.match(ln)
        if m:
            letter, name, start, end = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
            cur_cat = (letter, name, start, end)
            continue
        if cur_cat is None:
            print(f"WARN: 카테고리 없는 줄 (헤더 이전): {ln[:60]}")
            continue
        qid += 1
        questions.append({
            "id": qid,
            "category": cur_cat[0],
            "category_name": cur_cat[1],
            "category_range": [cur_cat[2], cur_cat[3]],
            "question": ln,
        })

    # 검증: ID 250, 카테고리 16
    assert len(questions) == 250, f"질문 수 불일치: {len(questions)} (250 기대)"
    cats = {q["category"] for q in questions}
    assert cats == set("ABCDEFGHIJKLMNOP"), f"카테고리 누락/추가: {cats}"

    # 카테고리 범위와 ID 정합성
    for q in questions:
        rng = q["category_range"]
        assert rng[0] <= q["id"] <= rng[1], f"Q{q['id']} 범위 위반: {rng}"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")

    # 카테고리별 통계
    from collections import Counter
    cat_counts = Counter(q["category"] for q in questions)
    print(f"Total {len(questions)} questions saved to {OUTPUT}")
    for cat in sorted(cat_counts):
        sample = next(q for q in questions if q["category"] == cat)
        # ASCII-only print to avoid Windows cp949 console issues
        name_safe = sample["category_name"].encode("ascii", "replace").decode("ascii")
        print(f"  {cat}: {cat_counts[cat]} questions [{name_safe}]")


if __name__ == "__main__":
    main()
