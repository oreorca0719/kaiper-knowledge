"""
JSON parse error로 실패한 3건의 라벨을 수동으로 수정.
"""
from __future__ import annotations
import json
from pathlib import Path

LABELS_FILE = Path(__file__).parent / "data" / "labels.json"

# 수동 수정 (출처에서 직접 추출)
MANUAL_FIXES = {
    83: {
        "answer": "이미지, 카드뉴스, 포스터",
        "answer_type": "list_n",
        "citation": {
            "doc_id": "AI교육_내부기준문서_v2",
            "snippet": "산출물 확정: 이미지 + 카드뉴스 + 포스터 / 숏폼 영상 + 자동화 시나리오 (Day 2)",
            "location": "Slide 3"
        },
        "alternative_answers": ["이미지 세트, 카드뉴스, 포스터", "이미지·카드뉴스·포스터"],
        "confidence": 1.0,
        "needs_review": False,
        "_reasoning": "수동 수정 — Slide 3의 Day 2 산출물 확정 셀에서 카드뉴스와 함께 나오는 시각 산출물 3가지: 이미지, 카드뉴스, 포스터",
    },
    102: {
        "answer": "보조 도구, 실습 시나리오, 실습 난이도",
        "answer_type": "list_n",
        "citation": {
            "doc_id": "AI교육_프로그램개요_강사전달용",
            "snippet": "메인 도구 Genspark 외의 보조 도구·실습 시나리오·실습 난이도 설계는 강사님과 협의하여 확정합니다",
            "location": "Page 5"
        },
        "alternative_answers": ["보조도구, 실습 시나리오, 실습 난이도", "보조 도구·실습 시나리오·실습 난이도"],
        "confidence": 0.9,
        "needs_review": False,
        "_reasoning": "수동 수정 — Page 5 하단에 명시된 강사와 협의하여 확정하는 3가지 항목",
    },
    153: {
        "answer": "내부 인력 강의 경험 전무",
        "answer_type": "exact_phrase",
        "citation": {
            "doc_id": "AI교육_내부기준문서_v2",
            "snippet": "핵심 변수: 내부 인력 강의 경험 전무. 소싱 과정에서 A/B/C 강의료 모두 확인 → 비용·리스크 비교 후 최종 결정",
            "location": "Slide 4"
        },
        "alternative_answers": ["내부 강의력 미검증", "내부 인력의 강의 경험 전무"],
        "confidence": 1.0,
        "needs_review": False,
        "_reasoning": "수동 수정 — Slide 4의 강사 투입 시나리오 하단에 '핵심 변수: 내부 인력 강의 경험 전무'로 명시",
    },
}


def main() -> None:
    labels = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
    fixed_count = 0
    for label in labels:
        if label["id"] in MANUAL_FIXES:
            fix = MANUAL_FIXES[label["id"]]
            for k, v in fix.items():
                label[k] = v
            # _error 키 제거
            label.pop("_error", None)
            print(f"FIXED Q{label['id']}: {label['answer']}")
            fixed_count += 1

    LABELS_FILE.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{fixed_count} labels fixed in {LABELS_FILE}")


if __name__ == "__main__":
    main()
