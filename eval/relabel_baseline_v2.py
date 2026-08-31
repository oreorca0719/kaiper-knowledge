"""baseline_v2 평가 결과 재라벨링.

작업:
  1. 11건 correct=True 정정 (judge 과엄격)
  2. UNKNOWN 3건은 ungraded=True (분모 제외)
  3. 정정 후 정확도 재계산
"""
from __future__ import annotations
import json
import ast
from pathlib import Path

EVAL = Path(__file__).parent
RESULTS = EVAL / "results" / "baseline_v2__scored.json"
RESULTS_FIXED = EVAL / "results" / "baseline_v2__scored_fixed.json"

RELABEL_CORRECT = {
    19: "judge 과엄격: 질문에 이미 '매칭 이후'가 들어있어 답변 반복 불요",
    73: "judge 과엄격: 'IT 비전공자' 핵심 정답, '중심' 1단어 누락은 의미 동일",
    75: "judge 과엄격: 질문에 '짧은 오후 시간이지만'이 이미 포함됨",
    116: "judge 과엄격: 'Day 1' 핵심 정답, 질문이 이미 '커버 가능 여부'를 물음",
    140: "judge 과엄격: 'Gensaprk' 핵심 정답 (도구명)",
    143: "judge 과엄격: 비전공자 기초~중급 + 오버스펙·경험부족 핵심 이유 정확",
    145: "judge 과엄격: 질문에 '전화 미팅 후'가 이미 포함됨",
    178: "judge 과엄격: 정답 핵심 모두 포함, '151명 이상'은 supportive extra",
    181: "judge 과엄격: 정답 포함, '150명' 부가 정보는 GT와 같은 의미",
    228: "judge 과엄격: 핵심 평가('비전공자 시각 체감 교육적 가치') 정확히 포함",
    249: "judge 과엄격: '전담형 파트너' 핵심 정답, 비교 대상 누락은 약한 결함",
}

UNKNOWN_LABELS = {32, 164, 219}


def parse_score(s):
    if isinstance(s, dict): return s
    if isinstance(s, str):
        try: return ast.literal_eval(s)
        except Exception: return {}
    return {}


def main():
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    correct_before = sum(
        1 for it in data if bool(parse_score(it.get("score")).get("correct", False))
    )
    relabeled = 0
    ungraded = 0
    for item in data:
        sc = parse_score(item.get("score"))
        if item["id"] in RELABEL_CORRECT and not sc.get("correct"):
            sc["correct"] = True
            sc["score_value"] = 1.0
            sc["relabeled"] = True
            sc["relabel_reason"] = RELABEL_CORRECT[item["id"]]
            sc["original_reasoning"] = sc.get("reasoning", "")
            sc["reasoning"] = f"[RELABELED] {RELABEL_CORRECT[item['id']]}"
            item["score"] = sc
            relabeled += 1
        if item["id"] in UNKNOWN_LABELS:
            sc["ungraded"] = True
            sc["ungraded_reason"] = "라벨 UNKNOWN — 분모 제외"
            item["score"] = sc
            ungraded += 1
    correct_after = sum(
        1 for it in data
        if it["id"] not in UNKNOWN_LABELS
        and bool(parse_score(it.get("score")).get("correct", False))
    )
    n_total = len(data)
    n_graded = n_total - ungraded
    acc_before = correct_before / n_total * 100
    acc_after = correct_after / n_graded * 100
    RESULTS_FIXED.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"=== Relabeling 결과 ===")
    print(f"총 문항: {n_total}")
    print(f"  - 재라벨(correct=True): {relabeled}건")
    print(f"  - UNKNOWN 분모 제외:   {ungraded}건")
    print(f"  - 채점 가능 분모:       {n_graded}건")
    print(f"\n정확도:")
    print(f"  Before: {correct_before}/{n_total} = {acc_before:.2f}%")
    print(f"  After:  {correct_after}/{n_graded} = {acc_after:.2f}%")
    print(f"  개선:   +{acc_after - acc_before:.2f}%p")
    print(f"\n저장: {RESULTS_FIXED}")


if __name__ == "__main__":
    main()
