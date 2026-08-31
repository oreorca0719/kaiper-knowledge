"""decision_path를 파싱해 grader max_score / cov / kept_ratio를 추출하고
score(correct) 와의 상관을 분석한다.

목적: relevant=0.5 / partial=0.3 임계가 적정한지 데이터로 확인.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

EVAL_DIR = Path(__file__).parent
RESULTS = EVAL_DIR / "results" / "v2_reranker_grader__scored.json"

# grade:pass(5rel/5kept/5total,max=0.50,cov=full)
# grade:all_irrelevant(0rel/5total,max=0.28,cov=none)
# grade:exhausted_after_3rewrites(...)
GRADE_RE = re.compile(
    r"grade:(?P<status>pass|all_irrelevant|exhausted_after_3rewrites|no_docs)"
    r"(?:\((?P<rel>\d+)rel/(?:(?P<kept>\d+)kept/)?(?P<total>\d+)total,"
    r"max=(?P<max>[\d.]+),cov=(?P<cov>full|partial|none)\))?"
)


def parse_grades(decision_path: list[str]) -> list[dict]:
    grades: list[dict] = []
    for step in decision_path:
        m = GRADE_RE.search(step)
        if not m:
            continue
        d = m.groupdict()
        grades.append(
            {
                "status": d["status"],
                "rel": int(d["rel"]) if d["rel"] else 0,
                "kept": int(d["kept"]) if d["kept"] else 0,
                "total": int(d["total"]) if d["total"] else 0,
                "max": float(d["max"]) if d["max"] else 0.0,
                "cov": d["cov"] or "none",
            }
        )
    return grades


def parse_score(score_field) -> bool:
    """scored.json의 score 필드는 dict 또는 str(repr(dict)) 둘 다 등장 가능."""
    if isinstance(score_field, dict):
        return bool(score_field.get("correct", False))
    if isinstance(score_field, str):
        # 안전하게 ast.literal_eval
        import ast
        try:
            return bool(ast.literal_eval(score_field).get("correct", False))
        except Exception:
            return False
    return False


def main() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))

    rows = []
    for item in data:
        grades = parse_grades(item.get("decision_path") or [])
        if not grades:
            continue
        # 마지막(=최종) grade가 채택 결과
        final = grades[-1]
        first = grades[0]
        correct = parse_score(item.get("score"))
        rows.append(
            {
                "id": item["id"],
                "category": item.get("category"),
                "correct": correct,
                "task_type": item.get("task_type"),
                "first_max": first["max"],
                "final_max": final["max"],
                "first_cov": first["cov"],
                "final_cov": final["cov"],
                "first_rel": first["rel"],
                "final_rel": final["rel"],
                "final_status": final["status"],
                "n_grades": len(grades),
            }
        )

    print(f"총 {len(rows)}건 (grade 로그 포함)")

    # 1. 첫 grade의 max_score 분포 — 정/오답별
    bins = [0.0, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.01]
    print("\n=== 첫 retrieve의 max(reranker score) 분포 (정답/오답) ===")
    print(f"{'bin':<14}{'correct':<10}{'wrong':<10}{'precision':<12}")
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = [r for r in rows if lo <= r["first_max"] < hi]
        c = sum(1 for r in in_bin if r["correct"])
        w = len(in_bin) - c
        prec = (c / len(in_bin)) if in_bin else 0.0
        print(f"[{lo:.2f},{hi:.2f}) {c:<10}{w:<10}{prec*100:>6.1f}%")

    # 2. cov 별 정답률
    print("\n=== 첫 retrieve의 coverage별 정답률 ===")
    for cov in ["full", "partial", "none"]:
        sub = [r for r in rows if r["first_cov"] == cov]
        c = sum(1 for r in sub if r["correct"])
        prec = (c / len(sub) * 100) if sub else 0.0
        print(f"cov={cov:<8} n={len(sub):<5} correct={c:<5} acc={prec:.1f}%")

    # 3. 임계 ablation 시뮬레이션 — 첫 grade의 max_score vs 정/오답
    print("\n=== 임계 후보별 시뮬레이션 (첫 retrieve 기준) ===")
    print("- pass: max>=relevant_threshold 인 케이스에서 정답률")
    print("- reject_n: max<partial_threshold 라 grader가 모두 떨어뜨릴 케이스")
    print(f"{'rel_thr':<10}{'par_thr':<10}{'pass_n':<8}{'pass_acc':<10}"
          f"{'reject_n':<10}{'rejected_correct(loss)':<25}")
    for rel_thr, par_thr in [
        (0.50, 0.30),  # 현재
        (0.55, 0.30),
        (0.55, 0.35),
        (0.60, 0.35),
        (0.60, 0.40),
        (0.65, 0.40),
        (0.70, 0.45),
    ]:
        # pass: first_max >= rel_thr 로 가정 (relevant 1개 이상 보장)
        passed = [r for r in rows if r["first_max"] >= rel_thr]
        rejected = [r for r in rows if r["first_max"] < par_thr]
        c_pass = sum(1 for r in passed if r["correct"])
        c_rej = sum(1 for r in rejected if r["correct"])
        pass_acc = (c_pass / len(passed) * 100) if passed else 0.0
        print(f"{rel_thr:<10.2f}{par_thr:<10.2f}"
              f"{len(passed):<8}{pass_acc:>6.1f}%   "
              f"{len(rejected):<10}{c_rej} (정답인데 rewrite로 갈 케이스)")

    # 4. 현재 잘못 통과시킨 의심 케이스 (max<0.6 인데 오답)
    print("\n=== 의심 케이스: max<0.6 + 오답 (grader가 약하게 통과시킨 것 추정) ===")
    suspects = sorted(
        [r for r in rows if not r["correct"] and r["first_max"] < 0.6],
        key=lambda r: r["first_max"],
    )
    print(f"총 {len(suspects)}건. 상위 20건:")
    for r in suspects[:20]:
        print(f"  Q{r['id']:<4} cat={r['category']} max={r['first_max']:.2f}"
              f" cov={r['first_cov']:<7} rel={r['first_rel']}/{r['final_rel']}"
              f" task={r['task_type']}")

    # 5. 임계 강화 시 잃을 케이스 (max<0.6 인데 정답) — 이걸 재 retrieval 통해 잃을 위험
    print("\n=== 임계 강화 시 잃을 위험: max<0.6 + 정답 ===")
    losers = sorted(
        [r for r in rows if r["correct"] and r["first_max"] < 0.6],
        key=lambda r: r["first_max"],
    )
    print(f"총 {len(losers)}건. 상위 20건:")
    for r in losers[:20]:
        print(f"  Q{r['id']:<4} cat={r['category']} max={r['first_max']:.2f}"
              f" cov={r['first_cov']:<7} task={r['task_type']}")


if __name__ == "__main__":
    main()
