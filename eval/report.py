"""
채점 결과 breakdown 리포트.

입력: eval/results/<run_name>__scored.json
출력: 콘솔 + eval/results/<run_name>__report.md

세부 분석:
  - 카테고리별 정확도
  - answer_type별 정확도
  - task_type별 정확도 (라우팅 결과별)
  - 상위 실패 케이스 20개

Usage:
    python eval/report.py --run baseline
    python eval/report.py --run baseline --compare lever3_prompt
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

EVAL_DIR = Path(__file__).parent
DATA_DIR = EVAL_DIR / "data"
RESULTS_DIR = EVAL_DIR / "results"
LABELS_FILE = DATA_DIR / "labels.json"


def load_scored(run_name: str) -> list[dict]:
    path = RESULTS_DIR / f"{run_name}__scored.json"
    if not path.exists():
        raise SystemExit(f"ERROR: {path} 없음. judge.py 먼저 실행.")
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(scored: list[dict], labels: list[dict]) -> dict:
    by_cat: dict[str, list[float]] = defaultdict(list)
    by_type: dict[str, list[float]] = defaultdict(list)
    by_task: dict[str, list[float]] = defaultdict(list)

    label_type_by_id = {l["id"]: l.get("answer_type", "reasoning") for l in labels}

    for r in scored:
        score = r["score"]["score_value"]
        by_cat[r["category"]].append(score)
        by_type[label_type_by_id.get(r["id"], "reasoning")].append(score)
        by_task[r.get("task_type") or "unknown"].append(score)

    return {
        "by_category": {k: (sum(v) / len(v), len(v)) for k, v in by_cat.items()},
        "by_answer_type": {k: (sum(v) / len(v), len(v)) for k, v in by_type.items()},
        "by_task_type": {k: (sum(v) / len(v), len(v)) for k, v in by_task.items()},
    }


def top_failures(scored: list[dict], n: int = 20) -> list[dict]:
    """score_value 낮은 순서 + 응답 길이 합리적 (잘못된 confidently wrong 우선)."""
    failures = [r for r in scored if not r["score"]["correct"]]
    # score_value 오름차순 → 0.0인 것 먼저
    failures.sort(key=lambda r: (r["score"]["score_value"], r["id"]))
    return failures[:n]


def format_table(rows: list[tuple], headers: tuple) -> str:
    """간단한 markdown 표."""
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def render_report(run_name: str, scored: list[dict], agg: dict, failures: list[dict]) -> str:
    overall_correct = sum(1 for r in scored if r["score"]["correct"])
    overall_total = len(scored)
    overall_acc = overall_correct / overall_total * 100 if overall_total else 0.0
    avg_score = sum(r["score"]["score_value"] for r in scored) / max(overall_total, 1)
    err_count = sum(1 for r in scored if r.get("error") or r["score"].get("method") == "system_error")

    lines: list[str] = []
    lines.append(f"# Eval Report — `{run_name}`")
    lines.append("")
    lines.append(f"- 총 문항: **{overall_total}**")
    lines.append(f"- 정답 (binary): **{overall_correct}/{overall_total} = {overall_acc:.1f}%**")
    lines.append(f"- 평균 점수 (partial): **{avg_score:.3f}**")
    lines.append(f"- 시스템 오류: {err_count}")
    lines.append("")

    # 카테고리별
    lines.append("## 카테고리별 정확도")
    rows = []
    for cat in sorted(agg["by_category"]):
        avg, n = agg["by_category"][cat]
        rows.append((cat, n, f"{avg*100:.1f}%"))
    lines.append(format_table(rows, ("Category", "N", "Avg Score")))
    lines.append("")

    # answer_type별
    lines.append("## Answer Type별 정확도")
    rows = []
    for t in sorted(agg["by_answer_type"]):
        avg, n = agg["by_answer_type"][t]
        rows.append((t, n, f"{avg*100:.1f}%"))
    lines.append(format_table(rows, ("Answer Type", "N", "Avg Score")))
    lines.append("")

    # task_type별
    lines.append("## Task Type (라우팅) 별 정확도")
    rows = []
    for t in sorted(agg["by_task_type"]):
        avg, n = agg["by_task_type"][t]
        rows.append((t, n, f"{avg*100:.1f}%"))
    lines.append(format_table(rows, ("Task Type", "N", "Avg Score")))
    lines.append("")

    # 실패 사례 상위
    lines.append(f"## 상위 실패 사례 ({len(failures)}건)")
    for f in failures:
        lines.append(f"### Q{f['id']} [{f['category']}]")
        lines.append(f"- **질문**: {f['question']}")
        lines.append(f"- **정답**: {f['score']['ground_truth']}")
        sys_ans = f.get("answer", "").replace("\n", " ")[:300]
        lines.append(f"- **시스템 답변**: {sys_ans}")
        lines.append(f"- **task_type**: `{f.get('task_type', '')}`")
        lines.append(f"- **judge**: score={f['score']['score_value']:.2f}, "
                     f"method={f['score']['method']}")
        lines.append(f"  - reasoning: {f['score']['reasoning']}")
        if f["score"].get("missing"):
            lines.append(f"  - missing: {f['score']['missing']}")
        if f["score"].get("extra"):
            lines.append(f"  - extra: {f['score']['extra']}")
        lines.append("")

    return "\n".join(lines)


def render_compare(base_name: str, base: list[dict], comp_name: str, comp: list[dict]) -> str:
    """두 run을 비교: 카테고리별 변화량, regression/improvement 케이스."""
    base_by_id = {r["id"]: r for r in base}
    comp_by_id = {r["id"]: r for r in comp}

    common_ids = sorted(set(base_by_id) & set(comp_by_id))

    base_correct = sum(1 for i in common_ids if base_by_id[i]["score"]["correct"])
    comp_correct = sum(1 for i in common_ids if comp_by_id[i]["score"]["correct"])

    improved = [i for i in common_ids
                if not base_by_id[i]["score"]["correct"] and comp_by_id[i]["score"]["correct"]]
    regressed = [i for i in common_ids
                 if base_by_id[i]["score"]["correct"] and not comp_by_id[i]["score"]["correct"]]

    lines: list[str] = []
    lines.append(f"# Compare — `{base_name}` vs `{comp_name}`")
    lines.append("")
    lines.append(f"- 공통 문항: {len(common_ids)}")
    lines.append(f"- {base_name}: {base_correct}/{len(common_ids)} = "
                 f"{base_correct/len(common_ids)*100:.1f}%")
    lines.append(f"- {comp_name}: {comp_correct}/{len(common_ids)} = "
                 f"{comp_correct/len(common_ids)*100:.1f}%")
    lines.append(f"- **변화: {comp_correct - base_correct:+d}**")
    lines.append("")
    lines.append(f"## 신규 정답 (개선 {len(improved)}건)")
    for i in improved[:20]:
        q = base_by_id[i]
        lines.append(f"- Q{i} [{q['category']}]: {q['question'][:80]}")
    lines.append("")
    lines.append(f"## 신규 오답 (회귀 {len(regressed)}건)")
    for i in regressed[:20]:
        q = base_by_id[i]
        lines.append(f"- Q{i} [{q['category']}]: {q['question'][:80]}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--compare", help="비교 대상 run name")
    parser.add_argument("--top", type=int, default=20, help="실패 사례 표시 개수")
    args = parser.parse_args()

    labels = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
    scored = load_scored(args.run)
    agg = aggregate(scored, labels)
    failures = top_failures(scored, n=args.top)
    md = render_report(args.run, scored, agg, failures)

    out_path = RESULTS_DIR / f"{args.run}__report.md"
    out_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n→ Saved: {out_path}")

    if args.compare:
        comp_scored = load_scored(args.compare)
        cmp_md = render_compare(args.run, scored, args.compare, comp_scored)
        cmp_path = RESULTS_DIR / f"compare__{args.run}__vs__{args.compare}.md"
        cmp_path.write_text(cmp_md, encoding="utf-8")
        print()
        print(cmp_md)
        print(f"\n→ Saved: {cmp_path}")


if __name__ == "__main__":
    main()
