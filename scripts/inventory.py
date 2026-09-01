#!/usr/bin/env python
"""사내 디스크 실태 파악 — 폴더 인벤토리 추출 (Phase 0).

【목적】
수만 개 문서를 인제스트하기 전에 규모·구성·위험을 숫자로 파악한다.
이게 없으면 비용도 일정도 권한 설계도 추정할 수 없다.

【원칙 — 파일 내용을 읽지 않는다】
파일명·크기·수정일 등 메타데이터만 수집한다. 따라서
  - 보안 위험이 없다 (내용이 프로세스에 올라오지 않음)
  - API 비용이 0 이다
  - 네트워크 드라이브에서도 빠르다
민감 키워드 판정도 **폴더명·파일명만** 보고 하며, 어디까지나 태그 지정 시
사람이 참고할 힌트다. 판정이 아니다.

【산출물】
  inventory_folders.csv   폴더별 집계 — 태그 지정 작업용 (사람이 채움)
  inventory_summary.json  전체 요약 — 규모·형식·중복 추정
  콘솔 리포트

【사용】
  python scripts/inventory.py --root "Z:/" --depth 2
  python scripts/inventory.py --root "Z:/" --root "D:/공유" --depth 3 --out ./inv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

# Windows 콘솔 기본 인코딩(cp949)이 일부 문자를 못 찍는다. UTF-8 로 고정한다.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── 형식 분류 ──────────────────────────────────────────────
# 현재 인제스트 파이프라인이 처리할 수 있는 것과 없는 것을 나눈다.
SUPPORTED = {".pdf", ".docx", ".pptx", ".xlsx", ".xlsm", ".txt", ".md"}
NEEDS_WORK = {".hwp", ".hwpx", ".doc", ".ppt", ".xls"}   # 변환기 추가 필요
IGNORE = {".lnk", ".tmp", ".ini", ".db", ".ds_store", ".url", ".exe", ".dll"}

# ── 민감 가능성 힌트 (이름만 보고 판단, 참고용) ──────────────
# 이 목록에 걸린다고 기밀인 것이 아니다. 태그 지정 시 사람이 우선 확인할
# 대상을 좁히는 용도다. 반대로 여기 안 걸려도 기밀일 수 있다.
SENSITIVE_HINTS = [
    "급여", "연봉", "인사", "평가", "고과", "징계", "개인정보", "주민",
    "이력서", "계약", "견적", "세금계산서", "매출", "원가", "정산",
    "임원", "이사회", "지분", "투자", "감사", "법무", "소송",
]

# 판본·중복 추정용 접미 패턴
_VER_PAT = re.compile(
    r"[\s_\-(]*(최종|final|fin|수정본?|복사본|사본|copy|백업|backup|"
    r"v\d+(\.\d+)*|ver\d*|\d{1,2}차|rev\d*|\(\d+\)|_\d{6,8})[\s_\-)]*",
    re.I,
)


def norm_name(name: str) -> str:
    """판본 접미를 걷어낸 이름. 중복 추정에 쓴다."""
    stem = os.path.splitext(name)[0]
    prev = None
    while prev != stem:
        prev = stem
        stem = _VER_PAT.sub("", stem).strip()
    return re.sub(r"\s+", "", stem).lower()


def bucket(ext: str) -> str:
    e = ext.lower()
    if e in SUPPORTED:
        return "지원"
    if e in NEEDS_WORK:
        return "변환필요"
    if e in IGNORE or not e:
        return "무시"
    return "기타"


def scan(roots: list[str], depth: int, max_files: int | None):
    """폴더 트리를 순회하며 메타데이터만 수집한다."""
    folders: dict[str, dict] = {}
    ext_all = Counter()
    dup_key = defaultdict(list)          # (정규화이름, 크기) -> 경로들
    denied: list[str] = []
    total_files = 0
    total_bytes = 0
    t0 = time.time()

    def group_of(path: str, root: str) -> str:
        """태그 지정 단위 — root 기준 depth 단계까지의 경로."""
        rel = os.path.relpath(path, root)
        if rel == ".":
            return root
        parts = rel.split(os.sep)[:depth]
        return os.path.join(root, *parts)

    for root in roots:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            print(f"  [건너뜀] 폴더가 아님: {root}", file=sys.stderr)
            continue
        for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: denied.append(str(e))):
            # 시스템·휴지통 계열 제외
            dirnames[:] = [d for d in dirnames
                           if not d.startswith(("$", "~", "."))
                           and d.lower() not in ("@recycle", "system volume information")]
            g = group_of(dirpath, root)
            f = folders.setdefault(g, {
                "group": g, "files": 0, "bytes": 0,
                "지원": 0, "변환필요": 0, "기타": 0, "무시": 0,
                "newest": 0, "oldest": 0, "exts": Counter(),
                "samples": [], "subdirs": set(), "hints": set(),
            })
            rel = os.path.relpath(dirpath, g)
            if rel != "." and rel.split(os.sep)[0]:
                f["subdirs"].add(rel.split(os.sep)[0])
            for h in SENSITIVE_HINTS:
                if h in dirpath:
                    f["hints"].add(h)

            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                b = bucket(ext)
                if b == "무시":
                    f["무시"] += 1
                    continue
                p = os.path.join(dirpath, name)
                try:
                    st = os.stat(p)
                except OSError as e:
                    denied.append(f"{p}: {e}")
                    continue
                total_files += 1
                total_bytes += st.st_size
                f["files"] += 1
                f["bytes"] += st.st_size
                f[b] += 1
                f["exts"][ext] += 1
                ext_all[ext] += 1
                m = int(st.st_mtime)
                f["newest"] = max(f["newest"], m)
                f["oldest"] = m if f["oldest"] == 0 else min(f["oldest"], m)
                if len(f["samples"]) < 8:
                    f["samples"].append(name)
                for h in SENSITIVE_HINTS:
                    if h in name:
                        f["hints"].add(h)
                if b in ("지원", "변환필요"):
                    dup_key[(norm_name(name), st.st_size)].append(p)

                if total_files % 2000 == 0:
                    print(f"  … {total_files:,}개 ({time.time()-t0:.0f}s) {dirpath[:70]}",
                          flush=True)
                if max_files and total_files >= max_files:
                    print(f"  [중단] --max-files {max_files} 도달", flush=True)
                    return folders, ext_all, dup_key, denied, total_files, total_bytes

    return folders, ext_all, dup_key, denied, total_files, total_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True, help="스캔할 최상위 경로 (여러 번 가능)")
    ap.add_argument("--depth", type=int, default=2, help="태그 지정 단위 깊이 (기본 2)")
    ap.add_argument("--out", default=".", help="결과 저장 폴더")
    ap.add_argument("--max-files", type=int, default=None, help="시험 실행용 상한")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    print(f"스캔 시작 — {a.root}  (depth={a.depth}, 파일 내용은 읽지 않음)\n", flush=True)
    folders, ext_all, dup_key, denied, n_files, n_bytes = scan(a.root, a.depth, a.max_files)

    # ── 중복 추정 ──
    dups = {k: v for k, v in dup_key.items() if len(v) > 1}
    dup_files = sum(len(v) for v in dups.values())
    dup_waste = sum(k[1] * (len(v) - 1) for k, v in dups.items())

    # ── CSV: 태그 지정 작업용 ──
    csv_path = os.path.join(a.out, "inventory_folders.csv")
    rows = sorted(folders.values(), key=lambda x: -x["files"])
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(["폴더", "파일수", "용량MB", "지원", "변환필요", "기타",
                    "최신수정", "최근3년내", "민감힌트", "하위폴더예시", "파일명예시",
                    "→ 태그", "→ 색인여부"])
        cutoff = time.time() - 3 * 365 * 86400
        for f in rows:
            if f["files"] == 0:
                continue
            w.writerow([
                f["group"], f["files"], round(f["bytes"] / 1024 / 1024, 1),
                f["지원"], f["변환필요"], f["기타"],
                datetime.fromtimestamp(f["newest"], timezone.utc).strftime("%Y-%m-%d") if f["newest"] else "",
                "예" if f["newest"] >= cutoff else "아니오",
                " ".join(sorted(f["hints"])),
                " / ".join(sorted(f["subdirs"])[:5]),
                " / ".join(f["samples"][:5]),
                "", "",          # 사람이 채우는 칸
            ])

    # ── JSON 요약 ──
    summary = {
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "roots": a.root, "depth": a.depth,
        "total_files": n_files, "total_gb": round(n_bytes / 1024**3, 2),
        "folder_groups": len([f for f in folders.values() if f["files"]]),
        "by_bucket": {b: sum(f[b] for f in folders.values())
                      for b in ("지원", "변환필요", "기타", "무시")},
        "top_ext": ext_all.most_common(20),
        "duplicate_groups": len(dups), "duplicate_files": dup_files,
        "duplicate_waste_gb": round(dup_waste / 1024**3, 2),
        "access_denied": len(denied),
    }
    with open(os.path.join(a.out, "inventory_summary.json"), "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    # ── 콘솔 리포트 ──
    sup = summary["by_bucket"]["지원"]
    conv = summary["by_bucket"]["변환필요"]
    print("\n" + "=" * 66)
    print(f"파일 {n_files:,}개, {summary['total_gb']:.1f}GB, 폴더그룹 {summary['folder_groups']}개")
    print(f"  인제스트 가능      {sup:,}개 ({sup/max(n_files,1)*100:.0f}%)")
    print(f"  변환기 필요        {conv:,}개 ({conv/max(n_files,1)*100:.0f}%)  ← HWP·구버전 오피스")
    print(f"  기타               {summary['by_bucket']['기타']:,}개")
    print(f"\n형식 분포 (상위 10)")
    for e, c in ext_all.most_common(10):
        print(f"    {e or '(없음)':10s} {c:7,}개")
    print(f"\n중복 추정  {summary['duplicate_groups']:,}개 그룹 / {dup_files:,}개 파일"
          f" / 중복분 {summary['duplicate_waste_gb']:.1f}GB")
    print(f"  (판본 접미 '최종·수정본·v1·(1)' 등을 제거한 이름 + 크기 일치 기준)")
    if denied:
        print(f"\n접근 불가 {len(denied)}건 — 권한 확인 필요")
        for d in denied[:5]:
            print(f"    {d[:100]}")
    hinted = [f for f in rows if f["hints"] and f["files"]]
    print(f"\n민감 가능성 폴더 {len(hinted)}개 — 태그 지정 시 우선 확인")
    for f in hinted[:10]:
        print(f"    [{' '.join(sorted(f['hints']))[:24]:24s}] {f['group'][:60]} ({f['files']}개)")
    print("=" * 66)
    print(f"\n다음 단계: {csv_path} 의 '→ 태그' / '→ 색인여부' 칸을 채우십시오.")
    print("  태그 예:  all(전사) / hr / exec / project_kb / education_biz")
    print("  색인여부: Y / N   ← 개인폴더·판단 애매한 곳은 N 으로 시작하는 편이 안전")


if __name__ == "__main__":
    main()
