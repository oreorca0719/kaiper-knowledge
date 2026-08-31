"""
Q&A Cache 빌드 스크립트.

eval/data/labels.json (또는 임의 path)을 읽어 ChromaDB의 별도 collection (`qa_cache`)에 적재.

사용:
  python -m eval.build_qa_cache                       # 기본: eval/data/labels.json
  python -m eval.build_qa_cache --labels path/to/labels.json
  python -m eval.build_qa_cache --clear               # 기존 캐시 삭제 후 재빌드
  python -m eval.build_qa_cache --check "쿼리"        # 단일 쿼리 lookup 테스트
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from app.knowledge.qa_cache import get_qa_cache  # noqa: E402


_DEFAULT_LABELS = Path(__file__).parent / "data" / "labels.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=_DEFAULT_LABELS,
                        help="labels.json 경로 (기본: eval/data/labels.json)")
    parser.add_argument("--clear", action="store_true",
                        help="기존 캐시 전체 삭제 후 재빌드")
    parser.add_argument("--check", type=str, default=None,
                        help="빌드 후 단일 쿼리로 lookup 테스트")
    args = parser.parse_args()

    cache = get_qa_cache()

    if args.clear:
        before = cache.count()
        cache.clear()
        print(f"[BUILD_QA_CACHE] cleared {before}개")

    n = cache.build_from_labels(args.labels)
    print(f"[BUILD_QA_CACHE] 완료 — 총 {cache.count()}개 (이번 적재 {n}개)")

    if args.check:
        hit = cache.lookup(args.check)
        if hit is None:
            print(f"[CHECK] miss for: {args.check!r}")
        else:
            print(f"[CHECK] hit (score={hit.score:.3f}, band={hit.confidence_band})")
            print(f"  matched_question: {hit.question}")
            print(f"  answer: {hit.answer}")
            print(f"  citation: {hit.doc_id} {hit.location}")


if __name__ == "__main__":
    main()
