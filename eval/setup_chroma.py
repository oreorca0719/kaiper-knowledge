"""
평가용 ChromaDB 구축.

12개 출처 문서 (C:\\Users\\User\\Desktop\\새 폴더\\)를 knowledge_data/로 복사하고
auto_ingest를 실행해 ChromaDB를 재구성한다.

이 스크립트를 1회 실행해야 runner.py가 의미있는 결과를 낸다.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SOURCE_DIR = Path(r"C:\Users\User\Desktop\새 폴더")
KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge_data"


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    os.environ.setdefault("AUTO_INGEST", "1")
    os.environ.setdefault("S3_KNOWLEDGE_BUCKET", "")  # S3 sync 비활성화

    # 1) 출처 디렉토리 → knowledge_data 복사
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in sorted(SOURCE_DIR.iterdir()):
        if not src.is_file():
            continue
        if src.suffix.lower() not in {".txt", ".pdf", ".docx", ".pptx"}:
            continue
        dest = KNOWLEDGE_DIR / src.name
        if dest.exists() and dest.stat().st_size == src.stat().st_size:
            print(f"SKIP (already copied): {src.name}")
            continue
        shutil.copy2(src, dest)
        print(f"COPIED: {src.name} ({src.stat().st_size} bytes)")
        copied += 1
    print(f"\nCopied {copied} files to {KNOWLEDGE_DIR}")

    # 2) ingest 실행
    from app.knowledge.ingest import auto_ingest_if_enabled
    print("\nRunning auto_ingest_if_enabled()...")
    auto_ingest_if_enabled()


if __name__ == "__main__":
    main()
