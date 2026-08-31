from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_chroma import Chroma

from app.core.config import (
    get_embeddings,
    KNOWLEDGE_DIR as _KNOWLEDGE_DIR,
    AUTO_INGEST,
    S3_KNOWLEDGE_BUCKET,
    S3_KNOWLEDGE_PREFIX,
    CHROMA_DB_PATH,
    CHROMA_COLLECTION,
    INGEST_CHUNK_MAX_CHARS,
    INGEST_CHUNK_OVERLAP,
)

# 지원 형식 (file_extractor와 동일 셋)
_SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".docx", ".xlsx", ".xlsm", ".pptx"}

_CHUNK_MAX_CHARS = INGEST_CHUNK_MAX_CHARS
_CHUNK_OVERLAP   = INGEST_CHUNK_OVERLAP
_STATE_FILE = ".ingest_state.json"


# ──────────────────────────────────────────────
# 텍스트 추출
# ──────────────────────────────────────────────

def _extract_text(p: Path) -> str:
    suffix = p.suffix.lower()
    if suffix in {".txt", ".md"}:
        return p.read_text(encoding="utf-8", errors="ignore").strip()

    # 바이너리 형식은 file_extractor 함수 재사용
    from app.graph.nodes.file_extractor import extract_text_from_file
    text, _ = extract_text_from_file(p)
    return text.strip()


def _extract_pdf_pages(p: Path) -> List[Tuple[int, str]]:
    """PDF를 페이지 단위로 추출. [(1-indexed page_num, text), ...] 반환."""
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as e:
        raise RuntimeError(f"PDF 추출을 위해 pypdf가 필요합니다: {e}")

    reader = PdfReader(str(p))
    result: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""
        if text:
            result.append((i, text))
    return result


# ──────────────────────────────────────────────
# 청킹
# ──────────────────────────────────────────────

def _chunk_text(
    text: str,
    max_chars: int = _CHUNK_MAX_CHARS,
    overlap: int = _CHUNK_OVERLAP,
) -> List[str]:
    """단락 기반 청킹. 청크 간 overlap 만큼 겹쳐 단락 경계 누락을 방지.

    - 큰 단락(>max_chars): 슬라이딩 윈도우로 절단(step = max_chars - overlap).
    - 누적 청크가 max_chars 초과 시: 확정 후 새 청크는 직전 청크 끝 overlap 글자로 시작.
    - overlap=0이면 종전 동작(오버랩 없음).
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        if len(para) > max_chars:
            if current:
                chunks.append("\n\n".join(current))
                current, current_len = [], 0
            step = max(max_chars - overlap, 1)
            for i in range(0, len(para), step):
                chunks.append(para[i : i + max_chars])
        elif current_len + len(para) + 2 > max_chars and current:
            confirmed = "\n\n".join(current)
            chunks.append(confirmed)
            if overlap > 0:
                tail = confirmed[-overlap:]
                current = [tail, para]
                current_len = len(tail) + len(para) + 2
            else:
                current = [para]
                current_len = len(para)
        else:
            current.append(para)
            current_len += len(para) + 2

    if current:
        chunks.append("\n\n".join(current))

    return chunks or ([text[:max_chars]] if text else [])


# ──────────────────────────────────────────────
# 파일 해시 (중복 방지)
# ──────────────────────────────────────────────

def _file_hash(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def _load_state(knowledge_dir: Path) -> Dict:
    path = knowledge_dir / _STATE_FILE
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_state(knowledge_dir: Path, state: Dict) -> None:
    try:
        (knowledge_dir / _STATE_FILE).write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        print(f"[INGEST] state 저장 실패 (non-fatal): {e}")


# ──────────────────────────────────────────────
# S3 동기화
# ──────────────────────────────────────────────

def _sync_from_s3(knowledge_dir: Path) -> None:
    bucket = S3_KNOWLEDGE_BUCKET.strip()
    if not bucket:
        return

    prefix = S3_KNOWLEDGE_PREFIX.strip()
    if not prefix.endswith("/"):
        prefix += "/"

    try:
        import boto3
        region = (os.getenv("AWS_REGION") or "ap-northeast-1").strip()
        s3 = boto3.client("s3", region_name=region)

        knowledge_dir.mkdir(parents=True, exist_ok=True)

        paginator = s3.get_paginator("list_objects_v2")
        count = 0
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                rel = key[len(prefix):]
                dest = knowledge_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, key, str(dest))
                count += 1

        print(f"[INGEST] S3 sync: {count}개 파일 → {knowledge_dir}")
    except Exception as e:
        print(f"[INGEST] S3 sync 실패 (non-fatal): {e}")


# ──────────────────────────────────────────────
# 메인 인제스트
# ──────────────────────────────────────────────

def auto_ingest_if_enabled() -> None:
    """
    startup 시 호출됩니다.
    1. S3_KNOWLEDGE_BUCKET 설정 시 S3 → KNOWLEDGE_DIR 동기화
    2. 파일 해시 비교 — 변경된 파일만 Chroma 재적재
    3. 삭제된 파일의 청크는 Chroma에서 제거
    """
    enabled = (AUTO_INGEST or "").strip() in ("1", "true", "True", "YES", "yes")
    if not enabled:
        return

    knowledge_dir = Path(_KNOWLEDGE_DIR)
    _sync_from_s3(knowledge_dir)

    if not knowledge_dir.exists() or not knowledge_dir.is_dir():
        return

    db_path = CHROMA_DB_PATH
    collection = CHROMA_COLLECTION
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name=collection,
    )

    state = _load_state(knowledge_dir)
    current_files: Dict[str, Path] = {}

    for p in sorted(knowledge_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() not in _SUPPORTED_SUFFIXES:
            continue
        rel = str(p.relative_to(knowledge_dir)).replace(os.sep, "/")
        current_files[rel] = p

    added = skipped = updated = deleted = 0

    # 삭제된 파일 청크 제거
    for rel in list(state.keys()):
        if rel not in current_files:
            old_ids = state[rel].get("chunk_ids", [])
            if old_ids:
                try:
                    vectorstore.delete(ids=old_ids)
                    print(f"[INGEST] 삭제: {rel} ({len(old_ids)}개 청크 제거)")
                    deleted += 1
                except Exception as e:
                    print(f"[INGEST] 청크 삭제 실패 {rel}: {e}")
            del state[rel]

    # 신규/변경 파일 인제스트
    for rel, p in current_files.items():
        try:
            current_hash = _file_hash(p)
        except Exception as e:
            print(f"[INGEST] 해시 실패 {rel}: {e}")
            continue

        if state.get(rel, {}).get("hash") == current_hash:
            skipped += 1
            continue

        # 기존 청크 삭제
        old_ids = state.get(rel, {}).get("chunk_ids", [])
        if old_ids:
            try:
                vectorstore.delete(ids=old_ids)
            except Exception:
                pass

        # ─── Phase G: 형식 인지 청킹 파이프라인 ───
        # 1. format 분류 → 2. extractor → 3. adaptive chunking → 4. metadata 부착 (코드 + LLM)
        try:
            from app.knowledge.chunking.pipeline import run_chunking_only
            from app.knowledge.chunking.tagger import tag_chunks

            doc_id = rel.replace(".pptx", "").replace(".pdf", "").replace(".docx", "").replace(".txt", "").replace(".xlsx", "").replace(".md", "")

            fmt, units = run_chunking_only(p)
            if not units:
                print(f"[INGEST] 추출 결과 없음 {rel}")
                continue

            # metadata 부착 (LLM doc_topic만 — chunk_question_types는 폐기, Q&A cache로 대체)
            chunk_objs = tag_chunks(
                units,
                doc_id=doc_id,
                doc_format=fmt,
                enable_llm_doc_topic=True,
            )
            if not chunk_objs:
                print(f"[INGEST] 태깅 결과 없음 {rel}")
                continue

            chunk_ids = [c.chunk_id for c in chunk_objs]
            chunks_text = [c.text for c in chunk_objs]
            metadatas = [c.metadata for c in chunk_objs]
        except Exception as e:
            print(f"[INGEST] 청킹 파이프라인 실패 {rel}: {type(e).__name__}: {e}")
            continue

        try:
            vectorstore.add_texts(texts=chunks_text, metadatas=metadatas, ids=chunk_ids)
            state[rel] = {"hash": current_hash, "chunk_ids": chunk_ids}
            if old_ids:
                print(f"[INGEST] 갱신: {rel} → {len(chunk_objs)}개 청크 (fmt={fmt})")
                updated += 1
            else:
                print(f"[INGEST] 신규: {rel} → {len(chunk_objs)}개 청크 (fmt={fmt})")
                added += 1
        except Exception as e:
            print(f"[INGEST] Chroma 적재 실패 {rel}: {e}")

    _save_state(knowledge_dir, state)
    print(f"[INGEST] 완료 - 신규:{added} 갱신:{updated} 스킵:{skipped} 삭제:{deleted}")

    if added + updated + deleted > 0:
        try:
            from app.graph.nodes.knowledge_search import invalidate_bm25_cache
            invalidate_bm25_cache()
            print("[INGEST] BM25 캐시 무효화 완료")
        except Exception as e:
            print(f"[INGEST] BM25 캐시 무효화 실패 (non-fatal): {e}")