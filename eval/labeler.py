"""
정답 라벨 자동 추출 스크립트.

Gemini API로 각 질문에 대한 정답을 mapped 출처 문서에서 추출한다.
출력: eval/data/labels.json

각 라벨 객체:
{
  "id": int,
  "category": "A",
  "question": str,
  "answer": str,                # 정답 (verbatim 우선)
  "answer_type": str,           # exact_phrase | numerical | list_n | fill_blank | reasoning
  "citation": {
    "doc_id": str,              # category_source_map의 doc_id
    "snippet": str,             # 답변 근거 발췌 (50~200자)
    "location": str             # "Slide 3", "Page 2" 등
  },
  "alternative_answers": [str], # 동등 표현 (채점 시 OR 매칭)
  "confidence": float,          # 0.0~1.0 — LLM이 자체 평가
  "needs_review": bool          # confidence<0.7 또는 추론 케이스 표시
}

Usage:
    GEMINI_API_KEY=... python eval/labeler.py
    GEMINI_API_KEY=... python eval/labeler.py --categories A,B  # 특정 카테고리만
    GEMINI_API_KEY=... python eval/labeler.py --resume          # 기존 labels.json 보존, 누락만 채움
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# 프로젝트 root에서 실행 가정 (sys.path 조정)
sys.path.insert(0, str(Path(__file__).parent.parent))


EVAL_DIR = Path(__file__).parent
DATA_DIR = EVAL_DIR / "data"
SOURCES_DIR = DATA_DIR / "sources"

QUESTIONS_FILE = DATA_DIR / "questions.json"
MAP_FILE = DATA_DIR / "category_source_map.json"
LABELS_FILE = DATA_DIR / "labels.json"


SYSTEM_PROMPT = """당신은 사내 문서에서 질문의 정답을 정확히 추출하는 라벨링 전문가입니다.

【원칙】
1. 답은 반드시 [출처] 텍스트에 명시된 내용에서만 추출하세요. 추론·외부 지식 금지.
2. 수치·고유명사·인용 문구는 원문 그대로 (verbatim). 임의 paraphrase 금지.
3. 답을 출처에서 찾을 수 없으면 answer="UNKNOWN", confidence=0.0으로 반환.
4. 빈칸 채우기 질문은 빈칸에 들어갈 단어/구만 답하세요 (전체 문장 아님).
5. "N가지/N개" 질문은 정확히 N개 항목을 출처 표현 그대로 콤마로 구분.

【출력 형식】 — JSON만, 다른 텍스트 금지
{
  "answer": "...",
  "answer_type": "exact_phrase | numerical | list_n | fill_blank | reasoning",
  "citation_snippet": "출처에서 답변 근거가 된 부분 50~200자 발췌",
  "citation_location": "Slide N | Page N | Section N 등",
  "alternative_answers": ["동등한 표현 1", "동등한 표현 2"],
  "confidence": 0.0~1.0,
  "reasoning": "한 줄 — 어디서 어떻게 찾았는지"
}

confidence 가이드:
- 1.0: 출처에 verbatim으로 존재
- 0.8~0.9: 출처에 분명히 있으나 일부 표현 차이
- 0.5~0.7: 출처에서 추론·종합 필요
- <0.5: 불확실하거나 부분 정보만 존재
- 0.0: 답을 찾을 수 없음
"""


def load_sources(doc_ids: list[str]) -> str:
    """매핑된 출처 문서들을 하나의 컨텍스트로 합침."""
    blocks: list[str] = []
    for doc_id in doc_ids:
        path = SOURCES_DIR / f"{doc_id}.txt"
        if not path.exists():
            print(f"WARN: source not found: {doc_id}")
            continue
        text = path.read_text(encoding="utf-8")
        blocks.append(f"=== [출처: {doc_id}] ===\n{text}")
    return "\n\n".join(blocks)


def call_gemini(question: str, sources_text: str, model_name: str = "gemini-2.5-flash") -> dict:
    """Gemini API 호출. structured JSON 응답 강제."""
    from app.core.config import get_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_llm(model_name)
    user_content = f"[질문]\n{question}\n\n[출처]\n{sources_text}"

    resp = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ])
    raw = resp.content
    if isinstance(raw, list):
        raw = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in raw)
    raw = str(raw).strip()

    # ```json fenced 제거
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip().rstrip("`").strip()

    return json.loads(raw)


def label_one(q: dict, source_map: dict, model_name: str) -> dict:
    """단일 질문에 대해 라벨 생성."""
    cat = q["category"]
    cat_info = source_map[cat]
    sources_text = load_sources(cat_info["sources"])

    parsed = call_gemini(q["question"], sources_text, model_name=model_name)

    # 메인 doc_id는 첫 번째 매핑된 source (필요 시 LLM이 어느 doc에서 왔는지 알려주도록 확장 가능)
    main_doc = cat_info["sources"][0]

    return {
        "id": q["id"],
        "category": cat,
        "question": q["question"],
        "answer": parsed.get("answer", "UNKNOWN"),
        "answer_type": parsed.get("answer_type", "reasoning"),
        "citation": {
            "doc_id": main_doc,
            "snippet": parsed.get("citation_snippet", ""),
            "location": parsed.get("citation_location", ""),
        },
        "alternative_answers": parsed.get("alternative_answers") or [],
        "confidence": float(parsed.get("confidence", 0.0)),
        "needs_review": float(parsed.get("confidence", 0.0)) < 0.7,
        "_reasoning": parsed.get("reasoning", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", help="콤마구분 (예: A,B,C). 없으면 전체.")
    parser.add_argument("--resume", action="store_true", help="기존 labels.json 보존, 누락만 추가.")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini 모델명")
    parser.add_argument("--sleep", type=float, default=0.3, help="API 호출 간 sleep (rate limit 회피)")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        print("ERROR: GOOGLE_API_KEY 또는 GEMINI_API_KEY가 설정되어 있지 않습니다.")
        sys.exit(1)

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    source_map = json.loads(MAP_FILE.read_text(encoding="utf-8"))

    target_cats = set(args.categories.split(",")) if args.categories else set("ABCDEFGHIJKLMNOP")
    targets = [q for q in questions if q["category"] in target_cats]

    existing: dict[int, dict] = {}
    if args.resume and LABELS_FILE.exists():
        for lab in json.loads(LABELS_FILE.read_text(encoding="utf-8")):
            existing[lab["id"]] = lab
        targets = [q for q in targets if q["id"] not in existing]
        print(f"Resume mode: {len(existing)} 기존 라벨 보존, {len(targets)} 신규 처리.")

    all_labels: list[dict] = list(existing.values())
    fail_count = 0

    for idx, q in enumerate(targets, 1):
        try:
            label = label_one(q, source_map, model_name=args.model)
            all_labels.append(label)
            print(f"[{idx}/{len(targets)}] Q{q['id']} ({q['category']}) "
                  f"conf={label['confidence']:.2f} {'[REVIEW]' if label['needs_review'] else 'OK'}")
        except Exception as e:
            print(f"[{idx}/{len(targets)}] Q{q['id']} FAIL: {type(e).__name__}: {e}")
            fail_count += 1
            # 실패 케이스도 stub으로 추가 (나중에 retry 가능)
            all_labels.append({
                "id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "answer": "ERROR",
                "answer_type": "reasoning",
                "citation": {"doc_id": "", "snippet": "", "location": ""},
                "alternative_answers": [],
                "confidence": 0.0,
                "needs_review": True,
                "_error": str(e),
            })
        time.sleep(args.sleep)

    all_labels.sort(key=lambda x: x["id"])
    LABELS_FILE.write_text(json.dumps(all_labels, ensure_ascii=False, indent=2), encoding="utf-8")

    review_count = sum(1 for l in all_labels if l.get("needs_review"))
    print(f"\nDone. {len(all_labels)} labels saved to {LABELS_FILE}")
    print(f"  - Review needed: {review_count}")
    print(f"  - Failed: {fail_count}")


if __name__ == "__main__":
    main()
