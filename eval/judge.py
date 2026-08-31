"""
LLM-as-judge 채점 스크립트.

입력:
  - eval/data/labels.json (정답)
  - eval/results/<run_name>.json (시스템 응답)

출력: eval/results/<run_name>__scored.json
각 항목에 score 필드 추가:
{
  ...기존 필드...,
  "score": {
    "correct": bool,         # 최종 판정 (True/False)
    "score_value": float,    # 0.0~1.0 (partial credit)
    "method": "exact" | "f1_overlap" | "llm_judge",
    "reasoning": str,        # judge가 왜 이렇게 판정했는지
    "missing": [str],        # 답변에 빠진 항목 (list_n 케이스)
    "extra": [str],          # 답변에 추가된 잘못된 항목
    "ground_truth": str,     # labels.json의 answer
  }
}

채점 전략:
  1. answer_type=numerical/exact_phrase/fill_blank → 정규화 후 exact match (alternatives 포함)
  2. answer_type=list_n → 항목 단위 set 비교 + LLM judge (순서 무관)
  3. answer_type=reasoning → LLM judge

Usage:
    GEMINI_API_KEY=... python eval/judge.py --run baseline
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

EVAL_DIR = Path(__file__).parent
DATA_DIR = EVAL_DIR / "data"
RESULTS_DIR = EVAL_DIR / "results"
LABELS_FILE = DATA_DIR / "labels.json"


JUDGE_SYSTEM_PROMPT = """당신은 사내 RAG 시스템 답변의 정확성을 채점하는 평가자입니다.
실사용자 만족 기준으로 평가하시오 — 표면 표현 차이보다 의미 전달 여부가 중요합니다.

【원칙】
1. 시스템 답변이 정답의 핵심 의미를 전달하면 정답 (의역·요약·축약 모두 인정).
2. 정답이 출처 verbatim일 때, 시스템 답변이 그 표현을 정확히 포함하면 정답.
3. 시스템 답변에 정답 외 정보를 추가했더라도, 그것이 정답을 부정하지 않고 사실이면 정답.
   - 잘못된 정보(hallucination)를 추가했을 때만 부분 감점 또는 오답.
4. "정보 없음" / "확인할 수 없음" 응답은 정답이 출처에 있을 때 오답, 출처에 없을 때 정답.
5. 빈칸 채우기는 빈칸 단어가 일치해야 정답 (긴 문장 답변도 핵심 단어 포함하면 정답).

【과엄격 금지 — 다음은 정답으로 인정】
- **질문에 이미 들어간 표현의 답변 누락**: 질문이 "매칭 이후 ___의 의미는?"이면, 답변이 "매칭 이후"를 또 반복하지 않아도 정답.
  예: 질문 "Day 1 시간 배치 의도(짧은 오후 시간이지만…)는?" → 답변 "업무 흐름 완주 → AI 첫 체험" → 정답 (질문에 이미 '짧은 오후 시간이지만'이 있음).
- **부수 단어 1~2개 누락**: 의미가 같으면 정답.
  예: 정답 "IT 비전공자 중심" / 답변 "IT 비전공자" → 정답.
  예: 정답 "Day 1 커버 가능 여부" / 답변 "Day 1" (커버 가능 여부는 질문이 묻는 것) → 정답.
- **정답을 포함하면서 supportive 추가 정보**: 추가 정보가 사실이면 정답.
  예: 정답 "금융기관 + SSO + 데이터 처리 정책" / 답변 "151명 이상 + 금융기관 + SSO + 데이터 처리 정책" → 정답 (151명 이상은 supportive sub-fact).
- **비교 대상 누락 (대조 관점)**: 핵심 정답 포함 시 정답.
  예: 정답 "개별 강사보다 전담형 파트너" / 답변 "전담형 파트너" → 정답.

【과관대 금지 — 다음은 오답】
- 정답의 핵심 항목 절반 이상 누락 (list_n에서 5개 중 2개만 등).
- 정답과 의미가 다른 답변 (예: 정답 "A 방식", 답변 "B 방식").
- 출처에 없는 사실을 추가한 hallucination.
- "정보 없음"인데 정답이 출처에 명백히 있는 경우.

【출력 형식】 — JSON만
{
  "correct": true | false,
  "score_value": 0.0 ~ 1.0,
  "missing": ["빠진 핵심 항목들"],
  "extra": ["답변에 추가된 잘못된 항목들 (사실인 supportive info는 제외)"],
  "reasoning": "1~2 문장으로 판정 이유. 과엄격 금지 규칙 적용 시 명시."
}
"""


def normalize(text: str) -> str:
    """정규화: 공백 압축, 특수기호 제거, lowercase."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()
    # 일반적인 구두점·괄호 제거
    text = re.sub(r"[\[\]\(\)\{\}『』「」'\"·,.;:!?]", "", text)
    return text


def exact_or_alt_match(system_ans: str, gold_ans: str, alternatives: list[str]) -> bool:
    """정규화 후 정답·대안 중 하나와 정확히 일치 OR 정답 포함."""
    sys_norm = normalize(system_ans)
    candidates = [normalize(gold_ans)] + [normalize(a) for a in alternatives]
    for cand in candidates:
        if not cand:
            continue
        if cand == sys_norm or cand in sys_norm:
            return True
    return False


def call_judge(question: str, gold_answer: str, gold_alternatives: list[str],
               system_answer: str, model_name: str = "gemini-2.5-flash") -> dict:
    """LLM 채점 호출."""
    from app.core.config import get_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_llm(model_name)
    alts = ", ".join(f'"{a}"' for a in gold_alternatives) if gold_alternatives else "(없음)"
    user_content = (
        f"[질문]\n{question}\n\n"
        f"[정답 (gold)]\n{gold_answer}\n\n"
        f"[정답 대안 표현]\n{alts}\n\n"
        f"[시스템 답변]\n{system_answer}"
    )

    resp = llm.invoke([
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ])
    raw = resp.content
    if isinstance(raw, list):
        raw = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in raw)
    raw = str(raw).strip()
    return _tolerant_json_parse(raw)


def _tolerant_json_parse(raw: str) -> dict:
    """LLM judge 응답을 robust하게 파싱 (Phase A-3).

    1) ```json fenced block 정리
    2) json.loads 시도
    3) 실패 시: 첫 { 부터 마지막 } 까지 잘라서 재시도
    4) 실패 시: 핵심 필드 정규식 추출 (correct, score_value, reasoning)
    5) 그래도 실패 시: 예외 raise
    """
    s = raw.strip()
    # 1) fenced code block
    if s.startswith("```"):
        try:
            s = s.split("```", 2)[1]
            if s.startswith("json"):
                s = s[4:]
            s = s.strip().rstrip("`").strip()
        except Exception:
            pass

    # 2) 직접 시도
    try:
        return json.loads(s)
    except Exception:
        pass

    # 3) 첫 { ~ 마지막 } 추출
    try:
        i = s.find("{")
        j = s.rfind("}")
        if i >= 0 and j > i:
            inner = s[i:j+1]
            # 흔한 LLM 실수: 문자열 안 unescaped 줄바꿈 → 줄바꿈을 \n 이스케이프로 변환
            inner_safe = re.sub(r'(?<!\\)\n', r'\\n', inner)
            return json.loads(inner_safe)
    except Exception:
        pass

    # 4) 정규식으로 핵심 필드만 추출
    try:
        correct_m = re.search(r'"correct"\s*:\s*(true|false)', s, re.I)
        score_m = re.search(r'"score_value"\s*:\s*([0-9.]+)', s)
        reason_m = re.search(r'"reasoning"\s*:\s*"([^"]{0,500})', s)
        if correct_m or score_m:
            return {
                "correct": (correct_m.group(1).lower() == "true") if correct_m else False,
                "score_value": float(score_m.group(1)) if score_m else 0.0,
                "missing": [],
                "extra": [],
                "reasoning": (reason_m.group(1) if reason_m else "regex-extracted (parse fallback)"),
            }
    except Exception:
        pass

    # 5) 최종 실패
    raise ValueError(f"judge JSON parse failed: {raw[:200]}")


def score_one(label: dict, result: dict, model_name: str) -> dict:
    """단일 응답 채점."""
    gold_ans = label.get("answer", "")
    alts = label.get("alternative_answers", []) or []
    sys_ans = result.get("answer", "")
    answer_type = label.get("answer_type", "reasoning")

    # 시스템 에러 → 0점
    if result.get("error"):
        return {
            "correct": False,
            "score_value": 0.0,
            "method": "system_error",
            "reasoning": f"시스템 오류: {result['error']}",
            "missing": [], "extra": [],
            "ground_truth": gold_ans,
        }

    # 정답이 UNKNOWN/ERROR이면 채점 불가 (라벨 부실)
    if gold_ans in ("UNKNOWN", "ERROR", ""):
        return {
            "correct": False,
            "score_value": 0.0,
            "method": "skip_no_label",
            "reasoning": "라벨이 UNKNOWN/ERROR로 채점 불가",
            "missing": [], "extra": [],
            "ground_truth": gold_ans,
        }

    # 1차: 빠른 exact/alternative 매칭 (numerical, fill_blank, exact_phrase)
    if answer_type in ("numerical", "fill_blank", "exact_phrase"):
        if exact_or_alt_match(sys_ans, gold_ans, alts):
            return {
                "correct": True,
                "score_value": 1.0,
                "method": "exact",
                "reasoning": "exact match (or alternative)",
                "missing": [], "extra": [],
                "ground_truth": gold_ans,
            }
        # exact 실패해도 LLM judge로 fallback (의미 매칭)

    # 2차: LLM judge
    try:
        verdict = call_judge(label["question"], gold_ans, alts, sys_ans, model_name=model_name)
        return {
            "correct": bool(verdict.get("correct", False)),
            "score_value": float(verdict.get("score_value", 0.0)),
            "method": "llm_judge",
            "reasoning": verdict.get("reasoning", ""),
            "missing": verdict.get("missing", []) or [],
            "extra": verdict.get("extra", []) or [],
            "ground_truth": gold_ans,
        }
    except Exception as e:
        return {
            "correct": False,
            "score_value": 0.0,
            "method": "judge_error",
            "reasoning": f"judge 호출 실패: {e}",
            "missing": [], "extra": [],
            "ground_truth": gold_ans,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="run name (e.g. baseline)")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--sleep", type=float, default=0.3)
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        print("ERROR: GOOGLE_API_KEY/GEMINI_API_KEY 미설정")
        sys.exit(1)

    labels = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
    labels_by_id = {l["id"]: l for l in labels}

    results_path = RESULTS_DIR / f"{args.run}.json"
    if not results_path.exists():
        print(f"ERROR: {results_path} 없음. runner.py를 먼저 실행하세요.")
        sys.exit(1)
    results = json.loads(results_path.read_text(encoding="utf-8"))

    scored: list[dict] = []
    out_path = RESULTS_DIR / f"{args.run}__scored.json"
    for idx, r in enumerate(results, 1):
        label = labels_by_id.get(r["id"])
        if not label:
            print(f"[{idx}/{len(results)}] Q{r['id']} SKIP (no label)", flush=True)
            r["score"] = {"correct": False, "score_value": 0.0, "method": "no_label",
                          "reasoning": "label 없음", "missing": [], "extra": [], "ground_truth": ""}
            scored.append(r)
            continue

        score = score_one(label, r, args.model)
        r["score"] = score
        scored.append(r)
        flag = "OK" if score["correct"] else "X"
        print(f"[{idx}/{len(results)}] Q{r['id']} ({r['category']}) "
              f"{score['method']} score={score['score_value']:.2f} {flag}", flush=True)
        if score["method"] == "llm_judge":
            time.sleep(args.sleep)

        # Phase A-4: incremental save 매 10개마다 (장시간 실행 중 중단 대비)
        if idx % 10 == 0 or idx == len(results):
            out_path.write_text(json.dumps(scored, ensure_ascii=False, indent=2), encoding="utf-8")

    # 요약
    correct_n = sum(1 for r in scored if r["score"]["correct"])
    avg_score = sum(r["score"]["score_value"] for r in scored) / max(len(scored), 1)
    print(f"\nScored {len(scored)} responses → {out_path}")
    print(f"  Accuracy (binary): {correct_n}/{len(scored)} = {correct_n/len(scored)*100:.1f}%")
    print(f"  Avg score (partial): {avg_score:.3f}")


if __name__ == "__main__":
    main()
