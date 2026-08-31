# RAG 평가 인프라

250문항 평가셋을 사용해 RAG 시스템 정확도를 측정하고 단계별 변화를 추적한다.

## 디렉토리 구조

```
eval/
├── data/
│   ├── questions.json              # 250문항 (KakaoTalk 평가셋 파싱 결과)
│   ├── category_source_map.json    # 카테고리 → 출처 doc 매핑
│   ├── labels.json                 # 정답 + citation (labeler.py 산출물)
│   └── sources/                    # 12개 출처 문서 텍스트 추출본
│       ├── *.txt
│       └── manifest.json
├── results/                        # 측정 결과 (gitignore 권장)
│   ├── <run>.json                  # runner.py 출력
│   ├── <run>__scored.json          # judge.py 출력
│   └── <run>__report.md            # report.py 출력
├── extract_sources.py              # 출처 12개 텍스트 추출 (1회)
├── parse_questions.py              # KakaoTalk → questions.json (1회)
├── labeler.py                      # 정답 자동 라벨링 (Gemini API)
├── runner.py                       # 250문항 → graph_app.invoke
├── judge.py                        # LLM-as-judge 채점
└── report.py                       # 카테고리/유형별 breakdown
```

## 사용 흐름

### 사전 준비 (1회)

1. `.env` 설정 — `GEMINI_API_KEY` 또는 `GOOGLE_API_KEY` 필요
2. ChromaDB 인제스트 완료 (앱 정상 부팅 가능 상태)
3. 출처 문서 추출 — `python eval/extract_sources.py` (이미 완료된 상태)
4. 질문 파싱 — `python eval/parse_questions.py` (이미 완료된 상태)

### 정답 라벨링

```bash
GEMINI_API_KEY=... python eval/labeler.py
# 또는 카테고리별
GEMINI_API_KEY=... python eval/labeler.py --categories A,B
# 누락분만 추가
GEMINI_API_KEY=... python eval/labeler.py --resume
```

라벨링 후 `data/labels.json`에 250개 항목 저장. `needs_review=true`인 항목은
사람 확인 후 수정 권장.

### Baseline 측정

```bash
# 250문항 전체
GEMINI_API_KEY=... python eval/runner.py --run-name baseline

# 카테고리 일부
GEMINI_API_KEY=... python eval/runner.py --run-name baseline_AB --categories A,B

# Smoke test (앞 10문항)
GEMINI_API_KEY=... python eval/runner.py --run-name smoke --limit 10
```

### 채점

```bash
GEMINI_API_KEY=... python eval/judge.py --run baseline
```

### 리포트

```bash
python eval/report.py --run baseline
# 두 run 비교
python eval/report.py --run baseline --compare lever3_prompt
```

## 단계별 측정 워크플로우

각 Lever 적용 시:

1. 코드 변경
2. `python eval/runner.py --run-name <lever_n>` — 새 run 이름으로 결과 수집
3. `python eval/judge.py --run <lever_n>` — 채점
4. `python eval/report.py --run <lever_n> --compare baseline` — 비교 리포트
5. `__report.md`에서 카테고리별 변화 + 회귀 케이스 확인
6. 정확도 목표 미달 시 다음 Lever로

## 주의

- `runner.py`는 `InMemorySaver`를 사용하여 DynamoDB 의존성을 회피한다 (평가 시 외부 인프라 영향 차단).
- 각 질문은 새 `thread_id`로 격리 실행 — 대화 컨텍스트 누적 영향 없음.
- 평가 시 시스템 fingerprint(라우팅 결정·검색 docs·재시도 횟수)도 함께 저장되므로 실패 원인 추적 가능.
- `labels.json`이 정확하지 않으면 모든 측정이 무의미 — labeler 1회 실행 후 반드시 spot-check.
