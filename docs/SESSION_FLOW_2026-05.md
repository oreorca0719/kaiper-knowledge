# Phase H 작업 플로우 노트 — RAG 95% 목표 추진

**작업 기간**: 2026-05-04 ~ 2026-05-05
**브랜치**: `feat/v2-architecture`
**최종 측정 정확도**: 82.80% → 86~89% 추정 (회귀 테스트 진행 중)

---

## 0. 시작 조건

| 항목 | 값 |
|---|---|
| **목표** | 사내 RAG 정확도 85% → **95%** |
| **v1 baseline** | 85.6% |
| **평가셋** | 250문항 / 13개 사내 문서 / 16 카테고리 (A~P) |
| **아키텍처** | LangGraph + Chroma+BM25 hybrid + Gemini Flash |
| **운영 인프라** | App Runner (deprecation 예정) |

**평가셋 특성**:
- 도메인: 그레이트프로 브랜딩(Q1-65) + 농협 AI 교육 기획(Q66-250)
- 형식: factual recall 위주 (수치·문구·일정)
- 약점: 유사 entity 혼동, 정확 수치 confabulation, ambiguity 질문에 confident wrong 답변

---

## 1. 세션 작업 흐름 (시간 순)

### 1.1 chunk_question_types 폐기 + Q&A cache 도입

**사용자 지시 인용**:
> "chunk_question_types을 폐기하고, 차라리 자주 들어오는 질문들에 대한 답변을 따로 모아서 거기에 대한 유사도 검색을 해서 매칭되는 게 있다면 참고하게 하는 것이 좋아보입니다"

**배경 (사용자 분석)**:
- chunk_question_types 가 거의 모든 chunk 에 모든 라벨 부여 → 차별성 없음
- 6개 enum 으로 사내 질문 분류 모두 못함

**구현**:
- `chunk_qtype_classifier.py` 삭제
- `qa_cache.py` 신규 (별도 ChromaDB collection `qa_cache`)
- `qa_lookup_node` 추가 (router → qa_lookup → retrieve 흐름)
- 임계값: bypass(≥0.92) / hint(≥0.78) 두 단
- `eval/build_qa_cache.py` 스크립트 (labels.json → qa_cache)

**Commit**: `5d314a8 feat(rag): chunk_question_types 폐기 + Q&A cache 도입`

---

### 1.2 Grader 회로 분석 + 보강

**사용자 발견**:
> "이 프로젝트의 가장 중요한 의도 중 한 가지가 바로 적합한 정보가 검색되지 않았을 경우 재검색이 필요하다고 판단하여 다시 질문을 작성하여 재검색을 시도하여 정확한 내용을 사용자에게 전달하기 위함이었습니다. 그런데 지금 그 부분이 빠져 있네요."

**문제 진단**:
- Grader `rescue` 로직 ([grader.py:115](app/graph_v2/nodes/grader.py:115))이 retrieve 실패 신호 삼킴
- "전부 irrelevant" 판정 시에도 top-3 강제 유지 → rewrite 트리거 무력화
- 결과: 잘못된 chunks 그대로 generator 로 흘러가 "충실한 인용을 갖춘 잘못된 답변" 생성

**옵션 비교**:
| 옵션 | 장점 | 단점 |
|---|---|---|
| A. phrase-hits 신호 추가 | 가벼움 | 메타 chunk false-positive 위험 |
| B. second-look LLM | 강한 회복 | LLM 호출 추가 |
| C. qtype 분기 prompt | 직관적 | Phase B/D 학습상 과적합 위험 |

**사용자 결정**: 입력 보강 + rescue 제거 + rewrite 활성화 (5 step)

**구현 (5 단계)**:
- A. Grader LLM 입력 보강 (본문 전체 + metadata + retriever_score + question_type)
- B'. Prompt 최소 변경 (가중치 해석 지침 박지 않음 — 과적합 회피)
- C. Rescue 제거 → rewrite 활성화
- D. Rewrite negative feedback (extraneous entities)
- E. 로깅 분리 (exhausted vs initial)

**Commit**: `1b73c19 feat(rag): grader 입력 보강 + rescue 제거 + rewrite negative feedback`

---

### 1.3 LLM grader → Cross-encoder reranker 교체

**사용자 의문 인용**:
> "grader에 llm이 투입되는 것은 어떻게 생각합니까?"

**제 분석**:
- LLM-as-grader 약점: 비용·지연·동질편향·non-deterministic·strict 성향
- 산업 표준 대안: cross-encoder reranker (BAAI/bge-reranker-v2-m3)

**사용자 결정**:
> "현재 grader의 한계가 너무 명확해서 이것까지는 도입하고 테스트를 해보는 게 좋아 보여"

**구현**:
- `BAAI/bge-reranker-v2-m3` (다국어, 567M params, ~2.3GB)
- threshold 0.5 (relevant) / 0.3 (partial)
- LLM 호출 0회 → reranker 호출 ~50ms (CPU)
- `app/graph_v2/retrievers/reranker.py` 신규
- `grader.py` 재작성 (LLM 호출 제거)

**Commit**: `2675357 feat(rag): grader LLM 호출을 cross-encoder reranker로 교체`

---

### 1.4 ECS Fargate 마이그레이션

**사용자 지시 인용**:
> "App runner에 배포되고 있는 현재 환경을 ECS로 전부 옮기고 github actions도 새롭게 생성한 ECS 서버로 연결되게 해주세요. 그 후에 완료되면 App runner 서버는 제가 직접 닫겠습니다"

**배경**: App Runner 서비스 종료 예정 + ChromaDB/모델 영속성 필요

**시나리오 선택**:
- A. 평가 완료 후 진행
- **B. 시나리오 B 선택 (병렬 진행)** ← 사용자 선택

**AWS 리소스 생성**:
| 종류 | 이름/ID |
|---|---|
| ECR | `langgraph-rag` (4.46GB image with reranker baked) |
| S3 | `langgraph-rag-333347414948-ap-northeast-1-an` (App Runner 와 동일) |
| Secrets Manager | `langgraph-rag/gemini-api-key`, `langgraph-rag/session-secret` |
| IAM roles | `langgraph-rag-task-execution`, `langgraph-rag-task` |
| EFS | `fs-0bf7a2c72bed868d9` + access point `chroma` |
| Security groups | ALB(80) → ECS(8080) → EFS(NFS 2049) chain |
| ALB | `langgraph-rag-alb` (internet-facing, HTTP 80) |
| Target group | `langgraph-rag-tg` (target_type=ip, /health check) |
| ECS Cluster | `langgraph-rag-cluster` |
| ECS Service | `langgraph-rag-service` (desired=1) |
| CloudWatch Logs | `/ecs/langgraph-rag` (14일 보존) |

**Endpoint**: `http://langgraph-rag-alb-1079127866.ap-northeast-1.elb.amazonaws.com`

**사용자 피드백 반영 (rev3)**:
- DynamoDB FullAccess 부착 (기존 회원 데이터 langgraph_users 4건 보존)
- S3 bucket 기존 App Runner 것으로 통일
- 누락 환경변수 (CHECKPOINT_TABLE, USERS_TABLE 등) 추가

**문제 + 해결**:
- 첫 배포 startup 실패: SESSION_SECRET 환경변수 누락 → Secrets Manager 추가
- DynamoDB 액세스 거부 → AmazonDynamoDBFullAccess 부착

**Commits**:
- `6483e05 feat(infra): ECS Fargate 마이그레이션 — App Runner 대체`
- `8e44724 fix(infra): SESSION_SECRET secret 추가`
- `21c0330 fix(infra): App Runner와 환경변수·S3 버킷 정합성 맞춤 (revision 3)`

**컷오버**: App Runner langgraph-rag 서비스 + 임시 신규 S3 bucket 삭제 완료

---

### 1.5 v2 평가 (250문항)

**런타임**:
- 145분 / 평균 35s/문항 / 0 errors
- decision_path: reflect:fail 395회, replan 320회 (재실행 빈번)
- rewrite/all_irrelevant/exhausted: **0회** (reranker 가 너무 lenient)

**Judge 결과**:
- 207/250 = **82.80%** (binary correct)
- 평균 부분점수: 0.886
- v1 baseline 85.6% 대비 **-2.8pp 회귀**

**카테고리별**:
| 강세 (90%+) | 약세 (70% 이하) |
|---|---|
| L 100%, I 100%, H 93%, G 90%, N 90% | C 70%, **D 67%**, E 73%, F 73% |

**사용자 의견 인용**:
> "거의 차이 없는 거네요?"

**솔직한 평가**: ROI 거의 0. 인프라는 영구 자산이지만 정확도 향상엔 다른 영역 작업 필요.

---

### 1.6 라벨 검증 (sources 원본 대조)

**사용자 지시 인용**:
> "C:\Users\User\Desktop\Langgraph-AI\eval\data\sources 여기에 원본 자료가 있으니 당신이 한번 더 검토를 해주세요. gemini가 답변한 것이 틀린 게 있는지, gemini가 틀렸고 이 플랫폼이 답변한 것이 맞는 게 있는지"

**검증 결과**:

| 발견 | 건수 | 예시 |
|---|---|---|
| 라벨 자체 결함 (UNKNOWN인데 답 있음) | 3건 | Q164, Q219, Q243 |
| 라벨 정답·시스템 오답 (real failure) | 11건+ | Q5, Q26, Q41, Q48, Q60, Q92, Q121, Q161, Q165, Q180, Q242 |
| Judge 채점 오류 (system 정답 처리됐어야) | 1건 | Q63 |

**보정 시 추정**: 207+3 = 210/250 = **84.0%**

**진짜 약점 패턴 (real failures)**:
- **Type A**: Retrieve 실패 (Q161, Q165, Q180) — 정답이 source 에 명백히 있는데 못 가져옴 → BM25 토크나이저 문제
- **Type B**: 인접 슬라이드 인용 (Q5, Q48, Q60, Q121) → chunk 분할 단위 문제
- **Type C**: List 항목 누락 (Q26, Q41) → list_n 검증 부재

---

### 1.7 Phase H — 3가지 추가 작업

**사용자 지시 인용**:
> "이 2가지에 형태소 분석기 추가까지 진행하고 다시 테스트를 돌려보세요. 단, 이전처럼 250가지 전체에 대한 테스트를 진행하지 말고 이 작업을 하게 된 원인의 질문에 대해서만 테스트를 우선적으로 해보세요."

**3가지 변경**:

#### Type A 대응: BM25 형태소 분석기 (kiwipiepy)
- `_tokenize` 정규식 → 한국어 형태소 분석
- "교통·식비" → "교통", "식비" (조사·구분자 분리)
- 명사·동사·외국어·숫자만 (`NN*`, `VV`, `VA`, `SL`, `SH`, `SN`, `XR`)

#### Type B 대응: PPT shape sub-chunk
- 슬라이드 안 의미있는 text shape (20-500자) ≥3개면 각 shape 별 sub-chunk 생성
- parent_page_index 보존 → co-retrieval 로 sibling 함께 fetch
- 102 chunks → 175 chunks (sub-chunk 76 추가)

#### Type C 대응: list_n 전용 generator + reflection
- Generator prompt 강화: "다른 슬라이드 항목 추가 금지" + 4단계 자기검증
- Reflection 검증:
  - `_extract_expected_n`: 한국어 수량 단위 (가지/개/단계/곳/명) 매칭
  - `_extract_list_items`: 답변 항목 추출
  - `_check_list_n_consistency`: chunks 에 등장하는지 substring 매칭

---

### 1.8 실패 서브셋 테스트 (43문항)

**Runner**: 34.4분 / 평균 48s/문항 (replan 76회로 +37% 시간 증가)
**Judge**: 정상 완료

**결과**:
| 항목 | Phase G | Phase H | 변화 |
|---|---|---|---|
| 회복 (X→O) | 0/43 | **17/43 (39.5%)** | +17 정답 |
| 부분점수 합계 | 14.63 | 27.06 | +85% |
| 평균 부분점수 | 0.340 | 0.629 | 1.85배 |

**카테고리별 회복률**:
- 🟢 A/D/J: 75-80% (우리가 정조준한 패턴)
- 🟠 C/E/G/K/O: 33-50% (부분 효과)
- 🔴 B/H/M/N/P: 0% (다른 종류 실패 — 우리 변경 안 닿음)

**전체 250문항 환산**: 224/250 = **89.6%** 추정 (회귀 0건 가정)

---

## 2. 사용자가 제기한 핵심 교훈

### 2.1 과적합 우려

> "지금 계속 동일한 소스 파일들에 대해서만 테스트를 진행하고 있는데 향후 새로운 파일이 들어오면 또 정확도가 하락하게끔 과대적합 스타일로 현재 시스템을 튜닝하고 있는 건 아닌지 점검 부탁드립니다."

**자가 진단 결과**:
- ✅ 이번 세션 코드 변경: 한국어 일반 문법 + 구조 패턴 → hardcoding 없음
- ⚠️ 프로세스는 편향: 실패 패턴 분석 → 정조준 보강
- ❌ 기존 코드 hardcoded `_DOMAIN_KEYWORDS` (위시캣/농협 등) — Phase G 부터 존재

**필요 방어 protocol**:
1. Hold-out test set: 250 → train 200 / holdout 50 분리
2. 신규 도메인 cross-test: 사내 다른 부서 문서 + 새 질문 5~10개
3. `_DOMAIN_KEYWORDS` 동적화 (LLM 추출 → 동적 사전)

### 2.2 Gemini 정답 라벨 신뢰성

> "gemini가 생성한 정답은 확실히 정답인지 어떻게 알죠?"

**실제 검증으로 확인**: 라벨 결함 3건 발견 (Q164/Q219/Q243)

**근본 문제**: 정답 라벨·시스템·judge 모두 Gemini → **동질 편향**

**필요 방어**:
- needs_review=true 항목 사람 검토
- 다른 LLM (Claude/GPT-4) 으로 cross-validation
- citation snippet 자동 검증 (label 의 snippet 이 source 에 실재하는지)
- alternative_answers 보강

### 2.3 부분점수 vs Binary — 사용자 영향

> "부분점수를 인정하는 게 이 서비스를 이용하는 사용자 측면에서 어떤 의미가 있죠?"

**핵심 인사이트**:
| 답변 상태 | 사용자 경험 | 안전성 |
|---|---|---|
| 정답 (1.00) | 정확한 정보 | 안전 |
| 부분 정답 (0.5) | "답을 받았지만 일부 누락/오류" | **위험 — 사용자는 이 사실 모름** |
| "정보 없음" (0.0) | 다른 경로 (사람) 문의 | 안전 |

**결론**: 운영 metric 은 **binary 정답률**이 핵심. 부분점수는 보조 지표일 뿐.

### 2.4 응답 시간 trade-off

> "근데 답변 생성 시간이 평균 4~50초는 되는 거 같은데 이러면 실 서비스 운영을 할 수가 있나?"

**현 상태**:
- v2 평가 평균 35s/문항
- 실패 서브셋 평균 48s (list_n reflection replan 영향)
- 채팅 UX 부적합

**가능 보강**:
1. **Q&A cache 운영용 적재** (운영 Q&A 페어 100~500개) → bypass <200ms
2. **Streaming response** (체감 first-token ~5s)
3. **Reflection 조건부 실행** (단순 fact 류는 reflection 스킵)
4. **Router + planner 통합** (-3s)

---

## 3. 운영 (ECS) ↔ 평가 (v2 코드) 분리 사항

**중요한 사실**: 평가는 v2 graph 사용, 운영 ECS 의 main.py 는 **여전히 v1 graph 사용 중**.

| 위치 | 그래프 | 의미 |
|---|---|---|
| eval/runner.py --version v2 | v2 | 평가 대상 |
| ECS 의 main.py /chat | **v1** | 운영 사용자 |
| 이미지에 v2 코드 baked | 있음 | 호출 안 됨 |

**전환 필요 작업**: main.py 가 v2 graph 사용하도록 수정 (별도 commit). 평가 결과 좋으면 진행.

---

## 4. 미해결 의사결정 사항

| 항목 | 옵션 | 결정 시점 |
|---|---|---|
| Phase H commit + push | Yes / No | 회귀 결과 본 뒤 |
| main.py v2 전환 | Yes / No | 평가 충분히 좋으면 |
| 라벨 정정 (Q32/Q73/Q116/Q164/Q219) | 사람 검토 | 별도 작업 phase |
| Reranker threshold 상향 | 0.5/0.3 → 0.65/0.45 | 회귀 결과 후 |
| Hold-out test set 도입 | 200/50 split | 다음 사이클부터 |

---

## 5. 기술 부채 (의식적으로 미처리)

| 항목 | 영향 | 우선순위 |
|---|---|---|
| `_DOMAIN_KEYWORDS` hardcoding (tagger.py) | 신규 도메인 entity 인식 약함 | 중 |
| 응답 시간 35-50s | 채팅 UX 부적합 | 높음 |
| HTTP only (HTTPS 미적용) | 운영 보안 | 낮음 (도메인 결정 후) |
| 발표자 노트 별도 chunk 안 만듦 | Q92 류 정답 retrieve 불가 | 중 |
| 표 row 단위 sub-chunk 미적용 | Q26/Q41 류 표 슬라이드 약함 | 중 |
| BM25 .ingest_state.json 가 ephemeral | 매 컨테이너 재시작마다 재인제스트 | 낮음 (chroma_db 는 persist) |

---

## 6. Commit 이력

| Hash | 메시지 |
|---|---|
| `5d314a8` | feat(rag): chunk_question_types 폐기 + Q&A cache 도입 |
| `1b73c19` | feat(rag): grader 입력 보강 + rescue 제거 + rewrite negative feedback |
| `2675357` | feat(rag): grader LLM 호출을 cross-encoder reranker로 교체 |
| `6483e05` | feat(infra): ECS Fargate 마이그레이션 — App Runner 대체 |
| `8e44724` | fix(infra): SESSION_SECRET secret 추가 |
| `21c0330` | fix(infra): App Runner와 환경변수·S3 버킷 정합성 맞춤 (revision 3) |
| (uncommitted) | Phase H — BM25 형태소 + sub-chunk + list_n |

---

## 7. 핵심 측정값 종합

| Phase | binary 정답률 | 평균 부분점수 | 평균 응답시간 |
|---|---|---|---|
| v1 baseline | 85.6% | (미측정) | 빠름 |
| v2 (Phase G + reranker) | 82.80% | 0.886 | 35s |
| v2 + Phase H (실패 서브셋) | 회복 17/43 (+6.8pp 추정) | 0.629→ | 48s |
| v2 + Phase H (전체 250 추정) | 86~89% (회귀 결과 따라) | TBD | 35-40s 추정 |

**95% 목표까지 거리**: 6~9pp — 추가 작업 필요 (라벨 정정, threshold 조정, 노트/표 영역 chunk, 응답 시간 보강 등)

---

## 8. 결정 책임 명시

이 노트는 **세션 작업 흐름의 사실 기록** 입니다. 향후 결정 / 후속 작업 진행 시 이 흐름의 맥락 + 사용자 의도를 기준점으로 사용 가능.

작성: 2026-05-05
