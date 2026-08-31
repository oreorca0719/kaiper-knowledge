# kaiper-knowledge

사내 지식 정보 관리 시스템 — LangGraph + Agentic RAG 기반 FastAPI 백엔드

**개발자: 김범준**

> **범위**: 사내 문서에서 사실을 찾아 출처와 함께 답하는 시스템입니다.
> 정보 검색 외의 요청(문서 작성·번역·일반 상담 등)은 정확도를 보장하지 않습니다.

---

## 이 저장소에 대하여

[`Langgraph-AI`](https://github.com/oreorca0719/Langgraph-AI) 에서 분기했습니다.
원본은 **2026-05 시점 형상(Google Gemini + ECS Fargate)** 그대로 보존되어 있으며,
이 저장소는 **2026-08 Claude 전환 이후**를 담습니다.

| | Langgraph-AI (원본) | kaiper-knowledge (현재) |
|---|---|---|
| 답변 생성 LLM | `gemini-3-flash-preview` | **`claude-sonnet-5`** |
| 임베딩 | `gemini-embedding-001` | `gemini-embedding-001` (동일) |
| 배포 | ECS Fargate + ALB + EFS | **EC2 단일 인스턴스 (t4g.medium)** |
| Chroma 저장소 | EFS | **로컬 EBS** |
| 의존성 | 버전 무핀 | **전 항목 고정** |
| 상태 | 아카이브 (참조용) | 활성 |

원본 대비 변경의 상세 근거는 최초 커밋 메시지를 참조하십시오.

---

## ⚠️ 현재 상태 — 테스트 환경, 미완성

**이 시스템은 지금 답변을 생성할 수 없습니다.** 두 가지가 비어 있습니다.

| 항목 | 상태 | 영향 |
|---|---|---|
| **지식 베이스 원본 문서** | **없음** | 검색할 대상이 없음. S3 `knowledge_data/` 가 비어 있음 |
| **`GEMINI_API_KEY`** | **미설정** | 임베딩 불가 → 검색·인젝션 탐지 전면 중단 |

`GEMINI_API_KEY` 가 없으면 `security_gate` 가 인젝션 판정을 내리지 못하고,
**설계상 fail-closed 이므로 모든 `/chat` 요청이 503 으로 차단됩니다.**
버그가 아니라 의도된 동작입니다 ([프롬프트 인젝션 방어](#프롬프트-인젝션-방어) 참조).

인프라와 앱은 정상 기동 상태이며 `/health`, `/login` 은 응답합니다.

---

## 운영 환경 (테스트)

| 항목 | 값 |
|---|---|
| 인스턴스 | `t4g.medium` — AWS Graviton2, 2 vCPU, 4 GB |
| AMI | Amazon Linux 2023 (arm64) |
| 스토리지 | gp3 40 GB |
| 리전 / AZ | `ap-northeast-1` / `ap-northeast-1a` |
| 접근 | HTTP `:8080` (단일 IP 화이트리스트) · 관리는 SSM Session Manager |
| 로드밸런서 | **없음** (단일 사용자 테스트 환경) |
| 컨테이너 | Docker, `--restart unless-stopped` |

### 왜 t4g.medium 인가

- **Graviton(ARM)**: `torch` · `chromadb` · `kiwipiepy` · `tokenizers` 전부 aarch64 wheel 확인.
  동급 x86(`t3.medium`, $0.0544/hr) 대비 **20.6% 저렴** ($0.0432/hr, Tokyo).
- **4 GB**: 이전 Fargate 실측 최대 메모리는 3072MB 의 41%(≈1.26GB)였으나 이는 정상 운영 구간이다.
  모델 로드 순간과 인제스트 중 스파이크는 별개이고 torch 로드만으로 1GB 이상 튄다.
  2GB 는 OOM 위험이 실재하고, 8GB 는 단일 사용자에 낭비다.
- **ALB 제거**: 단일 인스턴스에서 ALB 는 순수 낭비다 (월 $18).
- **EFS → 로컬 EBS**: EFS 는 Chroma 의 sqlite/mmap 워크로드에 최악의 조합이었다.
  부팅 지연의 상당 부분이 여기서 왔다.

### 비용

| 가동 방식 | 월 비용 |
|---|---|
| 24/7 | 약 **$39** (EC2 $31.5 + gp3 40GB $3.8 + IPv4 $3.7) |
| 평일 업무시간만 (≈220h) | 약 **$14** |

> 실제 서비스 운영이 확정되면 운영 환경 전면 재검토 예정.

---

## LLM 프로바이더 구성

**답변 생성과 임베딩이 서로 다른 프로바이더입니다.** 선택이 아니라 제약입니다.

```
답변 생성 (get_llm)        → Anthropic  claude-sonnet-5
임베딩   (get_embeddings)  → Google     gemini-embedding-001   ※ 교체 불가
```

**Anthropic 은 임베딩 API 를 제공하지 않습니다.** 벡터 인덱스 · Q&A 캐시 ·
인젝션 탐지가 모두 Gemini 임베딩에 묶여 있어 `GEMINI_API_KEY` 는 제거할 수 없습니다.

경계는 깨끗합니다 — 두 계층은 **평문 텍스트로만** 만납니다. 검색이 "어떤 chunk 를
고를지" 정하고, Claude 는 이미 골라진 텍스트만 받습니다. 벡터가 LLM 에 들어가거나
LLM 출력이 벡터 공간에서 비교되는 지점은 없습니다.

`LLM_PROVIDER=google` 로 되돌릴 수 있습니다. 회귀 측정 시 동일 코드에서
프로바이더만 바꿔 A/B 를 돌려야 "모델 교체 순효과"가 분리되기 때문입니다.

### ⚠️ `temperature` 는 전달하면 안 됩니다

`claude-sonnet-5` 는 `temperature` / `top_p` / `top_k` 를 제거했습니다.

```
400 invalid_request_error: `temperature` is deprecated for this model.
```

`get_llm()` 이 프로바이더별로 분기하여 Anthropic 경로에서는 전달하지 않습니다.
`LLM_TEMPERATURE` 는 `LLM_PROVIDER=google` 일 때만 적용됩니다.

파생 영향: `temperature=0` 이 사라지면서 **평가 재현성이 낮아집니다.**
250문항 회귀 측정은 단일 실행이 아니라 **N=3 반복**이 필요합니다.

### ⚠️ thinking 블록 처리

Claude 는 adaptive thinking 이 켜지면 (Sonnet 5 는 `thinking` 파라미터를 생략해도
adaptive 로 동작) 응답 content 에 thinking 블록을 함께 반환합니다.
이것이 텍스트에 섞이면 다음이 깨집니다:

| 위치 | 증상 |
|---|---|
| `_parse_rewrite_response` | 추론 텍스트의 첫 줄이 **재검색 질의**가 됨 |
| `reflection_node` | 추론 텍스트가 앞에 붙어 **`json.loads` 실패** → 검증 무력화 |
| `doc_topic_classifier` | 추론 텍스트가 **doc_topic 라벨로 저장** → 인덱스 메타 오염 |

`app/core/history_utils.py` 의 `extract_text_content()` 가 `type` 기준으로
텍스트 블록만 추출합니다. **LLM 응답은 반드시 이 함수를 경유해야 합니다.**

---

## 기술 스택

- **Backend**: FastAPI · Python 3.11
- **AI Orchestration**: LangGraph 1.0.8 (`StateGraph`, subgraph, DynamoDB checkpointer)
- **LLM**: Anthropic `claude-sonnet-5` (`langchain-anthropic` 1.4.1)
- **Embedding**: Google `gemini-embedding-001`
- **Vector DB**: Chroma (로컬 EBS) + BM25 (`rank-bm25`, 인메모리 싱글톤)
- **한국어 토큰화**: `kiwipiepy` (BM25 형태소 분석, POS 필터링)
- **Reranker**: `BAAI/bge-reranker-v2-m3` (cross-encoder, CPU 추론)
- **Document Store**: Amazon DynamoDB (사용자 · 체크포인터)
- **Auth**: 세션 쿠키 + CSRF 토큰
- **Secrets**: AWS SSM Parameter Store (SecureString)
- **Infra**: EC2 t4g.medium + S3 (문서 원본)

### 의존성 고정

`requirements.txt` 는 **전 항목 버전 고정**입니다. 무핀 상태에서는 재빌드마다
다른 스택이 설치되어 형상 재현이 불가능했습니다.

검증 조합: `langgraph 1.0.8` / `langchain-core 1.2.31` / `langchain-anthropic 1.4.1`
/ `pydantic 2.13.4` / `torch 2.10.0`

> `torch` 는 **CPU 전용 인덱스**에서 설치합니다. PyPI 기본 wheel 은 aarch64 에서도
> `nvidia-cudnn`(444MB) · `nvidia-cublas` · `triton` 등 CUDA 런타임을 의존성으로
> 끌고 오는데, CPU 전용 인스턴스에서는 한 번도 로드되지 않으면서 이미지만 부풀립니다.

---

## 주요 기능

| 기능 | 설명 |
|---|---|
| **사내 문서 검색 (Agentic RAG)** | Hybrid retrieval (Chroma + BM25 + 한국어 형태소, top 20 후보) → cross-encoder reranker → threshold 필터 → generator → reflection 자기검증 |
| **Q&A Cache** | 평가셋 라벨 기반 별도 ChromaDB collection. ⚠️ [측정 무결성](#-측정-무결성) 주의 |
| **파일 분석 (file_chat)** | PDF · DOCX · XLSX · PPTX · TXT 첨부 텍스트 추출 후 Q&A |
| **Question type 분류** | exact_phrase / numerical / list_n / fill_blank / reasoning / comparison — type별 generator prompt 분기 |
| **Multi-hop retrieval** | 복합 질문을 sub-question 으로 분해해 multi-query 검색 |
| **Self-correction** | grader fail 시 doc-aware query rewrite (`RETRIEVAL_MAX_REWRITES` 회) |
| **프롬프트 인젝션 방어** | 4계층 (아래 참조) |

---

## 그래프 구조

```
START
  │
  ▼
[security_gate] — 임베딩 유사도 + 슬라이딩 윈도우. fail-closed
  ├─(blocked)→ [rejected] → END
  └─(pass)
  ▼
[router] — LLM 분류 (routing_decision + question_type + sub_questions)
  ├─ no_retrieval → [no_retrieval_answer] → END
  ├─ ai_guide     → [ai_guide]            → END
  ├─ file_chat    → [file_chat]           → END
  ├─ rejected     → [rejected]            → END
  └─ single / multi_hop_retrieval
      ▼
  [qa_lookup] — Q&A 캐시 조회
      ├─(bypass hit)→ END
      └─(miss / hint)
      ▼
  [retrieve_subgraph]   query_planner → retrieve → grade → (rewrite ↺)
      ▼
  [generate_subgraph]   generator → reflection
      ▼
     END
```

`replan` 분기는 폐기되었습니다 — 효용 5% 미만 대비 응답 시간 +60초 부담.
1차 시도가 빗나가면 빠른 fail 후 사용자 재질문이 더 합리적입니다.

---

## 프롬프트 인젝션 방어

| 레이어 | 위치 | 방식 | 상태 |
|---|---|---|---|
| **1차** | `security_gate` (그래프 첫 노드) | 임베딩 유사도 + 슬라이딩 윈도우 | ✅ **fail-closed** |
| **2차** | `router` → `rejected` | LLM 분류 시 범위 외 차단 | ✅ |
| **3차** | `retrieve_node` (chunk sanitize) | RAG 문서 경유 간접 인젝션 차단 | ✅ |
| **4차** | `main.py` (응답 검증) | 민감 정보 노출 방지 | ✅ |

LLM 추가 호출 없이 오케스트레이션 레벨에서 동작합니다.

> 3차 계층은 원본 저장소의 v1→v2 전환에서 누락되어 프로덕션에서 동작하지
> 않았습니다. 이 저장소에서 복구했습니다.

### fail-closed 설계

인젝션 **판정 자체가 불가능한 상태**(임베딩 프로바이더 장애)를
"인젝션 아님"과 구분합니다.

```
security:pass                    정상 통과
security:blocked(injection)      인젝션 판정 → 차단
security:blocked(detector_down)  판정 불가 → 차단 (통과 아님) → HTTP 503
```

**근거**: LLM(Anthropic)과 임베딩(Google)이 분리된 뒤로는 임베딩만 단독 장애가
가능합니다. 이때 통과시키면 인젝션 방어가 꺼진 채 Claude 가 정상 응답을
만들어내므로 **장애가 관측되지 않습니다.** 실패는 시끄러워야 합니다.

`injection_detector.check()` 는 v1 호환을 위해 fail-open 을 유지하며,
v2 는 `check_strict()` 를 사용합니다.

---

## 환경변수

`/opt/langgraph/app.env` (컨테이너) 또는 로컬 `.env`. 시크릿은 SSM Parameter Store.

```env
# LLM (답변 생성)
LLM_PROVIDER=anthropic
ANTHROPIC_MODEL=claude-sonnet-5
ANTHROPIC_API_KEY=          # SSM /kaiper/anthropic-api-key
LLM_MAX_OUTPUT_TOKENS=4096
LLM_TIMEOUT_SEC=120
# LLM_TEMPERATURE 는 LLM_PROVIDER=google 일 때만 유효

# 임베딩 (검색·보안) — 제거 불가
GEMINI_API_KEY=

# 세션
SESSION_SECRET=             # SSM /kaiper/session-secret

# AWS
AWS_REGION=ap-northeast-1
USERS_TABLE=langgraph_users
CHECKPOINT_TABLE=langgraph_checkpoints
CREATE_USERS_TABLE=0

# Chroma (로컬 EBS 마운트)
CHROMA_DB_PATH=/mnt/chroma
CHROMA_COLLECTION=my_knowledge

# S3 (문서 인제스트)
S3_KNOWLEDGE_BUCKET=langgraph-rag-333347414948-ap-northeast-1
S3_KNOWLEDGE_PREFIX=knowledge_data/
AUTO_INGEST=0               # 문서 준비 후 1

# Retrieval
RETRIEVAL_CANDIDATE_TOP_K=20   # reranker 후보 풀
RETRIEVAL_FINAL_K=7            # generator 컨텍스트
RETRIEVAL_MAX_REWRITES=1       # rewrite 재시도 한도

# Reranker
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_RELEVANT_THRESHOLD=0.5
RERANK_PARTIAL_THRESHOLD=0.3
```

---

## 배포

CI/CD 파이프라인은 아직 없습니다. 원본의 `.github/workflows/deploy.yml` 은
삭제된 ECS 환경을 대상으로 하므로 동작하지 않습니다. 현재는 수동 배포입니다.

```bash
git archive --format=tar.gz -o app.tar.gz HEAD && aws s3 cp app.tar.gz s3://langgraph-rag-333347414948-ap-northeast-1/deploy/app.tar.gz --region ap-northeast-1
```

```bash
aws ssm send-command --region ap-northeast-1 --instance-ids i-0d76ba6d090ba5069 --document-name AWS-RunShellScript --parameters 'commands=["cd /opt/langgraph/app","aws s3 cp s3://langgraph-rag-333347414948-ap-northeast-1/deploy/app.tar.gz . --region ap-northeast-1","tar xzf app.tar.gz && rm app.tar.gz","docker build -t kaiper-knowledge:latest .","bash /opt/langgraph/start.sh"]'
```

인스턴스 접속은 SSH 가 아니라 SSM 입니다 (22번 포트 미개방).

```bash
aws ssm start-session --region ap-northeast-1 --target i-0d76ba6d090ba5069
```

### 로컬 개발

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
```

```bash
pip install -r requirements.txt
```

```bash
uvicorn main:app --reload --port 8000
```

> 첫 실행 시 cross-encoder 모델(~2.3GB) 다운로드 + Chroma 인덱싱이 발생합니다.

---

## 평가

`eval/` 에서 250문항 회귀 테스트를 수행합니다.

```bash
python eval/runner.py --run-name <name> --version v2
```

```bash
python eval/judge.py --run <name>
```

### ⚠️ 측정 무결성

`eval/build_qa_cache.py` 는 **`eval/data/labels.json`(= 정답 라벨)** 로 Q&A 캐시를
채웁니다. `qa_lookup` 노드는 유사도 0.92 이상이면 그 정답을 그대로 반환하고
retrieve 를 건너뜁니다. **캐시가 채워진 상태로 평가를 돌리면 정확도가
자기충족적이 됩니다.**

`runner.py` 는 실행 전 프로바이더 · 모델 · 캐시 항목 수를 출력하고,
캐시가 비어 있지 않으면 경고합니다. 깨끗한 측정이 필요하면:

```bash
python -c "from app.knowledge.qa_cache import get_qa_cache; get_qa_cache().clear()"
```

### 기준선

원본 저장소의 **91.20%(250문항, raw judge)** 는 `gemini-3-flash-preview` 기준이며
**이 저장소에 그대로 적용되지 않습니다.** generator · reflection · judge 프롬프트가
해당 모델의 응답 습성에 맞춰 튜닝된 결과이기 때문입니다.

Claude 기준선은 **프롬프트를 변경하지 않은 상태에서** 다시 측정해야 하며,
그래야 이후 프롬프트 개선 효과와 모델 교체 효과가 분리됩니다.
`temperature` 부재로 편차가 커지므로 **N=3** 을 권장합니다.

---

## 남은 작업

| 우선순위 | 작업 | 비고 |
|---|---|---|
| 1 | 지식 베이스 원본 문서 확보 → S3 업로드 | **차단 요인** |
| 2 | `GEMINI_API_KEY` 설정 | **차단 요인** |
| 3 | 전체 재인제스트 (LLM 메타데이터 통일) | 임베딩 프로바이더가 동일하므로 벡터 재계산 불필요 |
| 4 | 250문항 재측정 (N=3) — Claude 기준선 확정 | 약 $13/회, 2시간 |
| 5 | 프롬프트 축소 · structured output(router) | 신형 모델에 과도하게 규범적인 프롬프트는 역효과 |
| 6 | 응답 스트리밍 (SSE) | 논스트리밍 28초는 실사용 불가 |
| 7 | 검색 루프를 툴 루프로 전환 | `retrieve` 서브그래프 → 모델이 도구 반복 호출 |
| 8 | IAM 사용자 발급 (현재 root 액세스 키 사용) | |
| 9 | CI/CD 재구성 | 현재 수동 배포 |

---

## 디렉토리 구조

```
kaiper-knowledge/
├── main.py                            # FastAPI 앱 + v2 그래프 진입점
├── requirements.txt                   # 전 항목 버전 고정
├── Dockerfile                         # CPU 전용 torch + 모델 사전 bake
│
├── app/
│   ├── core/
│   │   ├── config.py                  # ★ 프로바이더 분기 (get_llm / get_embeddings)
│   │   └── history_utils.py           # ★ extract_text_content (thinking 블록 필터)
│   ├── graph_v2/                      # 운영 그래프
│   │   ├── builder.py
│   │   ├── states/state.py
│   │   ├── nodes/                     # security · router · qa_lookup · grader
│   │   │                              #   generator · reflection
│   │   ├── subgraphs/                 # retrieve · generate
│   │   └── retrievers/                # chroma_hybrid · reranker · base
│   ├── graph/                         # v1 (deprecated, 회귀 비교용)
│   ├── auth/                          # 인증 · 관리자 API
│   ├── checkpointer/                  # DynamoDB LangGraph checkpointer
│   ├── knowledge/                     # ingest · qa_cache · chunking
│   └── security/                      # ★ injection_detector (fail-closed)
│                                      #   content_sanitizer · output_validator
├── eval/                              # 250문항 평가 도구
├── docs/PHASE_I_RESULTS.md            # 원본 저장소 시절 노트 (Gemini 기준 — 참조용)
├── templates/  static/
└── infrastructure/
    └── ecs-task-definition.json       # ⚠️ 삭제된 ECS 환경 대상 — 현재 미사용
```
