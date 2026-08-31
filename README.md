# kaiper-knowledge

사내 지식 정보 관리 시스템 — LangGraph + Agentic RAG 기반 FastAPI 백엔드

**개발자: 김범준**

> **범위**: 사내 문서에서 사실을 찾아 출처와 함께 답하는 시스템입니다.
> 정보 검색 외의 요청은 정확도를 보장하지 않습니다.

---

## 현재 상태 — 동작 확인됨

테스트 환경에서 HTTP 실경로(브라우저와 동일)로 검증한 결과입니다.

| 항목 | 결과 |
|---|---|
| 22문항 회귀 | **22/22 정상 응답**, 출처 부착, 환각 0건 |
| 사실 정확도 (원문 대조 7문항) | **7/7** |
| 평균 응답 시간 | **16.7초** (최소 9.3 / 최대 28.3) |
| 프롬프트 내부 노출 | 0건 |

```
질의  "gitlab dap에 적용할 수 있는 모델들이 어떤 게 있나요?"
답변  - Claude 4 Sonnet [12]  - Claude Haiku 4.5 [12]
      - Claude Sonnet 4.6 [12]  - GPT-4 Turbo [12]  - GPT-oss-120B [3] ...

질의  "폐쇄망에서 GPU는 얼마나 필요한가요?"
답변  - 7B 모델(예: Mistral 7B): 1x NVIDIA A100 (40 GB), 최소 VRAM 35 GB [3]
      - Mixtral 8x22B: 8x NVIDIA A100 (80 GB), 최소 VRAM 526 GB [3]

질의  "우리 회사 연차 규정 알려줘"        (코퍼스에 없는 내용)
답변  "관련 사내 문서를 찾을 수 없습니다."   ← 환각 없음
```

---

## 이 저장소에 대하여

[`Langgraph-AI`](https://github.com/oreorca0719/Langgraph-AI) 에서 분기했습니다.
원본은 **2026-05 형상(Google Gemini + ECS Fargate)** 그대로 보존되어 있습니다.

| | Langgraph-AI (원본) | kaiper-knowledge (현재) |
|---|---|---|
| 답변 생성 | `gemini-3-flash-preview` | **`claude-sonnet-5`** |
| 임베딩 | `gemini-embedding-001` | `gemini-embedding-001` (동일) |
| 배포 | ECS Fargate + ALB + EFS | **EC2 단일 인스턴스 (t4g.medium)** |
| 이미지·표 슬라이드 | 텍스트 소실 | **VLM 전사로 복구** |
| 리랭커 | 항상 사용 | **기본 off** (측정 근거 아래) |
| 의존성 | 무핀 | 전 항목 고정 |

---

## 접속

```
http://54.64.176.30:8080/chat        admin@kaiper.local / admin1234
```

> 공인 IP는 인스턴스를 중지·재시작하면 바뀝니다 (Elastic IP 미사용).

---

## 핵심 설계 결정 — 전부 측정으로 정했습니다

### 1. 이미지 슬라이드를 VLM 으로 전사한다

PPT 를 PDF 로 내보낸 자료는 표·다이어그램이 **이미지로 박혀** 있고 `pypdf` 는
텍스트 레이어만 읽습니다. 대상 코퍼스 실측:

```
KB국민은행_GitLab_Duo_설명회자료 (64페이지)
  120자 미만 페이지  27개 / 64  (42%)
  p41 "지원 가능한 모델들"  70자(제목만) + 이미지 3개
      실제 내용: 지원 모델 12종 x 4개 기능 등급 표 + 호환 모델 13종
```

사용자가 "DAP 에 적용할 수 있는 모델"을 물었을 때 **정답 표가 인덱스에 없어**
답할 수 없었습니다. 검색·리랭킹을 아무리 고쳐도 없는 정보는 못 찾습니다.

`app/knowledge/chunking/extractors/vlm_page.py` 가 텍스트가 적고 이미지가 있는
페이지를 감지해 Claude 로 전사합니다 (인제스트 시점 1회).

```
p40   40자 -> 1,610자      p41   70자 -> 2,279자      p42   58자 -> 433자
전체 인제스트에서 31개 페이지 전사, 118 -> 145 chunks
```

`PDF_VLM_EXTRACT=0` 으로 끌 수 있습니다 (폐쇄망 등 LLM 사용 불가 환경).

### 2. 분할된 조각에 제목을 상속시킨다

긴 unit 을 자르면 2번째 이후 조각이 **제목을 잃습니다.** 임베딩도 BM25 도
그 조각을 원 주제와 연결하지 못해 검색에서 사라집니다.

```
chunk#119  152자  "-41 지원 가능한 모델들 ... ## 지원 모델"   split=1/4  제목만
chunk#120 1500자  "| Model family | Model | ..."          split=2/4  표만
```

질의 "지원 가능한 모델"이 **제목만 있는 빈 껍데기**를 찾고, 정작 답이 있는
조각은 그 문구가 없어 검색되지 않았습니다. 같은 슬라이드라도 493자로
안 쪼개진 GPU 표는 정상 검색됐습니다 — 분할이 원인임이 확인됩니다.

**원본 저장소에서 "PPT 비교 표 평탄화"로 8건 미해결로 남아 있던 결함의
실제 메커니즘이 이것입니다.**

추가로 마크다운 표는 **행 경계**에서 자르고 헤더 행을 각 조각에 반복합니다.
글자 수로 자르면 `| GPT | GPT-` 처럼 행 중간에서 끊겨 열 의미가 사라집니다.

### 3. 리랭커를 기본으로 끈다

| 구성 | 키워드 적중 | 평균 응답 |
|---|---|---|
| `FINAL_K=7` + 리랭커 replace | 10/24 (42%) | 59.9s |
| `FINAL_K=15` + off | 13/24 (54%) | 19.4s |
| `FINAL_K=15` + fuse | 12/24 (50%) | 107.1s |
| **`FINAL_K=20` + off** | **17/24 (71%)** | **21.4s** |
| `CAND=40 FINAL_K=35` + off | 16/24 (67%) | 19.5s |
| `CAND=60 FINAL_K=45` + off | 14/24 (58%) | 20.5s |

근거 넷:

1. **recall 에 영향이 없다.** `retrieve_node` 가 `FINAL_K` 로 자른 뒤 grader 에
   넘기므로, 어떤 문서가 generator 에 도달하는지는 RRF + co-retrieval + boost 가
   결정합니다. 리랭커는 그 안의 **순서만** 바꿉니다.
2. **비용이 크다.** 쌍당 4~6초 (2 vCPU ARM, `max_length=512`). 코드 주석의
   `~50ms/pair` 는 x86 기준으로 **80배** 차이납니다.
3. **이 코퍼스에서 변별력이 없다.** 20개 chunk 중 17개가 0.50~0.54 구간
   (`sigmoid(0)=0.5` = 판단 불가) 에 몰립니다. 슬라이드 평탄화 텍스트 +
   한국어 질의 + 영문 표 조합 때문입니다.
4. **메모리 2.3GB 를 회수한다.** off 이면 모델을 로드하지 않습니다.
   컨테이너 사용량 **858MiB**, 가용 2.5GB — 로컬 임베딩 모델을 올릴 여유가 생깁니다.

`RERANK_MODE=replace|fuse|off` 로 전환 가능합니다. 정제된 산문 코퍼스에서는
`replace` 가 유리할 수 있어 코드는 남겨 두고 기본값만 바꿨습니다.

### 4. `FINAL_K` 를 더 키우면 오히려 떨어진다

20이 최적점입니다. 그 이상은 관련 내용이 긴 컨텍스트에 묻히는 **context rot**
가 발생합니다 (위 표의 `FINAL_K=45` 참조).

### 5. 프롬프트에서 사고 절차 지시를 걷어낸다

`claude-sonnet-5` 가 *"답변 작성 전 4단계 자기검증 (반드시 수행)"* 지시를
**답변에 그대로 출력**했습니다:

```
**질문 entity**: GitLab DAP(인터넷 연결 가능시)에서 지원 가능한 모델 목록
**해당 모델들**:
```

`gemini-3-flash-preview` 시절 모델이 스스로 검증하지 못하던 것을 프롬프트로
보상한 장치입니다. 신형 모델은 지시 없이도 수행하며, 명시하면 내부 절차가
사용자에게 노출됩니다. 규칙을 **결과물의 성질**로만 남겼습니다.

```
_COMMON_PRINCIPLES  1,132자 -> 393자        list_n  831자 -> 378자
```

---

## LLM 프로바이더 구성

**답변 생성과 임베딩이 서로 다른 프로바이더입니다.** 제약이지 선택이 아닙니다.

```
답변 생성 (get_llm)        → Anthropic  claude-sonnet-5
임베딩   (get_embeddings)  → Google     gemini-embedding-001   ※ 교체 불가
```

Anthropic 은 임베딩 API 를 제공하지 않습니다. 벡터 인덱스 · Q&A 캐시 ·
인젝션 탐지가 모두 여기 묶여 있어 `GEMINI_API_KEY` 는 제거할 수 없습니다.

경계는 깨끗합니다 — 두 계층은 **평문 텍스트로만** 만납니다.
`LLM_PROVIDER=google` 로 되돌려 A/B 측정할 수 있습니다.

### ⚠️ `temperature` 는 전달하면 안 됩니다

```
400 invalid_request_error: `temperature` is deprecated for this model.
```

`get_llm()` 이 프로바이더별로 분기합니다. `LLM_TEMPERATURE` 는
`LLM_PROVIDER=google` 일 때만 적용됩니다.

파생 영향: `temperature=0` 이 사라져 **평가 재현성이 낮아집니다.**
회귀 측정은 단일 실행이 아니라 **N=3 반복**이 필요합니다.

### ⚠️ thinking 블록 처리

Claude 는 adaptive thinking 이 켜지면 (Sonnet 5 는 파라미터를 생략해도 adaptive)
응답에 thinking 블록을 함께 반환합니다. 텍스트에 섞이면 재검색 질의 오염,
reflection JSON 파싱 실패, doc_topic 라벨 오염이 발생합니다.

`app/core/history_utils.py` 의 `extract_text_content()` 가 타입 기준으로 걸러냅니다.
**LLM 응답은 반드시 이 함수를 경유해야 합니다.**

---

## 운영 환경 (테스트)

| 항목 | 값 |
|---|---|
| 인스턴스 | `t4g.medium` — Graviton2, 2 vCPU, 4 GB |
| AMI | Amazon Linux 2023 (arm64) |
| 스토리지 | gp3 40 GB |
| 리전 | `ap-northeast-1a` |
| 접근 | HTTP `:8080` (단일 IP) · 관리는 SSM Session Manager |
| 로드밸런서 | 없음 |
| 메모리 사용 | 컨테이너 858 MiB / 3.75 GB |

**비용**: 24/7 약 $39/월 · 평일 업무시간만(220h) 약 $14/월

Graviton 을 고른 이유: `torch` · `chromadb` · `kiwipiepy` · `tokenizers` 전부
aarch64 wheel 확인. 동급 x86(`t3.medium`) 대비 **20.6% 저렴**.

---

## 프롬프트 인젝션 방어

| 레이어 | 위치 | 상태 |
|---|---|---|
| 1차 | `security_gate` (그래프 첫 노드) | **fail-closed** |
| 2차 | `router` → `rejected` | 동작 |
| 3차 | `retrieve_node` (chunk sanitize) | **복구됨** (v1→v2 전환 시 누락돼 있었음) |
| 4차 | `main.py` (응답 검증) | 동작 |

### fail-closed

인젝션 **판정 자체가 불가능한 상태**(임베딩 프로바이더 장애)를 "인젝션 아님"과
구분합니다. LLM 과 임베딩이 분리된 뒤로는 임베딩만 단독 장애가 가능한데,
이때 통과시키면 방어가 꺼진 채 정상 응답이 나가 장애가 관측되지 않습니다.

```
security:pass                    정상 통과
security:blocked(injection)      인젝션 판정 → 차단
security:blocked(detector_down)  판정 불가 → 차단 → HTTP 503
```

### 명사구 단독으로 차단하지 않는다

`system prompt` / `instruction` / `override` 는 AI·DevOps 문서의 일상 용어입니다.
명사구만 매칭하면 정상 문서를 대량 오탐합니다 (실측: GitLab 제품 설명 2건이
차단돼 사용자 질의가 실패했습니다). **명령 동사와 결합될 때만** 차단합니다.

검증: 정상 문서 7건 오탐 0, 인젝션 13건 미탐 0.

---

## 기술 스택

- **Backend**: FastAPI · Python 3.11
- **Orchestration**: LangGraph 1.0.8
- **LLM**: Anthropic `claude-sonnet-5` (`langchain-anthropic` 1.4.1)
- **Embedding**: Google `gemini-embedding-001`
- **Vector DB**: Chroma (로컬 EBS) + BM25 (`rank-bm25`)
- **한국어 토큰화**: `kiwipiepy`
- **VLM 페이지 전사**: `pymupdf` + Claude 비전
- **Reranker**: `BAAI/bge-reranker-v2-m3` (기본 비활성)
- **Store**: DynamoDB (사용자 · 체크포인터)
- **Secrets**: SSM Parameter Store (SecureString)

`torch` 는 **CPU 전용 인덱스**에서 설치합니다. PyPI 기본 wheel 은 aarch64 에서도
`nvidia-cudnn`(444MB) 등 CUDA 런타임을 끌고 와 이미지를 부풀립니다
(8.32GB → **4.28GB**).

---

## 환경변수

```env
# LLM
LLM_PROVIDER=anthropic
ANTHROPIC_MODEL=claude-sonnet-5
ANTHROPIC_API_KEY=          # SSM /kaiper/anthropic-api-key
LLM_MAX_OUTPUT_TOKENS=4096
# LLM_TEMPERATURE 는 LLM_PROVIDER=google 일 때만 유효

# 임베딩 — 제거 불가
GEMINI_API_KEY=             # SSM /kaiper/gemini-api-key
SESSION_SECRET=             # SSM /kaiper/session-secret

# 검색 (측정으로 정한 값 — 위 '핵심 설계 결정' 참조)
RETRIEVAL_CANDIDATE_TOP_K=20
RETRIEVAL_FINAL_K=20
RETRIEVAL_MAX_REWRITES=1
RERANK_MODE=off             # replace | fuse | off

# 인제스트
AUTO_INGEST=1
PDF_VLM_EXTRACT=1           # 이미지 슬라이드 VLM 전사
PDF_VLM_MIN_TEXT_CHARS=200  # 이보다 적으면 이미지 지배 페이지로 간주
PDF_VLM_MIN_OUTPUT_CHARS=150

# 평가 무결성
QA_CACHE_ENABLED=1          # 운영 ON. 평가 러너가 자동으로 끈다
```

---

## 지원 문서 형식

`.pptx` · `.pdf` · `.docx` · `.xlsx` · `.xlsm` · `.txt` · `.md`

구형 형식(`.ppt` · `.doc` · `.xls`)은 ingest 단계에서 **조용히 걸러집니다**.
변환해서 넣으십시오.

파일명이 그대로 출처로 노출됩니다 (`doc_id` = 확장자 뗀 상대경로).

```bash
aws s3 cp ./문서폴더 s3://langgraph-rag-333347414948-ap-northeast-1/knowledge_data/ --recursive --region ap-northeast-1
```

---

## 배포

```bash
git archive --format=tar.gz -o app.tar.gz HEAD && aws s3 cp app.tar.gz s3://langgraph-rag-333347414948-ap-northeast-1/deploy/app.tar.gz --region ap-northeast-1
```

```bash
aws ssm send-command --region ap-northeast-1 --instance-ids i-0d76ba6d090ba5069 --document-name AWS-RunShellScript --parameters 'commands=["cd /opt/langgraph/app","aws s3 cp s3://langgraph-rag-333347414948-ap-northeast-1/deploy/app.tar.gz . --region ap-northeast-1","tar xzf app.tar.gz && rm app.tar.gz","docker restart kaiper"]'
```

소스는 `/opt/langgraph/app` → `/app` 으로 **볼륨 마운트**되어 있어 코드 변경에
이미지 재빌드가 필요 없습니다 (배포 7분 → 1분). 의존성이 바뀔 때만 재빌드합니다.

인스턴스 접속은 SSH 가 아니라 SSM 입니다 (22번 포트 미개방).

```bash
aws ssm start-session --region ap-northeast-1 --target i-0d76ba6d090ba5069
```

### 재인제스트

```bash
aws ssm send-command --region ap-northeast-1 --instance-ids i-0d76ba6d090ba5069 --document-name AWS-RunShellScript --parameters 'commands=["docker stop kaiper","rm -rf /opt/langgraph/chroma/*","docker start kaiper","sleep 20","docker exec kaiper rm -f /app/knowledge_data/.ingest_state.json","docker restart kaiper"]'
```

`.ingest_state.json` 을 지우지 않으면 Chroma 를 비워도 파일 해시가 같아
전부 스킵됩니다.

---

## 평가

```bash
python eval/runner.py --run-name <name> --version v2
```

### ⚠️ 측정 무결성

`eval/build_qa_cache.py` 는 **`labels.json`(정답 라벨)** 로 Q&A 캐시를 채우고,
`qa_lookup` 은 유사도 0.92 이상이면 그 정답을 그대로 반환합니다.
캐시가 켜진 채 평가하면 **정확도가 자기충족적**이 됩니다.

러너는 **기본적으로 캐시를 끕니다.** 켜려면 `--use-qa-cache` 를 명시해야 하며,
실행 전 프로바이더 · 모델 · 캐시 상태를 로그에 남깁니다.

### 기준선

원본 저장소의 **91.20%** 는 `gemini-3-flash-preview` + 다른 코퍼스 기준이며
**이 저장소에 적용되지 않습니다.** 프롬프트도 축소됐습니다.
새 코퍼스 기준의 평가셋을 별도로 구성해야 합니다.

---

## 알려진 미해결 항목

| 항목 | 내용 |
|---|---|
| **HTTPS 미적용** | HTTP 평문. 보안그룹이 단일 IP 로 막고 있음 |
| **로그인 시 자동 가입** | 미등록 이메일로 로그인하면 계정 생성 (승인 대기 상태). 외부 공유 전 반드시 차단 |
| **root 액세스 키 사용** | 최소권한 IAM 사용자로 교체 필요 |
| 응답 스트리밍 없음 | 평균 16.7초 동안 화면이 멈춘 것처럼 보임 |
| CI/CD 없음 | 수동 배포. 원본의 `deploy.yml` 은 삭제된 ECS 대상 |
| 공인 IP 고정 안 됨 | 인스턴스 재시작 시 변경. Cloudflare Tunnel 로 HTTPS 와 함께 해소 가능 |

---

## 디렉토리 구조

```
kaiper-knowledge/
├── main.py                              # FastAPI + v2 그래프 진입점
├── Dockerfile                           # CPU 전용 torch + 모델 사전 bake
│
├── app/
│   ├── core/
│   │   ├── config.py                    # ★ 프로바이더 분기 · 검색 파라미터
│   │   └── history_utils.py             # ★ extract_text_content (thinking 필터)
│   ├── graph_v2/                        # 운영 그래프
│   │   ├── builder.py                   # ★ _subgraph_node (경로 증분 반환)
│   │   ├── states/state.py              # ★ _reset_or_add reducer
│   │   ├── nodes/                       # security · router · qa_lookup
│   │   │                                #   grader(★ RERANK_MODE) · generator · reflection
│   │   ├── subgraphs/                   # retrieve · generate
│   │   └── retrievers/                  # chroma_hybrid · reranker · base
│   ├── checkpointer/
│   │   └── dynamo_checkpointer.py       # ★ checkpoint_ns 를 저장 키에 포함
│   ├── knowledge/
│   │   └── chunking/
│   │       ├── size_analyzer.py         # ★ 표 행 경계 분할 · 제목 상속
│   │       └── extractors/
│   │           ├── pdf_extractor.py     # ★ VLM 연동 · 제목 추정
│   │           └── vlm_page.py          # ★ 이미지 페이지 전사
│   └── security/
│       ├── injection_detector.py        # ★ fail-closed
│       └── content_sanitizer.py         # ★ 명령 동사 결합 시에만 차단
├── eval/                                # 평가 도구 (★ 캐시 기본 차단)
└── docs/PHASE_I_RESULTS.md              # 원본 저장소 시절 노트 (Gemini 기준)
```
