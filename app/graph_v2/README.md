# Langgraph-AI v2 Architecture

v1 (`app/graph/`)의 한계를 극복하기 위한 새 아키텍처. 원안 MD `agentic_rag_requirements_v0.1.md`의 Phase 1-5를 기반으로 설계.

## 디자인 결정 (확정)

| 영역 | 결정 |
|---|---|
| Framework | LangGraph 유지 |
| State | Pydantic v2 BaseModel + Annotated reducer |
| Retriever | Protocol 추상화 (`Retriever`) + Registry |
| 토폴로지 | Planner → Retrieve subgraph → Generate subgraph → (replan 1회) |
| Subgraph | Retrieval / Generation 모듈화 (LangGraph subgraph 패턴) |
| Security | 4계층 → 2계층 통합 (input + content) |
| 공통 | 인증·세션·체크포인터·intent_samples는 v1 그대로 사용 |

## 디렉토리 구조

```
app/graph_v2/
├── states/
│   └── state.py              # GraphState (Pydantic)
├── retrievers/
│   ├── base.py               # Retriever Protocol + Document + Registry
│   └── chroma_hybrid.py      # 기존 Chroma+BM25를 Protocol에 wrap
├── nodes/
│   ├── security.py           # 4→2 통합 security gate
│   ├── router.py             # Planner (no/single/multi 분류 + question_type)
│   ├── grader.py             # 검색 결과 라벨링 (relevant/partial/irrelevant)
│   ├── generator.py          # 답변 생성 + citation (question_type 분기)
│   └── reflection.py         # 자기검증 + 재계획
├── subgraphs/
│   ├── retrieve.py           # query_planner → retrieve → grade → (rewrite)
│   └── generate.py           # generate → reflect
└── builder.py                # main graph 조립
```

## 그래프 토폴로지

```
START
  ↓
[security_gate]
  ├─(blocked)→ [rejected] → END
  └─(pass)
  ↓
[router]
  ├─(no_retrieval)→ [no_retrieval_answer] → END
  ├─(ai_guide)→ [ai_guide] → END
  ├─(file_chat)→ [file_chat] → END
  └─(single/multi)
  ↓
[retrieve subgraph]
  └─ query_planner → retrieve → grade → (rewrite × N) → END
  ↓
[generate subgraph]
  └─ generate → reflect → END
  ↓
(verification.passed?)
  ├─(pass)→ END
  └─(fail + replan < 1)→ [router]   ← 재계획
```

## v1 대비 핵심 차이

| 항목 | v1 | v2 |
|---|---|---|
| State 결합도 | `task_args` dict 결합 | Pydantic 필드별 명시 |
| Retriever | 단일 hybrid 하드코딩 | Protocol + Registry (다중 retriever 가능) |
| 검색 후 평가 | `quality_check` (docs 유무만) | Grader (LLM 라벨링) |
| 답변 검증 | 없음 (output validator만) | Reflection (groundedness/relevance/hallucination) |
| 재계획 | 없음 | replan loop (max 1) |
| 보안 계층 | 4계층 (1개 미구현) | 2계층 통합 |
| 모듈화 | flat 13 nodes | main 8 nodes + 2 subgraphs |

## v1에서 학습한 것 (Phase A+B+C+D 시도, eval/auto_progress.md)

- ✅ Phase A clarification fix → A 카테고리 +8pp, I 카테고리 +20pp (v2 router에 반영)
- ✅ Phase A unknown fallback 강화 → 짧은 사실 질의 회복 (v2 router에 반영)
- ❌ Phase C multi-query 무조건 적용 → K 카테고리 -41pp (v2에서는 question_type 분기로 verbatim 질문엔 multi-query 끔)
- ❌ Phase B verbatim 일률 강제 → list_n -9.4pp (v2 generator에서 question_type별 분기)
- ⚠️ Phase D 자기교정 mixed → v2 reflection은 명확한 trigger만 (groundedness < 0.7 등)

## 다음 단계

- [ ] 각 node TODO 구현 (skeleton → 실제 로직)
- [ ] `/v2/chat` API 엔드포인트 (`main.py`에 추가)
- [ ] `eval/runner.py`가 v1·v2 그래프 모두 호출 가능하도록 옵션 추가
- [ ] v2 baseline 측정 → 95% 도달까지 튜닝
