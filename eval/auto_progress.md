# Auto Progress Log — RAG Accuracy 95% Push

**Started**: 2026-05-04 (자동 진행 모드)
**Goal**: 85.6% → 95%+
**Branch**: `feat/rag-accuracy-95`
**Operator**: Claude (자동, 사용자 부재)

## Phases Plan
- Phase A: 결함 수리 + 측정 신뢰도
- Phase B: Generation 재설계 (verbatim)
- Phase C: Indexing + Retrieval 재설계
- Phase D: 답변 검증 + 자기교정
- Final: full baseline 재측정

## Baseline (시작 시점)
- 정확도: 214/250 = **85.6%**
- 평균 score: 0.881
- 카테고리 약점: D (70.3%), I (79.5%), P (79%), O (80%), F (81.3%)
- Task type 약점: clarification (5건 모두 빈 답변 = 0%), ai_guide misroute (Q151)

---

## Phase A 완료 (2026-05-04)
- A-1: task_router clarification false-positive 수정 — `_has_file_mention` 추가
- A-2: `_unknown_fallback_route` 길이 기반 ai_guide 분기 제거
- A-3: judge.py `_tolerant_json_parse` 추가 (regex fallback)
- A-4: judge.py incremental save (매 10개)

## Phase B 완료
- answer_node 프롬프트 전면 재설계 (verbatim 강제 + 추가정보 차단 + per-fact citation + 정보없음 정확성)

## Phase C 완료
- C-1: chunk 1200 → 300, overlap 200 → 100
- C-2: ingest entity tagging (도메인 키워드 + 정규식)
- C-3: file_extractor가 이미 표 markdown + PPT 슬라이드 단위 처리 (변경 불필요)
- C-4: multi-query retrieval (LLM 변형 3개)
- C-5: LLM re-ranking (top_k=15 → top 5)
- C-6: entity-aware boost
- ChromaDB 재인제스트: 51 → **227 chunks**

## Phase D 완료
- 답변 검증 (수치/entity in chunks)
- no_info_misclaim 검출
- 자기교정 1회 재생성

## Smoke v2 결과
- 5/5 OK, 평균 25.69s/질의 (이전 14.85s에서 multi-query+rerank+verify로 증가)
- Q1-4: verbatim 정확 답변 (이전 부릿+추가설명에서 단답으로 개선)
- Q5: 02 PPT 간섭 잔존 (re-rank 부분 미흡)

## Commit
- `898370e` feat(rag): Phase A+B+C+D 완료 - 95% 정확도 push
- 16 files changed, 4428 insertions(+), 22 deletions(-)

## Baseline_v2 측정 진행 중
- 시작: ~107분 예상 (250 × 25.7s)

## Final Result (Baseline_v2 측정 완료)

**정확도: 200/250 = 80.0% (avg 0.849)** — 이전 baseline 85.6%에서 -5.6pp 회귀

### 카테고리별 (변화 큰 순)
- ❌ K: 84.0% → 42.7% (-41.3pp) — **Phase C multi-query/rerank로 잘못된 chunks**
- ❌ P: 79.0% → 48.0% (-31pp)
- ✅ I: 79.5% → 100% (+20.5pp) — **Phase A-2 fix 효과**
- ✅ D: 70.3% → 83.0% (+12.7pp) — **Phase B verbatim + Phase D 자기교정**
- ❌ B: 85.8% → 74.5% (-11.3pp)
- ✅ A: 87.0% → 95.4% (+8.4pp) — **Phase A-1 fix**

### Answer Type별
- list_n: 88% → 78.6% (-9.4pp) — **verbatim 프롬프트가 항목 누락 유발**
- exact_phrase, fill_blank, numerical: 거의 동일

### Task Type별
- knowledge_search: 90.5% → 86.4% (-4.1pp)
- unknown: 88.4% → 80.0% (-8.4pp)
- clarification: 0% → 0% (5건 → 2건, 빈도 감소)

### 결론
- **95% 미달, 오히려 회귀**.
- Phase A-1/A-2는 명백히 효과 (clarification false-positive 차단, unknown fallback 강화)
- Phase B verbatim은 양면 효과: 일부 정답 명확화, 일부 list_n 답변에서 항목 누락
- Phase C multi-query/rerank는 K 카테고리에 큰 부정 영향. 잘못된 chunk 끌어옴
- Phase D 자기교정은 mixed — 일부 회복, 일부 정답을 부정확하게 변경

### 다음 단계 권장
1. **Phase C/D 부분 롤백** — multi-query/rerank/자기교정 세분화 필요
2. **Phase A 유지** — 명확한 효과
3. **Phase B 미세 조정** — verbatim 강제를 list_n 케이스에 완화
4. **K 카테고리 별도 분석** — Genspark 자료 chunking 재검토

