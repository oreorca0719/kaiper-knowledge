"""
Kaiper AI — 중앙 설정 모듈

운영(App Runner)에서는 환경변수로 값을 주입합니다.
여기서 정의된 기본값은 로컬 개발 환경용 fallback입니다.
"""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


# ──────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────
#
# 프로바이더 분리 구조 (2026-08):
#   - 답변 생성 계층 (get_llm)      → Anthropic Claude
#   - 검색/보안 임베딩 (get_embeddings) → Google Gemini  ※ 교체 불가
#
# Anthropic 은 임베딩 API 를 제공하지 않는다. 벡터 인덱스·Q&A 캐시·인젝션
# 탐지가 모두 Gemini 임베딩에 묶여 있으므로 GEMINI_API_KEY 는 계속 필요하다.
#
# LLM_PROVIDER 로 되돌릴 수 있게 둔 이유: 250문항 회귀 측정 시 동일 코드에서
# 프로바이더만 바꿔 A/B 를 돌려야 "모델 교체 순효과"가 분리되기 때문.

LLM_PROVIDER = (os.getenv("LLM_PROVIDER", "anthropic") or "anthropic").strip().lower()

_DEFAULT_MODEL_BY_PROVIDER = {
    "anthropic": "claude-sonnet-5",
    "google": "gemini-3-flash-preview",
}


def has_gemini_api_key() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))


def has_anthropic_api_key() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def has_llm_api_key() -> bool:
    """현재 활성 프로바이더의 키가 있는지."""
    if LLM_PROVIDER == "anthropic":
        return has_anthropic_api_key()
    return has_gemini_api_key()


def missing_llm_key_name() -> str:
    return "ANTHROPIC_API_KEY" if LLM_PROVIDER == "anthropic" else "GOOGLE_API_KEY / GEMINI_API_KEY"


def _ensure_google_api_key_env() -> None:
    if os.getenv("GOOGLE_API_KEY"):
        return
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key


LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "4096"))


def _resolve_model(provider: str, expected_prefix: str) -> str:
    """활성 프로바이더에 맞는 모델 ID 결정.

    `LLM_MODEL` 은 프로바이더 중립적인 이름이라 전환 과도기에 오설정이 쉽다.
    (예: LLM_PROVIDER=anthropic 인데 ECS task def 에는 LLM_MODEL=gemini-3-flash-preview
     가 남아 있는 상태 → 전 요청 404)

    따라서 프로바이더 전용 변수를 우선하고, LLM_MODEL 은 접두사가 맞을 때만 채택한다.
    맞지 않으면 무시하고 기본값을 쓰되 경고를 남긴다 (조용한 오설정 방지).
    """
    explicit = os.getenv("ANTHROPIC_MODEL" if provider == "anthropic" else "GOOGLE_MODEL")
    if explicit and explicit.strip():
        return explicit.strip()

    generic = (os.getenv("LLM_MODEL") or "").strip()
    default = _DEFAULT_MODEL_BY_PROVIDER[provider]
    if not generic:
        return default
    if generic.startswith(expected_prefix):
        return generic
    print(
        f"[CONFIG] LLM_MODEL={generic!r} 은 provider={provider} 와 맞지 않아 무시합니다. "
        f"기본값 {default!r} 사용. (의도한 값이면 "
        f"{'ANTHROPIC_MODEL' if provider == 'anthropic' else 'GOOGLE_MODEL'} 로 지정하세요)"
    )
    return default


def get_llm(model_name: str | None = None) -> BaseChatModel:
    """활성 프로바이더의 chat 모델을 반환한다.

    Anthropic 경로에서 temperature 를 넘기지 않는 것은 선택이 아니라 필수다.
    claude-sonnet-5 는 temperature / top_p / top_k 를 제거했고, 넘기면 400 이다:
        400 invalid_request_error: `temperature` is deprecated for this model.
    사고 깊이는 temperature 가 아니라 effort / adaptive thinking 으로 제어한다
    (Sonnet 5 는 thinking 을 생략하면 adaptive 로 동작).
    """
    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model = model_name or _resolve_model("anthropic", "claude")
        return ChatAnthropic(
            model=model,
            max_tokens=LLM_MAX_OUTPUT_TOKENS,
            timeout=float(os.getenv("LLM_TIMEOUT_SEC", "120")),
        )

    # --- google (레거시 / A-B 비교용) ---
    _ensure_google_api_key_env()
    model = model_name or _resolve_model("google", "gemini")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
    )


# ──────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────

class GeminiRAGEmbeddings(GoogleGenerativeAIEmbeddings):
    """LangChain Google GenAI 임베딩 래퍼 (RAG용 task_type 분기)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.task_type = "retrieval_document"
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        self.task_type = "retrieval_query"
        return super().embed_query(text)


def get_embeddings() -> GeminiRAGEmbeddings:
    _ensure_google_api_key_env()
    return GeminiRAGEmbeddings(model="gemini-embedding-001")


# ──────────────────────────────────────────────
# Chroma / RAG
# ──────────────────────────────────────────────

CHROMA_DB_PATH    = os.getenv("CHROMA_DB_PATH",    "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "my_knowledge")

# v1 전용 (v2 미사용 — 운영 그래프는 reranker threshold 로 필터링한다)
RETRIEVAL_MIN_RELEVANCE = float(os.getenv("RETRIEVAL_MIN_RELEVANCE", "0.3"))
RETRIEVAL_MAX_DISTANCE  = float(os.getenv("RETRIEVAL_MAX_DISTANCE",  "0.75"))
RETRIEVAL_TOP_K         = int(os.getenv("RETRIEVAL_TOP_K", "5"))

# v2 운영 파라미터.
# 이전에는 retrieve.py 에 20 이 하드코딩되어 있어 task definition 의
# RETRIEVAL_TOP_K=10 이 아무 효과가 없었다 (형상 관리와 실제 동작 불일치).
#
# 【FINAL_K = 20 의 근거 — 실측】
# 원래 7 이었다. 컨텍스트가 비싸고 짧던 시절의 값이며, 답이 여러 chunk 에
# 걸치는 질문(표가 분할된 경우 등)에서 일부만 도달해 답변이 불완전했다.
# claude-sonnet-5 는 1M 컨텍스트라 20개를 넣어도 부담이 없다.
#
#   질문 5개 x 키워드 적중 / 평균 응답시간
#     CAND=20 FINAL=7  리랭커 on   10/24 (42%)   59.9s   <- 이전 기본값
#     CAND=20 FINAL=15 리랭커 off  13/24 (54%)   19.4s
#     CAND=20 FINAL=20 리랭커 off  17/24 (71%)   21.4s   <- 채택
#     CAND=40 FINAL=25 리랭커 off  15/24 (62%)   21.8s
#     CAND=40 FINAL=35 리랭커 off  16/24 (67%)   19.5s
#     CAND=60 FINAL=45 리랭커 off  14/24 (58%)   20.5s
#
# 더 넣으면 오히려 떨어진다 — 관련 내용이 긴 컨텍스트에 묻히는 context rot.
# 후보 풀과 최종 컷을 같은 값으로 두어 절단 없이 전달한다.
RETRIEVAL_CANDIDATE_TOP_K = int(os.getenv("RETRIEVAL_CANDIDATE_TOP_K", "20"))
RETRIEVAL_FINAL_K         = int(os.getenv("RETRIEVAL_FINAL_K", "20"))

# rewrite 재시도 한도. 이전에는 grader(3) / subgraph(1) / generator 로그 라벨(3) 로
# 흩어져 있어 실제 한도는 1 인데 grader 는 3 을 기준으로 라벨을 만들었다.
# 그 결과 "grade:exhausted_after_3rewrites" 는 영원히 찍히지 않았고,
# 이를 찾는 generator 분기도 도달 불가였다 (관측성 거짓).
RETRIEVAL_MAX_REWRITES = int(os.getenv("RETRIEVAL_MAX_REWRITES", "1"))


# ──────────────────────────────────────────────
# Ingest
# ──────────────────────────────────────────────

KNOWLEDGE_DIR        = os.getenv("KNOWLEDGE_DIR",        "./knowledge_data")
AUTO_INGEST          = os.getenv("AUTO_INGEST",          "1")
S3_KNOWLEDGE_BUCKET  = os.getenv("S3_KNOWLEDGE_BUCKET",  "")
S3_KNOWLEDGE_PREFIX  = os.getenv("S3_KNOWLEDGE_PREFIX",  "knowledge_data/")
INGEST_CHUNK_MAX_CHARS = int(os.getenv("INGEST_CHUNK_MAX_CHARS", "1200"))
INGEST_CHUNK_OVERLAP   = int(os.getenv("INGEST_CHUNK_OVERLAP",   "200"))


# ──────────────────────────────────────────────
# Semantic Router
# ──────────────────────────────────────────────

ROUTER_TOP1_MIN  = float(os.getenv("ROUTER_TOP1_MIN",  "0.62"))
ROUTER_MARGIN_MIN = float(os.getenv("ROUTER_MARGIN_MIN", "0.08"))


# ──────────────────────────────────────────────
# LLM Intent Fallback
# ──────────────────────────────────────────────

LLM_FALLBACK_ENABLED     = os.getenv("LLM_FALLBACK_ENABLED", "1")
LLM_FALLBACK_TIMEOUT_SEC = float(os.getenv("LLM_FALLBACK_TIMEOUT_SEC", "8"))
LLM_FALLBACK_CONFIDENCE_MIN = float(os.getenv("LLM_FALLBACK_CONFIDENCE_MIN", "0.5"))
