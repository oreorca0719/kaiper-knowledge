from __future__ import annotations

import asyncio
import os
import re
import uuid

from dotenv import load_dotenv
load_dotenv()
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.checkpointer.dynamo_checkpointer import DynamoDBCheckpointer, ensure_checkpoints_table

from app.core.history_utils import extract_text_content
from app.core.config import (
    LLM_PROVIDER,
    has_gemini_api_key,
    has_llm_api_key,
    missing_llm_key_name,
)
from app.security.content_sanitizer import sanitize as sanitize_content
from app.security.output_validator import validate as validate_output
from app.knowledge.ingest import auto_ingest_if_enabled

# v2 그래프 — Phase G/H 작업 결과 (Q&A cache, reranker, list_n strict, rewrite negative feedback 등)
from app.graph_v2.builder import build_main_graph

# Auth (DynamoDB)
from app.auth.deps import get_current_user, require_approved_user, require_admin_user
from app.auth.dynamo import ensure_admin_user, ensure_users_table_if_enabled
from app.auth.routes import router as auth_router, set_templates, set_graph_app
from app.auth.security import hash_password
from app.auth.routing_log import ensure_routing_log_table, save_routing_log
from app.auth.intent_samples import ensure_intent_samples_table, seed_intent_samples


# =========================
# LangGraph v2 구성
# =========================
# v1 (clarification + detail_search) 기능은 v2 에서 미지원. 운영 단순화 우선.
# - clarification (모호 질문 슬롯 묻기) → 제거
# - detail_search (후속 심화 검색) → 제거
# v2 가 가진 강점: Q&A cache bypass, cross-encoder reranker, list_n strict reflection,
#                  rewrite negative feedback, reflection 기반 replan
memory = DynamoDBCheckpointer()
graph_app = build_main_graph(checkpointer=memory)
set_graph_app(graph_app)


# =========================
# 템플릿 / 정적 파일
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================
# 초기 관리자 자동 생성 (DynamoDB)
# =========================
def ensure_initial_admin() -> None:
    """환경변수로 초기 관리자 계정을 DynamoDB에 자동 생성/보정합니다."""
    admin_email = os.getenv("ADMIN_EMAIL", "").strip().lower()
    admin_password = os.getenv("ADMIN_PASSWORD", "")
    admin_name = (os.getenv("ADMIN_NAME", "Admin") or "Admin").strip()

    if not admin_email or not admin_password:
        return

    try:
        pw_hash = hash_password(admin_password)
    except Exception as e:
        print(f"[INIT_ADMIN] SKIP: invalid ADMIN_PASSWORD ({e})")
        return

    try:
        ensure_admin_user(email=admin_email, name=admin_name, password_hash=pw_hash)
        print(f"[INIT_ADMIN] OK: {admin_email}")
    except Exception as e:
        print(f"[INIT_ADMIN] SKIP: ensure_admin_user failed ({e})")


# =========================
# Lifespan (startup/shutdown)
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    _secret = os.getenv("SESSION_SECRET", "")
    if not _secret or _secret == "dev-secret-change-me":
        raise RuntimeError(
            "[SECURITY] SESSION_SECRET 환경변수가 설정되지 않았거나 기본값입니다. "
            "안전한 무작위 문자열로 설정 후 서버를 재시작하세요."
        )
    ensure_users_table_if_enabled()
    ensure_routing_log_table()
    ensure_intent_samples_table()
    ensure_checkpoints_table()
    seed_intent_samples()
    ensure_initial_admin()
    auto_ingest_if_enabled()
    yield


# =========================
# FastAPI 앱 구성
# =========================
limiter = Limiter(key_func=get_remote_address, config_filename="__no_env__")
app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET", "dev-secret-change-me"),
    max_age=int(os.getenv("SESSION_MAX_AGE", "28800")),
)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.state.templates = templates

set_templates(templates)
app.include_router(auth_router)


# =========================
# Health / Status
# =========================
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/status")
def status(request: Request):
    user = get_current_user(request)
    require_admin_user(user)
    llm_key_present = has_llm_api_key()
    # 임베딩은 프로바이더와 무관하게 항상 Gemini — 별도로 노출해야 부분 장애가 보인다.
    embed_key_present = has_gemini_api_key()
    return {
        "ok": True,
        "llm_provider": LLM_PROVIDER,
        "llm_ready": llm_key_present,
        "embedding_provider": "google",
        "embedding_ready": embed_key_present,
        "reason": None if (llm_key_present and embed_key_present) else "missing_api_key",
        "auto_ingest": os.getenv("AUTO_INGEST", "1"),
        "knowledge_dir": os.getenv("KNOWLEDGE_DIR", "./knowledge_data"),
        "chroma_db_path": os.getenv("CHROMA_DB_PATH", "./chroma_db"),
        "chroma_collection": os.getenv("CHROMA_COLLECTION", "my_knowledge"),
        "users_table": os.getenv("USERS_TABLE", "langgraph_users"),
        "aws_region": os.getenv("AWS_REGION", "ap-northeast-1"),
        "thread_context_scope": os.getenv("THREAD_CONTEXT_SCOPE", "user"),
    }


def _ensure_llm_ready_or_503():
    """활성 LLM 프로바이더 키 + 임베딩 프로바이더 키를 모두 확인.

    임베딩(Gemini)은 LLM 프로바이더가 Anthropic 으로 바뀌어도 여전히 필요하다.
    검색·Q&A 캐시·인젝션 탐지가 모두 여기에 묶여 있으므로 별도로 검사한다.
    """
    if not has_llm_api_key():
        raise HTTPException(
            status_code=503,
            detail=(
                f"LLM API 키가 설정되지 않아 요청을 처리할 수 없습니다. "
                f"{missing_llm_key_name()} 를 설정해 주세요. (provider={LLM_PROVIDER})"
            ),
        )
    if not has_gemini_api_key():
        raise HTTPException(
            status_code=503,
            detail=(
                "임베딩 API 키가 설정되지 않아 요청을 처리할 수 없습니다. "
                "검색·보안 검사가 Gemini 임베딩에 의존하므로 "
                "GOOGLE_API_KEY 또는 GEMINI_API_KEY 가 필요합니다."
            ),
        )


# =========================
# Home Page
# =========================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    try:
        user = get_current_user(request)
        require_approved_user(user)
    except HTTPException as e:
        if e.status_code == 401:
            return RedirectResponse(url="/login", status_code=303)
        if e.status_code == 403:
            return RedirectResponse(url="/pending", status_code=303)
        raise

    from app.auth.dynamo import get_user_by_email
    from app.auth.routing_log import scan_user_recent_logs
    from datetime import datetime

    full_user = get_user_by_email(user.get("email", "")) or user
    user_id = str(full_user.get("user_id") or full_user.get("email", ""))
    recent_logs = scan_user_recent_logs(user_id, limit=5)
    for log in recent_logs:
        ts = log.get("timestamp", 0)
        log["ts_str"] = datetime.fromtimestamp(ts).strftime("%m-%d %H:%M") if ts else ""

    last_login = full_user.get("updated_at")
    last_login_str = datetime.fromtimestamp(float(last_login)).strftime("%Y-%m-%d %H:%M") if last_login else "-"

    return templates.TemplateResponse(request, "home.html", {
        "request": request,
        "user": full_user,
        "recent_logs": recent_logs,
        "last_login_str": last_login_str,
    })


# =========================
# Chat Page
# =========================
@app.get("/chat", response_class=HTMLResponse)
async def index(request: Request):
    try:
        user = get_current_user(request)
        require_approved_user(user)
    except HTTPException as e:
        if e.status_code == 401:
            return RedirectResponse(url="/login", status_code=303)
        if e.status_code == 403:
            return RedirectResponse(url="/pending", status_code=303)
        raise

    return templates.TemplateResponse(request, "index.html", {"request": request, "user": user})


# =========================
# Chat Reset Endpoint
# =========================
@app.post("/chat/reset")
async def chat_reset(request: Request):
    user = get_current_user(request)
    thread_id = str(user.get("user_id") or user.get("email"))
    memory.delete(thread_id)
    return {"ok": True}


# =========================
# Chat Endpoint (interrupt 패턴)
# =========================
@app.post("/chat")
async def chat_endpoint(request: Request):
    user = get_current_user(request)
    require_approved_user(user)

    data = await request.json()
    user_input = data.get("message", "")

    _max_input = int(os.getenv("CHAT_MAX_INPUT_CHARS", "4000"))
    if len(user_input) > _max_input:
        raise HTTPException(status_code=400, detail=f"입력이 너무 깁니다. 최대 {_max_input}자까지 허용됩니다.")

    _ensure_llm_ready_or_503()

    trace_id  = str(uuid.uuid4())
    thread_id = str(user.get("user_id") or user.get("email"))
    config    = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 25,
    }

    # ── v2 그래프 실행 ────────────────────────────────────────
    # LangGraph 체크포인터(thread_id=user_id)는 이전 턴의 state 전체를
    # DynamoDB 에 영속화한다. 출력·처리 관련 필드는 매 턴 새로 작성되어야
    # 하므로 invoke 시점에 명시적으로 빈 값으로 리셋. (잔존값 누출로 인한
    # 오답·잘못된 라우팅 방지)
    #
    # 리셋 제외 항목:
    #   - decision_path : Annotated[..., add] → reducer 가 append 만 수행하므로
    #                     [] 입력은 no-op. 누적 허용 (route_after_qa_lookup 은
    #                     [-1] 만 검사하므로 정상 동작).
    #   - messages      : add_messages reducer → 의도적 누적 (chat 히스토리)
    #   - original_input: router_node 가 매 턴 input_data 로 초기화
    #   - file_context*: 파일 업로드 세션 상태 — 유지
    inputs = {
        "trace_id":             trace_id,
        "input_data":           user_input,
        "input_embedding":      None,
        "answer":               "",
        "citations":            [],
        "verification":         None,
        "routing_decision":     "",
        "question_type":        "reasoning",
        "security_blocked":     False,
        "security_reason":      "",
        "sub_questions":        [],
        "retrieved_docs":       [],
        "llm_call_count":       0,
        "retrieval_iterations": 0,
        "replan_iterations":    0,
    }
    # 동기 invoke 를 스레드로 내보낸다.
    # graph_app.invoke 는 CPU/네트워크 블로킹 호출이 30초 가까이 이어지는데,
    # async 엔드포인트에서 직접 호출하면 그동안 이벤트 루프 전체가 멈춘다.
    # 단일 태스크 운영이므로 동시 사용자 2명이면 두 번째는 2배 대기하고,
    # /health 응답까지 막혀 ALB health check 실패 → 불필요한 재기동을 유발한다.
    result = await asyncio.to_thread(graph_app.invoke, inputs, config=config)

    # ── 라우팅 로그 저장 (v2 의 routing_decision 사용) ────────
    save_routing_log(
        user_id=str(user.get("user_id") or user.get("email")),
        input_text=user_input,
        final_task=(result.get("routing_decision") or "unknown"),
        routing_debug={"decision_path": result.get("decision_path", [])},
    )

    # ── 응답 포맷팅 ───────────────────────────────────────────
    routing_decision = (result.get("routing_decision") or "").strip()
    security_blocked = result.get("security_blocked", False)

    # 보안 검사 자체가 불가능했던 경우 → 200 이 아니라 503.
    # 200 으로 내리면 사용자는 질문을 바꿔 재시도하고, 운영자는 임베딩 프로바이더
    # 장애를 인지하지 못한다. 실패는 시끄러워야 한다.
    if result.get("security_reason") == "detector_unavailable":
        raise HTTPException(
            status_code=503,
            detail=(
                "보안 검사 모듈을 일시적으로 사용할 수 없어 요청을 처리하지 않았습니다. "
                "잠시 후 다시 시도해 주세요."
            ),
        )

    # 보안 차단 응답
    if security_blocked or routing_decision == "rejected":
        return {
            "type":    "chat",
            "answer":  result.get("answer") or "해당 질문은 사내 AI 어시스턴트의 지원 범위에 포함되지 않아 답변을 제공하지 않습니다. 사내 업무 관련 질문을 입력해 주세요.",
            "sources": [],
        }

    # v2 의 answer 필드 직접 사용 (generator/qa_lookup_node 가 채움)
    final_text = result.get("answer") or ""
    if not final_text:
        # fallback — messages 의 마지막 AIMessage
        msgs = result.get("messages") or []
        if msgs:
            raw_answer = msgs[-1].content
            if isinstance(raw_answer, list) and len(raw_answer) > 0:
                final_text = raw_answer[0].get("text", "")
            else:
                final_text = str(raw_answer)

    _, final_text = validate_output(final_text)

    if not final_text.strip():
        final_text = "응답을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요."

    def _extract_cited_ids(text: str) -> set[int]:
        return {int(x) for x in re.findall(r"\[(\d{1,3})\]", text or "")}

    cited_ids = _extract_cited_ids(final_text)
    # v2 의 citations 필드 (Citation Pydantic objects)
    raw_citations = result.get("citations") or []
    citations_dicts = []
    for c in raw_citations:
        if hasattr(c, "model_dump"):
            citations_dicts.append(c.model_dump())
        elif hasattr(c, "dict"):
            citations_dicts.append(c.dict())
        elif isinstance(c, dict):
            citations_dicts.append(c)
    filtered_sources = [
        s for s in citations_dicts
        if isinstance(s.get("id"), int) and s.get("id") in cited_ids
    ]

    return {
        "type":    "chat",
        "answer":  final_text,
        "sources": filtered_sources or [],
    }


# =========================
# 파일 업로드 / 파일 QA
# =========================
_UPLOAD_ALLOWED_SUFFIXES = {".txt", ".md", ".pdf", ".docx", ".xlsx", ".xlsm", ".pptx"}
_UPLOAD_MAX_BYTES = 20 * 1024 * 1024  # 20MB
_FILE_CONTEXT_MAX_CHARS = int(os.getenv("FILE_CONTEXT_MAX_CHARS", "8000"))

_MAGIC_SIGNATURES: dict[str, list[tuple[int, bytes]]] = {
    ".pdf":  [(0, b"%PDF")],
    ".docx": [(0, b"PK\x03\x04")],
    ".xlsx": [(0, b"PK\x03\x04")],
    ".xlsm": [(0, b"PK\x03\x04")],
    ".pptx": [(0, b"PK\x03\x04")],
    ".txt":  [],
    ".md":   [],
}


def _verify_mime(content: bytes, suffix: str) -> bool:
    sigs = _MAGIC_SIGNATURES.get(suffix, [])
    if not sigs:
        return True
    return any(content[off: off + len(magic)] == magic for off, magic in sigs)


def _get_chat_thread_config(user: dict) -> dict:
    base_thread = str(user.get("user_id") or user.get("email"))
    scope = (os.getenv("THREAD_CONTEXT_SCOPE", "user") or "user").strip().lower()
    thread_id = f"{base_thread}:chat" if scope == "task" else base_thread
    return {"configurable": {"thread_id": thread_id}}


def _extract_file_to_text(content: bytes, filename: str) -> tuple[str, dict]:
    import tempfile
    from pathlib import Path
    from app.graph.nodes.file_extractor import extract_text_from_file  # v1 file extractor 재사용 (포맷 추출 로직만 의존)

    suffix = os.path.splitext(filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        text, meta = extract_text_from_file(Path(tmp_path))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    return text, meta


@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    """파일만 첨부 시: 텍스트 추출 후 LLM 요약 반환."""
    user = get_current_user(request)
    require_approved_user(user)

    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in _UPLOAD_ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 파일 형식: {suffix}")

    content = await file.read()
    if len(content) > _UPLOAD_MAX_BYTES:
        raise HTTPException(status_code=400, detail="파일 크기는 20MB 이하여야 합니다.")

    if not _verify_mime(content, suffix):
        raise HTTPException(status_code=400, detail="파일 내용이 확장자와 일치하지 않습니다.")

    try:
        text, meta = _extract_file_to_text(content, file.filename or "file")
        meta["name"] = file.filename
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"파일 추출 실패: {e}")

    text = sanitize_content(text, source=f"upload:{file.filename}")

    summary = None
    keywords: list[str] = []
    if has_llm_api_key() and text.strip():
        try:
            import json as _json
            from app.core.config import get_llm
            from langchain_core.messages import HumanMessage, SystemMessage

            context = text.strip()[:_FILE_CONTEXT_MAX_CHARS]
            llm = get_llm()
            resp = llm.invoke([
                SystemMessage(content=(
                    "당신은 문서 분석 전문가입니다.\n"
                    "아래 문서를 분석하여 반드시 다음 JSON 형식으로만 응답하세요. "
                    "JSON 외 다른 텍스트는 절대 출력하지 마세요.\n\n"
                    "{\n"
                    '  "summary": "**문서 유형**: ...\\n**핵심 내용**:\\n• ...\\n**주요 수치/일정**: ...\\n**특이사항**: ...",\n'
                    '  "keywords": ["키워드1", "키워드2", ...]\n'
                    "}\n\n"
                    "keywords: 이 문서의 내용을 내부 문서 DB에서 유사 문서를 찾는 데 활용할 핵심 개념·용어 10개 이내."
                )),
                HumanMessage(content=context),
            ])
            raw = extract_text_content(resp.content)

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed = _json.loads(raw)
            summary = str(parsed.get("summary") or "").strip() or None
            keywords = [str(k) for k in (parsed.get("keywords") or []) if k]
        except Exception as e:
            print(f"[UPLOAD] LLM 분석 실패: {type(e).__name__}: {e}")

    try:
        config = _get_chat_thread_config(user)
        graph_app.update_state(config, {
            "file_context": text.strip()[:_FILE_CONTEXT_MAX_CHARS],
            "file_context_name": file.filename,
        })
    except Exception as e:
        print(f"[UPLOAD] file_context State 저장 실패 (non-fatal): {e}")

    return JSONResponse({"ok": True, "summary": summary, "text": text[:20000], "meta": meta})


@app.post("/chat-with-file")
async def chat_with_file(
    request: Request,
    file: UploadFile = File(...),
    message: str = Form(...),
):
    """파일 + 질문: 파일 내용을 컨텍스트로 LLM 답변 반환."""
    user = get_current_user(request)
    require_approved_user(user)
    _ensure_llm_ready_or_503()

    if not message.strip():
        raise HTTPException(status_code=400, detail="질문을 입력해 주세요.")

    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in _UPLOAD_ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 파일 형식: {suffix}")

    content = await file.read()
    if len(content) > _UPLOAD_MAX_BYTES:
        raise HTTPException(status_code=400, detail="파일 크기는 20MB 이하여야 합니다.")

    if not _verify_mime(content, suffix):
        raise HTTPException(status_code=400, detail="파일 내용이 확장자와 일치하지 않습니다.")

    try:
        raw_text, _ = _extract_file_to_text(content, file.filename or "file")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"파일 추출 실패: {e}")

    from app.core.config import get_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    raw_text = sanitize_content(raw_text, source=f"chat-with-file:{file.filename}")

    file_context = raw_text.strip()
    truncated = len(file_context) > _FILE_CONTEXT_MAX_CHARS
    if truncated:
        file_context = file_context[:_FILE_CONTEXT_MAX_CHARS]

    system_prompt = (
        "당신은 사용자가 첨부한 파일 내용을 분석하는 AI 어시스턴트입니다.\n"
        "아래 [파일 내용]만을 근거로 사용자의 질문에 답하세요.\n"
        "파일에 없는 내용은 '파일에서 확인할 수 없습니다'라고 답하세요.\n"
        "정중한 비즈니스 어투로 답하고, 필요 시 불렛(•)을 활용하세요.\n\n"
        f"[파일 내용]\n{file_context}"
    )

    try:
        llm = get_llm()
        resp = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=message.strip()),
        ])
        answer = extract_text_content(resp.content)
        if truncated:
            answer += f"\n\n※ 파일이 길어 앞부분 {_FILE_CONTEXT_MAX_CHARS}자만 참조했습니다."
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 호출 실패: {e}")

    try:
        config = _get_chat_thread_config(user)
        graph_app.update_state(config, {
            "file_context": raw_text.strip()[:_FILE_CONTEXT_MAX_CHARS],
            "file_context_name": file.filename,
        })
    except Exception as e:
        print(f"[CHAT-WITH-FILE] file_context State 저장 실패 (non-fatal): {e}")

    return JSONResponse({"type": "file_qa", "answer": answer, "sources": []})


@app.get("/favicon.ico")
def favicon():
    return HTMLResponse(status_code=204)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
