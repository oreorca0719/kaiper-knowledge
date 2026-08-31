from __future__ import annotations

import os
import re
import secrets
import threading
from pathlib import Path
from typing import Optional

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")
_NAME_RE  = re.compile(r"^[가-힣a-zA-Z]{2,20}$")

_READONLY_EMAIL = "testuser@test.co.kr"


def _is_readonly(user: dict) -> bool:
    return (user.get("email") or "").strip().lower() == _READONLY_EMAIL

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from jinja2 import TemplateNotFound
from slowapi import Limiter
from slowapi.util import get_remote_address

from .dynamo import (
    approve_user,
    create_user_if_not_exists,
    get_user_by_email,
    list_users,
    update_login_timestamp,
    set_admin,
    delete_user,
    set_department,
)
from .deps import get_current_user, require_admin_user
from .security import hash_password, verify_password

router = APIRouter()
_limiter = Limiter(key_func=get_remote_address, config_filename="__no_env__")

_templates = None
_graph_app = None


def set_graph_app(app) -> None:
    global _graph_app
    _graph_app = app


# ── CSRF 헬퍼 ────────────────────────────────────────────

def _get_csrf_token(request: Request) -> str:
    """세션에서 CSRF 토큰을 가져오거나 신규 생성합니다."""
    token = request.session.get("csrf_token")
    if not token:
        token = secrets.token_hex(32)
        request.session["csrf_token"] = token
    return token


def _verify_csrf(request: Request, form_token: str) -> None:
    """Form에서 전달된 토큰이 세션 토큰과 일치하지 않으면 403을 반환합니다."""
    session_token = request.session.get("csrf_token") or ""
    if not secrets.compare_digest(session_token, form_token or ""):
        raise HTTPException(status_code=403, detail="CSRF 토큰이 유효하지 않습니다.")


def set_templates(templates):
    global _templates
    _templates = templates


def _render(request: Request, name: str, context: dict) -> HTMLResponse:
    if _templates is None:
        return HTMLResponse("templates not set", status_code=500)
    try:
        return _templates.TemplateResponse(request, name, {"request": request, **context})
    except TemplateNotFound:
        return HTMLResponse(f"Template not found: {name}", status_code=500)


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return _render(request, "login.html", {"error": None})


@router.post("/login")
@_limiter.limit(os.getenv("LOGIN_RATE_LIMIT", "10/minute"))
def login_action(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    name: Optional[str] = Form(None),
):
    email_l = email.strip().lower()

    if not _EMAIL_RE.match(email_l):
        return _render(request, "login.html", {"error": "올바른 이메일 형식을 입력해 주세요."})

    if name and not _NAME_RE.match(name.strip()):
        return _render(request, "login.html", {"error": "이름은 한글 또는 영문 2~20자만 입력 가능합니다."})

    user = get_user_by_email(email_l)

    if user is None:
        pw_hash = hash_password(password)
        user = create_user_if_not_exists(email_l, name or "", pw_hash)
    else:
        if not verify_password(password, user.get("password_hash", "")):
            return _render(request, "login.html", {"error": "이메일 또는 비밀번호가 올바르지 않습니다."})

    update_login_timestamp(email_l)

    request.session["user"] = {
        "email": user.get("email"),
        "user_id": user.get("user_id"),
        "name": user.get("name"),
        "approved": bool(user.get("approved")),
        "is_admin": bool(user.get("is_admin")),
    }

    if not user.get("approved"):
        return RedirectResponse(url="/pending", status_code=303)

    return RedirectResponse(url="/", status_code=303)


@router.get("/pending", response_class=HTMLResponse)
def pending_page(request: Request):
    user = request.session.get("user")
    return _render(request, "pending.html", {"user": user})


@router.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


@router.get("/admin/users", response_class=HTMLResponse)
def admin_users(request: Request):
    user = get_current_user(request)
    require_admin_user(user)
    users = list_users(limit=300)
    users.sort(key=lambda x: (bool(x.get("approved", False)), x.get("email", "")))
    csrf_token = _get_csrf_token(request)
    return _render(request, "admin_users.html", {"user": user, "users": users, "csrf_token": csrf_token, "read_only": _is_readonly(user)})


@router.post("/admin/users/approve")
def admin_approve(request: Request, email: str = Form(...), csrf_token: str = Form(...)):
    user = get_current_user(request)
    require_admin_user(user)
    if _is_readonly(user):
        return RedirectResponse(url="/admin/users", status_code=303)
    _verify_csrf(request, csrf_token)
    approve_user(email=email, approved=True)
    return RedirectResponse(url="/admin/users", status_code=303)


@router.post("/admin/users/reject")
def admin_reject(request: Request, email: str = Form(...), csrf_token: str = Form(...)):
    user = get_current_user(request)
    require_admin_user(user)
    if _is_readonly(user):
        return RedirectResponse(url="/admin/users", status_code=303)
    _verify_csrf(request, csrf_token)
    if user.get("email") == email.strip().lower():
        return RedirectResponse(url="/admin/users", status_code=303)
    approve_user(email=email, approved=False)
    return RedirectResponse(url="/admin/users", status_code=303)


@router.post("/admin/users/toggle-admin")
def admin_toggle_admin(
    request: Request,
    email: str = Form(...),
    make_admin: str = Form(...),
    csrf_token: str = Form(...),
):
    user = get_current_user(request)
    require_admin_user(user)
    if _is_readonly(user):
        return RedirectResponse(url="/admin/users", status_code=303)
    _verify_csrf(request, csrf_token)
    target_email = email.strip().lower()
    if user.get("email") == target_email and make_admin.strip() != "1":
        return RedirectResponse(url="/admin/users", status_code=303)
    set_admin(target_email, is_admin=(make_admin.strip() == "1"))
    return RedirectResponse(url="/admin/users", status_code=303)


@router.post("/admin/users/set-department")
def admin_set_department(
    request: Request,
    email: str = Form(...),
    department: str = Form(...),
    csrf_token: str = Form(...),
):
    user = get_current_user(request)
    require_admin_user(user)
    if _is_readonly(user):
        return RedirectResponse(url="/admin/users", status_code=303)
    _verify_csrf(request, csrf_token)
    set_department(email=email, department=department)
    return RedirectResponse(url="/admin/users", status_code=303)


@router.post("/admin/users/delete")
def admin_delete(request: Request, email: str = Form(...), csrf_token: str = Form(...)):
    user = get_current_user(request)
    require_admin_user(user)
    if _is_readonly(user):
        return RedirectResponse(url="/admin/users", status_code=303)
    _verify_csrf(request, csrf_token)
    target_email = email.strip().lower()
    if user.get("email") == target_email:
        return RedirectResponse(url="/admin/users", status_code=303)
    delete_user(target_email)
    return RedirectResponse(url="/admin/users", status_code=303)


# ── 관리자 홈 ────────────────────────────────────────────

@router.get("/admin", response_class=HTMLResponse)
def admin_home(request: Request):
    user = get_current_user(request)
    require_admin_user(user)
    return _render(request, "admin_home.html", {"user": user, "read_only": _is_readonly(user)})


# ── 관리자 - 재인제스트 ────────────────────────────────────
# UI 카드(admin_home.html)에서 호출. 동시 호출은 lock으로 1회만 실행.
# 기본 동작: hash 비교로 변경된 파일만 재처리 (변경 없으면 skip, 신규/수정/삭제만 반영).
# force=true: .ingest_state.json 삭제 후 전체 재청킹·재임베딩 (청킹 로직 변경 시 1회성 사용).

_REINGEST_LOCK = threading.Lock()


@router.post("/admin/api/reingest")
def admin_reingest(request: Request, force: bool = False):
    user = get_current_user(request)
    require_admin_user(user)

    if _is_readonly(user):
        return JSONResponse(
            {"ok": False, "error": "읽기 전용 계정은 재인제스트를 실행할 수 없습니다."},
            status_code=403,
        )

    if not _REINGEST_LOCK.acquire(blocking=False):
        return JSONResponse(
            {"ok": False, "error": "이미 다른 재인제스트 작업이 진행 중입니다."},
            status_code=409,
        )

    try:
        from app.knowledge.ingest import auto_ingest_if_enabled

        if force:
            knowledge_dir = Path(os.getenv("KNOWLEDGE_DIR", "./knowledge_data"))
            state_file = knowledge_dir / ".ingest_state.json"
            if state_file.exists():
                try:
                    state_file.unlink()
                except OSError as e:
                    return JSONResponse(
                        {"ok": False, "error": f"ingest state 파일 삭제 실패: {e}"},
                        status_code=500,
                    )

        auto_ingest_if_enabled()

        # rewrite_node 의 doc 카탈로그 캐시 무효화 — 새 doc_summary/key_terms 반영
        try:
            from app.graph_v2.subgraphs.retrieve import _invalidate_doc_catalog
            _invalidate_doc_catalog()
            print("[ADMIN] doc 카탈로그 캐시 무효화 완료")
        except Exception as e:
            print(f"[ADMIN] doc 카탈로그 캐시 무효화 실패 (non-fatal): {e}")

        msg = "전체 강제 재인제스트가 완료되었습니다." if force else "재인제스트가 완료되었습니다. 변경된 파일만 재처리되었으며, 변경 없는 파일은 skip되었습니다."
        return JSONResponse({"ok": True, "message": msg, "force": force})

    except Exception as e:
        return JSONResponse(
            {"ok": False, "error": f"{type(e).__name__}: {e}"},
            status_code=500,
        )
    finally:
        _REINGEST_LOCK.release()

