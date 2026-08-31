from __future__ import annotations

from fastapi import HTTPException, Request


def get_current_user(request: Request):
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="not_authenticated")
    return user


def require_approved_user(user: dict):
    if not user.get("approved"):
        raise HTTPException(status_code=403, detail="not_approved")
    return user


def require_admin_user(user: dict):
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="not_admin")
    return user
