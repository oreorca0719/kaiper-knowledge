from __future__ import annotations

from passlib.context import CryptContext

# bcrypt는 72 bytes 제한 + 라이브러리 호환 이슈가 있어서 사용하지 않습니다.
# pbkdf2_sha256은 제한이 사실상 없고, 파이썬 순수 구현/호환성이 좋습니다.
_pwd = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def hash_password(password: str) -> str:
    if not isinstance(password, str) or not password:
        raise ValueError("password is required")
    return _pwd.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    if not password or not password_hash:
        return False
    try:
        return _pwd.verify(password, password_hash)
    except Exception:
        return False
