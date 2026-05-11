"""JWT-аутентификация: bcrypt-хэши, выдача и проверка токенов."""

from __future__ import annotations

import time
from typing import Any, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from service.config import get_settings
from service.db import User, get_db


# bcrypt - медленный и солёный, защита от brute-force паролей
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# auto_error=False, чтобы эндпоинт без токена не получал автоматический 401 -
# у нас часть ручек публичные, проверку токена делаем сами
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(subject: str, role: str, extra: Optional[dict[str, Any]] = None) -> tuple[str, int]:
    """Выпускает HS256 JWT. Возвращает (токен, срок_жизни_в_секундах).

    Здесь важная мелочь: `iat`/`exp` должны быть unix-timestamp в UTC.
    Если взять `datetime.utcnow().timestamp()`, на машинах с любым ненулевым
    UTC-offset получится сразу истёкший токен - `utcnow()` отдаёт naive
    datetime, а `.timestamp()` интерпретирует его как локальное время.
    Поэтому берём `time.time()`, он всегда UTC.
    """
    settings = get_settings()
    expire_seconds = settings.jwt_expire_minutes * 60
    now_ts = int(time.time())
    payload: dict[str, Any] = {
        "sub": subject,
        "role": role,
        "iat": now_ts,
        "exp": now_ts + expire_seconds,
    }
    if extra:
        payload.update(extra)
    token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return token, expire_seconds


def decode_token(token: str) -> dict[str, Any]:
    """Проверяет подпись и срок жизни. Падает 401, если что-то не так."""
    settings = get_settings()
    try:
        return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Невалидный токен: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """FastAPI-зависимость. Если токена нет - None, если есть невалидный - 401."""
    if token is None:
        return None
    payload = decode_token(token)
    username = payload.get("sub")
    if not username:
        return None
    return db.query(User).filter(User.username == username).first()


def require_admin(user: Optional[User] = Depends(get_current_user)) -> User:
    """FastAPI-зависимость для admin-only эндпоинтов."""
    if user is None:
        # 401 - токена нет, надо логиниться
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Требуется аутентификация",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if user.role != "admin":
        # 403 - токен есть, но прав не хватает
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Требуются права администратора",
        )
    return user


def ensure_admin_user(db: Session) -> None:
    """Создаёт админа из настроек, если такого пользователя ещё нет.

    Вызывается на старте сервиса. Логин/пароль для прода надо переопределить
    через env (ADMIN_USERNAME / ADMIN_PASSWORD), иначе по умолчанию admin/admin.
    """
    settings = get_settings()
    existing = db.query(User).filter(User.username == settings.admin_username).first()
    if existing is not None:
        return
    admin = User(
        username=settings.admin_username,
        hashed_password=hash_password(settings.admin_password),
        role="admin",
    )
    db.add(admin)
    db.commit()
