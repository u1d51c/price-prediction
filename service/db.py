"""ORM-модели сервиса и фабрики сессий SQLAlchemy."""

from __future__ import annotations

import datetime as dt
from typing import Iterator

from sqlalchemy import (
    Column, DateTime, Float, Integer, JSON, String, Text, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from service.config import get_settings


class Base(DeclarativeBase):
    """Базовый класс для всех ORM-моделей. Его metadata используется в Alembic."""


class RequestLog(Base):
    """Лог запросов к /forward и /forward/live.

    Сюда пишем каждый вызов: что пришло, что вернули, сколько обрабатывали,
    кто пользователь. По этой таблице потом считается /stats и /history.
    """

    __tablename__ = "request_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow, index=True, nullable=False)
    endpoint = Column(String(64), index=True, nullable=False)
    asset = Column(String(16), nullable=True)
    # тело запроса и ответ как JSON - SQLite и Postgres оба это поддерживают
    request_payload = Column(JSON, nullable=True)
    response_payload = Column(JSON, nullable=True)
    status_code = Column(Integer, nullable=False)
    # длина сериализованного JSON-входа в символах - для статистики
    input_length = Column(Integer, nullable=True)
    # длительность обработки на стороне сервера (без сетевых задержек)
    processing_ms = Column(Float, nullable=True)
    user = Column(String(64), nullable=True, index=True)
    error_text = Column(Text, nullable=True)


class User(Base):
    """Учётка для JWT-аутентификации.

    Хранится bcrypt-хэш пароля. Роль admin нужна для DELETE /history.
    """

    __tablename__ = "user"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    hashed_password = Column(String(256), nullable=False)
    # сейчас два значения: 'user' и 'admin'
    role = Column(String(16), nullable=False, default="user")
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)


# --- Engine и сессии ----------------------------------------------------

# держим engine как модульный синглтон - создаётся один раз при первом обращении
_engine = None
_SessionLocal: sessionmaker | None = None


def get_engine():
    """Ленивая инициализация SQLAlchemy engine."""
    global _engine, _SessionLocal
    if _engine is None:
        url = get_settings().database_url
        # SQLite в FastAPI нужно открывать с check_same_thread=False,
        # иначе сессии не смогут шарить соединение между запросами
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, connect_args=connect_args, future=True)
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False,
                                     bind=_engine, expire_on_commit=False)
    return _engine


def get_session_factory() -> sessionmaker:
    """Возвращает фабрику сессий - engine лениво создаст при первом вызове."""
    get_engine()
    assert _SessionLocal is not None
    return _SessionLocal


def get_db() -> Iterator[Session]:
    """FastAPI dependency: одна сессия на запрос, закрывается в finally."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Создаёт схему БД из metadata. Запасной вариант когда Alembic не прогоняли.

    В нормальном случае схема накатывается через `alembic upgrade head`.
    """
    engine = get_engine()
    Base.metadata.create_all(engine)
