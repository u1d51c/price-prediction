"""Настройки сервиса. Читаются из .env или переменных окружения процесса."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# корень проекта = на один уровень выше service/
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Все параметры сервиса в одном объекте.

    Все значения можно переопределить переменной окружения с тем же именем
    в верхнем регистре (например, JWT_SECRET_KEY).
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        # лишние переменные в .env не должны ломать сервис
        extra="ignore",
    )

    # БД: по умолчанию SQLite-файл рядом с проектом
    database_url: str = Field(
        default=f"sqlite:///{PROJECT_ROOT / 'data' / 'service.db'}",
        description="SQLAlchemy URL",
    )
    # секрет для подписи JWT - в проде ОБЯЗАТЕЛЬНО задать через env
    jwt_secret_key: str = Field(
        default="change-me-in-production-very-long-random-string",
        description="HS256 secret",
    )
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    # дефолтный админ - создаётся при первом старте, если в БД ещё нет такого пользователя
    admin_username: str = "admin"
    admin_password: str = "admin"
    # где лежат joblib-файлы обученных моделей
    models_dir: Path = PROJECT_ROOT / "models"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Возвращает один и тот же объект настроек на весь процесс."""
    return Settings()
