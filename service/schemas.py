"""Pydantic-схемы запросов и ответов сервиса.

Здесь же FastAPI берёт описание для автогенерации OpenAPI / Swagger UI.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


AssetKey = Literal["brent", "wti", "btc"]
DirectionClass = Literal["up", "down", "flat"]


# Pydantic v2 резервирует префикс model_ под свои внутренние нужды.
# У нас есть поле model_version - без этого конфига он бросит warning.
_RESP_CONFIG = ConfigDict(protected_namespaces=())


# --- /forward (готовый вектор фичей) ------------------------------------

class ForwardRequest(BaseModel):
    """Запрос с уже посчитанным вектором фичей.

    Клиент сам отвечает за состав и значения фичей. Список ожидаемых имён
    лежит в `models/<asset>.meta.json`, поле "features".
    """

    asset: AssetKey = Field(..., description="brent / wti / btc")
    features: dict[str, float] = Field(
        ..., description="{feature_name: value}; порядок не важен - сервис сам выстроит"
    )


class ForwardResponse(BaseModel):
    model_config = _RESP_CONFIG

    asset: AssetKey
    prediction: DirectionClass
    probabilities: dict[DirectionClass, float]
    # из meta.json - поле trained_at_utc, чтобы клиент видел, какая версия модели отвечала
    model_version: str = Field(..., description="trained_at_utc из meta.json")
    processing_ms: float


# --- /forward/live (по тикеру) -----------------------------------------

class LiveForwardRequest(BaseModel):
    """Запрос для live-инференса - только идентификатор актива.

    Сервис сам подтянет последние OHLCV с Yahoo Finance, посчитает фичи
    и вернёт прогноз на ближайший торговый день.
    """

    asset: AssetKey


class LiveForwardResponse(ForwardResponse):
    model_config = _RESP_CONFIG

    as_of: dt.datetime = Field(..., description="Дата последнего бара, на котором сделан прогноз")
    last_close: float


# --- /history ----------------------------------------------------------

class HistoryItem(BaseModel):
    # from_attributes - чтобы Pydantic мог собрать модель прямо из ORM-объекта
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: dt.datetime
    endpoint: str
    asset: Optional[str] = None
    status_code: int
    processing_ms: Optional[float] = None
    user: Optional[str] = None


class HistoryResponse(BaseModel):
    items: list[HistoryItem]
    total: int


class DeleteHistoryResponse(BaseModel):
    deleted: int


# --- /stats ------------------------------------------------------------

class QuantileStats(BaseModel):
    """Базовая статистика числовой колонки: среднее + 50/95/99-й перцентили."""

    mean: Optional[float] = None
    p50: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None
    count: int = 0


class StatsResponse(BaseModel):
    total_requests: int
    by_endpoint: dict[str, int]
    by_asset: dict[str, int]
    by_status: dict[int, int]
    processing_ms: QuantileStats
    input_length: QuantileStats


# --- /auth -------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in_seconds: int


# --- Унифицированная ошибка --------------------------------------------

class ErrorResponse(BaseModel):
    detail: str
