"""FastAPI-приложение: предсказание направления движения цены.

Эндпоинты (полное описание в Swagger UI на /docs):
  POST /forward          - прогноз по присланному вектору фичей
  POST /forward/live     - прогноз: тянем свежие OHLCV из yfinance и считаем фичи сами
  GET  /history          - история всех запросов с пагинацией
  DELETE /history        - очистить историю (нужен JWT с ролью admin)
  GET  /stats            - статистика: квантили времени обработки, длин входов
  POST /auth/login       - выдаёт JWT по логину/паролю
  GET  /healthz          - liveness probe

Коды ошибок: 400 - невалидный JSON, 401 - нет/невалиден токен,
403 - модель не смогла обработать данные, 500 - внутренняя ошибка.
"""

from __future__ import annotations

import json
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, Optional, Sequence

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import func
from sqlalchemy.orm import Session

from service.auth import (
    create_access_token, ensure_admin_user,
    get_current_user, require_admin, verify_password,
)
from service.config import get_settings
from service.db import RequestLog, User, get_db, get_session_factory, init_db
from service.inference import get_registry, predict_from_features, predict_live
from service.schemas import (
    DeleteHistoryResponse, ForwardRequest, ForwardResponse,
    HistoryItem, HistoryResponse,
    LiveForwardRequest, LiveForwardResponse,
    QuantileStats, StatsResponse, TokenResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Стартовый и завершающий хук: создаём схему БД, дефолтного админа и прогреваем модели."""
    init_db()
    SessionLocal = get_session_factory()
    with SessionLocal() as db:
        ensure_admin_user(db)
    # прогреваем ModelRegistry, чтобы первый запрос не ловил холодную загрузку joblib
    try:
        registry = get_registry()
        for a in registry.available_assets():
            registry.get(a)
    except Exception:
        # если моделей ещё нет - сервис всё равно стартанёт, /forward просто вернёт 403
        pass
    yield


app = FastAPI(
    title="Commodity & BTC direction-classification API",
    description=(
        "Прогноз направления (up/down/flat) дневной цены Brent / WTI / BTC "
        "по ценовым OHLCV и текстовым GDELT-сигналам."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# --- Логирование запросов в БД ------------------------------------------

def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(values, q))


def _log_request(
    db: Session,
    endpoint: str,
    asset: Optional[str],
    payload: Any,
    response: Any,
    status_code: int,
    processing_ms: float,
    user: Optional[str],
    error_text: Optional[str] = None,
) -> None:
    """Пишет одну строку в request_log. При ошибке записи делает rollback."""
    try:
        # длину входа считаем как длину сериализованного JSON - это число берётся в /stats
        payload_serialized = json.dumps(payload, ensure_ascii=False, default=str) if payload is not None else None
        input_length = len(payload_serialized) if payload_serialized else None
        entry = RequestLog(
            endpoint=endpoint, asset=asset,
            request_payload=payload, response_payload=response,
            status_code=status_code, input_length=input_length,
            processing_ms=processing_ms, user=user, error_text=error_text,
        )
        db.add(entry)
        db.commit()
    except Exception:
        # лог не должен ронять основной обработчик
        db.rollback()


# --- Обработка ошибок ---------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    """Pydantic-ошибки валидации превращаем в 400 с понятным телом 'bad request'."""
    return JSONResponse(status_code=400, content={"detail": "bad request", "errors": exc.errors()})


# --- /forward ----------------------------------------------------------

@app.post(
    "/forward",
    response_model=ForwardResponse,
    responses={
        400: {"description": "bad request"},
        403: {"description": "модель не смогла обработать данные"},
    },
)
def forward(
    payload: ForwardRequest,
    db: Session = Depends(get_db),
    user: Optional[User] = Depends(get_current_user),
):
    """Прогноз на готовом векторе фичей. Авторизация необязательна."""
    t0 = time.perf_counter()
    asset = payload.asset
    try:
        result = predict_from_features(asset, payload.features)
        elapsed = (time.perf_counter() - t0) * 1000
        result_payload = ForwardResponse(processing_ms=elapsed, **result).model_dump()
        _log_request(db, endpoint="/forward", asset=asset,
                     payload=payload.model_dump(), response=result_payload,
                     status_code=200, processing_ms=elapsed,
                     user=user.username if user else None)
        return result_payload
    except (FileNotFoundError, ValueError) as exc:
        # не нашли модель или не хватает фичей - это 403 "модель не смогла обработать"
        elapsed = (time.perf_counter() - t0) * 1000
        _log_request(db, endpoint="/forward", asset=asset,
                     payload=payload.model_dump(), response=None,
                     status_code=403, processing_ms=elapsed,
                     user=user.username if user else None, error_text=str(exc))
        raise HTTPException(status_code=403, detail="модель не смогла обработать данные") from exc
    except Exception as exc:
        # что-то совсем неожиданное - 500 + полный traceback в БД для разбора
        elapsed = (time.perf_counter() - t0) * 1000
        _log_request(db, endpoint="/forward", asset=asset,
                     payload=payload.model_dump(), response=None,
                     status_code=500, processing_ms=elapsed,
                     user=user.username if user else None,
                     error_text=f"{exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"internal error: {exc}") from exc


# --- /forward/live -----------------------------------------------------

@app.post("/forward/live", response_model=LiveForwardResponse,
          responses={403: {"description": "модель не смогла обработать данные"}})
def forward_live(
    payload: LiveForwardRequest,
    db: Session = Depends(get_db),
    user: Optional[User] = Depends(get_current_user),
):
    """Прогноз по тикеру: сами тянем свежие данные с yfinance и считаем фичи."""
    t0 = time.perf_counter()
    try:
        result = predict_live(payload.asset)
        elapsed = (time.perf_counter() - t0) * 1000
        full = LiveForwardResponse(processing_ms=elapsed, **result).model_dump(mode="json")
        _log_request(db, endpoint="/forward/live", asset=payload.asset,
                     payload=payload.model_dump(), response=full,
                     status_code=200, processing_ms=elapsed,
                     user=user.username if user else None)
        return full
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        # yfinance может не отдать данные, GDELT может быть недоступен, модели может не быть - всё 403
        elapsed = (time.perf_counter() - t0) * 1000
        _log_request(db, endpoint="/forward/live", asset=payload.asset,
                     payload=payload.model_dump(), response=None,
                     status_code=403, processing_ms=elapsed,
                     user=user.username if user else None, error_text=str(exc))
        raise HTTPException(status_code=403, detail="модель не смогла обработать данные") from exc


# --- /history ----------------------------------------------------------

@app.get("/history", response_model=HistoryResponse)
def get_history(
    limit: int = Query(default=50, le=500, ge=1),
    offset: int = Query(default=0, ge=0),
    endpoint: Optional[str] = Query(default=None),
    asset: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
):
    """История запросов: можно фильтровать по эндпоинту и активу, есть пагинация."""
    q = db.query(RequestLog)
    if endpoint is not None:
        q = q.filter(RequestLog.endpoint == endpoint)
    if asset is not None:
        q = q.filter(RequestLog.asset == asset)
    total = q.count()
    # сортируем по убыванию времени, чтобы свежие были сверху
    items = q.order_by(RequestLog.created_at.desc()).offset(offset).limit(limit).all()
    return HistoryResponse(items=[HistoryItem.model_validate(i) for i in items], total=total)


@app.delete("/history", response_model=DeleteHistoryResponse)
def delete_history(
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Полная очистка истории. Доступна только пользователям с ролью admin."""
    n = db.query(RequestLog).delete()
    db.commit()
    return DeleteHistoryResponse(deleted=n)


# --- /stats ------------------------------------------------------------

@app.get("/stats", response_model=StatsResponse)
def get_stats(db: Session = Depends(get_db)):
    """Общая статистика по всем логированным запросам."""
    total = db.query(RequestLog).count()

    def _group_count(col):
        """Группировка по колонке: возвращает словарь {значение: количество}."""
        rows = db.query(col, func.count()).group_by(col).all()
        return {(k if k is not None else "null"): n for k, n in rows}

    by_endpoint = _group_count(RequestLog.endpoint)
    by_asset = _group_count(RequestLog.asset)
    by_status_raw = db.query(RequestLog.status_code, func.count()).group_by(RequestLog.status_code).all()
    by_status = {int(k): n for k, n in by_status_raw}

    # значения числовых колонок для расчёта квантилей в Python
    proc = [r[0] for r in db.query(RequestLog.processing_ms)
            .filter(RequestLog.processing_ms.isnot(None)).all()]
    inlen = [r[0] for r in db.query(RequestLog.input_length)
             .filter(RequestLog.input_length.isnot(None)).all()]

    def _qstats(values) -> QuantileStats:
        if not values:
            return QuantileStats()
        return QuantileStats(
            mean=float(np.mean(values)),
            p50=_quantile(values, 0.5),
            p95=_quantile(values, 0.95),
            p99=_quantile(values, 0.99),
            count=len(values),
        )

    return StatsResponse(
        total_requests=total,
        by_endpoint=by_endpoint,
        by_asset=by_asset,
        by_status=by_status,
        processing_ms=_qstats(proc),
        input_length=_qstats(inlen),
    )


# --- /auth/login -------------------------------------------------------

@app.post("/auth/login", response_model=TokenResponse)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Стандартный OAuth2 password flow: логин/пароль -> JWT.

    В Swagger UI работает через кнопку "Authorize". Полученный токен потом
    передаётся в заголовке `Authorization: Bearer <token>`.
    """
    user = db.query(User).filter(User.username == form.username).first()
    if user is None or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Неверные логин или пароль",
                            headers={"WWW-Authenticate": "Bearer"})
    token, expires = create_access_token(user.username, user.role)
    return TokenResponse(access_token=token, expires_in_seconds=expires)


# --- /healthz ----------------------------------------------------------

@app.get("/healthz")
def healthz():
    """Liveness probe для Kubernetes/прочего: список доступных моделей и URL БД."""
    settings = get_settings()
    try:
        assets = get_registry().available_assets()
    except Exception:
        assets = []
    return {
        "status": "ok",
        "available_models": sorted(assets),
        # обрезаем username:password@ если он есть в URL - наружу пароль не отдаём
        "db_url": settings.database_url.split("@")[-1],
    }
