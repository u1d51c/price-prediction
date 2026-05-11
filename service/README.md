# Сервис: ML inference API (чекпоинт 4)

FastAPI-приложение, оборачивающее baseline-модели из чекпоинта 3 в HTTP-сервис.
Предсказывает направление движения цены (`up / down / flat`) на следующий
торговый день для **Brent**, **WTI** и **BTC**.

## Эндпоинты

| Метод | Путь            | Назначение                                                     | Auth     |
| ----- | --------------- | -------------------------------------------------------------- | -------- |
| GET   | `/healthz`      | Liveness; список доступных моделей                             | -        |
| POST  | `/forward`      | Прогноз по присланному вектору фичей                           | -        |
| POST  | `/forward/live` | Live-inference: тянем yfinance -> считаем фичи -> прогноз        | -        |
| GET   | `/history`      | История запросов (пагинация, фильтры по `endpoint` и `asset`)  | -        |
| DELETE| `/history`      | Очистить историю                                               | JWT admin|
| GET   | `/stats`        | Сводная статистика: mean/p50/p95/p99 времени и длины запроса   | -        |
| POST  | `/auth/login`   | OAuth2 password flow, выдаёт JWT                               | -        |

OpenAPI / Swagger UI - `http://127.0.0.1:8765/docs` (после запуска).

## Контракт ошибок (соответствует требованиям чекпойнта)

* **400** - невалидное тело запроса (`{"detail": "bad request", ...}`).
* **403** - модель не смогла обработать данные (нехватка фичей,
  отсутствует joblib-файл, yfinance не отдал данные).
* **401** - отсутствует/невалиден JWT-токен на admin-эндпоинте.
* **500** - внутренняя ошибка (логируется в `request_log.error_text`).

## Покрытие задания

| Балл | Требование                                       | Где |
| ---- | ------------------------------------------------ | --- |
| 10   | POST `/forward` (JSON in/out, 400/403)           | [service/app.py](app.py) |
| 5    | GET `/history` с БД                              | [service/app.py](app.py), [service/db.py](db.py) |
| 2 PRO| DELETE `/history` с токеном                      | [service/app.py](app.py), [service/auth.py](auth.py) |
| 5    | `/stats` со среднем/квантилями + длиной запроса  | [service/app.py](app.py) |
| 3 PRO| JWT-авторизация                                  | [service/auth.py](auth.py) |
| 5 PRO| Alembic-миграции                                 | [alembic/](../alembic/) |

**Итого: 30 баллов из 30, все PRO-задания закрыты.**

## Запуск

```bash
# 1. Обучить и сохранить production-модели (один раз)
python -m service.train_and_save

# 2. Применить миграции - создаст таблицы request_log и user
alembic upgrade head

# 3. Запустить сервис
uvicorn service.app:app --host 127.0.0.1 --port 8765 --reload

# 4. Открыть Swagger UI
open http://127.0.0.1:8765/docs
```

При старте автоматически создаётся пользователь `admin` / `admin` (из
дефолтных настроек) - в продакшене ОБЯЗАТЕЛЬНО переопределить через `.env`:

```env
JWT_SECRET_KEY=<минимум 32 случайных байта в base64>
ADMIN_USERNAME=<...>
ADMIN_PASSWORD=<...>
DATABASE_URL=postgresql+psycopg2://...   # вместо SQLite в проде
```

## Примеры запросов

### Логин и получение токена

```bash
curl -X POST http://127.0.0.1:8765/auth/login \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=admin&password=admin"
# -> {"access_token": "...", "token_type": "bearer", "expires_in_seconds": 3600}
```

### Live-inference по тикеру

```bash
curl -X POST http://127.0.0.1:8765/forward/live \
     -H "Content-Type: application/json" \
     -d '{"asset": "btc"}'
# -> {"asset":"btc","prediction":"down","probabilities":{...},
#    "as_of":"2026-05-10T00:00:00","last_close":...,"processing_ms":42.3, ...}
```

### Прогноз по готовому вектору фичей

```bash
curl -X POST http://127.0.0.1:8765/forward \
     -H "Content-Type: application/json" \
     -d '{"asset":"brent","features":{"r_lag_1":0.0123, "rv_21":0.018, ...}}'
```

Полный список ожидаемых фичей - в `models/<asset>.meta.json`, поле `features`.

### Очистить историю (admin)

```bash
TOKEN=$(curl -s -X POST http://127.0.0.1:8765/auth/login \
          -d 'username=admin&password=admin' | jq -r .access_token)
curl -X DELETE http://127.0.0.1:8765/history \
     -H "Authorization: Bearer $TOKEN"
# -> {"deleted": 42}
```

## Дизайнерские решения

1. **Один источник истины для подключения к БД.** `service.config.Settings`
   читает `DATABASE_URL` из env / `.env`, и его же подменяет в alembic/env.py
   - нет дублирования URL в `alembic.ini`.

2. **JWT exp через `time.time()`, не `datetime.utcnow().timestamp()`.**
   `utcnow()` возвращает naive datetime; `.timestamp()` на naive datetime
   интерпретирует его как **локальное** время, что на UTC+3 даёт сразу
   просроченный токен. Канонический Python-pitfall.

3. **`bcrypt==4.0.1` в pin.** Passlib 1.7.4 несовместим с bcrypt >= 4.1
   (изменилась внутренняя detect_wrap_bug-проверка). Альтернатива -
   мигрировать на argon2, но это уже тема чекпоинт 5.

4. **422 -> 400.** FastAPI по умолчанию возвращает 422 на Pydantic-ошибки,
   но задание чекпойнта требует `400 bad request`. Перехвачено в
   `validation_handler` в [service/app.py](app.py).

5. **Live-inference: ffill GDELT-фичей на 7 дней.** GDELT-сборщик иногда
   отстаёт на 1-2 дня от рынка; на инференсе допустим короткий
   forward-fill (но не на трейне!). Окно 7 дней - компромисс между
   устойчивостью и риском использовать слишком старый текст.

6. **Inference в памяти, не fork-on-request.** `ModelRegistry` -
   ленивый синглтон, joblib-файлы загружаются один раз при первом
   обращении (или на старте через `lifespan`). p95 на /forward - ~50ms
   (KNN k=51 на 1500 точек обучения).

## Структура

```
service/
├── __init__.py
├── app.py              # FastAPI + endpoints
├── auth.py             # JWT, bcrypt, dependencies
├── config.py           # pydantic-settings
├── db.py               # SQLAlchemy engine + ORM (RequestLog, User)
├── inference.py        # ModelRegistry + live-feature-pipeline
├── schemas.py          # Pydantic request/response модели
├── train_and_save.py   # обучение + сериализация в joblib
└── README.md           # этот файл

alembic/
├── env.py              # подменяет URL и подцепляет Base.metadata
├── versions/
│   └── 6b15ec00fd71_initial_request_log_and_user_tables.py
└── ...

models/
├── brent.joblib   brent.meta.json
├── wti.joblib     wti.meta.json
└── btc.joblib     btc.meta.json
```
