# Запуск полного стека сервиса

Три слоя: **MLflow + MinIO** (хранение моделей) → **FastAPI** (инференс) → **Streamlit** (интерфейс).

## 0. Предусловия

```bash
cd year_project
pip install -r requirements.txt          # зависимости проекта
pip install mlflow boto3 streamlit hydra-core chronos-forecasting
```

Модели уже должны быть обучены и залогированы в MLflow с тегом `stage=PRD`
(см. шаг 3, если нужно переобучить).

## 1. Поднять MLflow + MinIO (Docker)

```bash
cd docker
docker compose up -d
# MLflow UI:     http://localhost:5001
# MinIO консоль: http://localhost:9001  (minioadmin / minioadmin)
```

## 2. Запустить FastAPI

```bash
cd year_project
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
PYTHONPATH=. uvicorn service.app:app --host 127.0.0.1 --port 8766
```

Проверка:
```bash
curl http://127.0.0.1:8766/healthz
curl "http://127.0.0.1:8766/predict/btc?n_days=3"
```

Эндпоинты:
| Метод | Путь | Назначение |
| --- | --- | --- |
| GET | `/predict/{asset}` | прогноз PRD-моделью из MLflow (основной для чекпоинта 7) |
| POST | `/forward` | прогноз по присланному вектору фичей |
| GET | `/history` | история запросов (БД) |
| DELETE | `/history` | очистка истории (JWT admin) |
| GET | `/stats` | квантили времени обработки, объёмы |
| POST | `/auth/login` | выдача JWT |
| GET | `/docs` | Swagger UI |

## 3. Запустить интерфейс (Streamlit)

```bash
cd year_project
API_URL=http://127.0.0.1:8766 streamlit run service/streamlit_app.py --server.port 8501
# UI: http://localhost:8501
```

Пользователь выбирает актив → видит прогноз направления, вероятности классов
и таблицу за выбранный период.

## (Опционально) Переобучить и зарегистрировать PRD-модель

```bash
cd year_project
# обучение с логированием в MLflow + регистрация с тегом PRD (Hydra CLI)
PYTHONPATH=. python -m src.training.train_with_mlflow asset=btc \
    feature_set.text_advanced=true feature_set.regime=true feature_set.finbert_title=true

# тестовый инференс PRD-модели через CLI
PYTHONPATH=. python -m src.inference.predict_cli asset=btc last_n_days=5
```

## Схема стека

```
[ Пользователь ]
      │  браузер :8501
      ▼
[ Streamlit UI ]  service/streamlit_app.py
      │  HTTP GET /predict/{asset}
      ▼
[ FastAPI ]       service/app.py  →  src/inference/core.py
      │  models:/{asset}_direction_model@PRD
      ▼
[ MLflow Registry ]  :5001   ──artifacts──▶  [ MinIO S3 ]  :9000
```
