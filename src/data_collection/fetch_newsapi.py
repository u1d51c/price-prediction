"""Финансовые новости через NewsAPI.org.

Бесплатный план NewsAPI отдаёт только последние 30 дней (и не более 100
запросов в сутки). Для исторических данных лучше использовать GDELT,
а NewsAPI пригождается для live-сбора при инференсе.

Документация: https://newsapi.org/docs/endpoints/everything
Ключ кладём в .env переменную NEWSAPI_KEY.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import PROJECT_ROOT, get_data_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

# подтягиваем .env (если есть) - оттуда возьмём NEWSAPI_KEY
load_dotenv()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def _newsapi_get(endpoint: str, params: dict, api_key: str) -> dict:
    """Один запрос к NewsAPI с retry на сетевых ошибках."""
    headers = {"X-Api-Key": api_key}
    r = requests.get(f"https://newsapi.org/v2/{endpoint}", params=params, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_news_for_query(
    query: str,
    api_key: str,
    days_back: int = 30,
    page_size: int = 100,
    language: str = "en",
) -> pd.DataFrame:
    """Тянет новости по запросу за последние days_back дней.

    На бесплатном плане days_back <= 30. Возвращает DataFrame со стандартными
    полями NewsAPI: title, description, content, publishedAt, source, url.
    """
    to_date = datetime.utcnow().date()
    from_date = to_date - timedelta(days=days_back)

    params = {
        "q": query,
        "language": language,
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "sortBy": "publishedAt",
        "pageSize": min(page_size, 100),
        "page": 1,
    }
    rows: list[dict] = []

    # NewsAPI пагинирует - крутим страницы пока не кончатся (или пока не упёрлись в лимит)
    while True:
        payload = _newsapi_get("everything", params, api_key)
        if payload.get("status") != "ok":
            logger.warning("NewsAPI вернул статус %s", payload.get("status"))
            break
        articles = payload.get("articles", [])
        if not articles:
            break
        rows.extend(articles)
        # последняя страница - кончилось
        if len(articles) < params["pageSize"]:
            break
        params["page"] += 1
        # бесплатный план всё равно отдаёт максимум первые 100 - не упираемся в стену
        if params["page"] > 5:
            break

    if not rows:
        return pd.DataFrame()

    # выравниваем вложенные структуры (например, source.name) в плоские колонки
    df = pd.json_normalize(rows)
    if "publishedAt" in df.columns:
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], utc=True, errors="coerce")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачать новости через NewsAPI (последние 30 дней)")
    parser.add_argument("--days", type=int, default=30, help="Глубина окна в днях")
    args = parser.parse_args()

    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        # без ключа делать нечего - просто завершаемся
        logger.error("NEWSAPI_KEY не установлен в .env - пропуск.")
        return

    cfg = get_data_config()
    out_dir = PROJECT_ROOT / cfg["texts_storage"]["raw_dir"] / "newsapi"
    out_dir.mkdir(parents=True, exist_ok=True)

    for topic_key, query in cfg["texts"]["newsapi"]["queries"].items():
        logger.info("Тема %s, запрос: %s", topic_key, query)
        df = fetch_news_for_query(query=query, api_key=api_key, days_back=args.days)
        if df.empty:
            logger.warning("По теме %s ничего не нашлось", topic_key)
            continue
        # имя файла включает дату - чтобы перезапуски не затирали друг друга
        path = out_dir / f"newsapi_{topic_key}_{datetime.utcnow().date()}.parquet"
        df.to_parquet(path)
        logger.info("Тема %s: %d новостей -> %s", topic_key, len(df), path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
