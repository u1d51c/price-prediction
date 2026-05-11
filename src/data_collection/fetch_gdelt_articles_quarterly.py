"""Сбор GDELT-статей поквартально для FinBERT-инференса.

Базовый src/data_collection/fetch_gdelt.py с ArtList в режиме DateDesc возвращает только самые
свежие статьи - все в одной-двух датах. Чтобы получить временную развёртку
sentiment'а на всю историю, проходим API кварталами и склеиваем результаты.

Запуск:
    python -m src.data_collection.fetch_gdelt_articles_quarterly
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.data_collection.fetch_gdelt import fetch_article_sample
from src.utils.config import PROJECT_ROOT, get_data_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def collect(topic_key: str, query: str, start: str, end: str, out_path: Path,
            per_quarter: int = 75) -> None:
    """Идёт по кварталам, тянет до per_quarter статей в каждом, склеивает в один файл."""
    # границы кварталов в выбранном интервале
    qstart = pd.date_range(start=start, end=end, freq="QS").strftime("%Y-%m-%d").tolist()
    if not qstart or qstart[0] > start:
        qstart = [start] + qstart
    if qstart[-1] < end:
        qstart.append(end)
    chunks = list(zip(qstart[:-1], qstart[1:]))

    parts: list[pd.DataFrame] = []
    for chunk_start, chunk_end in chunks:
        try:
            part = fetch_article_sample(
                query=query, start=chunk_start, end=chunk_end,
                max_records=per_quarter, source_lang="english",
            )
        except Exception as exc:
            logger.warning("[%s] %s..%s: %s", topic_key, chunk_start, chunk_end, exc)
            # пауза на случай 429 - иначе следующий запрос точно упадёт
            time.sleep(8)
            continue
        if not part.empty:
            # запоминаем какой квартал из какой пачки - пригодится для отладки
            part["chunk_start"] = chunk_start
            part["chunk_end"] = chunk_end
            parts.append(part)
        logger.info("[%s] %s..%s: добавлено %d статей", topic_key, chunk_start, chunk_end, len(part))

    if not parts:
        logger.warning("[%s] ничего не собралось", topic_key)
        return
    # один URL может встретиться в нескольких квартальных запросах - дедупим
    combined = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["url"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path)
    logger.info("[%s] итого %d уникальных статей -> %s",
                topic_key, len(combined), out_path.relative_to(PROJECT_ROOT))


def main() -> None:
    cfg = get_data_config()
    out_dir = PROJECT_ROOT / cfg["texts_storage"]["raw_dir"] / "gdelt"
    start = cfg["prices"]["start_date"]
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    queries = cfg["texts"]["gdelt"]["queries"]
    for topic_key, query in queries.items():
        out = out_dir / f"gdelt_{topic_key}_articles_quarterly.parquet"
        collect(topic_key, query, start, end, out)


if __name__ == "__main__":
    main()
