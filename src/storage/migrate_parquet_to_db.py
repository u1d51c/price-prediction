"""Миграция: переливает уже собранные parquet-файлы в SQLite-хранилище.

Источники:
  data/raw/texts/gdelt/gdelt_<topic>_articles_all_quarters.parquet  - сами статьи + url
  data/processed/articles_with_body_<topic>_all.parquet              - те же статьи с body
  data/processed/finbert_<topic>_articles.parquet                    - FinBERT-скоры по title и body
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.storage.finbert_db import (
    get_engine, rebuild_daily_aggregate, stats,
    upsert_articles, upsert_finbert_scores,
)
from src.utils.config import PROJECT_ROOT
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def migrate_topic(topic: str) -> None:
    raw_dir = PROJECT_ROOT / "data/raw/texts/gdelt"
    proc_dir = PROJECT_ROOT / "data/processed"

    # 1. Заголовки + url из ArtList (quarterly + weekly если есть)
    for name in (f"gdelt_{topic}_articles_all_quarters.parquet",
                 f"gdelt_{topic}_articles_weekly.parquet"):
        p = raw_dir / name
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if df.empty:
            continue
        # колонки которые есть у GDELT ArtList
        keep = [c for c in ("url", "seendate", "domain", "title", "language", "sourcecountry")
                if c in df.columns]
        n = upsert_articles(df[keep], topic=topic, with_body=False)
        logger.info("[%s] из %s заинжещено %d статей", topic, name, n)

    # 2. Body из articles_with_body
    p = proc_dir / f"articles_with_body_{topic}_all.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        # фильтруем только те где body реально получен
        df = df[df["body"].notna() & (df["body"].astype(str).str.len() > 0)]
        if not df.empty:
            keep = [c for c in ("url", "seendate", "domain", "title", "language",
                                "sourcecountry", "body") if c in df.columns]
            n = upsert_articles(df[keep], topic=topic, with_body=True)
            logger.info("[%s] body заинжещено %d статей", topic, n)

    # 3. FinBERT-скоры
    p = proc_dir / f"finbert_{topic}_articles.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        # title скоры — всегда есть
        title_df = df[["url"]].copy()
        title_df["p_pos"] = df["title_p_pos"]
        title_df["p_neg"] = df["title_p_neg"]
        title_df["p_neu"] = df["title_p_neu"]
        title_df = title_df.dropna(subset=["p_pos", "p_neg", "p_neu"])
        n = upsert_finbert_scores(title_df["url"].tolist(),
                                  title_df[["p_pos", "p_neg", "p_neu"]], "title")
        logger.info("[%s] title-скоров заинжещено %d", topic, n)

        # body скоры - только те где есть
        body_df = df[df["body_p_pos"].notna()][["url"]].copy()
        body_df["p_pos"] = df.loc[body_df.index, "body_p_pos"]
        body_df["p_neg"] = df.loc[body_df.index, "body_p_neg"]
        body_df["p_neu"] = df.loc[body_df.index, "body_p_neu"]
        n = upsert_finbert_scores(body_df["url"].tolist(),
                                  body_df[["p_pos", "p_neg", "p_neu"]], "body")
        logger.info("[%s] body-скоров заинжещено %d", topic, n)

    # 4. Пересобираем дневные агрегаты
    for source in ("title", "body"):
        n_days = rebuild_daily_aggregate(topic, source)
        logger.info("[%s/%s] daily aggregate: %d дат", topic, source, n_days)


def main() -> None:
    get_engine()  # создаст БД и таблицы
    for topic in ("oil", "btc"):
        migrate_topic(topic)
    print("\n=== Итоговая статистика БД ===")
    print(stats().to_string(index=False))


if __name__ == "__main__":
    main()
