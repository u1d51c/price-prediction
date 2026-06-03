"""SQLite-хранилище для статей и FinBERT-скоров.

Три таблицы:
  articles        — сами статьи (заголовок + опционально body), key = url.
  finbert_scores  — скоры FinBERT для конкретной статьи и источника (title/body).
  daily_finbert   — материализованный кэш дневных агрегатов по (date, topic, source).

Используется и при сборе данных (через ingest_*), и при тренинге модели
(через get_daily_aggregate). База лежит в data/processed/finbert.db,
этот файл попадает под data/processed/* в .gitignore.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Index, Integer, String, Text,
    UniqueConstraint, create_engine, func, select,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.utils.config import PROJECT_ROOT


DB_PATH = PROJECT_ROOT / "data" / "processed" / "finbert.db"


class Base(DeclarativeBase):
    """Базовый класс для всех ORM-моделей хранилища."""


class Article(Base):
    """Одна статья из GDELT (заголовок + опционально полное body)."""

    __tablename__ = "articles"

    # url берём как первичный ключ — он уникален и совпадает с GDELT-id
    url = Column(String, primary_key=True)
    topic = Column(String(16), nullable=False, index=True)
    seendate = Column(DateTime, nullable=False, index=True)
    domain = Column(String(255), nullable=True)
    language = Column(String(32), nullable=True)
    sourcecountry = Column(String(32), nullable=True)
    title = Column(Text, nullable=True)
    body = Column(Text, nullable=True)
    body_len = Column(Integer, nullable=True)
    scraped_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_articles_topic_date", "topic", "seendate"),
    )


class FinbertScore(Base):
    """FinBERT-скор для (article, source). Source = 'title' | 'body'."""

    __tablename__ = "finbert_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    article_url = Column(String, ForeignKey("articles.url"), nullable=False, index=True)
    source = Column(String(8), nullable=False)
    p_pos = Column(Float, nullable=False)
    p_neg = Column(Float, nullable=False)
    p_neu = Column(Float, nullable=False)
    scored_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)

    __table_args__ = (
        # один скор на (статья, источник) — повторное вставление будет update
        UniqueConstraint("article_url", "source", name="uq_score_per_source"),
    )


class DailyFinbert(Base):
    """Материализованный кэш агрегатов по дню. Перестраивается при необходимости."""

    __tablename__ = "daily_finbert"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False)
    topic = Column(String(16), nullable=False)
    source = Column(String(8), nullable=False)
    n_texts = Column(Integer, nullable=False)
    mean_compound = Column(Float, nullable=False)
    pos_share = Column(Float, nullable=False)
    neg_share = Column(Float, nullable=False)
    neu_share = Column(Float, nullable=False)
    computed_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("date", "topic", "source", name="uq_daily_unique"),
        Index("ix_daily_topic_source_date", "topic", "source", "date"),
    )


# --- инициализация --------------------------------------------------------

_engine = None
_SessionLocal: sessionmaker | None = None


def get_engine(db_path: Path | None = None):
    """Ленивая инициализация engine."""
    global _engine, _SessionLocal
    if _engine is None:
        path = db_path or DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(
            f"sqlite:///{path}",
            connect_args={"check_same_thread": False},
            future=True,
        )
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False,
                                     expire_on_commit=False, future=True)
        # создаём таблицы при первом обращении
        Base.metadata.create_all(_engine)
    return _engine


def get_session() -> Session:
    get_engine()
    assert _SessionLocal is not None
    return _SessionLocal()


# --- ingest утилиты -------------------------------------------------------

def upsert_articles(df: pd.DataFrame, topic: str, with_body: bool = False) -> int:
    """Идемпотентно записывает статьи в БД. По url - update.

    df ожидаемые колонки: url, seendate, domain, language?, sourcecountry?,
    title; body — если with_body=True.
    """
    if df.empty:
        return 0
    engine = get_engine()
    work = df.copy()
    work["topic"] = topic
    work["seendate"] = pd.to_datetime(work["seendate"])
    rows = []
    for r in work.to_dict(orient="records"):
        row = {
            "url": r.get("url"),
            "topic": topic,
            "seendate": r.get("seendate"),
            "domain": r.get("domain"),
            "language": r.get("language"),
            "sourcecountry": r.get("sourcecountry"),
            "title": r.get("title"),
        }
        if with_body:
            b = r.get("body")
            row["body"] = b if isinstance(b, str) else None
            row["body_len"] = len(b) if isinstance(b, str) else None
            row["scraped_at"] = dt.datetime.utcnow()
        if not row["url"]:
            continue
        rows.append(row)
    if not rows:
        return 0
    # ON CONFLICT (url) DO UPDATE - sqlite-специфичный, но самый чистый
    with engine.begin() as conn:
        stmt = sqlite_insert(Article).values(rows)
        # обновляем все поля кроме url и created_at
        excluded = {c.name: stmt.excluded[c.name] for c in Article.__table__.columns
                    if c.name not in ("url", "created_at")}
        # если body=NULL приходит из ArtList без скрейпа, не затираем уже сохранённый body
        if not with_body:
            excluded.pop("body", None)
            excluded.pop("body_len", None)
            excluded.pop("scraped_at", None)
        stmt = stmt.on_conflict_do_update(index_elements=["url"], set_=excluded)
        conn.execute(stmt)
    return len(rows)


def upsert_finbert_scores(
    urls: Iterable[str],
    probs_df: pd.DataFrame,
    source: str,
) -> int:
    """Записывает FinBERT-скоры. probs_df имеет колонки p_pos, p_neg, p_neu в том же порядке, что urls."""
    assert source in ("title", "body")
    engine = get_engine()
    urls = list(urls)
    if not urls:
        return 0
    rows = []
    for url, (_, p) in zip(urls, probs_df.iterrows()):
        rows.append({
            "article_url": url,
            "source": source,
            "p_pos": float(p.get("p_pos", 0.0)),
            "p_neg": float(p.get("p_neg", 0.0)),
            "p_neu": float(p.get("p_neu", 0.0)),
            "scored_at": dt.datetime.utcnow(),
        })
    if not rows:
        return 0
    with engine.begin() as conn:
        stmt = sqlite_insert(FinbertScore).values(rows)
        excluded = {c.name: stmt.excluded[c.name] for c in FinbertScore.__table__.columns
                    if c.name not in ("id", "article_url", "source")}
        stmt = stmt.on_conflict_do_update(
            index_elements=["article_url", "source"], set_=excluded,
        )
        conn.execute(stmt)
    return len(rows)


# --- агрегаты -------------------------------------------------------------

def rebuild_daily_aggregate(topic: str, source: str) -> int:
    """Пересобирает daily_finbert для (topic, source) из articles + finbert_scores."""
    assert source in ("title", "body")
    engine = get_engine()
    sql = """
        SELECT date(a.seendate) AS date,
               COUNT(*) AS n_texts,
               AVG(s.p_pos - s.p_neg) AS mean_compound,
               AVG(CASE WHEN s.p_pos >= s.p_neg AND s.p_pos >= s.p_neu THEN 1.0 ELSE 0.0 END) AS pos_share,
               AVG(CASE WHEN s.p_neg >  s.p_pos AND s.p_neg >= s.p_neu THEN 1.0 ELSE 0.0 END) AS neg_share,
               AVG(CASE WHEN s.p_neu >  s.p_pos AND s.p_neu >  s.p_neg THEN 1.0 ELSE 0.0 END) AS neu_share
        FROM articles a
        JOIN finbert_scores s ON s.article_url = a.url AND s.source = :source
        WHERE a.topic = :topic
        GROUP BY date(a.seendate)
    """
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"topic": topic, "source": source})
    if df.empty:
        return 0
    df["date"] = pd.to_datetime(df["date"])
    df["topic"] = topic
    df["source"] = source
    # SQLAlchemy default не срабатывает при прямом to_sql, ставим явно
    df["computed_at"] = dt.datetime.utcnow()

    with engine.begin() as conn:
        # сносим старый агрегат для (topic, source) и пишем новый
        conn.execute(
            DailyFinbert.__table__.delete().where(
                (DailyFinbert.topic == topic) & (DailyFinbert.source == source)
            )
        )
        df.to_sql("daily_finbert", conn, if_exists="append", index=False)
    return len(df)


def get_daily_aggregate(topic: str, source: str) -> pd.DataFrame:
    """Возвращает дневной DataFrame для использования в тренинге.

    Колонки: date (index), finbert_compound, finbert_pos_share,
    finbert_neg_share, finbert_neutral_share, finbert_n_texts.
    """
    engine = get_engine()
    sql = """
        SELECT date, mean_compound AS finbert_compound,
               pos_share AS finbert_pos_share,
               neg_share AS finbert_neg_share,
               neu_share AS finbert_neutral_share,
               n_texts AS finbert_n_texts
        FROM daily_finbert
        WHERE topic = :topic AND source = :source
        ORDER BY date
    """
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"topic": topic, "source": source})
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


# --- статистика для диагностики ------------------------------------------

def stats() -> pd.DataFrame:
    """Краткая сводка: сколько статей и скоров по каждой паре (topic, source)."""
    engine = get_engine()
    sql = """
        SELECT a.topic, s.source,
               COUNT(DISTINCT a.url) AS n_articles,
               COUNT(DISTINCT date(a.seendate)) AS n_dates
        FROM articles a
        JOIN finbert_scores s ON s.article_url = a.url
        GROUP BY a.topic, s.source
        UNION ALL
        SELECT a.topic, 'no_score' AS source,
               COUNT(DISTINCT a.url) AS n_articles,
               COUNT(DISTINCT date(a.seendate)) AS n_dates
        FROM articles a
        WHERE a.url NOT IN (SELECT DISTINCT article_url FROM finbert_scores)
        GROUP BY a.topic
    """
    with engine.connect() as conn:
        return pd.read_sql(sql, conn)
