"""Производные текстовые фичи: лаги, дельты, z-score объёма новостей.

К базовым GDELT-сигналам (article_count / article_norm / avg_tone) добавляем:
лагированные значения (1, 3, 5 дней назад), их разности и z-score объёма.
Идея: реакция цены на новости часто запаздывает на день-два, поэтому
голый уровень "сегодняшнего" сигнала менее информативен, чем его динамика.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_text_lags_and_deltas(
    df: pd.DataFrame,
    lags: tuple[int, ...] = (1, 3, 5),
) -> pd.DataFrame:
    """Добавляет лаги, дельты и z21 для текстовых колонок GDELT.

    Если этих колонок в df нет - просто возвращает исходный df.
    """
    out = df.copy()
    text_cols = [c for c in ("article_count", "article_norm", "avg_tone") if c in out.columns]
    if not text_cols:
        # текстов нет - ничего считать не из чего
        return out

    for col in text_cols:
        for k in lags:
            out[f"{col}_lag_{k}"] = out[col].shift(k)
        # дельта за день и за неделю - улавливает резкие изменения новостного фона
        out[f"{col}_delta_1"] = out[col].diff(1)
        out[f"{col}_delta_5"] = out[col].diff(5)

    # z-score объёма относительно скользящего месячного фона - "необычно много новостей".
    # Окно 21 ~= торговый месяц.
    if "article_count" in out.columns:
        rolling_mean = out["article_count"].rolling(21).mean()
        rolling_std = out["article_count"].rolling(21).std()
        out["article_count_z21"] = (out["article_count"] - rolling_mean) / rolling_std.replace(0, np.nan)

    return out


# Имена всех расширенных текстовых фичей (удобно подключать в фичесеты)
TEXT_ADVANCED_COLS: tuple[str, ...] = (
    "article_count_lag_1", "article_count_lag_3", "article_count_lag_5",
    "article_norm_lag_1",  "article_norm_lag_3",  "article_norm_lag_5",
    "avg_tone_lag_1",      "avg_tone_lag_3",      "avg_tone_lag_5",
    "article_count_delta_1", "article_count_delta_5",
    "article_norm_delta_1",  "article_norm_delta_5",
    "avg_tone_delta_1",      "avg_tone_delta_5",
    "article_count_z21",
)


def daily_vader_from_article_sample(articles_df: pd.DataFrame) -> pd.DataFrame:
    """Считает VADER compound по заголовкам и агрегирует по дням.

    Используется как иллюстративная фича на выборке статей из GDELT
    (полноценный sentiment по всей истории появляется в src/features/finbert.py).
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    if articles_df.empty or "title" not in articles_df.columns:
        return pd.DataFrame()

    sia = SentimentIntensityAnalyzer()
    work = articles_df.copy()
    # compound - итоговый скаляр VADER в [-1, 1]
    work["compound"] = work["title"].fillna("").astype(str).map(
        lambda t: sia.polarity_scores(t)["compound"]
    )
    work["date"] = pd.to_datetime(work["seendate"]).dt.normalize()
    # пороги 0.1 / -0.1 - стандартная отсечка VADER для разделения нейтральных и окрашенных
    return work.groupby("date").agg(
        vader_compound_mean=("compound", "mean"),
        vader_compound_pos_share=("compound", lambda s: float((s > 0.1).mean())),
        vader_compound_neg_share=("compound", lambda s: float((s < -0.1).mean())),
        vader_articles_n=("compound", "size"),
    )
