"""Базовые текстовые фичи: дневные сигналы из GDELT и VADER-разметка.

Тут два простых уровня: подневное количество и тон статей из GDELT (это
агрегаты, которые GDELT сам считает по запросу) и VADER - лексический
sentiment по отдельным текстам. Это baseline-ы, потом на DL-этапе
к ним добавляется FinBERT.
"""

from __future__ import annotations

import pandas as pd


def daily_gdelt_signals(
    volraw_df: pd.DataFrame,
    tone_df: pd.DataFrame,
) -> pd.DataFrame:
    """Склеивает GDELT-объём и тон в одну дневную таблицу.

    На входе ожидает результаты fetch_timeline() с колонками datetime/value.
    Возвращает DataFrame с колонками article_count, article_norm, avg_tone.
    """
    if volraw_df.empty and tone_df.empty:
        return pd.DataFrame()

    # объём: число статей и нормировка на общий пул GDELT за день
    vol = volraw_df.copy()
    vol["date"] = pd.to_datetime(vol["datetime"]).dt.normalize()
    vol = vol.groupby("date")[["value", "norm"]].mean().rename(
        columns={"value": "article_count", "norm": "article_norm"}
    )

    # тон: средний sentiment по WordNet, который считает сам GDELT
    tone = tone_df.copy()
    tone["date"] = pd.to_datetime(tone["datetime"]).dt.normalize()
    tone = tone.groupby("date")["value"].mean().to_frame("avg_tone")

    out = vol.join(tone, how="outer").sort_index()
    out.index.name = "date"
    return out


def vader_sentiment_per_row(texts: pd.Series) -> pd.DataFrame:
    """VADER-скоры (neg/neu/pos/compound) по строкам."""
    # импорт внутри функции, чтобы модуль не падал, если NLP-зависимостей нет
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    rows = []
    for t in texts.fillna(""):
        scores = sia.polarity_scores(str(t))
        rows.append(scores)
    return pd.DataFrame(rows, index=texts.index)


def aggregate_daily_vader(
    df: pd.DataFrame,
    text_col: str,
    date_col: str,
    weight_col: str | None = None,
) -> pd.DataFrame:
    """Среднедневной VADER-compound. С весом - взвешенное среднее.

    Если есть weight_col (например, число лайков у Reddit-поста), важные
    посты получают приоритет, иначе шумные одиночные голоса перевесят.
    """
    scores = vader_sentiment_per_row(df[text_col])
    work = df[[date_col]].copy()
    work["compound"] = scores["compound"].values
    work["pos"] = scores["pos"].values
    work["neg"] = scores["neg"].values
    work["date"] = pd.to_datetime(work[date_col]).dt.normalize()

    if weight_col is not None and weight_col in df.columns:
        # клиппим, чтобы отрицательные веса не ломали среднее; +1 от нулей
        work["weight"] = df[weight_col].clip(lower=0).values + 1.0
        work["w_compound"] = work["compound"] * work["weight"]
        agg = work.groupby("date").agg(
            weighted_compound=("w_compound", "sum"),
            total_weight=("weight", "sum"),
            mean_compound=("compound", "mean"),
            mean_pos=("pos", "mean"),
            mean_neg=("neg", "mean"),
            n_texts=("compound", "size"),
        )
        # делим суммы - получаем нормированное взвешенное среднее
        agg["weighted_compound"] = agg["weighted_compound"] / agg["total_weight"]
        agg = agg.drop(columns=["total_weight"])
    else:
        agg = work.groupby("date").agg(
            mean_compound=("compound", "mean"),
            mean_pos=("pos", "mean"),
            mean_neg=("neg", "mean"),
            n_texts=("compound", "size"),
        )
    return agg
