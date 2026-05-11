"""Целевая переменная: направление дневного движения цены.

Три класса: up / down / flat. Flat вводим симметричным порогом tau вокруг
нуля - иначе на спокойных днях модель угадывала бы монетку, и flat -
естественный способ сказать "не открываем позицию". Работаем с
лог-доходностями: они аддитивны во времени и ближе к нормальным,
чем процентные.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(prices: pd.Series, horizon: int = 1) -> pd.Series:
    """Будущая лог-доходность: r_t = log(P_{t+horizon} / P_t)."""
    # WTI 20-04-2020 закрылся отрицательным - там log() даст NaN,
    # этот день просто не разметится таргетом.
    safe = prices.where(prices > 0)
    return np.log(safe.shift(-horizon) / safe)


def label_direction(
    log_returns: pd.Series,
    flat_threshold_sigma: float = 0.1,
) -> tuple[pd.Series, float]:
    """Размечает доходности на up / down / flat.

    flat_threshold_sigma - порог в долях стандартного отклонения. По умолчанию
    0.1*sigma даёт ~10-20% дней в классе flat на дневных данных.
    Возвращает кортеж (метки, фактический порог tau).
    """
    # Считаем sigma по всему ряду и берём от неё долю
    sigma = log_returns.std()
    tau = flat_threshold_sigma * sigma

    def _classify(r: float) -> str | float:
        if pd.isna(r):
            return np.nan
        if r > tau:
            return "up"
        if r < -tau:
            return "down"
        return "flat"

    labels = log_returns.apply(_classify)
    return labels, float(tau)


def build_target_frame(
    prices_df: pd.DataFrame,
    price_col: str = "Close",
    horizon: int = 1,
    flat_threshold_sigma: float = 0.1,
) -> pd.DataFrame:
    """Из OHLCV считает таргет и базовые доходности.

    На выходе добавляются колонки:
      log_return_t  - доходность за прошедший день,
      future_return - будущая доходность на горизонт прогноза,
      target        - метка up / down / flat.
    """
    df = prices_df.copy()
    # log_return_t - это уже произошедшее движение, его потом можно
    # использовать как фичу (а не как таргет).
    safe = df[price_col].where(df[price_col] > 0)
    df["log_return_t"] = np.log(safe / safe.shift(1))
    # future_return - на чём собственно учим модель угадывать знак
    df["future_return"] = compute_log_returns(df[price_col], horizon=horizon)
    df["target"], threshold = label_direction(df["future_return"], flat_threshold_sigma)
    # сохраняем порог tau в attrs, чтобы потом его же использовать для ARIMA
    df.attrs["flat_threshold"] = threshold
    df.attrs["horizon"] = horizon
    return df
