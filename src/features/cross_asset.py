"""Кросс-активные фичи: спред Brent − WTI.

Оба сорта нефти движутся почти синхронно, но их спред чувствителен
к региональным дисбалансам (сланцевая нефть США vs ближневосточный
экспорт) и поэтому несёт макроинформацию, которой нет в отдельных
ценах.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def brent_wti_spread(brent: pd.DataFrame, wti: pd.DataFrame) -> pd.DataFrame:
    """Считает три фичи спреда Brent − WTI.

    Возвращает DataFrame с колонками:
      spread_brent_wti - абсолютный спред в долларах,
      spread_rel       - спред в долях средней из двух цен (нормировка),
      spread_z21       - z-score спреда относительно скользящего месяца.
    """
    b = brent[["Close"]].rename(columns={"Close": "brent_close"})
    w = wti[["Close"]].rename(columns={"Close": "wti_close"})
    df = b.join(w, how="inner")
    # абсолютный спред
    df["spread_brent_wti"] = df["brent_close"] - df["wti_close"]
    # относительный - чтобы можно было сравнивать значения за разные годы
    df["spread_rel"] = df["spread_brent_wti"] / ((df["brent_close"] + df["wti_close"]) / 2)
    # z-score за 21 день - насколько текущий спред необычен
    rolling_mean = df["spread_brent_wti"].rolling(21).mean()
    rolling_std = df["spread_brent_wti"].rolling(21).std().replace(0, np.nan)
    df["spread_z21"] = (df["spread_brent_wti"] - rolling_mean) / rolling_std
    return df[["spread_brent_wti", "spread_rel", "spread_z21"]]


CROSS_ASSET_OIL_COLS: tuple[str, ...] = ("spread_brent_wti", "spread_rel", "spread_z21")
