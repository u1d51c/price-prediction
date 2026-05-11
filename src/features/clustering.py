"""Кластерные фичи: номер режима рынка как дополнительный признак.

K-means на двух осях - реализованная волатильность и z-score объёма
новостей - даёт метку режима (например, "спокойно", "буря",
"волатильность без новостей"). Древесные модели плохо ловят такие
нелинейные пересечения сами, поэтому полезно подсказать им явно.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def fit_regime_clusters(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int = 3,
    random_state: int = 42,
) -> tuple[KMeans, StandardScaler]:
    """Обучает k-means только на train и возвращает (модель, скейлер).

    Скейлер обязателен: без него k-means схватится за фичу с большим масштабом.
    """
    work = df_train[feature_cols].dropna()
    scaler = StandardScaler().fit(work.values)
    X = scaler.transform(work.values)
    # n_init=10 - несколько запусков с разной инициализацией, берём лучший
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit(X)
    return km, scaler


def assign_regime(
    df: pd.DataFrame,
    feature_cols: list[str],
    km: KMeans,
    scaler: StandardScaler,
) -> pd.Series:
    """Применяет уже обученные km и scaler к df, отдаёт колонку 'regime'.

    Строки с NaN в фичах получают NaN в результате - не пытаемся гадать.
    """
    out = pd.Series(np.nan, index=df.index, dtype="float64", name="regime")
    sub = df[feature_cols].dropna()
    if sub.empty:
        return out
    X = scaler.transform(sub.values)
    labels = km.predict(X)
    out.loc[sub.index] = labels
    return out
