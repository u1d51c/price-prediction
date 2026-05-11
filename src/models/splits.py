"""Разбиения временных рядов на train/test без утечки во времени.

Обычный KFold на финансовых данных приведёт к тому, что модель будет учиться
на завтрашних данных и тестироваться на вчерашних - это data leakage. Здесь
два варианта корректного разбиения: одиночный по дате и walk-forward.
"""

from __future__ import annotations

from typing import Iterator

import pandas as pd


def time_train_test_split(
    df: pd.DataFrame,
    split_date: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Делит df по дате: всё до split_date - train, остальное - test."""
    split_ts = pd.Timestamp(split_date)
    train = df.loc[df.index < split_ts].copy()
    test = df.loc[df.index >= split_ts].copy()
    return train, test


def expanding_window_splits(
    df: pd.DataFrame,
    initial_train_size: int,
    test_size: int,
    step: int | None = None,
    max_folds: int | None = None,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Генератор walk-forward фолдов: train растёт, test фиксированной длины.

    Параметры:
      initial_train_size - длина стартового train в строках,
      test_size - длина каждого test-окна,
      step - на сколько строк сдвигаемся между фолдами (по умолчанию = test_size),
      max_folds - ограничение на число фолдов (если хочется быстрее).

    Так модель оценивается так, как реально работала бы в проде:
    каждый раз ей доступна вся история до текущего момента, прогноз - на ближайшее окно.
    """
    if step is None:
        step = test_size

    n = len(df)
    start_test = initial_train_size
    fold = 0
    while start_test + test_size <= n:
        train = df.iloc[:start_test]
        test = df.iloc[start_test:start_test + test_size]
        yield train, test
        fold += 1
        if max_folds is not None and fold >= max_folds:
            return
        # сдвигаемся вперёд на step строк; обычно step == test_size
        start_test += step
