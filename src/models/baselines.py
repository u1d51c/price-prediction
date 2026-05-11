"""Baseline-классификаторы: тривиальные + ARIMA по доходности.

Любая модель должна сначала пройти проверку на трёх простых эталонах:
угадывание мажоритарного класса, случайный прогноз по частотам, повторение
вчерашней метки. Плюс ARIMA на лог-доходностях с дискретизацией знака -
это эконометрический аналог логистической регрессии на лагах.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.models.metrics import CLASSES


class MajorityClassifier:
    """Всегда возвращает класс, который чаще встречается в train."""

    def __init__(self) -> None:
        self.cls_: str | None = None

    def fit(self, X, y: pd.Series) -> "MajorityClassifier":
        self.cls_ = pd.Series(y).value_counts().idxmax()
        return self

    def predict(self, X) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        return np.array([self.cls_] * n)


class StratifiedRandomClassifier:
    """Случайный класс с вероятностью пропорционально его частоте в train."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.rng_: np.random.Generator | None = None
        self.classes_: list[str] = []
        self.probs_: np.ndarray | None = None

    def fit(self, X, y: pd.Series) -> "StratifiedRandomClassifier":
        counts = pd.Series(y).value_counts(normalize=True)
        # фиксированный порядок CLASSES - для воспроизводимости random.choice
        self.classes_ = list(CLASSES)
        self.probs_ = np.array([counts.get(c, 0.0) for c in self.classes_])
        # если какого-то класса в train не было, нормируем оставшиеся, чтобы сумма = 1
        s = self.probs_.sum()
        self.probs_ = self.probs_ / s if s > 0 else np.ones_like(self.probs_) / len(self.probs_)
        self.rng_ = np.random.default_rng(self.random_state)
        return self

    def predict(self, X) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        return self.rng_.choice(self.classes_, size=n, p=self.probs_)


class PersistenceClassifier:
    """Прогноз = метка предыдущего дня (или мажоритарный класс на первый день)."""

    def __init__(self) -> None:
        self.last_train_label_: str | None = None
        self.fallback_: str | None = None

    def fit(self, X, y: pd.Series) -> "PersistenceClassifier":
        y = pd.Series(y).dropna()
        # запоминаем последнюю метку train, чтобы взять её для первого дня test
        self.last_train_label_ = str(y.iloc[-1]) if len(y) else "flat"
        self.fallback_ = y.value_counts().idxmax() if len(y) else "flat"
        return self

    def predict_with_y_test(self, y_test: pd.Series) -> np.ndarray:
        """Predict, который смотрит в фактическую метку прошлого дня test.

        Стандартный sklearn-интерфейс .predict(X) сюда не подходит - нам нужны
        предыдущие настоящие метки, поэтому выделил отдельный метод.
        """
        out: list[str] = []
        prev = self.last_train_label_ or self.fallback_ or "flat"
        for label in y_test:
            out.append(prev)
            if isinstance(label, str):
                prev = label
        return np.array(out)


class ARIMAClassifier:
    """SARIMA по лог-доходности с дискретизацией прогноза в класс по порогу tau.

    Параметры (p, d, q) по умолчанию (1, 0, 1): d=0, потому что доходности
    уже стационарны (это разность логарифмов цен).

    Важная деталь: для оценки на test используется rolling_predict() - 1-step
    walk-forward. Многошаговый прогноз ARIMA быстро уходит в безусловное
    среднее, и тогда все метки на test получаются одинаковыми.
    """

    def __init__(self, order: tuple[int, int, int] = (1, 0, 1), tau: float | None = None) -> None:
        self.order = order
        self.tau = tau
        self.results_ = None

    def fit(self, log_returns_train: pd.Series, tau: float | None = None) -> "ARIMAClassifier":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        if tau is not None:
            self.tau = tau
        series = pd.Series(log_returns_train).dropna().astype(float).values
        self.train_series_ = series
        # enforce_*=False - пусть solver сам найдёт коэффициенты, иначе SARIMAX
        # часто падает на reject-проверке стационарности на дневных доходностях.
        self.results_ = SARIMAX(
            series, order=self.order,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        return self

    def rolling_predict(self, log_returns_test: pd.Series) -> np.ndarray:
        """1-step прогноз для каждого дня test: forecast -> знак -> метка.

        После каждого шага дописываем фактическую точку в state Калман-фильтра
        через .append(refit=False) - это быстро (без переобучения параметров).
        """
        if self.results_ is None:
            raise RuntimeError("ARIMAClassifier не обучен")
        results = self.results_
        tau = self.tau if self.tau is not None else 0.0
        out: list[str] = []
        test_vals = pd.Series(log_returns_test).astype(float).values

        for actual in test_vals:
            fc = float(results.forecast(steps=1)[0])
            # дискретизация по тому же tau, что использовался при разметке таргета
            out.append("up" if fc > tau else "down" if fc < -tau else "flat")
            # дописываем фактическое наблюдение в state, чтобы следующий forecast его учитывал
            results = results.append([actual], refit=False)
        return np.array(out)

    def predict(self, n: int) -> np.ndarray:
        """Многошаговый прогноз - оставлен для совместимости.

        Использовать только для диагностики: на длинных горизонтах все метки
        схлопнутся в один класс (forecast сходится к безусловному среднему).
        """
        if self.results_ is None:
            raise RuntimeError("ARIMAClassifier не обучен")
        forecast = self.results_.get_forecast(steps=n).predicted_mean
        forecast = np.asarray(forecast)
        tau = self.tau if self.tau is not None else 0.0
        return np.where(forecast > tau, "up",
                np.where(forecast < -tau, "down", "flat"))


def predict_persistence(y_train: pd.Series, y_test: pd.Series) -> np.ndarray:
    """Шорткат: создать PersistenceClassifier и сразу получить прогнозы."""
    clf = PersistenceClassifier().fit(None, y_train)
    return clf.predict_with_y_test(y_test)
