"""Признаки на основе OHLCV-цены: лаги, волатильность, индикаторы, объём.

Без TA-Lib (чтобы не тащить нативную либу) - всё через pandas.
Формулы стандартные: RSI и ATR по Wilder, MACD по Appel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --- Доходности и лаги ---------------------------------------------------

def add_returns(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """Добавляет лог- и процентную доходности за один день."""
    out = df.copy()
    # отрицательные цены WTI 2020 -> NaN, иначе log упадёт
    safe = out[price_col].where(out[price_col] > 0)
    out["log_return_t"] = np.log(safe / safe.shift(1))
    out["pct_return_t"] = safe.pct_change(fill_method=None)
    return out


def add_return_lags(df: pd.DataFrame, lags: tuple[int, ...] = (1, 2, 3, 5, 10)) -> pd.DataFrame:
    """Лаги доходностей: r_lag_k = r_{t-k}.

    Берём короткие лаги - после 10 на дневных данных уже один шум.
    """
    out = df.copy()
    for k in lags:
        out[f"r_lag_{k}"] = out["log_return_t"].shift(k)
    return out


# --- Волатильность -------------------------------------------------------

def add_rolling_vol(df: pd.DataFrame, windows: tuple[int, ...] = (5, 10, 21)) -> pd.DataFrame:
    """Скользящая сигма доходности и среднее |r| - две прокси волатильности.

    Окно 21 ~= торговый месяц. 5 и 10 ловят более короткие режимы.
    """
    out = df.copy()
    for w in windows:
        out[f"rv_{w}"] = out["log_return_t"].rolling(w).std()
        out[f"abs_ret_ma_{w}"] = out["log_return_t"].abs().rolling(w).mean()
    return out


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """ATR по Уайлдеру.

    True Range = max(H-L, |H-prev_C|, |L-prev_C|), потом сглаживаем EMA
    с alpha = 1/window. Это устойчивее обычного стандартного отклонения,
    так как учитывает гэпы между днями.
    """
    out = df.copy()
    high, low, prev_close = out["High"], out["Low"], out["Close"].shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    out[f"atr_{window}"] = tr.ewm(alpha=1 / window, adjust=False).mean()
    # относительный ATR - нормируем на цену, чтобы можно было сравнивать активы
    out[f"atr_rel_{window}"] = out[f"atr_{window}"] / out["Close"]
    return out


# --- Тренд и моментум -----------------------------------------------------

def add_rsi(df: pd.DataFrame, window: int = 14, price_col: str = "Close") -> pd.DataFrame:
    """RSI по Уайлдеру: RSI = 100 - 100/(1 + avg_gain/avg_loss)."""
    out = df.copy()
    delta = out[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder smoothing = EMA с alpha=1/window
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out[f"rsi_{window}"] = 100 - 100 / (1 + rs)
    return out


def add_macd(
    df: pd.DataFrame,
    fast: int = 12, slow: int = 26, signal: int = 9,
    price_col: str = "Close",
) -> pd.DataFrame:
    """MACD = EMA(12) - EMA(26), сигнальная линия - EMA(9) от MACD."""
    out = df.copy()
    ema_fast = out[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = out[price_col].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = macd_line - signal_line
    # делим на цену, чтобы MACD у активов с разными порядками был сопоставим
    out["macd_rel"] = macd_line / out[price_col]
    return out


def add_bollinger(
    df: pd.DataFrame, window: int = 20, k: float = 2.0, price_col: str = "Close",
) -> pd.DataFrame:
    """Bollinger Bands + percent-B (нормированная позиция внутри коридора)."""
    out = df.copy()
    ma = out[price_col].rolling(window).mean()
    sd = out[price_col].rolling(window).std()
    upper = ma + k * sd
    lower = ma - k * sd
    # %B = где цена относительно полосы: 0 - на нижней, 1 - на верхней
    out[f"bb_pct_{window}"] = (out[price_col] - lower) / (upper - lower).replace(0, np.nan)
    # ширина полосы в долях цены - прокси волатильности
    out[f"bb_width_{window}"] = (upper - lower) / ma
    return out


# --- Объём ---------------------------------------------------------------

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Лог-объём + объём относительно 20-дневного среднего."""
    out = df.copy()
    vol = out["Volume"].replace(0, np.nan)
    # лог, чтобы убрать тяжёлые хвосты - у объёма распределение почти log-normal
    out["log_volume"] = np.log(vol)
    out["volume_rel_20"] = vol / vol.rolling(20).mean()
    return out


# --- Календарь -----------------------------------------------------------

def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """День недели и месяц - слабый, но дешёвый признак."""
    out = df.copy()
    idx = pd.to_datetime(out.index)
    out["dow"] = idx.dayofweek
    out["month"] = idx.month
    return out


# --- Сборка всего сразу ---------------------------------------------------

def build_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Прогоняет весь набор функций в одной цепочке.

    NaN от роллинг-окон не убираем - пусть потребитель сам решит, какой
    drop_na делать (он зависит от ширины окна используемых фичей).
    """
    out = add_returns(df)
    out = add_return_lags(out)
    out = add_rolling_vol(out)
    out = add_atr(out)
    out = add_rsi(out)
    out = add_macd(out)
    out = add_bollinger(out)
    out = add_volume_features(out)
    out = add_calendar(out)
    return out


# Имена всех ценовых фичей - удобно для feature-set ablation в ноутбуках
PRICE_FEATURE_COLS: tuple[str, ...] = (
    "r_lag_1", "r_lag_2", "r_lag_3", "r_lag_5", "r_lag_10",
    "rv_5", "rv_10", "rv_21",
    "abs_ret_ma_5", "abs_ret_ma_10", "abs_ret_ma_21",
    "atr_14", "atr_rel_14",
    "rsi_14",
    "macd", "macd_signal", "macd_hist", "macd_rel",
    "bb_pct_20", "bb_width_20",
    "log_volume", "volume_rel_20",
    "dow", "month",
)
