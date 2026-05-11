"""Загрузка сериализованных моделей и live-инференс по тикеру."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from service.config import get_settings
from src.features.labels import build_target_frame
from src.features.technical import build_price_features
from src.features.text_signals import daily_gdelt_signals
from src.utils.config import PROJECT_ROOT, get_data_config


@dataclass
class LoadedModel:
    """Контейнер с одним обученным пайплайном и его метаданными."""

    asset: str
    pipeline: Any
    features: list[str]
    classes: list[str]
    meta: dict


class ModelRegistry:
    """Ленивый кэш моделей: первый get(asset) подгружает .joblib и .meta.json."""

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self._cache: dict[str, LoadedModel] = {}

    def get(self, asset: str) -> LoadedModel:
        if asset not in self._cache:
            self._load(asset)
        return self._cache[asset]

    def _load(self, asset: str) -> None:
        model_path = self.models_dir / f"{asset}.joblib"
        meta_path = self.models_dir / f"{asset}.meta.json"
        if not (model_path.exists() and meta_path.exists()):
            raise FileNotFoundError(
                f"Не найдены файлы модели для актива '{asset}'. "
                f"Запустите: python -m service.train_and_save --asset {asset}"
            )
        pipeline = joblib.load(model_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._cache[asset] = LoadedModel(
            asset=asset,
            pipeline=pipeline,
            features=meta["features"],
            classes=meta["classes"],
            meta=meta,
        )

    def available_assets(self) -> list[str]:
        """Список активов, для которых есть joblib-файл в каталоге моделей."""
        return [p.stem for p in self.models_dir.glob("*.joblib")]


# держим registry как процесс-локальный синглтон
_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry(get_settings().models_dir)
    return _registry


# --- Инференс на готовом векторе фичей ----------------------------------

def _vector_from_dict(features: dict[str, float], required: list[str]) -> np.ndarray:
    """Собирает фичи из dict в массив строго в том порядке, в котором обучалась модель.

    Если какой-то фичи нет - кидаем ValueError, чтобы handler вернул 403.
    """
    missing = [f for f in required if f not in features]
    if missing:
        raise ValueError(f"Не хватает признаков: {missing}")
    return np.array([[features[f] for f in required]], dtype=float)


def predict_from_features(asset: str, features: dict[str, float]) -> dict:
    """Прогноз на готовом векторе фичей. Возвращает dict под ForwardResponse."""
    loaded = get_registry().get(asset)
    X = _vector_from_dict(features, loaded.features)
    pred = loaded.pipeline.predict(X)[0]
    proba = loaded.pipeline.predict_proba(X)[0]
    return {
        "asset": asset,
        "prediction": str(pred),
        "probabilities": {cls: float(p) for cls, p in zip(loaded.classes, proba)},
        "model_version": loaded.meta.get("trained_at_utc", "unknown"),
    }


# --- Live-инференс по тикеру -------------------------------------------

# для какого актива какие тематические GDELT-сигналы подключать
TOPIC_OF = {"brent": "oil", "wti": "oil", "btc": "btc"}


def _fetch_recent_prices(ticker: str, lookback_days: int = 120) -> pd.DataFrame:
    """Скачивает последние ~lookback_days дневных свечей с Yahoo Finance.

    120 дней с запасом покрывают самое широкое окно из наших фичей (MACD 26).
    """
    end = date.today()
    # +30 дней про запас под выходные/гэпы, иначе при коротких праздниках можем недотянуть
    start = end - timedelta(days=lookback_days + 30)
    df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(),
                     interval="1d", auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance вернул пусто для {ticker}")
    # yfinance возвращает MultiIndex колонки, если запросов было несколько - выпрямим
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def _attach_recent_gdelt(df: pd.DataFrame, topic: str) -> pd.DataFrame:
    """Подмешивает GDELT-сигналы с диска, если есть. Иначе возвращает df как есть.

    Live-сбор GDELT при каждом запросе не делаем - rate-limit. Берём то, что
    собрано офлайновым скриптом src/data_collection/fetch_gdelt.py. На самые свежие даты сигналов
    может не быть, но это нормально.
    """
    cfg = get_data_config()
    gdelt_dir = PROJECT_ROOT / cfg["texts_storage"]["raw_dir"] / "gdelt"
    vol_path = gdelt_dir / f"gdelt_{topic}_TimelineVolRaw.parquet"
    tone_path = gdelt_dir / f"gdelt_{topic}_TimelineTone.parquet"
    if not (vol_path.exists() and tone_path.exists()):
        return df
    sig = daily_gdelt_signals(pd.read_parquet(vol_path), pd.read_parquet(tone_path))
    sig.index = pd.to_datetime(sig.index).tz_localize(None)
    out = df.copy()
    # нормализуем обе оси к датам без часов, иначе join промахнётся
    out.index = pd.to_datetime(out.index).normalize()
    sig.index = pd.to_datetime(sig.index).normalize()
    return out.join(sig, how="left")


def predict_live(asset: str) -> dict:
    """Полный live-инференс: цены -> фичи -> модель -> прогноз на следующий день."""
    cfg = get_data_config()
    loaded = get_registry().get(asset)
    ticker = cfg["assets"][asset]["ticker"]

    prices = _fetch_recent_prices(ticker)
    labeled = build_target_frame(
        prices,
        horizon=cfg["target"]["horizon_days"],
        flat_threshold_sigma=cfg["target"]["flat_threshold_sigma"],
    )
    feats = build_price_features(labeled)
    feats = _attach_recent_gdelt(feats, TOPIC_OF[asset])

    # ffill текстовых фичей до недели назад: GDELT-поток меняется не каждый
    # день, и пары последних дней могут быть NaN. На train так делать нельзя
    # (исказит распределение), а на inference это разумно - просто берём
    # последний известный sentiment.
    text_cols = [c for c in ("article_count", "article_norm", "avg_tone") if c in feats.columns]
    if text_cols:
        feats[text_cols] = feats[text_cols].ffill(limit=7)

    # последний день, на котором есть весь нужный модели вектор - это и есть
    # точка инференса (прогноз делается на следующий торговый день)
    needed = loaded.features
    available = feats[needed].dropna()
    if available.empty:
        raise RuntimeError("Не удалось собрать полный вектор признаков из live-данных")
    row = available.iloc[[-1]]
    as_of_ts = row.index[-1]
    last_close = float(feats.loc[as_of_ts, "Close"])

    pred = loaded.pipeline.predict(row.values)[0]
    proba = loaded.pipeline.predict_proba(row.values)[0]
    return {
        "asset": asset,
        "prediction": str(pred),
        "probabilities": {cls: float(p) for cls, p in zip(loaded.classes, proba)},
        "model_version": loaded.meta.get("trained_at_utc", "unknown"),
        "as_of": as_of_ts.to_pydatetime(),
        "last_close": last_close,
    }
