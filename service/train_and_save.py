"""Обучает модели и сохраняет в models/<asset>.joblib для использования сервисом.

Сначала оцениваем качество на hold-out (после 2024-01-01) - это идёт в
meta.json. Потом обучаем заново на всей доступной истории и сохраняем
итоговый pipeline.

Запуск:
    python -m service.train_and_save
    python -m service.train_and_save --asset btc
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.labels import build_target_frame
from src.features.technical import build_price_features, PRICE_FEATURE_COLS
from src.features.text_signals import daily_gdelt_signals
from src.models.metrics import score
from src.utils.config import PROJECT_ROOT, get_data_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
TOPIC_OF = {"brent": "oil", "wti": "oil", "btc": "btc"}


def _safe_filename(ticker: str) -> str:
    """Тикер вроде BZ=F нельзя класть в имя файла как есть."""
    return ticker.replace("=", "_").replace("/", "_")


def build_dataset(asset_key: str, cfg: dict) -> tuple[pd.DataFrame, float]:
    """Цены -> таргет -> технические фичи -> джойн GDELT (если есть)."""
    asset = cfg["assets"][asset_key]
    prices_dir = PROJECT_ROOT / cfg["prices"]["cache_dir"]
    raw = pd.read_parquet(prices_dir / f"{_safe_filename(asset['ticker'])}.parquet")
    raw.index = pd.to_datetime(raw.index).tz_localize(None)

    labeled = build_target_frame(
        raw, price_col="Close",
        horizon=cfg["target"]["horizon_days"],
        flat_threshold_sigma=cfg["target"]["flat_threshold_sigma"],
    )
    tau = labeled.attrs["flat_threshold"]
    feats = build_price_features(labeled)

    # GDELT подмешиваем, только если оба файла на диске (volume + tone)
    topic = TOPIC_OF[asset_key]
    gdelt_dir = PROJECT_ROOT / cfg["texts_storage"]["raw_dir"] / "gdelt"
    vol_path = gdelt_dir / f"gdelt_{topic}_TimelineVolRaw.parquet"
    tone_path = gdelt_dir / f"gdelt_{topic}_TimelineTone.parquet"
    if vol_path.exists() and tone_path.exists():
        sig = daily_gdelt_signals(pd.read_parquet(vol_path), pd.read_parquet(tone_path))
        sig.index = pd.to_datetime(sig.index).tz_localize(None)
        feats.index = pd.to_datetime(feats.index).normalize()
        sig.index = pd.to_datetime(sig.index).normalize()
        feats = feats.join(sig, how="left")

    return feats, tau


def make_pipeline() -> Pipeline:
    """Production-пайплайн: StandardScaler + KNN k=51 (лучшее из baseline-моделей)."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=51, weights="distance", n_jobs=-1)),
    ])


def train_one(asset_key: str, holdout_date: str = "2024-01-01") -> dict:
    """Полный цикл для одного актива: оценка на hold-out + итоговый fit + сохранение."""
    cfg = get_data_config()
    df, tau = build_dataset(asset_key, cfg)

    # Список фичей: базовые ценовые + GDELT, если он есть и не пустой
    base = list(PRICE_FEATURE_COLS)
    text = ["article_count", "article_norm", "avg_tone"]
    if all(c in df.columns and df[c].notna().any() for c in text):
        features = base + text
    else:
        features = base
        logger.warning("[%s] GDELT-фичи недоступны, обучаем только на цене.", asset_key)

    work = df[features + ["target"]].dropna()
    holdout_ts = pd.Timestamp(holdout_date)
    train_part = work.loc[work.index < holdout_ts]
    test_part = work.loc[work.index >= holdout_ts]

    if len(train_part) < 200:
        # 200 - порог, ниже которого учить вообще нет смысла
        raise RuntimeError(f"[{asset_key}] слишком мало данных: {len(train_part)} строк в train")

    # шаг 1: учим только на train и оцениваем на hold-out - это пойдёт в meta
    pipe_eval = make_pipeline()
    pipe_eval.fit(train_part[features], train_part["target"])
    eval_score = score(test_part["target"], pipe_eval.predict(test_part[features]),
                       name=f"{asset_key}/holdout")
    logger.info("[%s] hold-out macro-F1 = %.4f", asset_key, eval_score.macro_f1)

    # шаг 2: итоговый fit на ВСЕЙ истории - для продакшена хотим максимум данных
    pipe_final = make_pipeline()
    pipe_final.fit(work[features], work["target"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{asset_key}.joblib"
    meta_path = MODELS_DIR / f"{asset_key}.meta.json"

    # Всё, что нужно сервису для inference - лежит в meta рядом с моделью
    meta = {
        "asset": asset_key,
        "ticker": cfg["assets"][asset_key]["ticker"],
        "features": features,
        "classes": list(pipe_final.classes_),
        "tau": tau,
        "horizon_days": cfg["target"]["horizon_days"],
        "train_start": work.index.min().isoformat(),
        "train_end": work.index.max().isoformat(),
        "n_train": int(len(work)),
        "holdout_macro_f1": float(eval_score.macro_f1),
        "holdout_accuracy": float(eval_score.accuracy),
        "holdout_balanced_accuracy": float(eval_score.balanced_accuracy),
        "trained_at_utc": datetime.utcnow().isoformat() + "Z",
        "model_class": "sklearn.Pipeline(StandardScaler+KNN k=51 distance)",
    }
    joblib.dump(pipe_final, model_path, compress=3)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("[%s] сохранено -> %s (%d признаков, %d строк train)",
                asset_key, model_path.relative_to(PROJECT_ROOT), len(features), len(work))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучить и сохранить production-модели")
    parser.add_argument("--asset", default="all", choices=["all", "brent", "wti", "btc"])
    parser.add_argument("--holdout-date", default="2024-01-01")
    args = parser.parse_args()

    cfg = get_data_config()
    assets = list(cfg["assets"].keys()) if args.asset == "all" else [args.asset]
    for k in assets:
        try:
            train_one(k, args.holdout_date)
        except Exception as exc:
            # Один сломанный актив не должен ронять обучение остальных
            logger.error("[%s] не удалось обучить: %s", k, exc)


if __name__ == "__main__":
    main()
