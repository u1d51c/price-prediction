"""Общая логика inference: загрузка PRD-модели из MLflow + сборка фичей.

Используется и CLI (predict_cli.py), и FastAPI-сервисом. Гарантирует, что
набор признаков при инференсе ровно тот же, что при обучении (train_with_mlflow.py).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TOPIC_OF = {"brent": "oil", "wti": "oil", "btc": "btc"}


def _find_prd_version(client, model_name: str):
    """Возвращает версию модели с тегом stage=PRD или None."""
    for v in client.search_model_versions(f"name='{model_name}'"):
        if (v.tags or {}).get("stage") == "PRD":
            return v
    return None


@lru_cache(maxsize=8)
def load_prd(asset: str) -> tuple[Any, dict]:
    """Грузит PRD-модель и meta из MLflow Registry. Кэшируется по активу."""
    import mlflow
    from src.utils.mlflow_setup import setup_mlflow

    setup_mlflow()
    client = mlflow.tracking.MlflowClient()
    model_name = f"{asset}_direction_model"
    prd = _find_prd_version(client, model_name)
    if prd is None:
        raise FileNotFoundError(f"Нет PRD-версии модели {model_name} в MLflow")
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{prd.version}")
    meta_path = client.download_artifacts(prd.run_id, "model/meta.json", "/tmp")
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    meta["_version"] = prd.version
    meta["_run_id"] = prd.run_id
    return model, meta


def build_features(asset: str, meta: dict) -> pd.DataFrame:
    """Собирает полный фичесет для актива ровно как в обучении.

    Повторяет шаги train_with_mlflow.py: цены -> таргет -> технические ->
    GDELT -> лаги/дельты -> cross-asset (нефть) -> регим -> FinBERT (если в фичах).
    """
    from src.features.labels import build_target_frame
    from src.features.technical import build_price_features
    from src.features.text_signals import daily_gdelt_signals
    from src.features.text_advanced import add_text_lags_and_deltas
    from src.features.cross_asset import brent_wti_spread
    from src.features.clustering import fit_regime_clusters, assign_regime
    from src.utils.config import get_data_config, PROJECT_ROOT

    cfg = get_data_config()
    target_cfg = cfg["target"]
    topic = TOPIC_OF[asset]
    feat_cols = meta["features"]

    def _safe(t):
        return t.replace("=", "_").replace("/", "_")

    raw = pd.read_parquet(
        PROJECT_ROOT / cfg["prices"]["cache_dir"] /
        f"{_safe(cfg['assets'][asset]['ticker'])}.parquet"
    )
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    labeled = build_target_frame(raw, horizon=target_cfg["horizon_days"],
                                 flat_threshold_sigma=target_cfg["flat_threshold_sigma"])
    feats = build_price_features(labeled)

    gdir = PROJECT_ROOT / cfg["texts_storage"]["raw_dir"] / "gdelt"
    vol_p = gdir / f"gdelt_{topic}_TimelineVolRaw.parquet"
    tone_p = gdir / f"gdelt_{topic}_TimelineTone.parquet"
    if vol_p.exists() and tone_p.exists():
        sig = daily_gdelt_signals(pd.read_parquet(vol_p), pd.read_parquet(tone_p))
        sig.index = pd.to_datetime(sig.index).tz_localize(None).normalize()
        feats.index = pd.to_datetime(feats.index).normalize()
        feats = feats.join(sig, how="left")

    feats = add_text_lags_and_deltas(feats)

    if asset in ("brent", "wti"):
        b = pd.read_parquet(PROJECT_ROOT / cfg["prices"]["cache_dir"] / "BZ_F.parquet")
        w = pd.read_parquet(PROJECT_ROOT / cfg["prices"]["cache_dir"] / "CL_F.parquet")
        b.index = pd.to_datetime(b.index).tz_localize(None)
        w.index = pd.to_datetime(w.index).tz_localize(None)
        sp = brent_wti_spread(b, w)
        sp.index = pd.to_datetime(sp.index).normalize()
        feats.index = pd.to_datetime(feats.index).normalize()
        feats = feats.join(sp, how="left")

    if {"rv_21", "article_count_z21"}.issubset(feats.columns):
        split = pd.Timestamp(meta["split_date"])
        train_p = feats.loc[feats.index < split]
        km, sc = fit_regime_clusters(train_p, ["rv_21", "article_count_z21"],
                                     n_clusters=3, random_state=meta["seed"])
        feats["regime"] = assign_regime(feats, ["rv_21", "article_count_z21"], km, sc)

    # FinBERT из БД с ffill, если модель его использует
    need_title = any(c.startswith("title_finbert_") for c in feat_cols)
    need_body = any(c.startswith("body_finbert_") for c in feat_cols)
    if need_title or need_body:
        from src.storage.finbert_db import get_daily_aggregate

        def _ffill(fb, idx):
            if fb.empty:
                return fb
            return fb.reindex(fb.index.union(idx).sort_values()).ffill(limit=30).reindex(idx)

        if need_title:
            fb = get_daily_aggregate(topic, "title")
            if not fb.empty:
                fb.index = pd.to_datetime(fb.index).tz_localize(None).normalize()
                feats = feats.join(_ffill(fb, feats.index).add_prefix("title_"), how="left")
        if need_body:
            fb = get_daily_aggregate(topic, "body")
            if not fb.empty:
                fb.index = pd.to_datetime(fb.index).tz_localize(None).normalize()
                feats = feats.join(_ffill(fb, feats.index).add_prefix("body_"), how="left")

    return feats


def predict_latest(asset: str, n_days: int = 1) -> dict:
    """Прогноз PRD-модели на последние n_days дней с полным вектором фичей.

    Возвращает dict: asset, model_version, последние строки с prediction и
    вероятностями классов.
    """
    model, meta = load_prd(asset)
    feats = build_features(asset, meta)
    feat_cols = meta["features"]
    X = feats[feat_cols].dropna()
    if X.empty:
        raise RuntimeError("Не удалось собрать полный вектор фичей")
    X_recent = X.tail(n_days)
    pred = model.predict(X_recent)
    proba = model.predict_proba(X_recent)
    rows = []
    for date, p, pr in zip(X_recent.index, pred, proba):
        rows.append({
            "date": date.date().isoformat(),
            "prediction": str(p),
            "probabilities": {c: float(x) for c, x in zip(model.classes_, pr)},
        })
    return {
        "asset": asset,
        "model_version": str(meta.get("_version", "?")),
        "split_date": meta["split_date"],
        "n_features": len(feat_cols),
        "predictions": rows,
    }
