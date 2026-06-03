"""Загружает PRD-модель из MLflow и делает тестовый predict.

Запуск:
    cd year_project
    PYTHONPATH=. python -m src.inference.predict_cli asset=btc last_n_days=5
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def _find_prd_version(client, model_name: str, tag_key: str, tag_value: str):
    """Перебирает версии модели и возвращает ту, у которой нужный тег."""
    for v in client.search_model_versions(f"name='{model_name}'"):
        tags = v.tags or {}
        if tags.get(tag_key) == tag_value:
            return v
    return None


@hydra.main(config_path="../../configs/inference", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    import mlflow
    import pandas as pd

    from src.features.labels import build_target_frame
    from src.features.technical import build_price_features, PRICE_FEATURE_COLS
    from src.features.text_signals import daily_gdelt_signals
    from src.features.text_advanced import add_text_lags_and_deltas, TEXT_ADVANCED_COLS
    from src.features.cross_asset import brent_wti_spread, CROSS_ASSET_OIL_COLS
    from src.features.clustering import fit_regime_clusters, assign_regime
    from src.utils.config import get_data_config, PROJECT_ROOT
    from src.utils.logging_setup import get_logger
    from src.utils.mlflow_setup import setup_mlflow

    logger = get_logger("predict")

    setup_mlflow()
    client = mlflow.tracking.MlflowClient()

    # ищем версию модели с PRD-тегом
    model_name = cfg.model_name
    prd = _find_prd_version(client, model_name, cfg.prd_tag_key, cfg.prd_tag_value)
    if prd is None:
        raise RuntimeError(
            f"В MLflow Registry нет версии {model_name} с тегом {cfg.prd_tag_key}={cfg.prd_tag_value}"
        )
    logger.info("PRD версия: %s/v%s, run_id=%s", model_name, prd.version, prd.run_id)

    # загружаем модель и meta.json из артефактов того же run-а
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{prd.version}")
    meta_local = client.download_artifacts(prd.run_id, "model/meta.json", "/tmp")
    meta = json.loads(Path(meta_local).read_text(encoding="utf-8"))
    feat_cols = meta["features"]
    logger.info("Загружена модель: features=%d, classes=%s",
                len(feat_cols), meta["classes"])

    # подготавливаем последний срез данных, как при обучении
    asset = cfg.asset
    topic = {"brent": "oil", "wti": "oil", "btc": "btc"}[asset]
    data_cfg = get_data_config()
    target_cfg = data_cfg["target"]

    def _safe(t): return t.replace("=", "_").replace("/", "_")
    raw = pd.read_parquet(
        PROJECT_ROOT / data_cfg["prices"]["cache_dir"] /
        f"{_safe(data_cfg['assets'][asset]['ticker'])}.parquet"
    )
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    labeled = build_target_frame(
        raw,
        horizon=target_cfg["horizon_days"],
        flat_threshold_sigma=target_cfg["flat_threshold_sigma"],
    )
    feats = build_price_features(labeled)

    gdir = PROJECT_ROOT / data_cfg["texts_storage"]["raw_dir"] / "gdelt"
    vol_p = gdir / f"gdelt_{topic}_TimelineVolRaw.parquet"
    tone_p = gdir / f"gdelt_{topic}_TimelineTone.parquet"
    if vol_p.exists() and tone_p.exists():
        sig = daily_gdelt_signals(pd.read_parquet(vol_p), pd.read_parquet(tone_p))
        sig.index = pd.to_datetime(sig.index).tz_localize(None).normalize()
        feats.index = pd.to_datetime(feats.index).normalize()
        feats = feats.join(sig, how="left")

    feats = add_text_lags_and_deltas(feats)

    if asset in ("brent", "wti"):
        brent_raw = pd.read_parquet(PROJECT_ROOT / data_cfg["prices"]["cache_dir"] / "BZ_F.parquet")
        wti_raw = pd.read_parquet(PROJECT_ROOT / data_cfg["prices"]["cache_dir"] / "CL_F.parquet")
        brent_raw.index = pd.to_datetime(brent_raw.index).tz_localize(None)
        wti_raw.index = pd.to_datetime(wti_raw.index).tz_localize(None)
        spread = brent_wti_spread(brent_raw, wti_raw)
        spread.index = pd.to_datetime(spread.index).normalize()
        feats.index = pd.to_datetime(feats.index).normalize()
        feats = feats.join(spread, how="left")

    # кластерный режим: используем train (до split_date из meta) для fit
    if {"rv_21", "article_count_z21"}.issubset(feats.columns):
        split_ts = pd.Timestamp(meta["split_date"])
        train_part = feats.loc[feats.index < split_ts]
        km, sc = fit_regime_clusters(train_part, ["rv_21", "article_count_z21"],
                                     n_clusters=3, random_state=meta["seed"])
        feats["regime"] = assign_regime(feats, ["rv_21", "article_count_z21"], km, sc)

    # FinBERT подмешиваем, если был в обучении
    for suffix, prefix in (("title", "title"), ("body", "body")):
        path = PROJECT_ROOT / "data/processed" / f"finbert_{topic}_{suffix}_daily.parquet"
        if path.exists() and any(c.startswith(f"{prefix}_finbert_") for c in feat_cols):
            fb = pd.read_parquet(path)
            fb.index = pd.to_datetime(fb.index).tz_localize(None).normalize()
            feats = feats.join(fb.add_prefix(f"{prefix}_"), how="left")

    X = feats[feat_cols].dropna().tail(cfg.last_n_days)
    if X.empty:
        raise RuntimeError("Нет полного вектора фичей для последних дней")

    y_pred = model.predict(X)
    proba = model.predict_proba(X)

    print(f"\n=== PRD-предикт по {asset} (последние {len(X)} дн) ===\n")
    for date, pred, p in zip(X.index, y_pred, proba):
        probs_str = ", ".join(f"{c}={pr:.2f}" for c, pr in zip(model.classes_, p))
        print(f"  {date.date()}  →  {pred:5s}  [{probs_str}]")
    print()


if __name__ == "__main__":
    main()
