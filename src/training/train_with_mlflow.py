"""Переобучает выбранную модель с логированием в MLflow.

Запускается через Hydra-CLI:

    cd year_project
    PYTHONPATH=. python -m src.training.train_with_mlflow asset=btc

Что делается:
  1. Собираем датасет (цены + GDELT + опц. FinBERT) для выбранного актива.
  2. Разделяем train/test по configs/training/default.yaml split_date.
  3. Учим RandomForest на train, считаем метрики на train и test.
  4. В MLflow логируем: params, train+test metrics, confusion matrix (картинка),
     feature importances (csv), 5 примеров predict (csv) и сам pipeline.
  5. Регистрируем модель под именем asset_direction_model с тегом PRD.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


@hydra.main(config_path="../../configs/training", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    import io
    import joblib
    import matplotlib.pyplot as plt
    import mlflow
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
    )

    from src.features.labels import build_target_frame
    from src.features.technical import build_price_features, PRICE_FEATURE_COLS
    from src.features.text_signals import daily_gdelt_signals
    from src.features.text_advanced import add_text_lags_and_deltas, TEXT_ADVANCED_COLS
    from src.features.cross_asset import brent_wti_spread, CROSS_ASSET_OIL_COLS
    from src.features.clustering import fit_regime_clusters, assign_regime
    from src.utils.config import get_data_config, PROJECT_ROOT
    from src.utils.logging_setup import get_logger
    from src.utils.mlflow_setup import setup_mlflow

    logger = get_logger("train")
    logger.info("Конфиг:\n%s", OmegaConf.to_yaml(cfg))

    np.random.seed(cfg.seed)
    data_cfg = get_data_config()
    target_cfg = data_cfg["target"]
    asset = cfg.asset
    topic = {"brent": "oil", "wti": "oil", "btc": "btc"}[asset]

    # ---------- сборка датасета ----------
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
    tau = labeled.attrs["flat_threshold"]
    feats = build_price_features(labeled)

    # GDELT timeline
    gdir = PROJECT_ROOT / data_cfg["texts_storage"]["raw_dir"] / "gdelt"
    vol_p = gdir / f"gdelt_{topic}_TimelineVolRaw.parquet"
    tone_p = gdir / f"gdelt_{topic}_TimelineTone.parquet"
    if vol_p.exists() and tone_p.exists():
        sig = daily_gdelt_signals(pd.read_parquet(vol_p), pd.read_parquet(tone_p))
        sig.index = pd.to_datetime(sig.index).tz_localize(None).normalize()
        feats.index = pd.to_datetime(feats.index).normalize()
        feats = feats.join(sig, how="left")

    if cfg.feature_set.text_advanced:
        feats = add_text_lags_and_deltas(feats)

    if cfg.feature_set.cross_asset and asset in ("brent", "wti"):
        brent_raw = pd.read_parquet(PROJECT_ROOT / data_cfg["prices"]["cache_dir"] / "BZ_F.parquet")
        wti_raw = pd.read_parquet(PROJECT_ROOT / data_cfg["prices"]["cache_dir"] / "CL_F.parquet")
        brent_raw.index = pd.to_datetime(brent_raw.index).tz_localize(None)
        wti_raw.index = pd.to_datetime(wti_raw.index).tz_localize(None)
        spread = brent_wti_spread(brent_raw, wti_raw)
        spread.index = pd.to_datetime(spread.index).normalize()
        feats.index = pd.to_datetime(feats.index).normalize()
        feats = feats.join(spread, how="left")

    if cfg.feature_set.regime and {"rv_21", "article_count_z21"}.issubset(feats.columns):
        split_ts = pd.Timestamp(cfg.split_date)
        train_part = feats.loc[feats.index < split_ts]
        km, sc = fit_regime_clusters(train_part, ["rv_21", "article_count_z21"],
                                     n_clusters=3, random_state=cfg.seed)
        feats["regime"] = assign_regime(feats, ["rv_21", "article_count_z21"], km, sc)

    # FinBERT подмешиваем из SQLite (data/processed/finbert.db).
    # На дни без FinBERT делаем forward-fill в пределах 30 дней - sentiment
    # медленно меняющийся сигнал, и днём раньше можно считать представительным.
    # Так делаем и при live-inference (service/inference.py).
    if cfg.feature_set.finbert_title or cfg.feature_set.finbert_body:
        from src.storage.finbert_db import get_daily_aggregate

        def _ffill_finbert(fb: pd.DataFrame, full_index) -> pd.DataFrame:
            if fb.empty:
                return fb
            return fb.reindex(fb.index.union(full_index).sort_values()).ffill(limit=30).reindex(full_index)

        if cfg.feature_set.finbert_title:
            fb = get_daily_aggregate(topic, "title")
            if not fb.empty:
                fb.index = pd.to_datetime(fb.index).tz_localize(None).normalize()
                fb = _ffill_finbert(fb, feats.index)
                feats = feats.join(fb.add_prefix("title_"), how="left")
        if cfg.feature_set.finbert_body:
            fb = get_daily_aggregate(topic, "body")
            if not fb.empty:
                fb.index = pd.to_datetime(fb.index).tz_localize(None).normalize()
                fb = _ffill_finbert(fb, feats.index)
                feats = feats.join(fb.add_prefix("body_"), how="left")

    # формируем feature_cols по селекторам в конфиге
    feat_cols: list[str] = []
    if cfg.feature_set.base:
        feat_cols += list(PRICE_FEATURE_COLS)
    if cfg.feature_set.text_basic:
        feat_cols += ["article_count", "article_norm", "avg_tone"]
    if cfg.feature_set.text_advanced:
        feat_cols += list(TEXT_ADVANCED_COLS)
    if cfg.feature_set.cross_asset and asset in ("brent", "wti"):
        feat_cols += list(CROSS_ASSET_OIL_COLS)
    if cfg.feature_set.regime:
        feat_cols += ["regime"]
    if cfg.feature_set.finbert_title:
        feat_cols += [c for c in feats.columns if c.startswith("title_finbert_")]
    if cfg.feature_set.finbert_body:
        feat_cols += [c for c in feats.columns if c.startswith("body_finbert_")]

    feat_cols = [c for c in feat_cols if c in feats.columns]
    work = feats[feat_cols + ["target"]].dropna()
    split_ts = pd.Timestamp(cfg.split_date)
    X_train = work.loc[work.index < split_ts, feat_cols]
    y_train = work.loc[work.index < split_ts, "target"]
    X_test = work.loc[work.index >= split_ts, feat_cols]
    y_test = work.loc[work.index >= split_ts, "target"]

    logger.info("X_train: %s, X_test: %s, фичей: %d", X_train.shape, X_test.shape, len(feat_cols))

    # ---------- MLflow ----------
    setup_mlflow()
    with mlflow.start_run(run_name=cfg.mlflow.run_name) as run:
        # теги
        for k, v in cfg.mlflow.tags.items():
            mlflow.set_tag(k, str(v))
        mlflow.set_tag("asset", asset)

        # параметры
        mlflow.log_params({
            "asset": asset,
            "seed": cfg.seed,
            "split_date": cfg.split_date,
            "n_features": len(feat_cols),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "tau": float(tau),
            **{f"model_{k}": v for k, v in OmegaConf.to_container(cfg.model).items()},
            **{f"feat_{k}": v for k, v in OmegaConf.to_container(cfg.feature_set).items()},
        })

        # обучение
        model = RandomForestClassifier(
            n_estimators=cfg.model.n_estimators,
            max_depth=cfg.model.max_depth,
            min_samples_leaf=cfg.model.min_samples_leaf,
            class_weight=cfg.model.class_weight,
            random_state=cfg.seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # метрики на train
        y_tr_pred = model.predict(X_train)
        # метрики на test
        y_te_pred = model.predict(X_test)

        classes = list(model.classes_)
        for split_name, y_true, y_pred in [("train", y_train, y_tr_pred),
                                            ("test", y_test, y_te_pred)]:
            mlflow.log_metric(f"{split_name}_macro_f1",
                              f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))
            mlflow.log_metric(f"{split_name}_accuracy", accuracy_score(y_true, y_pred))
            mlflow.log_metric(f"{split_name}_balanced_accuracy",
                              balanced_accuracy_score(y_true, y_pred))
            for cls in classes:
                f1c = f1_score(y_true, y_pred, labels=[cls], average="macro", zero_division=0)
                mlflow.log_metric(f"{split_name}_f1_{cls}", f1c)

        # confusion matrix как картинка
        cm = confusion_matrix(y_test, y_te_pred, labels=classes)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{asset} - confusion matrix (test)")
        fig.colorbar(im, ax=ax)
        cm_path = Path("/tmp/cm.png")
        fig.savefig(cm_path, dpi=120, bbox_inches="tight"); plt.close(fig)
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        # feature importances
        imp = pd.DataFrame({
            "feature": feat_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        imp_path = Path("/tmp/feature_importances.csv")
        imp.to_csv(imp_path, index=False)
        mlflow.log_artifact(str(imp_path), artifact_path="diagnostics")

        # примеры предсказаний
        sample = X_test.head(10).copy()
        sample["y_true"] = y_test.iloc[:10].values
        sample["y_pred"] = y_te_pred[:10]
        sp = Path("/tmp/sample_predictions.csv")
        sample.to_csv(sp)
        mlflow.log_artifact(str(sp), artifact_path="examples")

        # сохраняем pipeline как sklearn модель + параллельно как joblib
        mlflow.sklearn.log_model(model, artifact_path="model",
                                 registered_model_name=f"{asset}_direction_model")
        jb = Path("/tmp/model.joblib")
        joblib.dump(model, jb, compress=3)
        mlflow.log_artifact(str(jb), artifact_path="model_joblib")

        # сохраняем список фичей и tau (нужно для inference)
        meta = {
            "asset": asset,
            "features": feat_cols,
            "classes": classes,
            "tau": float(tau),
            "split_date": cfg.split_date,
            "seed": cfg.seed,
        }
        mp = Path("/tmp/meta.json")
        mp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        mlflow.log_artifact(str(mp), artifact_path="model")

        logger.info("Готово. Run %s, см. http://localhost:5001/#/experiments/%s/runs/%s",
                    run.info.run_id, run.info.experiment_id, run.info.run_id)
        logger.info("Test macro-F1: %.4f",
                    f1_score(y_test, y_te_pred, labels=classes, average="macro", zero_division=0))

        # помечаем модель тегом PRD в Model Registry
        try:
            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(f"{asset}_direction_model")
            if versions:
                v = versions[0].version
                client.set_model_version_tag(f"{asset}_direction_model", v, "stage", "PRD")
                client.set_model_version_tag(f"{asset}_direction_model", v, "checkpoint", "7")
                logger.info("Model version %s помечен stage=PRD", v)
        except Exception as exc:
            logger.warning("Не удалось проставить PRD-тег: %s", exc)


if __name__ == "__main__":
    main()
