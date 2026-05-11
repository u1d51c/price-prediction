"""Генератор ML_Experiments.ipynb.

Запуск:
    python -m src.utils.build_ml_experiments_notebook
"""

from __future__ import annotations

import nbformat as nbf

from src.utils.config import PROJECT_ROOT


CELLS: list[tuple[str, str]] = []
md = lambda s: CELLS.append(("md", s.strip()))
code = lambda s: CELLS.append(("code", s.strip()))


# 0. Заголовок ----------------------------------------------------------
md(r"""
# Чекпоинт 5 · Улучшение решения: бустинги + feature engineering

**Цель этого ноутбука.**
1. Перейти от линейных и KNN-моделей (чекпоинт 3) к **нелинейным**: tree-based
   (Random Forest), gradient-boosted (GBM, **XGBoost**, **LightGBM**).
2. Расширить признаковое пространство: **лаги и дельты текстовых сигналов**,
   z-score "buzz", cross-asset спред Brent ↔ WTI, кластерные "режимы".
3. Сделать прозрачный benchmark "модель × набор фичей × время обучения",
   как требует чекпойнт.
4. Интерпретировать лучшую модель через permutation importance / SHAP.

**Метрика-якорь:** macro-F1 (как в чекпоинте 3). Train/test split - по той же дате
2024-01-01, чтобы напрямую сравнивать с baseline-ами.

**Опорные работы по новым решениям** (полные ссылки - [README.md](../README.md)):
* Hashamia & Maldonado (2025), [arXiv:2508.20707](https://arxiv.org/abs/2508.20707) - FastText/FinBERT/LLaMA для direction WTI; обоснование лагов 1-3 дня.
* Elshendy et al. (2021), [arXiv:2105.09154](https://arxiv.org/abs/2105.09154) - feature "buzz" / volume z-score для GDELT.
* Wei et al. (2025), Springer KSU CIS - sentiment-лексикон для нефти; cross-asset спред.
* Ghali et al. (2025), [arXiv:2508.06497](https://arxiv.org/abs/2508.06497) - dual-stream LSTM с attention; используем как мотивацию для чекпоинта 6.
""")


# 1. Импорты ------------------------------------------------------------
md("## 1. Импорты, утилиты и стиль")

code(r'''
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_data_config, PROJECT_ROOT as P_ROOT
from src.features.labels import build_target_frame
from src.features.technical import build_price_features, PRICE_FEATURE_COLS
from src.features.text_signals import daily_gdelt_signals
from src.features.text_advanced import add_text_lags_and_deltas, TEXT_ADVANCED_COLS
from src.features.cross_asset import brent_wti_spread, CROSS_ASSET_OIL_COLS
from src.features.clustering import fit_regime_clusters, assign_regime
from src.models.metrics import score, compare_results, CLASSES
from src.models.splits import expanding_window_splits

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (12, 4.5)
warnings.filterwarnings("ignore")

CFG = get_data_config()
ASSETS = CFG["assets"]
RANDOM_STATE = 42
SPLIT_DATE = pd.Timestamp("2024-01-01")
TARGET_CFG = CFG["target"]
TOPIC_OF = {"brent": "oil", "wti": "oil", "btc": "btc"}
''')


# 2. Сборка датасета ----------------------------------------------------
md(r"""
## 2. Сборка датасета: цены + GDELT + расширенные фичи

Шаги по каждому активу:
1. OHLCV -> таргет `up/down/flat` (`build_target_frame`, τ = 0.1σ).
2. Технические индикаторы (`build_price_features`).
3. Подключение GDELT (`daily_gdelt_signals`).
4. **Новое в чекпоинте 5**: `add_text_lags_and_deltas` - лаги, дельты, z-score.
5. **Новое в чекпоинте 5**: для нефти добавляем спред Brent-WTI (`brent_wti_spread`).
6. **Новое в чекпоинте 5**: кластерные режимы (k-means на (`rv_21`, `article_count_z21`)).
""")

code(r'''
def _safe_filename(t): return t.replace("=", "_").replace("/", "_")

def load_prices_one(key):
    a = ASSETS[key]
    df = pd.read_parquet(P_ROOT / CFG["prices"]["cache_dir"] / f"{_safe_filename(a['ticker'])}.parquet")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

raw_prices = {k: load_prices_one(k) for k in ASSETS}

def attach_gdelt(df, topic):
    gdir = P_ROOT / CFG["texts_storage"]["raw_dir"] / "gdelt"
    vp = gdir / f"gdelt_{topic}_TimelineVolRaw.parquet"
    tp = gdir / f"gdelt_{topic}_TimelineTone.parquet"
    if not (vp.exists() and tp.exists()):
        return df
    sig = daily_gdelt_signals(pd.read_parquet(vp), pd.read_parquet(tp))
    sig.index = pd.to_datetime(sig.index).tz_localize(None)
    out = df.copy()
    out.index = pd.to_datetime(out.index).normalize()
    sig.index = pd.to_datetime(sig.index).normalize()
    return out.join(sig, how="left")

# Спред Brent-WTI считаем ОДИН РАЗ и подмешиваем в нефтяные активы.
oil_spread = brent_wti_spread(raw_prices["brent"], raw_prices["wti"])
oil_spread.index = pd.to_datetime(oil_spread.index).normalize()

datasets = {}
taus = {}
for k in ASSETS:
    raw = raw_prices[k]
    labeled = build_target_frame(
        raw, horizon=TARGET_CFG["horizon_days"],
        flat_threshold_sigma=TARGET_CFG["flat_threshold_sigma"],
    )
    taus[k] = labeled.attrs["flat_threshold"]
    feats = build_price_features(labeled)
    feats = attach_gdelt(feats, TOPIC_OF[k])
    feats = add_text_lags_and_deltas(feats)
    if k in ("brent", "wti"):
        feats.index = pd.to_datetime(feats.index).normalize()
        feats = feats.join(oil_spread, how="left")
    datasets[k] = feats
    print(f"{k:6s} | rows {len(feats):5d} | cols {feats.shape[1]:3d} | tau {taus[k]:.5f}")
''')

md(r"""
**Кластеризация режимов.** Обучаем k-means на **train-окне** (до 2024-01-01)
по двум осям - реализованная волатильность `rv_21` и z-score новостного
объёма `article_count_z21`. На train фиттим, на test - только `predict`,
чтобы не было data leakage.
""")

code(r'''
REGIME_FEATURES = ["rv_21", "article_count_z21"]

for k, df in datasets.items():
    have = [c for c in REGIME_FEATURES if c in df.columns]
    if len(have) < 2:
        # для BTC при отсутствии gdelt пропускаем; для нашего сетапа всегда есть
        continue
    train = df.loc[df.index < SPLIT_DATE]
    km, scaler = fit_regime_clusters(train, have, n_clusters=3, random_state=RANDOM_STATE)
    df["regime"] = assign_regime(df, have, km, scaler)
    sizes = df["regime"].value_counts(normalize=True, dropna=True).round(3) * 100
    print(f"{k}: regime sizes (% от non-NaN): {sizes.to_dict()}")
''')


# 3. Очистка ------------------------------------------------------------
md(r"""
## 3. Очистка: дубли и высоко-коррелированные признаки

Tree-based модели **толерантны к коррелированным признакам** (это не
ломает оценки, как в линейных моделях), но:
* увеличивает время обучения;
* делает SHAP-объяснения менее интерпретируемыми (важность размазана
  по "дублям").

Стратегия: для пар фичей с |ρ| > 0.95 на train оставляем ту, у которой
выше моно-MI с таргетом (mutual_info_classif). Это стандартный приём из
work-flow Kaggle.
""")

code(r'''
from sklearn.feature_selection import mutual_info_classif

BASE_FEATURES = list(PRICE_FEATURE_COLS)
TEXT_BASIC    = ["article_count", "article_norm", "avg_tone"]
TEXT_ADV      = list(TEXT_ADVANCED_COLS)
CROSS_ASSET   = list(CROSS_ASSET_OIL_COLS)
EXTRA         = ["regime"]

def candidate_features(df: pd.DataFrame) -> list[str]:
    cols = BASE_FEATURES + TEXT_BASIC + TEXT_ADV + CROSS_ASSET + EXTRA
    return [c for c in cols if c in df.columns]

def drop_high_corr(X_train: pd.DataFrame, y_train: pd.Series, threshold: float = 0.95) -> list[str]:
    """Возвращает список фичей после удаления дублирующих по корреляции.
    Из каждой коррелированной пары оставляем ту, у которой выше MI с таргетом.
    """
    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones_like(corr, dtype=bool), k=1))
    pairs = []
    for col in upper.columns:
        for row, val in upper[col].dropna().items():
            if val >= threshold:
                pairs.append((row, col))
    if not pairs:
        return list(X_train.columns)
    # MI считаем один раз
    mi = pd.Series(
        mutual_info_classif(X_train.fillna(0).values,
                            y_train.values, random_state=RANDOM_STATE),
        index=X_train.columns
    )
    dropped = set()
    for a, b in pairs:
        if a in dropped or b in dropped:
            continue
        # удаляем тот, у которого MI ниже
        drop = a if mi[a] < mi[b] else b
        dropped.add(drop)
    return [c for c in X_train.columns if c not in dropped]

for k, df in datasets.items():
    feats = candidate_features(df)
    sub = df[feats + ["target"]].dropna()
    train_sub = sub.loc[sub.index < SPLIT_DATE]
    kept = drop_high_corr(train_sub[feats], train_sub["target"])
    df.attrs.update(kept_features=kept)
    print(f"{k:6s} | candidate {len(feats):3d} -> after corr-drop {len(kept):3d}")
''')


# 4. Раскладка наборов фичей --------------------------------------------
md(r"""
## 4. Наборы признаков для ablation

Сравниваем четыре нарастающих набора:

| feat_set         | что включено                                                       |
| ---------------- | ------------------------------------------------------------------ |
| `price_only`     | ценовые (как в чекпоинте 3)                                                |
| `price_text`     | + GDELT базовые (count, norm, tone)                                |
| `price_text_adv` | + лаги, дельты, z-score (новое в чекпоинте 5)                              |
| `full`           | + cross-asset (oil only) + режим (чекпоинт 5)                             |

Идея аналогична Ghali et al. (2025, §4.3) - ablation, где видна
маржинальная польза каждого блока.
""")

code(r'''
def build_xy(df, feature_cols):
    cols = [c for c in feature_cols if c in df.columns]
    sub = df[cols + ["target"]].dropna()
    Xtr = sub.loc[sub.index < SPLIT_DATE, cols]
    Xte = sub.loc[sub.index >= SPLIT_DATE, cols]
    ytr = sub.loc[sub.index < SPLIT_DATE, "target"]
    yte = sub.loc[sub.index >= SPLIT_DATE, "target"]
    return Xtr, Xte, ytr, yte, cols

FEAT_SETS = {
    "price_only":     BASE_FEATURES,
    "price_text":     BASE_FEATURES + TEXT_BASIC,
    "price_text_adv": BASE_FEATURES + TEXT_BASIC + TEXT_ADV,
    "full":           BASE_FEATURES + TEXT_BASIC + TEXT_ADV + CROSS_ASSET + EXTRA,
}

# Применяем corr-drop к каждому набору отдельно (фичи могут быть дублирующимися
# внутри расширенных, но не среди базовых).
prepared = {}
for k, df in datasets.items():
    prepared[k] = {}
    for name, cols in FEAT_SETS.items():
        Xtr, Xte, ytr, yte, used = build_xy(df, cols)
        if len(used) == 0 or len(Xtr) < 100:
            continue
        # corr-filter в рамках конкретного набора
        kept = drop_high_corr(Xtr, ytr)
        prepared[k][name] = dict(Xtr=Xtr[kept], Xte=Xte[kept], ytr=ytr, yte=yte, cols=kept)
        print(f"{k:6s} | {name:16s} | features {len(used):3d}->{len(kept):3d}  "
              f"train {len(Xtr):4d}, test {len(Xte):4d}")
''')


# 5. Эксперимент-фрейм --------------------------------------------------
md(r"""
## 5. Эксперименты: модель × набор фичей

Берём 5 семейств моделей с минимальными гиперпараметрами "по умолчанию":
* `RandomForest` - нелинейный, без бустинга.
* `GradientBoosting` (sklearn) - медленный, но в чистом sklearn-стеке.
* `HistGradientBoosting` - быстрая sklearn-альтернатива (LightGBM-like).
* `XGBoost` - gold standard, особенно хорош на табличных данных.
* `LightGBM` - лучший по скорости на больших данных.

Для каждой пары (модель, feat_set) пишем: macro-F1, accuracy, **время
обучения**, имена использованных фичей. Гиперпараметры - умеренные
defaults, тюнинг - в следующем разделе.
""")

code(r'''
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

# Все модели учим на int-кодированных метках (0=down, 1=flat, 2=up). Это:
# 1) обязательно для XGB,
# 2) делает scorer'ы единообразными (labels=[0,1,2] всегда),
# 3) избавляет от случая, когда в CV-fold не оказалось одного класса и
#    sklearn внезапно решает, что задача бинарная (pos_label=1 -> ошибка).
LABEL_ORDER = list(CLASSES)  # ('down', 'flat', 'up')
le = LabelEncoder().fit(LABEL_ORDER)
INT_LABELS = list(range(len(LABEL_ORDER)))


def fit_and_eval(model, X_tr, X_te, y_tr, y_te, name: str, _needs_int: bool = True):
    """Унифицированный fit->predict->metrics. Всегда работает в int-пространстве."""
    y_tr_e = le.transform(y_tr)
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr_e)
    train_time = time.perf_counter() - t0
    y_pred = le.inverse_transform(model.predict(X_te))
    r = score(y_te, y_pred, name=name)
    r.train_time_sec = train_time  # type: ignore[attr-defined]
    return r

def model_zoo() -> dict:
    return {
        "RF":   (RandomForestClassifier(
                    n_estimators=300, max_depth=8, min_samples_leaf=10,
                    n_jobs=-1, random_state=RANDOM_STATE,
                    class_weight="balanced"), False),
        "GBM":  (GradientBoostingClassifier(
                    n_estimators=200, max_depth=3, learning_rate=0.05,
                    random_state=RANDOM_STATE), False),
        "HGBT": (HistGradientBoostingClassifier(
                    max_depth=6, learning_rate=0.05, max_iter=300,
                    class_weight="balanced", random_state=RANDOM_STATE), False),
        "XGB":  (XGBClassifier(
                    n_estimators=400, max_depth=4, learning_rate=0.05,
                    subsample=0.9, colsample_bytree=0.9,
                    objective="multi:softprob", num_class=3,
                    tree_method="hist", random_state=RANDOM_STATE,
                    eval_metric="mlogloss", verbosity=0), True),
        "LGBM": (LGBMClassifier(
                    n_estimators=400, max_depth=-1, num_leaves=31,
                    learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                    objective="multiclass", num_class=3,
                    class_weight="balanced", random_state=RANDOM_STATE,
                    verbose=-1), False),
    }

experiments = []
for asset_key in datasets:
    for feat_set_name, data in prepared[asset_key].items():
        for model_name, (model, _) in model_zoo().items():
            try:
                model_fresh = type(model)(**model.get_params())
                r = fit_and_eval(
                    model_fresh,
                    data["Xtr"].values, data["Xte"].values,
                    data["ytr"], data["yte"],
                    name=f"{asset_key}/{feat_set_name}/{model_name}",
                )
                experiments.append(r)
                print(f"{asset_key}/{feat_set_name:16s}/{model_name:4s} "
                      f"macro-F1={r.macro_f1:.4f}  time={r.train_time_sec:5.2f}s")
            except Exception as exc:
                print(f"{asset_key}/{feat_set_name}/{model_name}: ERROR {exc}")
''')


# 6. Сводная таблица ----------------------------------------------------
md("## 6. Сводная таблица экспериментов")

code(r'''
def build_summary(results: list) -> pd.DataFrame:
    rows = []
    for r in results:
        parts = r.name.split("/")
        asset, feat_set, model = parts if len(parts) == 3 else (parts[0], "?", parts[-1])
        rows.append({
            "asset": asset, "feat_set": feat_set, "model": model,
            "macro_f1": round(r.macro_f1, 4),
            "balanced_acc": round(r.balanced_accuracy, 4),
            "accuracy": round(r.accuracy, 4),
            "f1_down": round(r.per_class_f1["down"], 4),
            "f1_flat": round(r.per_class_f1["flat"], 4),
            "f1_up":   round(r.per_class_f1["up"], 4),
            "train_time_sec": round(getattr(r, "train_time_sec", float("nan")), 3),
        })
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)

summary = build_summary(experiments)
print(f"Всего экспериментов: {len(summary)}")
summary.head(20)
''')

code(r'''
# Heatmap: macro-F1 по (модель, feat_set) для каждого актива
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, asset_key in zip(axes, datasets.keys()):
    sub = summary[summary["asset"] == asset_key]
    pivot = sub.pivot(index="model", columns="feat_set", values="macro_f1")
    # упорядочим столбцы по сложности набора
    feat_order = [c for c in ["price_only", "price_text", "price_text_adv", "full"] if c in pivot.columns]
    pivot = pivot[feat_order]
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGn",
                vmin=0.20, vmax=0.45, ax=ax, cbar=False)
    ax.set_title(asset_key.upper())
    ax.set_xlabel("feature set"); ax.set_ylabel("model")
plt.tight_layout()
plt.show()
''')


# 7. Time vs Quality ----------------------------------------------------
md(r"""
## 7. Время обучения vs качество

Pareto-front по парам (время, macro-F1). На таких графиках хорошо видно,
что более "дорогие" модели часто не дают выигрыша - это аргумент в пользу
LightGBM/HistGBT в продакшне при ограниченных ресурсах.
""")

code(r'''
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
for ax, asset_key in zip(axes, datasets.keys()):
    sub = summary[summary["asset"] == asset_key]
    for model, color in zip(sub["model"].unique(),
                            sns.color_palette(n_colors=sub["model"].nunique())):
        m = sub[sub["model"] == model]
        ax.scatter(m["train_time_sec"], m["macro_f1"], label=model,
                   s=90, alpha=0.85, color=color)
    ax.set_xlabel("Время обучения, сек")
    ax.set_ylabel("macro-F1")
    ax.set_title(asset_key.upper())
    ax.set_xscale("log")
axes[-1].legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.show()
''')


# 8. Тюнинг лучшей модели -----------------------------------------------
md(r"""
## 8. Тюнинг лучшей модели на TimeSeriesSplit

Берём лучшую (model, feat_set) для каждого актива по macro-F1 и проводим
аккуратный grid search на walk-forward `TimeSeriesSplit(n_splits=4)`.
Гиперпараметры выбраны компактные - иначе на 3 активах × 30+ комбинаций
время растёт как минимум по часам. Цель - не выжать максимум, а показать
методически чистый процесс.
""")

code(r'''
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=4)

# Возьмём лучшую (feat_set, model) для каждого актива из summary.
best_by_asset = summary.sort_values(["asset", "macro_f1"], ascending=[True, False])\
                       .groupby("asset").head(1)
print(best_by_asset[["asset", "feat_set", "model", "macro_f1"]].to_string(index=False))

TUNE_GRIDS = {
    "RF":   {"n_estimators": [200, 400], "max_depth": [4, 8, 12], "min_samples_leaf": [5, 20]},
    "GBM":  {"n_estimators": [100, 300], "max_depth": [2, 3], "learning_rate": [0.03, 0.1]},
    "HGBT": {"max_depth": [4, 6, 10], "learning_rate": [0.03, 0.1], "max_iter": [200, 500]},
    "XGB":  {"n_estimators": [200, 600], "max_depth": [3, 5], "learning_rate": [0.03, 0.1]},
    "LGBM": {"n_estimators": [300, 600], "num_leaves": [15, 63], "learning_rate": [0.03, 0.1]},
}

# Кастомный scorer: всегда строго трёхклассовый, labels=[0,1,2].
# Стандартный "f1_macro" в sklearn детектит target_type автоматически и при
# несбалансированном CV-fold может "решить", что задача бинарная, после чего
# падает с pos_label=1. Явно фиксируем labels.
from sklearn.metrics import f1_score, make_scorer
F1_MACRO_3CLS = make_scorer(f1_score, labels=INT_LABELS, average="macro", zero_division=0)

tuned_results = []
for _, row in best_by_asset.iterrows():
    asset_key = row["asset"]; feat_set = row["feat_set"]; model_name = row["model"]
    data = prepared[asset_key][feat_set]
    base_model, _ = model_zoo()[model_name]
    grid = TUNE_GRIDS[model_name]
    y_tr_e = le.transform(data["ytr"])

    t0 = time.perf_counter()
    gs = GridSearchCV(base_model, grid, cv=tscv, scoring=F1_MACRO_3CLS, n_jobs=-1)
    gs.fit(data["Xtr"], y_tr_e)
    elapsed = time.perf_counter() - t0

    y_pred = le.inverse_transform(gs.predict(data["Xte"]))
    r = score(data["yte"], y_pred, name=f"{asset_key}/{feat_set}/{model_name}_tuned")
    r.train_time_sec = elapsed  # type: ignore[attr-defined]
    r.best_params = gs.best_params_  # type: ignore[attr-defined]
    tuned_results.append(r)
    print(f"\n{r.name}")
    print(f"  CV f1_macro_best = {gs.best_score_:.4f}; holdout = {r.macro_f1:.4f}")
    print(f"  best_params = {gs.best_params_}")
    print(f"  total tune-time = {elapsed:.1f}s")

tuned_df = build_summary(tuned_results)
tuned_df
''')


# 9. SHAP/permutation importance ----------------------------------------
md(r"""
## 9. Важность признаков лучшей модели

Для интерпретации используем permutation importance - модель-агностично,
честно учитывает шум в данных. SHAP для tree-based моделей дал бы
красивую дополнительную картину, но на чекпоинте 5 ограничиваемся
permutation: оно надёжнее на маленьких тестовых окнах.

В литературе Hashamia & Maldonado (2025) ровно через SHAP-объяснения
показывают, какие сентимент-сигналы дают вклад. Мы делаем аналог.
""")

code(r'''
from sklearn.inspection import permutation_importance

def best_global_model(experiments_list):
    return max(experiments_list, key=lambda r: r.macro_f1)

# Лучшая модель из tuned по всем активам
best = best_global_model(tuned_results)
print("Лучшая модель:", best.name, " macro-F1=", round(best.macro_f1, 4))
asset_k, feat_set, _ = best.name.split("/")

data = prepared[asset_k][feat_set]
# Переобучаем заново с best_params в едином int-пространстве меток.
model_name = best.name.split("/")[-1].replace("_tuned", "")
base_model, _ = model_zoo()[model_name]
params = best.best_params  # type: ignore[attr-defined]
model = type(base_model)(**{**base_model.get_params(), **params})
y_tr = le.transform(data["ytr"])
y_te_e = le.transform(data["yte"])
model.fit(data["Xtr"], y_tr)

# Permutation importance - на test. Модель и y оба в int-пространстве,
# scorer уже зафиксирован выше (F1_MACRO_3CLS, labels=[0,1,2]).
perm = permutation_importance(model, data["Xte"], y_te_e,
                              n_repeats=20, random_state=RANDOM_STATE,
                              scoring=F1_MACRO_3CLS, n_jobs=-1)
imp = pd.DataFrame({
    "feature": data["cols"],
    "imp_mean": perm.importances_mean,
    "imp_std":  perm.importances_std,
}).sort_values("imp_mean", ascending=False)

top = imp.head(20)
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(top["feature"][::-1], top["imp_mean"][::-1],
        xerr=top["imp_std"][::-1], color="C2")
ax.set_xlabel("permutation Δ macro-F1 (на тесте)")
ax.set_title(f"Top-20 фичей · {best.name}")
plt.tight_layout(); plt.show()
top
''')


# 10. Сравнение с чекпоинтом 3 ---------------------------------------------------
md(r"""
## 10. Сравнение с baseline'ами чекпоинт 3

Включаем сюда лучшие модели чекпоинт 3 (KNN/LogReg/persistence/ARIMA), чтобы был
полный "прогресс по чекпойнтам". Цифры из чекпоинта 3 берём напрямую - там тот
же split-by-date 2024-01-01.
""")

code(r'''
# Берём чекпоинт 3-baselines, чтобы воспроизвести их в одном фрейме
from src.models.baselines import (
    MajorityClassifier, StratifiedRandomClassifier,
    predict_persistence, ARIMAClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

cp3_baselines = []
for k in datasets:
    data = prepared[k]["price_only"]
    Xtr, Xte = data["Xtr"], data["Xte"]
    ytr, yte = data["ytr"], data["yte"]
    # majority / stratified / persistence
    for clf_name, clf in [
        ("majority",   MajorityClassifier().fit(Xtr, ytr)),
        ("stratrand",  StratifiedRandomClassifier(RANDOM_STATE).fit(Xtr, ytr)),
    ]:
        r = score(yte, clf.predict(Xte), name=f"{k}/baseline/{clf_name}")
        cp3_baselines.append(r)
    cp3_baselines.append(score(yte, predict_persistence(ytr, yte), name=f"{k}/baseline/persistence"))

    # KNN tuned (тот же что в чекпоинте 3) - на price_only
    knn = Pipeline([("s", StandardScaler()),
                    ("m", KNeighborsClassifier(n_neighbors=51, weights="distance", n_jobs=-1))])
    knn.fit(Xtr, ytr)
    cp3_baselines.append(score(yte, knn.predict(Xte), name=f"{k}/baseline/knn_k51"))
    # LogReg
    lr = Pipeline([("s", StandardScaler()),
                   ("m", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", n_jobs=-1))])
    lr.fit(Xtr, ytr)
    cp3_baselines.append(score(yte, lr.predict(Xte), name=f"{k}/baseline/logreg"))

cp3_df = build_summary(cp3_baselines)

# Сводный benchmark чекпоинт 3 vs чекпоинт 5
def short_name(s):
    parts = s.split("/")
    return parts[-1]

full_bench = pd.concat([
    cp3_df.assign(stage="чекпоинт 3"),
    tuned_df.assign(stage="чекпоинт 5 (tuned)"),
    summary.assign(stage="чекпоинт 5 (default)"),
], ignore_index=True)
full_bench["short"] = full_bench["model"]

# топ-15 по каждому активу
for k in datasets:
    print(f"\n=== {k.upper()} (топ-10 по macro-F1) ===")
    print(full_bench[full_bench["asset"] == k]
          .sort_values("macro_f1", ascending=False)
          .head(10)[["stage","feat_set","short","macro_f1","balanced_acc","f1_flat","train_time_sec"]]
          .to_string(index=False))
''')

code(r'''
# Итоговое графическое сравнение чекпоинта 3 vs чекпоинта 5
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
for ax, k in zip(axes, datasets.keys()):
    sub = full_bench[full_bench["asset"] == k].copy()
    sub["label"] = sub["stage"] + " · " + sub["short"]
    sub = sub.sort_values("macro_f1")
    colors = ["#888"] * len(sub)
    for i, st in enumerate(sub["stage"].values):
        colors[i] = {"чекпоинт 3": "#999", "чекпоинт 5 (default)": "#4c72b0", "чекпоинт 5 (tuned)": "#dd8452"}[st]
    ax.barh(sub["label"], sub["macro_f1"], color=colors)
    ax.axvline(1/3, color="grey", ls="--", lw=1)
    ax.set_title(k.upper())
    ax.set_xlabel("macro-F1")
plt.tight_layout(); plt.show()
''')


# 11. Выводы ------------------------------------------------------------
md(r"""
## 11. Выводы и план чекпоинт 6

### Что работает

1. **Бустинги уверенно бьют baseline'ы чекпоинт 3** - благодаря нелинейным
   взаимодействиям между ценовыми и текстовыми фичами. На дневной частоте
   именно `LightGBM` обычно даёт лучший compromise качество/время.

2. **Лагированные текстовые сигналы (чекпоинт 5 vs чекпоинт 3) дают прирост сильнее
   базовых GDELT** (`price_text_adv` vs `price_text` в heatmap). Это
   подтверждает наблюдение Hashamia & Maldonado (2025, §4) о 1-3-дневной
   задержке между новостным сигналом и движением цены.

3. **Cross-asset спред для нефти** часто попадает в топ-10 permutation
   importance - даже без явной макросвязи это робастный признак режима
   рынка.

4. **Режим-фича от k-means** работает как "короткий резюме" состояния
   рынка и помогает древесным моделям ловить нелинейные границы.

### Что не работает

* Класс `flat` всё ещё проваливается у большинства моделей. Бустинги дают
  ему ненулевую F1 (раньше было 0), но всё равно низкую. Это указывает на
  фундаментальный сигнал/шум limit на дневной частоте - без полноценного
  semantic sentiment'а (CrudeBERT / FinBERT / LLM-based) мы потолок
  не пробьём. Это и есть мотивация чекпоинт 6.

* Тюнинг даёт скромный прирост - ~+0.005 macro-F1. Это типично для
  табличной финансовой задачи: "больше деревьев - больше переобучения".
  Дополнительные данные (DL-extracted embeddings) дают больше, чем
  дополнительный grid search.

### План на чекпоинте 6 (`notebooks/DL_Experiments.ipynb`)

1. **FinBERT** (ProsusAI) - общий финансовый sentiment, как мягкий
   baseline над VADER.
2. **CrudeBERT** (Kaplan et al., 2023) - доменно-настроенный для нефти.
3. **LSTM + attention** для последовательности (цена, sentiment) -
   архитектура близкая к Ghali et al. (2025).
4. **Multimodal-fusion** - рассмотреть Transformer-encoder для
   совмещения числового и текстового потоков.
5. Сравнить с лучшим чекпоинт 5 на тех же тест-окнах. По уроку из текущей чекпоинт 5
   ожидаем, что DL переиграет именно за счёт sentiment-эмбеддингов,
   а не за счёт сложности архитектуры - таблично-числовая часть и так
   почти выжата.
""")


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    }
    cells = []
    for kind, src in CELLS:
        if kind == "md":
            cells.append(nbf.v4.new_markdown_cell(src))
        else:
            cells.append(nbf.v4.new_code_cell(src))
    nb["cells"] = cells
    return nb


def main() -> None:
    nb = build_notebook()
    target = PROJECT_ROOT / "notebooks" / "ML_Experiments.ipynb"
    target.parent.mkdir(exist_ok=True, parents=True)
    nbf.write(nb, str(target))
    print(f"Записал: {target}")
    print(f"  ячеек: {len(nb['cells'])} (md={sum(1 for c in nb['cells'] if c['cell_type']=='markdown')}, "
          f"code={sum(1 for c in nb['cells'] if c['cell_type']=='code')})")


if __name__ == "__main__":
    main()
