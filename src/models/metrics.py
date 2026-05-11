"""Метрики качества для 3-классовой задачи.

Основная метрика - macro-F1. Классы у нас несбалансированы (flat редкий),
и accuracy в такой ситуации преувеличивает успехи модели на мажоритарных
классах. Macro-F1 усредняет F1 по классам без весов, поэтому даёт
равноценную оценку каждой категории.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


# фиксированный порядок классов - везде используем такой, чтобы матрицы совпадали
CLASSES: tuple[str, ...] = ("down", "flat", "up")


@dataclass
class ClassificationResult:
    """Все метрики одной модели на одном split-е, в одном объекте."""

    name: str
    macro_f1: float
    accuracy: float
    balanced_accuracy: float
    per_class_f1: dict[str, float]
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    confusion: pd.DataFrame
    n_samples: int

    def to_row(self) -> dict[str, float | str]:
        """Превращает результат в плоскую строку для сводной таблицы."""
        row: dict[str, float | str] = {
            "model": self.name,
            "macro_f1": round(self.macro_f1, 4),
            "accuracy": round(self.accuracy, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "n": self.n_samples,
        }
        # дописываем per-class F1, чтобы видеть, какой класс модель проваливает
        for cls in CLASSES:
            row[f"f1_{cls}"] = round(self.per_class_f1.get(cls, np.nan), 4)
        return row


def score(y_true: Iterable, y_pred: Iterable, name: str = "model") -> ClassificationResult:
    """Считает все метрики разом. y_true / y_pred - строковые метки."""
    y_true = list(y_true)
    y_pred = list(y_pred)

    # Все вызовы с labels=CLASSES - чтобы класс flat не пропадал, если модель
    # его ни разу не предсказала (тогда sklearn по умолчанию его выкидывает).
    macro_f1 = f1_score(y_true, y_pred, labels=list(CLASSES), average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(CLASSES), zero_division=0
    )
    per_f1 = dict(zip(CLASSES, f1.tolist()))
    per_p = dict(zip(CLASSES, prec.tolist()))
    per_r = dict(zip(CLASSES, rec.tolist()))

    cm = confusion_matrix(y_true, y_pred, labels=list(CLASSES))
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in CLASSES],
                         columns=[f"pred_{c}" for c in CLASSES])

    return ClassificationResult(
        name=name,
        macro_f1=macro_f1,
        accuracy=acc,
        balanced_accuracy=bal_acc,
        per_class_f1=per_f1,
        per_class_precision=per_p,
        per_class_recall=per_r,
        confusion=cm_df,
        n_samples=len(y_true),
    )


def text_report(result: ClassificationResult) -> str:
    """Печатный отчёт по одному результату (для print() в ноутбуке)."""
    lines = [
        f"=== {result.name} ===",
        f"  n={result.n_samples}",
        f"  macro-F1 = {result.macro_f1:.4f}",
        f"  accuracy = {result.accuracy:.4f}",
        f"  balanced-acc = {result.balanced_accuracy:.4f}",
        "  per-class F1:",
    ]
    for cls in CLASSES:
        lines.append(
            f"    {cls:5s}  F1={result.per_class_f1[cls]:.3f}  "
            f"P={result.per_class_precision[cls]:.3f}  R={result.per_class_recall[cls]:.3f}"
        )
    lines.append("  confusion (rows=true, cols=pred):")
    lines.append(result.confusion.to_string())
    return "\n".join(lines)


def compare_results(results: list[ClassificationResult]) -> pd.DataFrame:
    """Список результатов -> отсортированная по macro-F1 сводная таблица."""
    df = pd.DataFrame([r.to_row() for r in results])
    return df.sort_values("macro_f1", ascending=False).reset_index(drop=True)


def sklearn_classification_report(y_true, y_pred) -> str:
    """Тонкая обёртка над стандартным sklearn-отчётом - иногда нужна в ноутбуке."""
    return classification_report(y_true, y_pred, labels=list(CLASSES), zero_division=0)
