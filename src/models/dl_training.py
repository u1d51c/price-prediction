"""Тренировочный цикл для DL-моделей: early stopping, sliding windows.

Один общий fit для всех архитектур: данные подаются numpy-массивами,
ранняя остановка по валидационному macro-F1, последние 15% train идут
в валидацию (в test никогда не лезем).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    """Гиперпараметры обучения. Дефолты подобраны под маленькую выборку."""

    lr: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 60
    patience: int = 8          # сколько эпох без улучшения val-метрики до остановки
    val_fraction: float = 0.15 # доля train для валидации
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cpu"        # 'cpu' / 'mps' / 'cuda'


# --- Sliding window -----------------------------------------------------

def build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Превращает (T, D) в (T - seq_len + 1, seq_len, D).

    Каждое окно - последние seq_len дней до момента t включительно;
    таргет берётся с дня t (последнего в окне). В будущее не смотрим.
    """
    if len(X) <= seq_len:
        # данных меньше окна - отдаём пустые массивы корректной формы
        return np.empty((0, seq_len, X.shape[1])), np.empty((0,), dtype=y.dtype)
    out_X = np.stack([X[i - seq_len + 1: i + 1] for i in range(seq_len - 1, len(X))])
    out_y = y[seq_len - 1:]
    return out_X, out_y


# --- Trainer ------------------------------------------------------------

def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    """Macro-F1 на numpy без вызова sklearn - чтобы валидация в цикле была быстрой."""
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fp == 0 or tp + fn == 0 or tp == 0:
            f1s.append(0.0)
            continue
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1s.append(2 * p * r / (p + r))
    return sum(f1s) / n_classes


def train_model(
    model_factory: Callable[[], nn.Module],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: TrainConfig,
) -> dict:
    """Обучает модель и возвращает прогноз на test + диагностику.

    model_factory - callable без аргументов, каждый вызов создаёт свежую модель.
    X может быть либо плоским (B, F) для MLP, либо оконным (B, T, F) для остальных.

    На выходе словарь с y_pred, train_time_sec, лучшим val_f1, числом эпох
    до остановки, кривыми train_loss / val_f1 и числом параметров модели.
    """
    # фиксируем сиды и для torch, и для numpy - повторяемость
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # отрезаем валидацию от конца train (хронологически)
    n = len(X_train)
    n_val = max(1, int(n * cfg.val_fraction))
    X_tr, X_val = X_train[: n - n_val], X_train[n - n_val:]
    y_tr, y_val = y_train[: n - n_val], y_train[n - n_val:]

    device = torch.device(cfg.device)
    model = model_factory().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # переносим тензоры на устройство один раз - для val и test держим там до конца
    Xtr_t = torch.from_numpy(X_tr).float()
    ytr_t = torch.from_numpy(y_tr).long()
    Xv_t  = torch.from_numpy(X_val).float().to(device)
    yv_np = y_val
    Xte_t = torch.from_numpy(X_test).float().to(device)

    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=cfg.batch_size, shuffle=True)

    # переменные для early stopping
    best_val_f1, best_state, stale = -1.0, None, 0
    train_loss_curve: list[float] = []
    val_f1_curve: list[float] = []
    stopped_epoch = cfg.max_epochs

    t0 = time.perf_counter()
    for epoch in range(cfg.max_epochs):
        # --- train ---
        model.train()
        loss_sum, n_seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * xb.size(0)
            n_seen += xb.size(0)
        train_loss_curve.append(loss_sum / max(n_seen, 1))

        # --- validation ---
        model.eval()
        with torch.no_grad():
            val_pred = model(Xv_t).argmax(dim=1).cpu().numpy()
        val_f1 = _macro_f1(yv_np, val_pred)
        val_f1_curve.append(val_f1)

        # --- early stopping ---
        if val_f1 > best_val_f1 + 1e-6:
            # новый рекорд - сохраняем веса
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            # ухудшение или плато - считаем подряд
            stale += 1
            if stale >= cfg.patience:
                stopped_epoch = epoch + 1
                break

    train_time = time.perf_counter() - t0

    # откатываем модель к лучшему по валидации состоянию и делаем итоговый predict
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred = model(Xte_t).argmax(dim=1).cpu().numpy()

    return {
        "y_pred": y_pred,
        "train_time_sec": train_time,
        "best_val_f1": best_val_f1,
        "stopped_epoch": stopped_epoch,
        "train_loss_curve": train_loss_curve,
        "val_f1_curve": val_f1_curve,
        "n_params": sum(p.numel() for p in model.parameters()),
    }
