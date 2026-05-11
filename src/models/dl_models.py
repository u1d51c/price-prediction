"""DL-архитектуры: MLP, 1D-CNN, LSTM, мини-трансформер.

Все модели намеренно мелкие - тысячи параметров, не миллионы. У нас
обучающих примеров порядка тысячи, и большие модели здесь только
переобучаются.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# --- 1. MLP --------------------------------------------------------------

class MLPClassifier(nn.Module):
    """Полносвязная сеть на плоском векторе фичей.

    Структура: входной вектор -> 128 -> 64 -> 32 -> n_classes.
    BatchNorm после линейных слоёв нужен, потому что у нас фичи разных
    масштабов (цена, объём, sentiment). Dropout - обычная регуляризация.
    """

    def __init__(self, n_features: int, n_classes: int = 3, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),         nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32),          nn.BatchNorm1d(32),  nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --- 2. 1D-CNN на временном окне фичей -----------------------------------

class Conv1DSequenceClassifier(nn.Module):
    """1D-свёртки по временному окну.

    Вход - окно из seq_len дней с n_features фичами на каждый день.
    Перед conv транспонируем, чтобы каналы = фичи, длина = время.
    Два слоя с разными kernel_size: 3 (короткие паттерны) и 5 (на неделю).
    После - global average pool и линейная голова.
    """

    def __init__(self, seq_len: int, n_features: int, n_classes: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
        self.bn1, self.bn2 = nn.BatchNorm1d(32), nn.BatchNorm1d(32)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # переставляем оси: (B, seq_len, F) -> (B, F, seq_len), как ждёт Conv1d
        x = x.transpose(1, 2)
        h = self.act(self.bn1(self.conv1(x)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.dropout(h)
        # global avg pool: усредняем вдоль временной оси, получаем (B, 32)
        h = self.pool(h).squeeze(-1)
        return self.fc(h)


# --- 3. LSTM -------------------------------------------------------------

class LSTMClassifier(nn.Module):
    """Однонаправленный LSTM поверх последовательности фичей.

    Берём последний hidden state и из него считаем логиты. Bi-LSTM здесь
    не делаем: на коротких окнах смотреть "в будущее" внутри окна больше
    добавляет шума, чем сигнала.
    """

    def __init__(self, n_features: int, hidden: int = 32, n_classes: int = 3,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden,
                            num_layers=1, batch_first=True, dropout=0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # h_n имеет форму (1, B, hidden) - снимаем первую ось
        _, (h_n, _) = self.lstm(x)
        last = h_n.squeeze(0)
        return self.fc(self.dropout(last))


# --- 4. Mini-Transformer encoder ----------------------------------------

class TransformerEncoderClassifier(nn.Module):
    """Лёгкий self-attention поверх временного окна.

    Каждый временной шаг проецируется в d_model, добавляется sinusoidal-
    позиция, затем пара слоёв стандартного nn.TransformerEncoder.
    Голова работает по mean-pool (на табличных финансах он стабильнее
    CLS-токена из NLP).
    """

    def __init__(self, n_features: int, seq_len: int, d_model: int = 32,
                 n_heads: int = 4, n_layers: int = 2, n_classes: int = 3,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        # позиционный embedding делаем фиксированным, чтобы не учился отдельно
        self.pos = nn.Parameter(self._build_pos(seq_len, d_model), requires_grad=False)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, n_classes)

    @staticmethod
    def _build_pos(seq_len: int, d_model: int) -> torch.Tensor:
        """Классический синусоидальный positional embedding из оригинальной статьи."""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        # чётные позиции - sin, нечётные - cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x) + self.pos[: x.size(1)]
        h = self.encoder(h)
        # mean-pool по времени, потом одна линейная голова
        return self.fc(h.mean(dim=1))


# --- Фабрика -------------------------------------------------------------

def make_model(name: str, **kwargs) -> nn.Module:
    """По имени возвращает свежую модель - удобно вызывать в ноутбуках."""
    name = name.lower()
    if name == "mlp":
        return MLPClassifier(**kwargs)
    if name == "cnn":
        return Conv1DSequenceClassifier(**kwargs)
    if name == "lstm":
        return LSTMClassifier(**kwargs)
    if name == "transformer":
        return TransformerEncoderClassifier(**kwargs)
    raise ValueError(f"Unknown model: {name}")
