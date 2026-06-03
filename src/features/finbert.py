"""FinBERT для sentiment'а заголовков финансовых новостей.

Берём предобученную модель ProsusAI/finbert с HuggingFace -
BERT-base, дообученный на Financial PhraseBank. Возвращает три
вероятности: positive / negative / neutral. На вход - заголовки
статей из GDELT, на выход - дневные агрегаты для добавления
в фичесет.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch


def _device(preferred: str = "auto") -> torch.device:
    """Выбирает устройство: auto -> MPS на Apple Silicon, иначе CPU."""
    if preferred != "auto":
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FinBertScorer:
    """Обёртка над HuggingFace-моделью. Загружает веса лениво при первом вызове."""

    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self, device: str = "auto", batch_size: int = 32, max_length: int = 64) -> None:
        self.device = _device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        # модель и токенайзер подгрузим только когда реально позовут score()
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        """Подгружает веса с диска / HuggingFace при первом вызове."""
        if self._model is not None:
            return
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self._model.eval().to(self.device)
        # id2label у этой модели: {0: positive, 1: negative, 2: neutral}
        self.id2label = self._model.config.id2label

    @torch.no_grad()
    def score(self, texts: Iterable[str]) -> pd.DataFrame:
        """Прогоняет тексты через модель батчами, возвращает вероятности классов.

        Подходит для коротких текстов (заголовки до 64 токенов). Для длинных
        body статей используйте score_long().
        """
        self._ensure_loaded()
        texts = [str(t) if t is not None else "" for t in texts]
        if not texts:
            return pd.DataFrame(columns=["p_pos", "p_neg", "p_neutral"])

        out = []
        # бьём на батчи, чтобы не упереться в память
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            enc = self._tokenizer(batch, padding=True, truncation=True,
                                   max_length=self.max_length, return_tensors="pt").to(self.device)
            logits = self._model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            out.append(probs)
        probs = np.vstack(out)
        # сопоставляем колонки именам классов модели; на случай если порядок поменяется в будущем
        cols = {f"p_{self.id2label[i][:3]}": probs[:, i] for i in range(probs.shape[1])}
        return pd.DataFrame(cols)

    @torch.no_grad()
    def score_long(self, texts: Iterable[str], chunk_tokens: int = 510) -> pd.DataFrame:
        """Прогоняет длинные тексты с chunking-усреднением.

        BERT-base принимает максимум 512 токенов. Для статей это мало, поэтому
        режем body на куски длиной chunk_tokens, считаем FinBERT по каждому
        куску и усредняем softmax-вероятности. Возвращает те же три колонки.
        """
        self._ensure_loaded()
        rows = []
        for t in texts:
            t = str(t) if t else ""
            if not t.strip():
                rows.append({"p_pos": 0.0, "p_neg": 0.0, "p_neu": 0.0})
                continue

            # токенизируем без обрезания, чтобы получить весь набор token_ids
            ids = self._tokenizer.encode(t, add_special_tokens=False)
            if not ids:
                rows.append({"p_pos": 0.0, "p_neg": 0.0, "p_neu": 0.0})
                continue

            # режем на чанки по chunk_tokens
            chunks = [ids[i: i + chunk_tokens] for i in range(0, len(ids), chunk_tokens)]
            chunk_probs = []
            for c in chunks:
                # обрамляем CLS/SEP, чтобы модель видела стандартные spec-токены
                input_ids = torch.tensor(
                    [[self._tokenizer.cls_token_id] + c + [self._tokenizer.sep_token_id]],
                    device=self.device,
                )
                attn = torch.ones_like(input_ids)
                logits = self._model(input_ids=input_ids, attention_mask=attn).logits
                chunk_probs.append(torch.softmax(logits, dim=-1).cpu().numpy()[0])
            avg = np.mean(chunk_probs, axis=0)

            label_to_prob = {self.id2label[i]: float(avg[i]) for i in range(len(avg))}
            rows.append({
                "p_pos": label_to_prob.get("positive", 0.0),
                "p_neg": label_to_prob.get("negative", 0.0),
                "p_neu": label_to_prob.get("neutral", 0.0),
            })
        return pd.DataFrame(rows)


def daily_finbert_signals(
    articles_df: pd.DataFrame,
    scorer: FinBertScorer | None = None,
) -> pd.DataFrame:
    """Считает FinBERT по статьям и агрегирует по дате.

    На входе ожидает колонки seendate и title. На выходе:
      finbert_pos_share / neg_share / neutral_share - доли классов за день,
      finbert_compound - среднее (p_pos − p_neg),
      finbert_n_texts - сколько статей попало в агрегат.
    """
    if articles_df.empty or "title" not in articles_df.columns:
        return pd.DataFrame()

    if scorer is None:
        scorer = FinBertScorer()

    scores = scorer.score(articles_df["title"].fillna("").tolist())
    work = articles_df[["seendate"]].copy()
    work["date"] = pd.to_datetime(work["seendate"]).dt.normalize()

    # колонки могут называться p_pos/p_neg/p_neu или иначе - нормализуем
    rename = {}
    for c in scores.columns:
        if "pos" in c: rename[c] = "p_pos"
        elif "neg" in c: rename[c] = "p_neg"
        elif "neu" in c: rename[c] = "p_neu"
    scores = scores.rename(columns=rename)
    for col in ("p_pos", "p_neg", "p_neu"):
        if col not in scores.columns:
            scores[col] = 0.0

    # для каждой статьи берём argmax (предсказанный класс) и compound = p_pos - p_neg
    work["pred"] = scores[["p_pos", "p_neg", "p_neu"]].values.argmax(axis=1)
    work["compound"] = scores["p_pos"].values - scores["p_neg"].values

    grouped = work.groupby("date")
    out = pd.DataFrame({
        "finbert_pos_share":     grouped["pred"].apply(lambda s: float((s == 0).mean())),
        "finbert_neg_share":     grouped["pred"].apply(lambda s: float((s == 1).mean())),
        "finbert_neutral_share": grouped["pred"].apply(lambda s: float((s == 2).mean())),
        "finbert_compound":      grouped["compound"].mean(),
        "finbert_n_texts":       grouped.size(),
    })
    out.index.name = "date"
    return out


def score_and_cache(
    articles_path: Path,
    cache_path: Path,
    scorer: FinBertScorer | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Считает FinBERT по статьям и кэширует в parquet. Повторный вызов читает кэш."""
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)
    articles = pd.read_parquet(articles_path)
    daily = daily_finbert_signals(articles, scorer=scorer)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(cache_path)
    return daily
