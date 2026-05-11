"""Сбор сигналов из GDELT 2.0 Doc API.

GDELT - открытый сервис мониторинга мировых новостей. Дёргаем три вещи:
число статей в день по запросу (TimelineVolRaw), их средний тон по WordNet
(TimelineTone), и небольшую выборку самих статей (ArtList) для FinBERT'а
позже.

Документация API: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
Ключ не нужен, но есть жёсткие rate-limit (поэтому здесь throttling и retry).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import PROJECT_ROOT, get_data_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# Эмпирически: чаще одного запроса в ~5 секунд GDELT отдаёт 429.
# У ArtList лимит ещё строже, но мы там обходимся одной попыткой за запуск.
GDELT_MIN_INTERVAL_SEC = 5.0
_last_request_ts: float = 0.0


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=4, min=4, max=120))
def _gdelt_request(params: dict) -> dict:
    """Один запрос к GDELT с rate-limiting и retry на 429/5xx."""
    global _last_request_ts
    # ждём, чтобы между запросами было не меньше GDELT_MIN_INTERVAL_SEC секунд
    elapsed = time.monotonic() - _last_request_ts
    if elapsed < GDELT_MIN_INTERVAL_SEC:
        time.sleep(GDELT_MIN_INTERVAL_SEC - elapsed)
    _last_request_ts = time.monotonic()

    response = requests.get(GDELT_DOC_API, params=params, timeout=60)
    if response.status_code == 429:
        # tenacity сам ретрайнет, но логируем чтобы было видно в console
        logger.warning("GDELT rate limit 429, retry...")
    response.raise_for_status()
    # на запросах без совпадений GDELT возвращает пустую строку - это не JSON, парсить нельзя
    text = response.text.strip()
    if not text:
        return {}
    try:
        return response.json()
    except json.JSONDecodeError:
        logger.warning("GDELT вернул не-JSON для %s: %s", params.get("mode"), text[:200])
        return {}


def fetch_timeline(
    query: str,
    mode: str,
    start: str,
    end: str,
    source_lang: str = "english",
) -> pd.DataFrame:
    """Один timeline-режим GDELT (VolRaw / Tone / Lang) за период start..end."""
    start_dt = datetime.strptime(start, "%Y-%m-%d").strftime("%Y%m%d%H%M%S")
    end_dt = datetime.strptime(end, "%Y-%m-%d").strftime("%Y%m%d%H%M%S")

    # фильтр по языку - берём только англоязычные источники, иначе попадут переводы и куча шума
    full_query = f'{query} sourcelang:{source_lang}'

    params = {
        "query": full_query,
        "mode": mode,
        "format": "json",
        "startdatetime": start_dt,
        "enddatetime": end_dt,
        # timezoom=yes - попросить детализацию в зависимости от длины окна
        "timezoom": "yes",
    }
    payload = _gdelt_request(params)

    if not payload or "timeline" not in payload:
        logger.warning("Пустой ответ GDELT для mode=%s, query=%s", mode, query[:60])
        return pd.DataFrame()

    rows = []
    for series in payload["timeline"]:
        series_name = series.get("series", "value")
        for point in series.get("data", []):
            rows.append(
                {
                    "datetime": point["date"],
                    "series": series_name,
                    "value": point["value"],
                    # norm - нормировка на общее число статей в GDELT в этот день
                    # (есть только в VolRaw, фоновый шум по разным дням разный)
                    "norm": point.get("norm"),
                }
            )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # формат даты у GDELT: YYYYMMDDTHHMMSSZ
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%dT%H%M%SZ", errors="coerce")
    df = df.dropna(subset=["datetime"])
    return df


def fetch_article_sample(
    query: str,
    start: str,
    end: str,
    max_records: int = 250,
    source_lang: str = "english",
) -> pd.DataFrame:
    """Возвращает до max_records заголовков статей по запросу (ArtList mode).

    На больших периодах API режет результаты, реальный лимит - около 250 записей
    на запрос. Используется для качественного анализа и для FinBERT-инференса.
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d").strftime("%Y%m%d%H%M%S")
    end_dt = datetime.strptime(end, "%Y-%m-%d").strftime("%Y%m%d%H%M%S")

    params = {
        "query": f"{query} sourcelang:{source_lang}",
        "mode": "ArtList",
        "format": "json",
        "startdatetime": start_dt,
        "enddatetime": end_dt,
        "maxrecords": min(max_records, 250),
        # DateDesc - берём свежие сначала
        "sort": "DateDesc",
    }
    payload = _gdelt_request(params)
    if not payload or "articles" not in payload:
        return pd.DataFrame()

    df = pd.DataFrame(payload["articles"])
    if "seendate" in df.columns:
        df["seendate"] = pd.to_datetime(df["seendate"], format="%Y%m%dT%H%M%SZ", errors="coerce")
    return df


def _semiannual_chunks(start: str, end: str) -> list[tuple[str, str]]:
    """Разбивает интервал на полугодовые куски (для пакетной заливки)."""
    bounds = pd.date_range(start=start, end=end, freq="2QS").strftime("%Y-%m-%d").tolist()
    if not bounds or bounds[0] > start:
        bounds = [start] + bounds
    if bounds[-1] < end:
        bounds.append(end)
    return list(zip(bounds[:-1], bounds[1:]))


def fetch_for_topic(
    topic_key: str,
    query: str,
    start: str,
    end: str,
    out_dir: Path,
    source_lang: str = "english",
    article_sample_window_months: int = 12,
    article_sample_max: int = 250,
) -> None:
    """Полная заливка одной темы: оба timeline за всю историю + сэмпл статей.

    Timeline тащим полугодовыми кусками - на таком окне GDELT не теряет
    подневное разрешение. Уже скачанные куски не перекачиваем (resume-логика),
    поэтому при обрыве можно перезапустить без потери прогресса.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # промежуточные парты складываем в отдельный подкаталог, потом склеиваем
    chunks_dir = out_dir / f"_chunks_{topic_key}"
    chunks_dir.mkdir(exist_ok=True)

    timelines = ["TimelineVolRaw", "TimelineTone"]
    chunks = _semiannual_chunks(start, end)

    for chunk_start, chunk_end in chunks:
        for mode in timelines:
            chunk_path = chunks_dir / f"{mode}_{chunk_start}_{chunk_end}.parquet"
            # resume: если файл уже есть - не качаем повторно
            if chunk_path.exists():
                continue
            try:
                df = fetch_timeline(query=query, mode=mode, start=chunk_start,
                                    end=chunk_end, source_lang=source_lang)
            except Exception as exc:
                # одна сломанная пачка не должна валить всё - просто пропускаем
                logger.error("[%s] %s %s..%s: %s", topic_key, mode, chunk_start, chunk_end, exc)
                continue
            if not df.empty:
                df["chunk_start"] = chunk_start
                df["chunk_end"] = chunk_end
                df.to_parquet(chunk_path)
        logger.info("[%s] полугодие %s..%s готово", topic_key, chunk_start, chunk_end)

    # склеиваем парты в один итоговый parquet на каждый timeline-режим
    for mode in timelines:
        files = sorted(chunks_dir.glob(f"{mode}_*.parquet"))
        if not files:
            continue
        combined = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        combined = combined.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
        out_path = out_dir / f"gdelt_{topic_key}_{mode}.parquet"
        combined.to_parquet(out_path)
        logger.info("Сохранено %s: %d строк -> %s", mode, len(combined), out_path.relative_to(PROJECT_ROOT))

    # отдельно - выборка статей за последние ~12 мес (нужна для FinBERT)
    art_end = pd.to_datetime(end)
    art_start = art_end - pd.DateOffset(months=article_sample_window_months)
    try:
        art = fetch_article_sample(
            query=query,
            start=art_start.strftime("%Y-%m-%d"),
            end=art_end.strftime("%Y-%m-%d"),
            max_records=article_sample_max,
            source_lang=source_lang,
        )
    except Exception as exc:
        # сэмпл - необязательная штука, если упало - продолжаем без него
        logger.warning("[%s] ArtList упал: %s", topic_key, exc)
        art = pd.DataFrame()

    if not art.empty:
        path = out_dir / f"gdelt_{topic_key}_articles_sample.parquet"
        art.to_parquet(path)
        logger.info("Сэмпл статей %s: %d строк -> %s", topic_key, len(art), path.relative_to(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачать сигналы GDELT 2.0 для нефти и BTC")
    parser.add_argument("--start", default=None, help="Дата начала YYYY-MM-DD; по умолчанию из конфига")
    parser.add_argument("--end", default=None, help="Дата конца YYYY-MM-DD; по умолчанию сегодня")
    args = parser.parse_args()

    cfg = get_data_config()
    if not cfg["texts"]["gdelt"]["enabled"]:
        logger.warning("GDELT отключён в configs/data.yaml")
        return

    start = args.start or cfg["prices"]["start_date"]
    end = args.end or datetime.today().strftime("%Y-%m-%d")

    out_dir = PROJECT_ROOT / cfg["texts_storage"]["raw_dir"] / "gdelt"
    queries = cfg["texts"]["gdelt"]["queries"]

    for topic_key, query in queries.items():
        logger.info("=== Тема %s ===", topic_key)
        fetch_for_topic(topic_key=topic_key, query=query, start=start, end=end, out_dir=out_dir)


if __name__ == "__main__":
    main()
