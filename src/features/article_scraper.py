"""Скрейпер тела статей по URL.

GDELT в ArtList отдаёт только title/url/seendate. Чтобы получить body
статьи, ходим напрямую по URL'у через requests + trafilatura.
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd
import requests
import trafilatura

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


# вежливый User-Agent — без него часть сайтов отдаёт 403
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_body(url: str, timeout: int = 15) -> Optional[str]:
    """Скачивает страницу и извлекает основной текст через trafilatura.

    Возвращает body или None, если страница не открылась / не парсится.
    """
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
    except Exception as exc:
        logger.debug("HTTP error %s: %s", url[:80], exc)
        return None

    if r.status_code != 200 or not r.text:
        return None

    # trafilatura умеет выкидывать навигацию, рекламу, шапку/подвал
    body = trafilatura.extract(
        r.text,
        include_comments=False,
        include_tables=False,
        favor_recall=False,  # предпочитаем точность тексту
    )
    if not body:
        return None
    return body.strip()


def fetch_bodies_for_articles(
    articles_df: pd.DataFrame,
    url_col: str = "url",
    rate_limit_sec: float = 0.3,
    max_articles: int | None = None,
) -> pd.DataFrame:
    """Качает body последовательно (медленно, но проще для отладки).

    Для production используйте fetch_bodies_parallel().
    """
    df = articles_df.copy().reset_index(drop=True)
    if max_articles is not None:
        df = df.head(max_articles)

    bodies: list[Optional[str]] = []
    ok = 0
    for i, url in enumerate(df[url_col], 1):
        body = fetch_body(url)
        bodies.append(body)
        if body:
            ok += 1
        if i % 25 == 0 or i == len(df):
            logger.info("Скачано %d/%d (успешно: %d)", i, len(df), ok)
        time.sleep(rate_limit_sec)

    df["body"] = bodies
    return df


def fetch_bodies_parallel(
    articles_df: pd.DataFrame,
    url_col: str = "url",
    max_workers: int = 10,
    timeout_per_url: int = 15,
    log_every: int = 50,
) -> pd.DataFrame:
    """Параллельный скрейпинг body через ThreadPoolExecutor.

    Скорость ~ 1/(max_workers / 2) сек на статью. Для 2000 URL и 10 потоков
    обычно 5-10 минут.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    df = articles_df.copy().reset_index(drop=True)
    bodies = [None] * len(df)
    done = 0
    ok = 0

    def _task(idx: int, url: str):
        return idx, fetch_body(url, timeout=timeout_per_url)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_task, i, u) for i, u in enumerate(df[url_col])]
        for f in as_completed(futures):
            try:
                i, body = f.result()
            except Exception:
                continue
            bodies[i] = body
            if body:
                ok += 1
            done += 1
            if done % log_every == 0 or done == len(df):
                logger.info("Скачано %d/%d (успешно: %d, %.1f%%)",
                            done, len(df), ok, 100*ok/max(done, 1))

    df["body"] = bodies
    return df
