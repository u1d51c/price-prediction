"""Скачивает дневные OHLCV-цены через yfinance и кэширует в parquet.

Один раз скачали - больше не дёргаем сеть, если только не передали --refresh.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils.config import PROJECT_ROOT, get_data_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def _safe_filename(ticker: str) -> str:
    """Тикеры типа BZ=F или BRENT/USD нельзя класть в имя файла как есть."""
    return ticker.replace("=", "_").replace("/", "_").replace("\\", "_")


def fetch_one(
    ticker: str,
    start_date: str,
    end_date: str | None,
    interval: str,
    cache_dir: Path,
    refresh: bool = False,
) -> pd.DataFrame:
    """Один тикер: проверяем кэш, иначе качаем с Yahoo Finance и сохраняем.

    end_date=None - значит "до сегодня". refresh=True игнорирует кэш.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_safe_filename(ticker)}.parquet"

    # если кэш есть и не просили --refresh - отдаём его
    if cache_path.exists() and not refresh:
        df = pd.read_parquet(cache_path)
        if not df.empty:
            logger.info("Кэш-хит для %s (%d строк)", ticker, len(df))
            return df

    effective_end = end_date or date.today().isoformat()
    logger.info("Скачиваю %s: %s -> %s, шаг %s", ticker, start_date, effective_end, interval)

    df = yf.download(
        ticker,
        start=start_date,
        end=effective_end,
        interval=interval,
        # auto_adjust=False - нам нужны сырые цены, корректировки на сплиты применим сами при необходимости
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        raise RuntimeError(f"yfinance вернул пустой ответ для {ticker}")

    # yfinance иногда отдаёт MultiIndex колонки даже на один тикер - спрямляем
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # приводим колонки и индекс к стандартному виду
    df = df.rename(columns=str)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    df.to_parquet(cache_path)
    logger.info("Сохранён %s -> %s (%d строк)", ticker, cache_path.relative_to(PROJECT_ROOT), len(df))
    return df


def fetch_all(refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Качает все активы из configs/data.yaml."""
    cfg = get_data_config()
    assets = cfg["assets"]
    prices_cfg = cfg["prices"]

    cache_dir = PROJECT_ROOT / prices_cfg["cache_dir"]
    out: dict[str, pd.DataFrame] = {}

    for asset_key, asset in assets.items():
        df = fetch_one(
            ticker=asset["ticker"],
            start_date=prices_cfg["start_date"],
            end_date=prices_cfg["end_date"],
            interval=prices_cfg["interval"],
            cache_dir=cache_dir,
            refresh=refresh,
        )
        out[asset_key] = df

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачать дневные цены Brent/WTI/BTC")
    parser.add_argument("--refresh", action="store_true", help="Игнорировать кэш и перекачать заново")
    args = parser.parse_args()

    prices = fetch_all(refresh=args.refresh)
    for key, df in prices.items():
        logger.info(
            "%s: строк=%d, период %s - %s",
            key,
            len(df),
            df.index.min().date(),
            df.index.max().date(),
        )


if __name__ == "__main__":
    main()
