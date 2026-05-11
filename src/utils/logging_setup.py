"""Общая настройка логирования с одинаковым форматом во всех модулях."""

from __future__ import annotations

import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Возвращает логгер с компактным форматом. Повторный вызов не дублирует хендлеры."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        # формат: время | уровень | имя_модуля | сообщение - этого хватает для отладки сбора данных
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(level)
        # отключаем propagate, чтобы не получать дубли через root-logger
        logger.propagate = False
    return logger
