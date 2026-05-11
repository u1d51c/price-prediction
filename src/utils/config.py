"""Пути проекта и загрузка YAML-конфигов."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# корень проекта = на два уровня выше src/utils/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Грузит YAML. Относительные пути берутся от корня проекта."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_data_config() -> dict[str, Any]:
    """Шорткат: configs/data.yaml."""
    return load_yaml(CONFIG_DIR / "data.yaml")
