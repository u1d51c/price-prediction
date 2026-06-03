"""Настройка клиента MLflow для подключения к локальному tracking-серверу.

Все параметры лежат в configs/mlflow.yaml. Артефакты пишутся в MinIO
(S3-совместимое хранилище), для общения с ним нужны переменные окружения
MLFLOW_S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY -
устанавливаем их программно из конфига.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow

from src.utils.config import PROJECT_ROOT, load_yaml


def setup_mlflow(config_path: str | Path = "configs/mlflow.yaml") -> dict[str, Any]:
    """Читает конфиг, ставит переменные окружения для S3 и tracking_uri.

    Возвращает словарь с настройками (на случай если нужны в коде).
    """
    cfg = load_yaml(config_path)["mlflow"]

    # переменные для boto3, чтобы артефакты улетали в MinIO
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = cfg["s3_endpoint_url"]
    os.environ["AWS_ACCESS_KEY_ID"] = cfg["aws_access_key_id"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = cfg["aws_secret_access_key"]

    mlflow.set_tracking_uri(cfg["tracking_uri"])

    # создаём эксперимент, если ещё нет
    exp = mlflow.get_experiment_by_name(cfg["experiment_name"])
    if exp is None:
        # явно указываем артефакт-локацию, иначе MLflow попытается mlflow-artifacts:/...
        mlflow.create_experiment(
            cfg["experiment_name"],
            artifact_location="s3://mlflow/",
        )
    mlflow.set_experiment(cfg["experiment_name"])
    return cfg
