"""Alembic env: подгружает URL и metadata из настроек сервиса.

Зачем кастом-env. По умолчанию Alembic читает URL из alembic.ini, что
дублирует знание о подключении. Мы хотим один источник истины -
`service.config.Settings`. Поэтому здесь:
1. Импортируем `Base` из `service.db`, чтобы Alembic видел все таблицы.
2. Подменяем `sqlalchemy.url` значением из настроек, перед тем как
   создать engine.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Корень проекта в PYTHONPATH - для импорта service.* работал из alembic/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from service.config import get_settings  # noqa: E402
from service.db import Base  # noqa: E402  - импорт для регистрации моделей в metadata
import service.db  # noqa: F401, E402

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Подменяем URL - он переопределит то, что лежит в alembic.ini.
config.set_main_option("sqlalchemy.url", get_settings().database_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=url.startswith("sqlite"),  # ALTER TABLE для SQLite требует batch
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        is_sqlite = str(connection.engine.url).startswith("sqlite")
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=is_sqlite,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
