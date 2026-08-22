"""Alembic environment for Relief-RL."""

from __future__ import annotations

import os

from alembic import context
from sqlalchemy import engine_from_config, pool

from backend.app.db import Base
from backend.app.db_models import HazardRecord  # noqa: F401 - registers model metadata

config = context.config
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = os.getenv("DATABASE_URL", config.get_main_option("sqlalchemy.url"))
    if not url:
        raise RuntimeError("DATABASE_URL is required for database migrations")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True,
                      dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    url = os.getenv("DATABASE_URL")
    if url:
        configuration["sqlalchemy.url"] = url
    if not configuration.get("sqlalchemy.url"):
        raise RuntimeError("DATABASE_URL is required for database migrations")
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
