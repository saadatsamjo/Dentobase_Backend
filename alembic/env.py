# alembic/env.py

from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import os
import sys

# Add your project root to Python path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Import your configuration from the core.config file
from config.appconfig import settings

# import the Base from the connection file
from app.database.connection import Base

# 👇  Registering All Models BEFORE creating tables
import app.model_registry


# this is the Alembic Config object
config = context.config

# Override the sqlalchemy.url with the SYNC database URL from your config
config.set_main_option("sqlalchemy.url", settings.DB_URL_SYNC)

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target_metadata for autogenerate support
target_metadata = Base.metadata

print("\n🗂️ 🗂️ 🗂️ 🗂️ 🗂️ 🗂️ 🗂️ 🗂️ 🗂️ 🗂️ 🗂️ 🗂️ 🗂️ 🗂️\n \ntables found:", list(target_metadata.tables.keys()), "\n\n🗂️ 🗂️ 🗂️ 🗂️ 🗂️ 🗂️ 🗂️ 🗂️ 🗂️ 🗂️ 🗂️ 🗂️ 🗂️ 🗂️\n")


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # optional but handy
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
