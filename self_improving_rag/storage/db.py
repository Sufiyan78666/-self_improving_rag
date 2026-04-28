"""
Database connection and initialization for the Self-Improving RAG system.

Uses aiosqlite for async SQLite access (development).
Schema is PostgreSQL-compatible for production migration.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

import aiosqlite

from self_improving_rag.core.config import DB_PATH
from self_improving_rag.core.exceptions import DatabaseConnectionError

logger = logging.getLogger(__name__)

# Path to migration files
MIGRATIONS_DIR = Path(__file__).parent / "migrations"

# Module-level connection pool (single connection for SQLite)
_db_connection: Optional[aiosqlite.Connection] = None


async def get_connection() -> aiosqlite.Connection:
    """
    Get the shared async database connection.

    Returns the existing connection if already initialized,
    otherwise creates and initializes a new one.

    Returns:
        aiosqlite.Connection: An active async database connection.

    Raises:
        DatabaseConnectionError: If the connection cannot be established.
    """
    global _db_connection
    if _db_connection is None:
        _db_connection = await _create_connection()
    return _db_connection


async def _create_connection() -> aiosqlite.Connection:
    """
    Create a new aiosqlite connection and apply all pending migrations.

    Returns:
        aiosqlite.Connection: A newly created and migrated database connection.

    Raises:
        DatabaseConnectionError: If connection or migration fails.
    """
    try:
        # Ensure the database directory exists
        db_path = Path(DB_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = await aiosqlite.connect(str(db_path), timeout=20)
        # Enable WAL for better concurrent read performance
        await conn.execute("PRAGMA journal_mode=WAL;")
        # Enable foreign keys
        await conn.execute("PRAGMA foreign_keys=ON;")
        conn.row_factory = aiosqlite.Row
        logger.info(f"Connected to database at {db_path}")

        await _run_migrations(conn)
        return conn
    except Exception as exc:
        raise DatabaseConnectionError(
            f"Failed to connect to database at {DB_PATH}: {exc}"
        ) from exc


async def _run_migrations(conn: aiosqlite.Connection) -> None:
    """
    Apply all SQL migration files in order from the migrations directory.

    Migrations are applied in filename-sorted order. Only .sql files
    are considered.

    Args:
        conn: An active aiosqlite connection.

    Raises:
        DatabaseConnectionError: If a migration fails to apply.
    """
    migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    for migration_path in migration_files:
        logger.info(f"Applying migration: {migration_path.name}")
        try:
            sql = migration_path.read_text(encoding="utf-8")
            await conn.executescript(sql)
            await conn.commit()
            logger.info(f"Migration applied: {migration_path.name}")
        except Exception as exc:
            raise DatabaseConnectionError(
                f"Migration {migration_path.name} failed: {exc}"
            ) from exc


async def close_connection() -> None:
    """
    Close the shared database connection gracefully.

    Safe to call even if no connection is open.
    """
    global _db_connection
    if _db_connection is not None:
        await _db_connection.close()
        _db_connection = None
        logger.info("Database connection closed.")


async def init_db() -> None:
    """
    Public entry point: initialize the database and run migrations.

    Should be called once at application startup.

    Raises:
        DatabaseConnectionError: If the database cannot be initialized.
    """
    await get_connection()
    logger.info("Database initialized successfully.")
