"""
Experiment tracker for the Self-Improving RAG system.

Logs training run statistics and performance metrics to the database
to monitor improvement over time.
"""

import logging
import uuid
import asyncio
from datetime import datetime
from typing import Optional

from self_improving_rag.storage.db import get_connection
from self_improving_rag.core.exceptions import DatabaseConnectionError

logger = logging.getLogger(__name__)


async def log_experiment_run(
    model_name: str,
    dataset_size: int,
    epochs: int,
    ndcg_before: Optional[float] = None,
    ndcg_after: Optional[float] = None,
    mrr_after: Optional[float] = None,
) -> str:
    """
    Log a training run experiment with metrics to the database.

    Args:
        model_name: Name/path of the base model.
        dataset_size: Total number of training pairs used.
        epochs: Number of training epochs.
        ndcg_before: Initial performance metric.
        ndcg_after: Final performance metric.
        mrr_after: Reciprocal rank metric.

    Returns:
        str: UUID of the logged run.

    Raises:
        DatabaseConnectionError: If storage fails.
    """
    run_id = str(uuid.uuid4())
    try:
        conn = await get_connection()
        await conn.execute(
            """
            INSERT INTO experiment_runs
                (id, model_name, dataset_size, epochs, ndcg_before, ndcg_after, mrr_after, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                model_name,
                dataset_size,
                epochs,
                ndcg_before,
                ndcg_after,
                mrr_after,
                datetime.utcnow().isoformat(),
            ),
        )
        await conn.commit()
        logger.info(f"Logged experiment run {run_id}: dataset_size={dataset_size}, NDCG_after={ndcg_after}")
        return run_id
    except Exception as exc:
        raise DatabaseConnectionError(f"Failed to log experiment run: {exc}") from exc


async def get_latest_experiments(limit: int = 5) -> list[dict]:
    """
    Retrieve the most recent experiment runs.

    Returns:
        List of dicts containing run details.
    """
    try:
        conn = await get_connection()
        async with conn.execute(
            "SELECT * FROM experiment_runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception as exc:
        logger.error(f"Failed to fetch experiments: {exc}")
        return []
