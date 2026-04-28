"""
Database query functions for the Self-Improving RAG system.

Provides typed async functions for all common database operations:
- Session creation
- Retrieval logging
- Feedback event insertion
- Training pair extraction
"""

import uuid
import logging
from datetime import datetime
from typing import List, Optional, Tuple, Any

import aiosqlite

from self_improving_rag.storage.db import get_connection
from self_improving_rag.core.exceptions import DatabaseConnectionError

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Session operations
# ──────────────────────────────────────────────

async def insert_session(
    query: str,
    query_embedding: Optional[bytes] = None,
    session_id: Optional[str] = None,
) -> str:
    """
    Insert a new RAG query session into the database.

    Args:
        query: The raw user query string.
        query_embedding: Serialized numpy bytes of the query embedding (optional).
        session_id: Optional UUID. If not provided, a new one is generated.

    Returns:
        str: The UUID of the session.
    """
    session_id = session_id or str(uuid.uuid4())
    try:
        conn: aiosqlite.Connection = await get_connection()
        await conn.execute(
            """
            INSERT OR IGNORE INTO rag_sessions (id, query, query_embedding, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, query, query_embedding, datetime.utcnow().isoformat()),
        )
        await conn.commit()
        logger.debug(f"Inserted session {session_id}")
        return session_id
    except Exception as exc:
        raise DatabaseConnectionError(f"insert_session failed: {exc}") from exc


# ──────────────────────────────────────────────
# Retrieval operations
# ──────────────────────────────────────────────

async def insert_retrieval(
    session_id: str,
    chunk_id: str,
    chunk_text: str,
    rank_shown: int,
    score: float,
    reranker_score: float,
    source_doc: Optional[str] = None,
) -> str:
    """
    Insert a single chunk retrieval record for a session.

    Args:
        session_id: UUID of the parent session.
        chunk_id: Unique identifier for the chunk.
        chunk_text: Full text content of the chunk.
        rank_shown: 1-indexed position shown to the user.
        score: Raw retrieval score (vector or hybrid).
        reranker_score: CrossEncoder score after reranking.
        source_doc: Source document filename (optional).

    Returns:
        str: The UUID of the newly created retrieval record.

    Raises:
        DatabaseConnectionError: If the insert fails.
    """
    retrieval_id = str(uuid.uuid4())
    try:
        conn: aiosqlite.Connection = await get_connection()
        await conn.execute(
            """
            INSERT INTO rag_retrievals
                (id, session_id, chunk_id, chunk_text, source_doc, rank_shown,
                 vector_score, reranker_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                retrieval_id,
                session_id,
                chunk_id,
                chunk_text,
                source_doc,
                rank_shown,
                score,
                reranker_score,
                datetime.utcnow().isoformat(),
            ),
        )
        await conn.commit()
        logger.debug(f"Inserted retrieval {retrieval_id} for session {session_id}")
        return retrieval_id
    except Exception as exc:
        raise DatabaseConnectionError(f"insert_retrieval failed: {exc}") from exc


async def insert_retrievals_batch(
    session_id: str,
    retrievals: List[dict],
) -> List[str]:
    """
    Batch-insert multiple retrieval records in a single transaction.

    Args:
        session_id: UUID of the parent session.
        retrievals: List of dicts with keys: chunk_id, chunk_text,
                    rank_shown, vector_score, reranker_score, source_doc (optional).

    Returns:
        List[str]: UUIDs of the inserted retrieval records.

    Raises:
        DatabaseConnectionError: If the batch insert fails.
    """
    ids: List[str] = []
    rows: List[Tuple] = []
    for r in retrievals:
        rid = str(uuid.uuid4())
        ids.append(rid)
        rows.append((
            rid,
            session_id,
            r["chunk_id"],
            r.get("text", ""),
            r.get("source_doc"),
            r.get("rank_shown", r.get("rank", 0)),
            r.get("score", 0.0),
            r.get("reranker_score", 0.0),
            datetime.utcnow().isoformat(),
        ))
    try:
        conn: aiosqlite.Connection = await get_connection()
        await conn.executemany(
            """
            INSERT INTO rag_retrievals
                (id, session_id, chunk_id, chunk_text, source_doc, rank_shown,
                 vector_score, reranker_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        await conn.commit()
        logger.debug(f"Batch inserted {len(rows)} retrievals for session {session_id}")
        return ids
    except Exception as exc:
        raise DatabaseConnectionError(f"insert_retrievals_batch failed: {exc}") from exc


# ──────────────────────────────────────────────
# Feedback operations
# ──────────────────────────────────────────────

async def insert_feedback(
    session_id: str,
    chunk_id: str,
    signal_type: str,
    value: float,
) -> str:
    """
    Insert a single feedback event into the database.

    Args:
        session_id: UUID of the associated session.
        chunk_id: UUID/ID of the chunk the feedback is about.
        signal_type: One of: thumbs_up, thumbs_down, citation_click, dwell, re_query.
        value: Raw signal value (typically 1.0 for binary signals, seconds for dwell).

    Returns:
        str: The UUID of the newly created feedback record.

    Raises:
        DatabaseConnectionError: If the insert fails.
    """
    feedback_id = str(uuid.uuid4())
    try:
        conn: aiosqlite.Connection = await get_connection()
        await conn.execute(
            """
            INSERT INTO feedback_events (id, session_id, chunk_id, signal_type, value, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (feedback_id, session_id, chunk_id, signal_type, value, datetime.utcnow().isoformat()),
        )
        await conn.commit()
        logger.debug(f"Inserted feedback {feedback_id}: {signal_type}={value} for chunk {chunk_id}")
        return feedback_id
    except Exception as exc:
        raise DatabaseConnectionError(f"insert_feedback failed: {exc}") from exc


async def get_feedback_count() -> int:
    """
    Get the total number of feedback events stored in the database.

    Returns:
        int: Total count of feedback events.

    Raises:
        DatabaseConnectionError: If the query fails.
    """
    try:
        conn: aiosqlite.Connection = await get_connection()
        async with conn.execute("SELECT COUNT(*) FROM feedback_events") as cursor:
            row = await cursor.fetchone()
            return int(row[0]) if row else 0
    except Exception as exc:
        raise DatabaseConnectionError(f"get_feedback_count failed: {exc}") from exc


# ──────────────────────────────────────────────
# Training pair extraction
# ──────────────────────────────────────────────

async def get_training_pairs(
    min_relevance: float = -1.0,
) -> List[Tuple[str, str, float]]:
    """
    Retrieve aggregated (query, chunk, relevance_score) training pairs.

    Uses the `training_pairs` view defined in the migration.
    Relevance scores are clamped to [-1, 1].

    Args:
        min_relevance: Minimum absolute relevance to include a pair.
                       Use -1.0 to include all pairs.

    Returns:
        List of (query_text, chunk_text, relevance_score) tuples.

    Raises:
        DatabaseConnectionError: If the query fails.
    """
    try:
        conn: aiosqlite.Connection = await get_connection()
        async with conn.execute(
            """
            SELECT
                query_text,
                chunk_text,
                MAX(-1.0, MIN(1.0, relevance_score_raw)) AS relevance_score
            FROM training_pairs
            WHERE ABS(MAX(-1.0, MIN(1.0, relevance_score_raw))) >= ?
            """,
            (abs(min_relevance),),
        ) as cursor:
            rows = await cursor.fetchall()
            return [(row[0], row[1], float(row[2])) for row in rows]
    except Exception as exc:
        raise DatabaseConnectionError(f"get_training_pairs failed: {exc}") from exc


async def get_last_ndcg_score() -> Optional[float]:
    """
    Retrieve the most recent NDCG score logged in experiment tracking.

    Falls back gracefully if the experiment_runs table does not yet exist.

    Returns:
        Optional[float]: The last NDCG score, or None if not available.
    """
    try:
        conn: aiosqlite.Connection = await get_connection()
        async with conn.execute(
            """
            SELECT ndcg_after FROM experiment_runs
            ORDER BY created_at DESC LIMIT 1
            """
        ) as cursor:
            row = await cursor.fetchone()
            return float(row[0]) if row else None
    except Exception:
        # Table may not exist yet — this is expected during early usage
        return None
