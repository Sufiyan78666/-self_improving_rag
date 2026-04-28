"""
Feedback capture logic for the Self-Improving RAG system.

Provides functions to validate and persist user feedback signals
to the backend database.
"""

import logging
from typing import Optional

from self_improving_rag.core.exceptions import FeedbackProcessingError
from self_improving_rag.feedback.schema import FeedbackEvent, SignalType
from self_improving_rag.storage.queries import insert_feedback

logger = logging.getLogger(__name__)


async def log_feedback(
    session_id: str,
    chunk_id: str,
    signal_type: str,
    value: float = 1.0
) -> str:
    """
    Validate and store a feedback event.

    Args:
        session_id: UUID of the session.
        chunk_id: ID of the targeted chunk.
        signal_type: Type of signal (validated against SignalType).
        value: Numeric value (e.g. 1.0 for click, 30 for dwell).

    Returns:
        str: ID of the created database record.

    Raises:
        FeedbackProcessingError: If the signal is invalid or storage fails.
    """
    # 1. Type validation
    try:
        sig = SignalType(signal_type)
    except ValueError:
        raise FeedbackProcessingError(f"Invalid signal type: {signal_type}")

    # 2. Basic range validation
    if sig in [SignalType.THUMBS_UP, SignalType.THUMBS_DOWN, SignalType.CITATION_CLICK] and value != 1.0:
        logger.warning(f"Binary signal {sig} logged with non-standard value {value}. Overriding to 1.0.")
        value = 1.0

    # 3. Persistence
    try:
        event_id = await insert_feedback(
            session_id=session_id,
            chunk_id=chunk_id,
            signal_type=sig.value,
            value=value
        )
        logger.info(f"Captured feedback: {sig} for chunk {chunk_id} in {session_id}")
        return event_id
    except Exception as exc:
        raise FeedbackProcessingError(f"Failed to persist feedback: {exc}") from exc
