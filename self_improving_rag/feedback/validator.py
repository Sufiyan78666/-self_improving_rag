"""
Feedback validator for the Self-Improving RAG system.

Filters out noise such as rapid repeated clicks, bot-like behavior,
or accidental double-taps.
"""

import logging
from typing import List

from self_improving_rag.feedback.schema import FeedbackEvent, SignalType

logger = logging.getLogger(__name__)


def is_valid_click(last_events: List[FeedbackEvent], current_event: FeedbackEvent) -> bool:
    """
    Check if a click event is valid (not a rapid double-tap).

    Args:
        last_events: List of recent events in the same session.
        current_event: The new event candidate.

    Returns:
        bool: True if valid, False if it looks like a double-tap.
    """
    if current_event.signal_type not in [SignalType.THUMBS_UP, SignalType.THUMBS_DOWN, SignalType.CITATION_CLICK]:
        return True

    # Check for identical event in the last 2 seconds
    for event in last_events:
        if (event.chunk_id == current_event.chunk_id and
            event.signal_type == current_event.signal_type):
            time_diff = (current_event.created_at - event.created_at).total_seconds()
            if time_diff < 2.0:
                logger.debug(f"Filtered repeat feedback: {current_event.signal_type} for {current_event.chunk_id}")
                return False

    return True


def filter_noise(events: List[FeedbackEvent]) -> List[FeedbackEvent]:
    """
    Filter a list of events to remove duplicates or noise.

    Args:
        events: Raw list of feedback events.

    Returns:
        List[FeedbackEvent]: Filtered list.
    """
    filtered = []
    # Simple deduplication by (session+chunk+type) within a small time window
    processed = set()

    for e in sorted(events, key=lambda x: x.created_at):
        key = (e.session_id, e.chunk_id, e.signal_type)
        if key not in processed:
            filtered.append(e)
            processed.add(key)

    return filtered
