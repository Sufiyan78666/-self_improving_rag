"""
Feedback processor for the Self-Improving RAG system.

Aggregates multiple feedback signals for a single (query, chunk) pair
into a normalized relevance score in the range [-1.0, 1.0].
"""

import logging
from typing import Dict, List

from self_improving_rag.feedback.schema import FeedbackEvent, SignalType

logger = logging.getLogger(__name__)

# Weight configuration for different signals
# (User requested weights in project description)
SIGNAL_WEIGHTS = {
    SignalType.THUMBS_UP: 1.0,
    SignalType.THUMBS_DOWN: -1.0,
    SignalType.CITATION_CLICK: 0.8,
    SignalType.DWELL: 0.3,      # typically applied if dwell > 30s
    SignalType.RE_QUERY: -0.4,
}


def compute_relevance_score(events: List[FeedbackEvent]) -> float:
    """
    Aggregate a list of feedback events into a total relevance score.

    The score is calculated as a weighted sum of signals and then
    clamped to the range [-1.0, 1.0].

    Args:
        events: List of FeedbackEvent objects for a specific chunk.

    Returns:
        float: Normalized relevance score in [-1.0, 1.0].
    """
    if not events:
        return 0.0

    total_score = 0.0
    for event in events:
        weight = SIGNAL_WEIGHTS.get(event.signal_type, 0.0)

        # For dwell, we only apply the weight if it's over the threshold
        if event.signal_type == SignalType.DWELL:
            if event.value >= 30.0:
                total_score += weight
        else:
            # For thumbs up/down and clicks, we multiply by the weight
            total_score += weight * event.value

    # Clamp the result to [-1, 1]
    final_score = max(-1.0, min(1.0, total_score))
    return final_score


def aggregate_session_feedback(events: List[FeedbackEvent]) -> Dict[str, float]:
    """
    Group events by chunk_id and compute the total relevance score for each.

    Args:
        events: List of all feedback events in a session.

    Returns:
        Dict[str, float]: Mapping of chunk_id -> relevance_score.
    """
    chunk_groups: Dict[str, List[FeedbackEvent]] = {}
    for event in events:
        if event.chunk_id not in chunk_groups:
            chunk_groups[event.chunk_id] = []
        chunk_groups[event.chunk_id].append(event)

    return {
        chunk_id: compute_relevance_score(chunk_events)
        for chunk_id, chunk_events in chunk_groups.items()
    }
