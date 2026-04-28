"""
Feedback schemas for the Self-Improving RAG system.

Defines the structure of feedback events captured from the UI.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class SignalType(str, Enum):
    """Supported feedback signal types."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    CITATION_CLICK = "citation_click"
    DWELL = "dwell"
    RE_QUERY = "re_query"


@dataclass
class FeedbackEvent:
    """
    Data structure representing a single user feedback signal.

    Attributes:
        session_id: UUID of the RAG session.
        chunk_id: ID of the specific chunk this feedback relates to.
        signal_type: Type of signal (thumbs_up, dwell, etc.).
        value: Numeric value (1.0 for binary, seconds for dwell).
        created_at: Timestamp of the event.
    """
    session_id: str
    chunk_id: str
    signal_type: SignalType
    value: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert the event to a dictionary for database storage."""
        return {
            "session_id": self.session_id,
            "chunk_id": self.chunk_id,
            "signal_type": self.signal_type.value,
            "value": self.value,
            "created_at": self.created_at.isoformat(),
        }
