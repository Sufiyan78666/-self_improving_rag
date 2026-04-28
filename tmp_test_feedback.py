"""Test script for the feedback module (Module 6)."""
import sys, asyncio
from datetime import datetime, timedelta
sys.path.insert(0, '.')

from self_improving_rag.feedback.schema import FeedbackEvent, SignalType
from self_improving_rag.feedback.validator import is_valid_click, filter_noise
from self_improving_rag.feedback.processor import compute_relevance_score, aggregate_session_feedback

async def test_logic():
    # 1. Test Aggregation Logic
    events = [
        FeedbackEvent("s1", "c1", SignalType.THUMBS_UP),          # +1.0
        FeedbackEvent("s1", "c1", SignalType.CITATION_CLICK),    # +0.8
        FeedbackEvent("s1", "c2", SignalType.THUMBS_DOWN),        # -1.0
        FeedbackEvent("s1", "c3", SignalType.DWELL, 45.0),       # +0.3
        FeedbackEvent("s1", "c3", SignalType.DWELL, 10.0),       # 0.0 (too short)
    ]
    
    # Chunk c1 score: 1.0 + 0.8 = 1.8 -> clamped to 1.0
    c1_score = compute_relevance_score([events[0], events[1]])
    print(f"c1 score: {c1_score}")
    assert c1_score == 1.0

    # Chunk c3 score: 0.3 + 0.0 = 0.3
    c3_score = compute_relevance_score([events[3], events[4]])
    print(f"c3 score: {c3_score}")
    assert c3_score == 0.3

    # All session aggregation
    session_scores = aggregate_session_feedback(events)
    print(f"Session scores: {session_scores}")
    assert session_scores["c2"] == -1.0

    # 2. Test Validator Logic
    e1 = FeedbackEvent("s1", "c1", SignalType.CITATION_CLICK, created_at=datetime.utcnow())
    e2 = FeedbackEvent("s1", "c1", SignalType.CITATION_CLICK, created_at=datetime.utcnow() + timedelta(seconds=1))
    
    is_valid = is_valid_click([e1], e2)
    print(f"Is second click valid? {is_valid}")
    assert is_valid == False # Too fast

    print("ALL FEEDBACK LOGIC TESTS PASSED")

if __name__ == "__main__":
    asyncio.run(test_logic())
