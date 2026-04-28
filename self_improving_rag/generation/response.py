"""
Response schema for the Self-Improving RAG system.

Defines the structure of the payload returned to the UI or API clients.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RAGResponse:
    """
    Final output object for a user query.

    Attributes:
        session_id: UUID of the session for feedback tracking.
        query: The user's original question.
        answer: The LLM-generated answer text.
        cited_chunks: List of chunk metadata provided by the retriever/reranker
                      that were actually cited in the answer.
        all_retrieved_chunks: The complete list of chunks retrieved/reranked,
                              including those not cited.
        tokens_used: Optional count of tokens consumed (if available).
        metadata: Any extra info (e.g., model name, latency).
    """
    session_id: str
    query: str
    answer: str
    cited_chunks: List[dict] = field(default_factory=list)
    all_retrieved_chunks: List[dict] = field(default_factory=list)
    tokens_used: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert the response to a JSON-serializable dictionary."""
        return {
            "session_id": self.session_id,
            "query": self.query,
            "answer": self.answer,
            "cited_chunks": self.cited_chunks,
            "all_retrieved_chunks": self.all_retrieved_chunks,
            "tokens_used": self.tokens_used,
            "metadata": self.metadata,
        }
