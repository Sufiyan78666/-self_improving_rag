"""
A/B testing support for the Self-Improving RAG system.

Allows routing users between different reranker models (e.g., pretrained vs fine-tuned)
and comparing their aggregate feedback performance.
"""

import hashlib
import logging
from typing import List, Optional

from self_improving_rag.reranker.registry import registry

logger = logging.getLogger(__name__)


class ABTester:
    """
    Groups users into A/B buckets and tracks metric differences.
    """

    @staticmethod
    def get_bucket(session_id: str) -> str:
        """
        Deterministically assign a session to bucket A or B.

        Args:
            session_id: UUID string.

        Returns:
            str: 'A' or 'B'
        """
        # Hash the session ID and take modulo 2
        hash_val = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
        return "A" if hash_val % 2 == 0 else "B"

    @staticmethod
    def route_reranker(session_id: str, model_a: str = "pretrained", model_b: str = "fine-tuned") -> str:
        """
        Determine which reranker model to use for a given session.

        Args:
            session_id: UUID string.
            model_a: Registry name for variant A.
            model_b: Registry name for variant B.

        Returns:
            str: The name of the model to use.
        """
        bucket = ABTester.get_bucket(session_id)
        
        # Check if model_b exist in registry, fall back to A
        available = registry.list_registered()
        if bucket == "B" and model_b in available:
            return model_b
        return model_a
