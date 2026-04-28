"""
Feature builder for the CrossEncoder reranker.

Constructs (query, chunk_text) sentence pairs in the exact format
expected by the HuggingFace CrossEncoder.predict() API.
"""

from typing import List, Tuple


def build_pairs(query: str, chunks: List[dict]) -> List[Tuple[str, str]]:
    """
    Build (query, passage) pairs from a query and a list of chunk dicts.

    This is the input format required by CrossEncoder.predict().

    Args:
        query: The raw user query string.
        chunks: List of chunk dicts, each with at minimum a 'text' key.

    Returns:
        List of (query, chunk_text) 2-tuples, one per chunk.
    """
    return [(query, chunk.get("text", "")) for chunk in chunks]


def build_training_pairs(
    examples: List[Tuple[str, str, float]]
) -> List[Tuple[List[str], float]]:
    """
    Convert (query, chunk_text, relevance_score) triples into the format
    expected by CrossEncoder.fit() — InputExample-style.

    Args:
        examples: List of (query_text, chunk_text, relevance_score) tuples.
                  relevance_score is in [-1, 1]; we rescale to [0, 1] for MSE loss.

    Returns:
        List of ({'texts': [query, chunk]}, label) tuples.
        The CrossEncoder.fit() method accepts these as dicts with 'texts' keys.
    """
    output = []
    for query, chunk_text, score in examples:
        # Rescale [-1, 1] → [0, 1] for MSE regression training
        label = (score + 1.0) / 2.0
        label = max(0.0, min(1.0, label))  # clamp to [0, 1]
        output.append(({"texts": [query, chunk_text]}, label))
    return output
