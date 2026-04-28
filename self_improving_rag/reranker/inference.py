"""
CrossEncoder inference: rerank retrieved chunks.

Takes raw FAISS retrieval results and returns them sorted by
the CrossEncoder's relevance scores, keeping only top-N.
"""

import logging
from typing import List

from self_improving_rag.core.config import RERANK_TOP_N
from self_improving_rag.core.exceptions import RerankingFailedError
from self_improving_rag.reranker.model import get_model
from self_improving_rag.reranker.features import build_pairs

logger = logging.getLogger(__name__)


def rerank(
    query: str,
    chunks: List[dict],
    top_n: int = RERANK_TOP_N,
) -> List[dict]:
    """
    Rerank a list of retrieved chunks using the CrossEncoder model.

    Each chunk dict is enriched with a 'reranker_score' field.
    Returns the top-N chunks sorted by descending reranker score.

    Args:
        query: Raw user query string.
        chunks: List of chunk dicts from the retriever. Each must have 'text'.
        top_n: Number of top chunks to return after reranking.

    Returns:
        List of dicts (at most top_n) sorted by descending reranker_score.
        Each dict contains all original retrieval fields plus:
            - reranker_score (float)
            - reranked_rank (int, 1-indexed)

    Raises:
        RerankingFailedError: If the CrossEncoder predict call fails.
    """
    if not chunks:
        return []

    model = get_model()
    pairs = build_pairs(query, chunks)

    try:
        scores = model.predict(pairs, show_progress_bar=False)
    except Exception as exc:
        raise RerankingFailedError(f"CrossEncoder predict failed: {exc}") from exc

    # Attach reranker scores to each chunk
    scored_chunks = []
    for chunk, score in zip(chunks, scores):
        enriched = dict(chunk)
        enriched["reranker_score"] = float(score)
        scored_chunks.append(enriched)

    # Sort by descending reranker score
    scored_chunks.sort(key=lambda x: x["reranker_score"], reverse=True)

    # Assign reranked positions and limit to top_n
    top_chunks = scored_chunks[:top_n]
    for rank, chunk in enumerate(top_chunks, start=1):
        chunk["reranked_rank"] = rank

    logger.info(
        f"Reranked {len(chunks)} chunks → top {len(top_chunks)} | "
        f"Best score: {top_chunks[0]['reranker_score']:.4f}"
    )
    return top_chunks
