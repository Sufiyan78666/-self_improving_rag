"""
Top-K retriever for the Self-Improving RAG system.

Embeds an incoming query and retrieves the most similar chunks
from the FAISS vector store, returning enriched metadata dicts.
"""

import logging
from typing import List, Optional

import numpy as np

from self_improving_rag.core.config import RETRIEVE_K, HYBRID_ENABLED
from self_improving_rag.core.exceptions import RetrievalFailedError, EmbeddingFailedError
from self_improving_rag.retrieval.ingest import embed_query
from self_improving_rag.retrieval.vector_store import get_store
from self_improving_rag.retrieval.hybrid import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


def retrieve(
    query: str,
    k: int = RETRIEVE_K,
    precomputed_embedding: Optional[np.ndarray] = None,
) -> List[dict]:
    """
    Retrieve the top-K most relevant chunks for a query using hybrid search.

    Combines dense FAISS search and sparse BM25 search via Reciprocal Rank Fusion (RRF).
    """
    store = get_store()

    if store.total_vectors == 0:
        raise RetrievalFailedError(
            "Vector store is empty. Ingest documents first."
        )

    # 1. Coordinate query vector
    try:
        if precomputed_embedding is not None:
            query_vec = precomputed_embedding
        else:
            query_vec = embed_query(query)
    except Exception as exc:
        raise EmbeddingFailedError(f"Query embedding failed: {exc}") from exc

    # 2. Perform searches
    if not HYBRID_ENABLED:
        try:
            results = store.search_dense(query_vec, k=k)
            # Standardize output
            return [{
                "chunk_id": item["chunk_id"],
                "text": item["text"],
                "source_doc": item["source_doc"],
                "score": float(item["score"]),
                "rank": int(item["rank"])
            } for item in results]
        except Exception as exc:
            raise RetrievalFailedError(f"Dense search failed: {exc}") from exc

    # Hybrid Flow
    try:
        dense_results = store.search_dense(query_vec, k=k)
        sparse_results = store.search_sparse(query, k=k)
        
        fused_results = reciprocal_rank_fusion(dense_results, sparse_results)
        
        # Standardize for RAG Pipeline
        output = []
        for item in fused_results[:k]:
            output.append({
                "chunk_id": item["chunk_id"],
                "text": item.get("text", ""),
                "source_doc": item.get("source_doc", ""),
                "score": float(item.get("rrf_score", 0.0)),
                "rank": int(item.get("fused_rank", 0))
            })
        
        logger.info(f"Hybrid retrieval returned {len(output)} chunks.")
        return output

    except Exception as exc:
        raise RetrievalFailedError(f"Hybrid search failed: {exc}") from exc
