"""
Hybrid Search module for the Self-Improving RAG system.

Combines results from Dense Retrieval (FAISS) and Sparse Retrieval (BM25)
using Reciprocal Rank Fusion (RRF).
"""

import logging
from typing import List, Dict
from self_improving_rag.core.config import RRF_K

logger = logging.getLogger(__name__)

def reciprocal_rank_fusion(
    dense_results: List[Dict], 
    sparse_results: List[Dict], 
    k: int = RRF_K
) -> List[Dict]:
    """
    Fuses two ranked lists of results using Reciprocal Rank Fusion.
    
    Formula: score = sum(1 / (k + rank_i))
    
    Args:
        dense_results: List of chunks from dense search with 'chunk_id' and 'rank'.
        sparse_results: List of chunks from sparse search with 'chunk_id' and 'rank'.
        k: Smoothing constant for RRF (default 60).
        
    Returns:
        Fused and re-ranked list of chunks.
    """
    fused_scores = {} # chunk_id -> rrf_score
    chunk_map = {}   # chunk_id -> metadata
    
    # Process Dense Results
    for rank, doc in enumerate(dense_results, start=1):
        chunk_id = doc["chunk_id"]
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        if chunk_id not in chunk_map:
            chunk_map[chunk_id] = doc

    # Process Sparse Results
    for rank, doc in enumerate(sparse_results, start=1):
        chunk_id = doc["chunk_id"]
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        if chunk_id not in chunk_map:
            chunk_map[chunk_id] = doc

    # Sort by fused score descending
    sorted_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    final_results = []
    for rank, (chunk_id, score) in enumerate(sorted_ids, start=1):
        doc = chunk_map[chunk_id].copy()
        doc["rrf_score"] = score
        doc["fused_rank"] = rank
        final_results.append(doc)
        
    logger.info(f"Hybrid Fusion complete. Combined {len(dense_results)} dense and {len(sparse_results)} sparse results into {len(final_results)} unique docs.")
    return final_results
