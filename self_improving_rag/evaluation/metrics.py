"""
Information Retrieval (IR) metrics for the Self-Improving RAG system.

Implements standard ranking metrics:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)@K
"""

import math
from typing import List, Set, Union


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Calculate Precision at Rank K.
    P@k = (Number of relevant items in top k) / k

    Args:
        retrieved_ids: List of chunk IDs returned by the system (sorted by rank).
        relevant_ids: Set of ground-truth relevant chunk IDs.
        k: The rank threshold.

    Returns:
        float: Precision@k score in [0, 1].
    """
    if k <= 0:
        return 0.0
    
    top_k = retrieved_ids[:k]
    relevant_retrieved = [rid for rid in top_k if rid in relevant_ids]
    return len(relevant_retrieved) / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Calculate Recall at Rank K.
    R@k = (Number of relevant items in top k) / (Total relevant items)

    Args:
        retrieved_ids: List of chunk IDs returned by the system.
        relevant_ids: Set of ground-truth relevant chunk IDs.
        k: The rank threshold.

    Returns:
        float: Recall@k score in [0, 1].
    """
    if not relevant_ids:
        return 0.0
    
    top_k = retrieved_ids[:k]
    relevant_retrieved = [rid for rid in top_k if rid in relevant_ids]
    return len(relevant_retrieved) / len(relevant_ids)


def mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    Calculate Mean Reciprocal Rank.
    MRR = 1 / (rank of first relevant item)

    Args:
        retrieved_ids: List of chunk IDs returned by the system.
        relevant_ids: Set of ground-truth relevant chunk IDs.

    Returns:
        float: Reciprocal rank (0.0 if no relevant item found).
    """
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevance_scores: List[float], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at Rank K.
    NDCG = DCG / IDCG

    Args:
        retrieved_ids: List of chunk IDs returned (not strictly used if scores are provided).
        relevance_scores: List of relevance labels/scores for items in retrieved_ids.
                          Must be ordered same as retrieved_ids.
        k: Rank threshold.

    Returns:
        float: NDCG@k score in [0, 1].
    """
    if not relevance_scores or k <= 0:
        return 0.0

    scores = relevance_scores[:k]
    
    # Calculate DCG: sum( (2^rel - 1) / log2(i + 1) )
    dcg = 0.0
    for i, rel in enumerate(scores, start=1):
        dcg += (2**rel - 1) / math.log2(i + 1)

    # Calculate IDCG: DCG of the perfectly sorted scores
    ideal_scores = sorted(relevance_scores, reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_scores, start=1):
        idcg += (2**rel - 1) / math.log2(i + 1)

    if idcg == 0:
        return 0.0
        
    return dcg / idcg
