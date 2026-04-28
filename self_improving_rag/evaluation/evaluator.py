"""
Evaluation controller for the Self-Improving RAG system.

Runs a fixed evaluation set (query -> expected chunks) through the
full retrieval and reranking pipeline and reports aggregate metrics.
"""

import logging
from typing import Dict, List, Set, Tuple

from self_improving_rag.evaluation.metrics import ndcg_at_k, mrr, precision_at_k, recall_at_k

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Handles systematic evaluation of the RAG pipeline.

    Attributes:
        eval_set: List of (query, set_of_relevant_chunk_ids) tuples.
        k: Default K for metrics calculation.
    """

    def __init__(self, eval_set: List[Tuple[str, Set[str]]], k: int = 3) -> None:
        self.eval_set = eval_set
        self.k = k

    async def evaluate_pipeline(self, pipeline_func) -> Dict[str, float]:
        """
        Run the evaluation set through a provided pipeline function.

        Args:
            pipeline_func: An async function that takes a query string and
                           returns a list of reranked chunk dicts.

        Returns:
            Dict[str, float]: Aggregated metrics (average across eval set).
        """
        total_ndcg = 0.0
        total_mrr = 0.0
        total_p_at_k = 0.0
        total_r_at_k = 0.0
        n = len(self.eval_set)

        if n == 0:
            return {}

        logger.info(f"Starting evaluation on {n} queries...")

        for query, relevant_ids in self.eval_set:
            # 1. Run pipeline
            results = await pipeline_func(query)
            retrieved_ids = [r["chunk_id"] for r in results]
            
            # For NDCG, we use binary relevance (1 if relevant, 0 otherwise)
            # though in a real scenario we might have graded relevance.
            scores = [1.0 if rid in relevant_ids else 0.0 for rid in retrieved_ids]

            # 2. Compute metrics
            total_ndcg += ndcg_at_k(retrieved_ids, scores, self.k)
            total_mrr += mrr(retrieved_ids, relevant_ids)
            total_p_at_k += precision_at_k(retrieved_ids, relevant_ids, self.k)
            total_r_at_k += recall_at_k(retrieved_ids, relevant_ids, self.k)

        metrics = {
            "ndcg_at_k": total_ndcg / n,
            "mrr": total_mrr / n,
            "precision_at_k": total_p_at_k / n,
            "recall_at_k": total_r_at_k / n,
        }

        logger.info(f"Evaluation complete: NDCG@{self.k} = {metrics['ndcg_at_k']:.4f}")
        return metrics
