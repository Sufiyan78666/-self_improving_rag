"""Test script for the evaluation module (Module 8)."""
import sys, asyncio
sys.path.insert(0, '.')

from self_improving_rag.evaluation.metrics import precision_at_k, recall_at_k, mrr, ndcg_at_k
from self_improving_rag.evaluation.evaluator import RAGEvaluator
from self_improving_rag.evaluation.ab_test import ABTester
from self_improving_rag.evaluation.report import generate_report_markdown

async def test_metrics():
    # 1. Test Metrics
    retrieved = ["c1", "c2", "c3"]
    relevant = {"c1", "c3"}
    
    p = precision_at_k(retrieved, relevant, 2)
    print(f"P@2: {p}") # Expect 1/2 = 0.5
    assert p == 0.5
    
    r = recall_at_k(retrieved, relevant, 2)
    print(f"R@2: {r}") # Expect 1/2 = 0.5
    assert r == 0.5
    
    m = mrr(retrieved, relevant)
    print(f"MRR: {m}") # Expect 1/1 = 1.0
    assert m == 1.0
    
    # NDCG: c1=1, c2=0, c3=1
    n = ndcg_at_k(retrieved, [1.0, 0.0, 1.0], 3)
    print(f"NDCG@3: {n:.4f}")
    assert n > 0.7

    # 2. Test Evaluator (Mock Pipeline)
    async def mock_pipeline(query):
        return [{"chunk_id": "c1"}, {"chunk_id": "c4"}, {"chunk_id": "c3"}]
    
    eval_set = [("query1", {"c1", "c3"})]
    evaluator = RAGEvaluator(eval_set, k=3)
    results = await evaluator.evaluate_pipeline(mock_pipeline)
    print(f"Evaluator results: {results}")
    assert results["ndcg_at_k"] > 0.0

    # 3. Test AB Tester
    bucket1 = ABTester.get_bucket("session-1")
    bucket2 = ABTester.get_bucket("session-2")
    print(f"Buckets: {bucket1}, {bucket2}")

    # 4. Test Report
    report = generate_report_markdown(results, "Mock Model")
    print("\n--- Report Preview ---")
    print(report)
    
    print("\nALL EVALUATION MODULE TESTS PASSED")

if __name__ == "__main__":
    asyncio.run(test_metrics())
