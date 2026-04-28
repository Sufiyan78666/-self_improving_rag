"""Test script for the reranker module (Module 4)."""
import sys
sys.path.insert(0, '.')

# Simulate retrieved chunks (as retriever would return them)
chunks = [
    {"chunk_id": "a1", "text": "The sky is blue due to Rayleigh scattering of sunlight.", "vector_score": 0.7, "source_doc": "physics.txt"},
    {"chunk_id": "a2", "text": "Retrieval Augmented Generation improves LLM factual accuracy.", "vector_score": 0.65, "source_doc": "ml.txt"},
    {"chunk_id": "a3", "text": "Cross-encoder models score query-passage pairs jointly.", "vector_score": 0.6, "source_doc": "ml.txt"},
    {"chunk_id": "a4", "text": "The capital of France is Paris.", "vector_score": 0.55, "source_doc": "geo.txt"},
]

query = "How does RAG improve language models?"

# Test features
from self_improving_rag.reranker.features import build_pairs, build_training_pairs
pairs = build_pairs(query, chunks)
print(f"Pairs built: {len(pairs)}")

training_input = [("How does RAG work?", "RAG combines retrieval with generation.", 1.0)]
formatted = build_training_pairs(training_input)
print(f"Training pairs formatted: {formatted[0]}")

# Test inference
from self_improving_rag.reranker.inference import rerank
print("Loading CrossEncoder model (first-time download may take a moment)...")
top_chunks = rerank(query, chunks, top_n=2)
print(f"Top {len(top_chunks)} chunks after reranking:")
for c in top_chunks:
    print(f"  rank={c['reranked_rank']} score={c['reranker_score']:.4f} text={c['text'][:60]}")

# Test registry
from self_improving_rag.reranker.registry import registry
print(f"Registered rerankers: {registry.list_registered()}")
print(f"Active reranker: {registry.get_active_config().name}")

# Test model source
from self_improving_rag.reranker.model import get_model_source
print(f"Model source: {get_model_source()}")

print("ALL RERANKER TESTS PASSED")
