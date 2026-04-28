"""
Integration test for Hybrid Search (Dense + Sparse).
"""
import asyncio
import sys
import os
import numpy as np
sys.path.insert(0, '.')

from self_improving_rag.retrieval.ingest import ingest_file
from self_improving_rag.retrieval.retriever import retrieve
from self_improving_rag.retrieval.vector_store import get_store

async def test_hybrid():
    print("Step 1: Cleaning and Re-Ingesting data...")
    store = get_store()
    # Manual reset for clean test
    if os.path.exists("data/chunks"):
        import shutil
        shutil.rmtree("data/chunks")
    store.__init__() # Reset singleton state
    
    test_file = "test_hybrid_data.txt"
    with open(test_file, "w") as f:
        f.write("The quick brown fox jumps over the lazy dog.\n")
        f.write("Quantum computing relies on qubits to perform complex calculations.\n")
        f.write("Deep learning models like Transformers have revolutionized natural language processing.\n")
    
    ingest_file(test_file)
    print("Ingestion complete.")
    
    print("\nStep 2: Testing Sparse-focused query (Keywords)...")
    results = retrieve("Quantum qubits calculations")
    print(f"Found {len(results)} results.")
    for res in results:
        print(f"- [{res.get('chunk_id')}] Score: {res.get('rrf_score', 'N/A'):.4f} | Source: {res.get('source_doc')} | Text: {res.get('text')[:60]}...")
    
    print("\nStep 3: Testing Dense-focused query (Semantics)...")
    results = retrieve("artificial intelligence and machine learning progress")
    print(f"Found {len(results)} results.")
    for res in results:
        print(f"- [{res.get('chunk_id')}] Score: {res.get('rrf_score', 'N/A'):.4f} | Text: {res.get('text')[:60]}...")

if __name__ == "__main__":
    asyncio.run(test_hybrid())
