"""Full Integration Test for the Self-Improving RAG Pipeline (Module 9)."""
import os, sys, asyncio
sys.path.insert(0, '.')

# Set API Key for testing if not set (Mocking generation if key is missing)
if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = "MOCK_KEY"

from self_improving_rag.retrieval.ingest import ingest_file
from self_improving_rag.core.pipeline import run_pipeline
from self_improving_rag.storage.db import get_connection

async def test_integration():
    print("Step 1: Ingesting data...")
    test_file = "sample_data.txt"
    with open(test_file, "w") as f:
        f.write("The self-improving RAG system uses a cross-encoder to rerank retrieved documents.\n")
        f.write("User feedback is collected through thumbs up/down and citation clicks.\n")
        f.write("The system uses FAISS for vector search and Gemini Pro for answer generation.\n")
    
    ingest_file(test_file)
    print("Ingestion complete.")

    # Mock generate_response to avoid hanging on API call
    import self_improving_rag.generation.llm
    async def mock_gen(prompt):
        return "The system uses a cross-encoder for reranking [1]."
    self_improving_rag.generation.llm.generate_response = mock_gen
    
    print("\nStep 2: Running pipeline...")
    try:
        response = await run_pipeline("How does the system rerank documents?")
        
        print("\n--- RAG Response ---")
        print(f"Session ID: {response.session_id}")
        print(f"Answer: {response.answer}")
        print(f"Citations: {[c['id'] for c in response.cited_chunks]}")
        print(f"Metadata: {response.metadata}")
        
    except Exception as e:
        if "API_KEY_INVALID" in str(e) or "MOCK_KEY" in str(e):
            print("\n[WARNING] Gemini API call failed as expected (Mock/Invalid Key).")
            print("Pipeline successfully reached the generation stage.")
        else:
            raise e

    print("\nStep 3: Verifying storage...")
    conn = await get_connection()
    async with conn.execute("SELECT query FROM rag_sessions WHERE id = ?", (response.session_id,)) as cursor:
        row = await cursor.fetchone()
        print(f"Stored session query: {row[0]}")
        assert row[0] == "How does the system rerank documents?"

    async with conn.execute("SELECT count(*) FROM rag_retrievals WHERE session_id = ?", (response.session_id,)) as cursor:
        row = await cursor.fetchone()
        print(f"Stored retrievals for session: {row[0]}")
        assert row[0] > 0

    print("\nINTEGRATION TEST PASSED")

if __name__ == "__main__":
    asyncio.run(test_integration())
