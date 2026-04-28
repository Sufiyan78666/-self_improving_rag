"""Test script for the generation module (Module 5)."""
import sys, asyncio
sys.path.insert(0, '.')

from self_improving_rag.generation.prompt import format_prompt
from self_improving_rag.generation.citations import extract_citations, map_citations_to_chunks
from self_improving_rag.generation.response import RAGResponse

# 1. Test Prompt Formatting
chunks = [
    {"chunk_id": "c1", "text": "RAG is cool.", "source_doc": "doc1.txt"},
    {"chunk_id": "c2", "text": "Gemini is fast.", "source_doc": "doc2.txt"},
]
query = "What is cool and fast?"
prompt = format_prompt(query, chunks)
print("--- Formatted Prompt ---")
print(prompt)
print("------------------------")

# 2. Test Citation Extraction
answer = "Based on the context, RAG is cool [c1] and Gemini is fast [c2]."
citations = extract_citations(answer)
print(f"Extracted Citations: {citations}")
assert "c1" in citations and "c2" in citations

# 3. Test Citation Mapping
cited_chunks = map_citations_to_chunks(answer, chunks)
print(f"Cited Chunks: {[c['chunk_id'] for c in cited_chunks]}")
assert len(cited_chunks) == 2

# 4. Test Response Object
resp = RAGResponse(
    session_id="session-123",
    query=query,
    answer=answer,
    cited_chunks=cited_chunks,
    all_retrieved_chunks=chunks,
    metadata={"model": "gemini-mock"}
)
print(f"RAGResponse: {resp.to_dict()['answer']}")

print("ALL GENERATION LOGIC TESTS PASSED (Mocked API)")
