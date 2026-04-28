"""
Citation mapping logic for the Self-Improving RAG system.

Extracts [ID] tags from the LLM-generated answer and maps them
back to the original source chunks for UI display and feedback.
"""

import re
from typing import Dict, List, Set, Tuple


def extract_citations(text: str) -> Set[str]:
    """
    Extract all unique citation IDs from the text in [ID] format.

    Args:
        text: The LLM-generated answer text.

    Returns:
        Set[str]: A set of unique chunk IDs found in the text.
    """
    # Regex matches patterns like [3], [abc-123], [5][7]
    pattern = r"\[([a-zA-Z0-9\-_]+)\]"
    matches = re.findall(pattern, text)
    return set(matches)


def map_citations_to_chunks(
    text: str,
    chunks: List[dict]
) -> List[dict]:
    """
    Filter the list of retrieved chunks to only include those cited in the text.

    Args:
        text: The LLM-generated answer text.
        chunks: The list of chunks that were provided to the LLM.

    Returns:
        List[dict]: A list of chunks that were actually cited in the response.
    """
    cited_ids = extract_citations(text)
    cited_chunks = []
    
    # Map numerical IDs back to chunks by their 1-indexed position
    for i, chunk in enumerate(chunks, start=1):
        if str(i) in cited_ids:
            # We add a temporary display_id for the UI
            enriched_chunk = dict(chunk)
            enriched_chunk["display_id"] = i
            cited_chunks.append(enriched_chunk)
            
    return cited_chunks


def get_citation_indices(text: str) -> List[Tuple[int, int, str]]:
    """
    Find all citation locations and IDs in the text.
    Useful for highlighting in the UI or building clickable areas.

    Args:
        text: The LLM-generated answer text.

    Returns:
        List of (start_idx, end_idx, citation_id) tuples.
    """
    pattern = r"\[([a-zA-Z0-9\-_]+)\]"
    matches = []
    for m in re.finditer(pattern, text):
        matches.append((m.start(), m.end(), m.group(1)))
    return matches
