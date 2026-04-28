"""
Prompt templates for the Self-Improving RAG system.

Defines the system instructions and the format for injecting
retrieved context chunks into the LLM prompt.
"""

# System instructions to guide the LLM's behavior
SYSTEM_PROMPT = (
    "You are a factual and concise research assistant. "
    "Your goal is to answer the user's question using ONLY the provided context chunks. "
    "If the answer is not in the context, say you don't know; do not make up information.\n\n"
    "CITATION RULES:\n"
    "1. Every claim must be cited using the format [ID] at the end of the sentence.\n"
    "2. Each source chunk has a unique ID provided in the header.\n"
    "3. Use multiple citations like [1][2] if a sentence draws from multiple sources.\n"
    "4. Keep the output in professional, academic markdown format."
)

# Template for human message containing query and context
HUMAN_PROMPT_TEMPLATE = (
    "CONTEXT CHUNKS:\n"
    "{context_blocks}\n\n"
    "USER QUESTION: {query}\n\n"
    "ANSWER:"
)

# Template for an individual context block
CONTEXT_BLOCK_TEMPLATE = (
    "--- SOURCE ID: {chunk_id} ---\n"
    "From: {source_doc}\n"
    "Content: {text}\n"
)


def format_prompt(query: str, chunks: list[dict]) -> str:
    """
    Format the human prompt with retrieved context and the user query.

    Args:
        query: User's question.
        chunks: List of reranked chunk dicts with 'chunk_id', 'source_doc', and 'text'.

    Returns:
        str: Formatted human prompt string.
    """
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        block = CONTEXT_BLOCK_TEMPLATE.format(
            chunk_id=i,
            source_doc=chunk.get("source_doc", "unknown"),
            text=chunk.get("text", "")
        )
        context_blocks.append(block)

    return HUMAN_PROMPT_TEMPLATE.format(
        context_blocks="\n".join(context_blocks),
        query=query
    )
