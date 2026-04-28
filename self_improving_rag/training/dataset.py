"""
Dataset preparation for CrossEncoder fine-tuning.

Extracts training pairs from the storage layer and prepares them
for the sentence-transformers training loop.
"""

import logging
from typing import List

from sentence_transformers import InputExample
from torch.utils.data import DataLoader

from self_improving_rag.storage.queries import get_training_pairs
from self_improving_rag.reranker.features import build_training_pairs

logger = logging.getLogger(__name__)


async def get_training_examples(
    min_relevance: float = 0.3
) -> List[InputExample]:
    """
    Fetch training pairs from the database and prepare them
    for the sentence-transformers training loop.

    Only pairs with an absolute relevance score >= min_relevance are included
    to focus on clear signals.

    Args:
        min_relevance: Threshold for including a pair in training.

    Returns:
        List[InputExample]: Ready for use in CrossEncoder.fit().
    """
    # 1. Fetch raw pairs (query, chunk_text, relevance_score)
    raw_pairs = await get_training_pairs(min_relevance=min_relevance)
    
    if not raw_pairs:
        logger.warning("No training pairs found in database meeting criteria.")
        return []

    # 2. Convert to ( {texts: [q, p]}, label ) format
    # build_training_pairs also rescales [-1, 1] -> [0, 1]
    formatted_data = build_training_pairs(raw_pairs)
    
    # 3. Create InputExample objects
    examples = [
        InputExample(texts=item[0]["texts"], label=item[1])
        for item in formatted_data
    ]

    logger.info(f"Prepared {len(examples)} training samples.")
    return examples
