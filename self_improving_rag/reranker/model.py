"""
CrossEncoder model wrapper for the Self-Improving RAG system.

Handles loading, caching, and swapping the HuggingFace CrossEncoder model.
Supports both the pretrained checkpoint and fine-tuned model saved locally.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from sentence_transformers.cross_encoder import CrossEncoder

from self_improving_rag.core.config import (
    RERANKER_MODEL,
    RERANKER_SAVED_PATH,
)
from self_improving_rag.core.exceptions import RerankingFailedError

logger = logging.getLogger(__name__)

# Module-level singleton
_cross_encoder: Optional[CrossEncoder] = None


def _fine_tuned_exists() -> bool:
    """
    Check whether any fine-tuned model checkpoint exists on disk.
    """
    if not os.path.exists(RERANKER_SAVED_PATH):
        return False
    
    # Check for direct model files or checkpoint subdirectories
    if (Path(RERANKER_SAVED_PATH) / "config.json").exists():
        return True
        
    checkpoints = [d for d in os.listdir(RERANKER_SAVED_PATH) if d.startswith("checkpoint_")]
    return len(checkpoints) > 0


def load_model(force_pretrained: bool = False) -> CrossEncoder:
    """
    Load the CrossEncoder model, preferring the fine-tuned checkpoint if available.

    On first call the model is cached in the module-level singleton.
    Subsequent calls return the cached model.

    Args:
        force_pretrained: If True, always load the pretrained HuggingFace model
                          and ignore any fine-tuned checkpoint.

    Returns:
        CrossEncoder: A ready-to-use cross-encoder model.

    Raises:
        RerankingFailedError: If the model fails to load.
    """
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder

    model_path = _choose_model_path(force_pretrained)
    try:
        logger.info(f"Loading CrossEncoder from: {model_path}")
        _cross_encoder = CrossEncoder(model_path, max_length=512)
        logger.info("CrossEncoder loaded successfully.")
        return _cross_encoder
    except Exception as exc:
        raise RerankingFailedError(f"Failed to load CrossEncoder from {model_path}: {exc}") from exc


def _choose_model_path(force_pretrained: bool) -> str:
    """
    Determine which model path to load from.
    
    Priority:
      1. Newest fine-tuned checkpoint subdirectory
      2. Direct fine-tuned model files in RERANKER_SAVED_PATH
      3. Pretrained HuggingFace model
    """
    if force_pretrained or not os.path.exists(RERANKER_SAVED_PATH):
        logger.info(f"Using pretrained model: {RERANKER_MODEL}")
        return RERANKER_MODEL

    # Look for the newest checkpoint folder
    checkpoints = [
        d for d in os.listdir(RERANKER_SAVED_PATH) 
        if d.startswith("checkpoint_") and os.path.isdir(os.path.join(RERANKER_SAVED_PATH, d))
    ]
    
    if checkpoints:
        # Sort by name (which has timestamp) and take the latest
        newest = sorted(checkpoints)[-1]
        path = os.path.join(RERANKER_SAVED_PATH, newest)
        logger.info(f"Using newest fine-tuned checkpoint: {path}")
        return path

    # Fallback to direct files in the folder
    if (Path(RERANKER_SAVED_PATH) / "config.json").exists():
        logger.info(f"Fine-tuned model found at root of {RERANKER_SAVED_PATH}. Loading it.")
        return str(RERANKER_SAVED_PATH)

    logger.info(f"No fine-tuned model found in {RERANKER_SAVED_PATH}. Using pretrained: {RERANKER_MODEL}")
    return RERANKER_MODEL


def reload_model() -> CrossEncoder:
    """
    Force-reload the model, clearing the cached instance first.

    Useful after fine-tuning completes to pick up the new checkpoint.

    Returns:
        CrossEncoder: Freshly loaded model (fine-tuned if available).
    """
    global _cross_encoder
    _cross_encoder = None
    return load_model()


def get_model() -> CrossEncoder:
    """
    Return the cached model, loading it if not yet initialized.

    Returns:
        CrossEncoder: Active cross-encoder model.
    """
    return load_model()


def get_model_source() -> str:
    """
    Return a human-readable string indicating which model is active.

    Returns:
        str: 'fine-tuned (local)' or 'pretrained (HuggingFace)'.
    """
    if _fine_tuned_exists():
        return f"fine-tuned ({RERANKER_SAVED_PATH})"
    return f"pretrained ({RERANKER_MODEL})"
