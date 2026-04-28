"""
CrossEncoder fine-tuning trainer for the Self-Improving RAG system.

Orchestrates the fine-tuning process using user feedback collected
in the database. Implements MSE regression training.
"""

import logging
import os
from datetime import datetime
from typing import Optional

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.losses import CosineSimilarityLoss  # Not used for CE, CE uses internal MSE usually

from self_improving_rag.core.config import (
    RERANKER_SAVED_PATH,
    TRAINING_EPOCHS,
    MIN_FEEDBACK_TO_TRAIN,
)
from self_improving_rag.core.exceptions import ModelTrainingError
from self_improving_rag.reranker.model import get_model, reload_model
from self_improving_rag.storage.queries import get_feedback_count
from self_improving_rag.training.dataset import get_training_examples

logger = logging.getLogger(__name__)


async def train_reranker(epochs: int = TRAINING_EPOCHS, batch_size: int = 16) -> str:
    """
    Fine-tune the active CrossEncoder on accumulated user feedback.

    The model is trained using MSE loss on the [0, 1] rescaled relevance scores.
    The fine-tuned checkpoint is saved to a timestamped subdirectory and 
    then symlinked (or copied) to the primary RERANKER_SAVED_PATH.

    Returns:
        str: Path to the saved model checkpoint.

    Raises:
        ModelTrainingError: If training fails or insufficient data is available.
    """
    # 1. Check data availability
    feedback_count = await get_feedback_count()
    if feedback_count < MIN_FEEDBACK_TO_TRAIN:
        msg = f"Insufficient feedback ({feedback_count}/{MIN_FEEDBACK_TO_TRAIN}) to start training."
        logger.info(msg)
        return ""

    # 2. Get Data
    train_examples = await get_training_examples()
    if not train_examples:
        raise ModelTrainingError("No training samples found despite feedback count check.")

    # 3. Load Model
    model: CrossEncoder = get_model()

    # 4. Prepare DataLoader
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        train_examples, 
        shuffle=True, 
        batch_size=batch_size,
        collate_fn=model.smart_batching_collate
    )

    # 5. Run Training
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(RERANKER_SAVED_PATH, f"checkpoint_{timestamp}")
        
        logger.info(f"Starting CrossEncoder fine-tuning for {epochs} epochs...")
        
        # Ensure the directory exists
        abs_output_path = os.path.abspath(output_path)
        os.makedirs(abs_output_path, exist_ok=True)
        
        # CrossEncoder.fit defaults to MSE for float labels
        model.fit(
            train_dataloader=train_dataloader,
            epochs=epochs,
            output_path=abs_output_path,
            show_progress_bar=True
        )
        
        # Explicitly save if the directory is empty (some versions of ST fit don't save automatically)
        if not os.listdir(abs_output_path):
            logger.info(f"Directory empty after fit. Explicitly saving to {abs_output_path}")
            model.save(abs_output_path)
        
        logger.info(f"Training complete. Model saved to {abs_output_path}")

        # Note: We no longer copy files to a 'latest' folder because 
        # model.py now automatically finds the newest checkpoint folder.
        
        # Force reload the singleton to use the new model
        reload_model()
        
        return abs_output_path

    except Exception as e:
        raise ModelTrainingError(f"Fine-tuning failed: {e}") from e


def _update_latest_model(checkpoint_path: str) -> None:
    """
    Deprecated: Model loading now finds newest checkpoint automatically.
    """
    pass
