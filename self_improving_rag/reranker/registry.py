"""
Reranker model registry for the Self-Improving RAG system.

Provides a central place to register, list, and swap reranker model
configurations without touching any other code. Swap models via
config.py or environment variable — no code changes required.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from self_improving_rag.core.config import RERANKER_MODEL, RERANKER_SAVED_PATH

logger = logging.getLogger(__name__)


@dataclass
class RerankerConfig:
    """
    Descriptor for a registered reranker model.

    Attributes:
        name: Human-readable name for the configuration.
        model_path: HuggingFace model ID or local path.
        description: Short description of the model.
        max_length: Maximum token sequence length.
        is_default: Whether this is the default model.
    """
    name: str
    model_path: str
    description: str
    max_length: int = 512
    is_default: bool = False


class RerankerRegistry:
    """
    Registry that maps names to RerankerConfig entries.

    Allows swapping the active reranker at runtime by name.
    The active model is loaded lazily via model.get_model().
    """

    def __init__(self) -> None:
        self._configs: Dict[str, RerankerConfig] = {}
        self._active_name: Optional[str] = None

    def register(self, config: RerankerConfig) -> None:
        """
        Register a new reranker configuration.

        Args:
            config: RerankerConfig to register.
        """
        self._configs[config.name] = config
        if config.is_default or self._active_name is None:
            self._active_name = config.name
        logger.info(f"Registered reranker: '{config.name}' → {config.model_path}")

    def set_active(self, name: str) -> None:
        """
        Set the active reranker by registered name.

        Args:
            name: Name of a previously registered RerankerConfig.

        Raises:
            KeyError: If the name is not registered.
        """
        if name not in self._configs:
            raise KeyError(
                f"Reranker '{name}' not in registry. "
                f"Available: {list(self._configs.keys())}"
            )
        self._active_name = name
        # Clear the model singleton so the new model is loaded lazily
        import self_improving_rag.reranker.model as _model_mod
        _model_mod._cross_encoder = None
        logger.info(f"Active reranker set to '{name}'.")

    def get_active_config(self) -> RerankerConfig:
        """
        Return the currently active RerankerConfig.

        Returns:
            RerankerConfig: Active configuration.

        Raises:
            RuntimeError: If no reranker is registered.
        """
        if self._active_name is None:
            raise RuntimeError("No reranker registered. Call registry.register() first.")
        return self._configs[self._active_name]

    def list_registered(self) -> List[str]:
        """
        Return a list of all registered reranker names.

        Returns:
            List[str]: Registered model names.
        """
        return list(self._configs.keys())

    def get_active_model_path(self) -> str:
        """
        Return the model_path of the currently active reranker.

        Returns:
            str: HuggingFace ID or local path.
        """
        return self.get_active_config().model_path


# ──────────────────────────────────────────
# Global singleton registry + built-in configs
# ──────────────────────────────────────────

registry = RerankerRegistry()

# Register the pretrained baseline
registry.register(RerankerConfig(
    name="pretrained",
    model_path=RERANKER_MODEL,
    description="Pretrained ms-marco MiniLM cross-encoder (HuggingFace baseline).",
    is_default=True,
))

# Register the fine-tuned slot (points at local save path)
registry.register(RerankerConfig(
    name="fine-tuned",
    model_path=RERANKER_SAVED_PATH,
    description="Fine-tuned cross-encoder from user feedback.",
))

# Auto-promote to fine-tuned if checkpoint exists
import os as _os
if _os.path.exists(_os.path.join(RERANKER_SAVED_PATH, "config.json")):
    registry.set_active("fine-tuned")
    logger.info("Auto-promoted to fine-tuned reranker from saved checkpoint.")
