"""Isolate model loading."""
import logging
logging.basicConfig(level=logging.INFO)
from self_improving_rag.reranker.model import get_model
print("Attempting to load CrossEncoder model...")
model = get_model()
print("Model loaded successfully.")
