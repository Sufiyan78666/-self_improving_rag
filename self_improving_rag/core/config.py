"""
Configuration settings for the Self-Improving RAG system.
This module serves as the single source of truth for all configurable parameters.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Model Configurations ---
# SentenceTransformer model for embeddings
EMBED_MODEL: str = "all-MiniLM-L6-v2"

# HuggingFace CrossEncoder model for reranking
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Path to save/load the fine-tuned reranker
RERANKER_SAVED_PATH: str = os.path.join("models", "reranker")

# LLM Provider (gemini or ollama)
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()

# LLM model for generation
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")

# Ollama specific settings
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# NVIDIA specific settings
NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY")

# --- Retrieval & Reranking Settings ---
# Number of chunks to retrieve from the vector store initially
RETRIEVE_K: int = 10

# Number of chunks to keep after reranking
RERANK_TOP_N: int = 3

# Chunk size for document ingestion
CHUNK_SIZE: int = 500

# Chunk overlap for document ingestion
CHUNK_OVERLAP: int = 50

# --- Training Configurations ---
# Minimum number of feedback pairs required before triggering fine-tuning
MIN_FEEDBACK_TO_TRAIN: int = 50

# Number of epochs for fine-tuning
TRAINING_EPOCHS: int = 3

# Minimum NDCG improvement required to accept new model
NDCG_IMPROVEMENT_THRESHOLD: float = 0.01

# --- Hybrid Search Settings ---
HYBRID_ENABLED: bool = True
RRF_K: int = 60  # Typical constant for Reciprocal Rank Fusion
BM25_K1: float = 1.5
BM25_B: float = 0.75

# --- Additional Core settings ---
# Default database path (relative to project root)
DB_PATH: str = os.getenv("DB_PATH", os.path.join("data", "rag.db"))
