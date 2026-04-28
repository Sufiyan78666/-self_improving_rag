"""
Custom exceptions for the Self-Improving RAG system.
"""

class RAGException(Exception):
    """Base exception for all RAG system errors."""
    pass

class ChunkNotFoundError(RAGException):
    """Raised when a specific document chunk cannot be found in the store."""
    pass

class EmbeddingFailedError(RAGException):
    """Raised when text embedding generation fails."""
    pass

class RetrievalFailedError(RAGException):
    """Raised when the vector database retrieval fails."""
    pass

class RerankingFailedError(RAGException):
    """Raised when the cross-encoder fails to rerank chunks."""
    pass

class GenerationFailedError(RAGException):
    """Raised when the LLM fails to generate a response."""
    pass

class FeedbackProcessingError(RAGException):
    """Raised when feedback events cannot be processed or stored."""
    pass

class ModelTrainingError(RAGException):
    """Raised when the fine-tuning process encounters an error."""
    pass

class DatabaseConnectionError(RAGException):
    """Raised when connecting or executing queries on the database fails."""
    pass
