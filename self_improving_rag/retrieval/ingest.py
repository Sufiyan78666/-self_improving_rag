"""
Document ingestion pipeline for the Self-Improving RAG system.

Supports PDF and plain-text (.txt) files.
Flow: load file → extract text → chunk → embed → store in FAISS → save JSON cache.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from self_improving_rag.core.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBED_MODEL,
    OCR_ENABLED,
    OCR_DPI,
    TESSERACT_CMD,
)
from self_improving_rag.core.exceptions import EmbeddingFailedError
from self_improving_rag.retrieval.vector_store import get_store

logger = logging.getLogger(__name__)

_CHUNKS_DIR = os.path.join("data", "chunks")
_CHUNKS_CACHE = os.path.join(_CHUNKS_DIR, "chunks_cache.json")

# Lazy-loaded embedding model (loaded once per process)
_embed_model: Optional[SentenceTransformer] = None


def _get_embed_model() -> SentenceTransformer:
    """
    Load and cache the SentenceTransformer embedding model.

    Returns:
        SentenceTransformer: The shared embedding model instance.
    """
    global _embed_model
    if _embed_model is None:
        logger.info(f"Loading embedding model: {EMBED_MODEL}")
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


# ──────────────────────────────────────────
# Text extraction
# ──────────────────────────────────────────

def _extract_text_from_txt(path: str) -> str:
    """
    Read a plain-text file and return its content.

    Args:
        path: Absolute or relative path to the .txt file.

    Returns:
        str: Full text content of the file.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _extract_text_from_pdf(path: str) -> str:
    """
    Extract text from a PDF file using PyPDF2.

    Falls back to empty string per page if extraction fails.

    Args:
        path: Absolute or relative path to the .pdf file.

    Returns:
        str: Concatenated text of all pages.
    """
    try:
        import PyPDF2  # type: ignore
    except ImportError:
        raise ImportError("PyPDF2 is required for PDF ingestion. Run: pip install PyPDF2")

    text_parts: List[str] = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            try:
                text_parts.append(page.extract_text() or "")
            except Exception:
                text_parts.append("")
    return "\n".join(text_parts)


def _extract_text_from_pdf_ocr(path: str) -> str:
    """
    Extract text from a PDF using OCR (Tesseract via pdf2image).

    Returns empty string if OCR deps are missing.
    """
    try:
        from pdf2image import convert_from_path  # type: ignore
        import pytesseract  # type: ignore
    except ImportError:
        logger.warning("OCR dependencies not installed. Skipping OCR extraction.")
        return ""

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    text_parts: List[str] = []
    try:
        images = convert_from_path(path, dpi=OCR_DPI)
        for image in images:
            text_parts.append(pytesseract.image_to_string(image) or "")
    except Exception as exc:
        logger.warning(f"OCR extraction failed for {path}: {exc}")
        return ""

    return "\n".join(text_parts)


def extract_text(path: str) -> str:
    """
    Dispatch text extraction based on file extension.

    Args:
        path: Path to a .pdf or .txt file.

    Returns:
        str: Extracted text.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        text = _extract_text_from_pdf(path)
        if not text.strip() and OCR_ENABLED:
            logger.info(f"No text found in {path}; attempting OCR.")
            text = _extract_text_from_pdf_ocr(path)
        return text
    elif ext == ".txt":
        return _extract_text_from_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .txt")


# ──────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    source_doc: str = "",
) -> List[dict]:
    """
    Split text into overlapping fixed-size character chunks.

    Each chunk is assigned a stable deterministic ID based on a hash of the content.

    Args:
        text: Full document text to chunk.
        chunk_size: Target size of each chunk in characters.
        overlap: Number of overlapping characters between consecutive chunks.
        source_doc: Source document name to attach as metadata.

    Returns:
        List of dicts with keys: chunk_id, text, source_doc, char_start, char_end.
    """
    chunks: List[dict] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_text_content = text[start:end].strip()

        if chunk_text_content:
            chunk_id = hashlib.md5(
                f"{source_doc}:{start}:{chunk_text_content}".encode()
            ).hexdigest()
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text_content,
                "source_doc": source_doc,
                "char_start": start,
                "char_end": end,
            })

        start += chunk_size - overlap

    logger.info(f"Chunked '{source_doc}' into {len(chunks)} chunks.")
    return chunks


# ──────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────

def embed_chunks(chunks: List[dict], batch_size: int = 64) -> List[np.ndarray]:
    """
    Generate embeddings for a list of chunk dicts.

    Args:
        chunks: List of chunk dicts with a 'text' key.
        batch_size: Number of texts to embed per model batch call.

    Returns:
        List[np.ndarray]: One embedding array per chunk.

    Raises:
        EmbeddingFailedError: If the embedding model call fails.
    """
    model = _get_embed_model()
    texts = [c["text"] for c in chunks]
    try:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            convert_to_numpy=True,
            normalize_embeddings=True,  # pre-normalize for cosine similarity
        )
        return [embeddings[i] for i in range(len(embeddings))]
    except Exception as exc:
        raise EmbeddingFailedError(f"Embedding generation failed: {exc}") from exc


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.

    Args:
        query: User query text.

    Returns:
        np.ndarray: Normalized embedding vector.

    Raises:
        EmbeddingFailedError: If the embedding fails.
    """
    model = _get_embed_model()
    try:
        vec = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vec[0]
    except Exception as exc:
        raise EmbeddingFailedError(f"Query embedding failed: {exc}") from exc


# ──────────────────────────────────────────
# Cache
# ──────────────────────────────────────────

def _load_chunk_cache() -> dict:
    """Load the JSON chunk cache from disk, or return empty dict."""
    if os.path.exists(_CHUNKS_CACHE):
        with open(_CHUNKS_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_chunk_cache(cache: dict) -> None:
    """Persist the chunk cache dict to disk."""
    Path(_CHUNKS_DIR).mkdir(parents=True, exist_ok=True)
    with open(_CHUNKS_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def clear_chunk_cache(delete_file: bool = True) -> None:
    """Clear the chunk cache file or reset it to an empty dict."""
    if delete_file and os.path.exists(_CHUNKS_CACHE):
        os.remove(_CHUNKS_CACHE)
        return
    _save_chunk_cache({})


# ──────────────────────────────────────────
# Main ingest entry point
# ──────────────────────────────────────────

def ingest_file(path: str, force: bool = False) -> int:
    """
    Ingest a single PDF or .txt file into the vector store.

    Skips files already present in the chunk cache unless force=True.

    Args:
        path: Path to the document file.
        force: If True, re-ingest even if file was already processed.

    Returns:
        int: Number of new chunks added.
    """
    source_doc = Path(path).name
    cache = _load_chunk_cache()

    if source_doc in cache and not force:
        logger.info(f"'{source_doc}' already ingested. Use force=True to re-ingest.")
        return 0

    logger.info(f"Ingesting: {path}")
    text = extract_text(path)
    if not text.strip():
        logger.warning(f"No text extracted from {path}")
        return 0

    chunks = chunk_text(text, source_doc=source_doc)
    embeddings = embed_chunks(chunks)

    store = get_store()
    entries = [
        (c["chunk_id"], emb, {"text": c["text"], "source_doc": c["source_doc"]})
        for c, emb in zip(chunks, embeddings)
    ]
    store.add_batch(entries)
    store.save()

    # Cache chunk metadata by source doc
    cache[source_doc] = [
        {"chunk_id": c["chunk_id"], "text": c["text"], "source_doc": c["source_doc"]}
        for c in chunks
    ]
    _save_chunk_cache(cache)

    logger.info(f"Ingested {len(chunks)} chunks from '{source_doc}'.")
    return len(chunks)


def ingest_directory(directory: str, force: bool = False) -> int:
    """
    Ingest all PDF and .txt files in a directory.

    Args:
        directory: Path to the directory containing documents.
        force: If True, re-ingest already-processed files.

    Returns:
        int: Total number of new chunks added across all files.
    """
    supported_exts = {".pdf", ".txt"}
    total = 0
    for filepath in Path(directory).iterdir():
        if filepath.suffix.lower() in supported_exts:
            total += ingest_file(str(filepath), force=force)
    logger.info(f"Directory ingestion complete. Total chunks added: {total}")
    return total
