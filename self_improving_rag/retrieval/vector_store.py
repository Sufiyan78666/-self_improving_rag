import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from self_improving_rag.core.config import BM25_K1, BM25_B
from self_improving_rag.core.exceptions import EmbeddingFailedError, ChunkNotFoundError

logger = logging.getLogger(__name__)

# Paths for persisting the FAISS index, ID map, and sparse corpus
_INDEX_DIR = os.path.join("data", "chunks")
_INDEX_FILE = os.path.join(_INDEX_DIR, "faiss.index")
_META_FILE = os.path.join(_INDEX_DIR, "id_map.json")
_SPARSE_FILE = os.path.join(_INDEX_DIR, "bm25_corpus.pkl")


def _tokenize(text: str) -> List[str]:
    """Tokenize text for BM25 (lowercase and split)."""
    return text.lower().split()


class FAISSVectorStore:
    """
    Thin wrapper around a FAISS IndexFlatIP index with integrated BM25 sparse retrieval.

    Attributes:
        dimension: Embedding dimension.
        index: The underlying faiss.Index object.
        id_to_meta: Dict mapping integer FAISS row id → chunk metadata dict.
        tokenized_corpus: List of tokenized chunks for BM25.
        bm25: The BM25 index instance.
    """

    def __init__(self) -> None:
        self.dimension: Optional[int] = None
        self.index: Optional[faiss.Index] = None
        self.id_to_meta: Dict[int, dict] = {}
        self.chunk_id_to_row: Dict[str, int] = {}
        
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    def _ensure_index(self, dim: int) -> None:
        if self.index is None:
            self.dimension = dim
            self.index = faiss.IndexFlatIP(dim)
            logger.info(f"FAISS index initialized with dim={dim}")

    def _update_bm25(self) -> None:
        """Re-initialize the BM25 index with the current corpus."""
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus, k1=BM25_K1, b=BM25_B)
            logger.debug(f"BM25 index updated with {len(self.tokenized_corpus)} docs.")

    def add(self, chunk_id: str, embedding: np.ndarray, metadata: Optional[dict] = None) -> int:
        try:
            vec = np.array(embedding, dtype=np.float32).flatten()
            self._ensure_index(vec.shape[0])

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            row_id = self.index.ntotal
            self.index.add(vec.reshape(1, -1))
            
            meta = metadata or {}
            meta["chunk_id"] = chunk_id
            self.id_to_meta[row_id] = meta
            self.chunk_id_to_row[chunk_id] = row_id

            # Update Sparse Index
            text = meta.get("text", "")
            self.tokenized_corpus.append(_tokenize(text))
            self._update_bm25()

            return row_id
        except Exception as exc:
            raise EmbeddingFailedError(f"Failed to add embedding: {exc}") from exc

    def add_batch(self, entries: List[Tuple[str, np.ndarray, Optional[dict]]]) -> List[int]:
        if not entries:
            return []

        try:
            vecs = []
            row_ids = []
            start_row = self.index.ntotal if self.index else 0

            for i, (chunk_id, embedding, metadata) in enumerate(entries):
                vec = np.array(embedding, dtype=np.float32).flatten()
                self._ensure_index(vec.shape[0])
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                vecs.append(vec)
                
                row_id = start_row + i
                meta = metadata or {}
                meta["chunk_id"] = chunk_id
                self.id_to_meta[row_id] = meta
                self.chunk_id_to_row[chunk_id] = row_id
                row_ids.append(row_id)
                
                # Update Sparse Corpus
                text = meta.get("text", "")
                self.tokenized_corpus.append(_tokenize(text))

            matrix = np.vstack(vecs).astype(np.float32)
            self.index.add(matrix)
            
            self._update_bm25()
            return row_ids
        except Exception as exc:
            raise EmbeddingFailedError(f"Batch add failed: {exc}") from exc

    def search_dense(self, query_embedding: np.ndarray, k: int) -> List[dict]:
        if self.index is None or self.index.ntotal == 0:
            return []

        vec = np.array(query_embedding, dtype=np.float32).flatten()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        effective_k = min(k, self.index.ntotal)
        scores, indices = self.index.search(vec.reshape(1, -1), effective_k)

        results = []
        for rank, (score, row_id) in enumerate(zip(scores[0], indices[0]), start=1):
            if row_id == -1: continue
            meta = self.id_to_meta.get(int(row_id), {}).copy()
            meta["score"] = float(score)
            meta["rank"] = rank
            meta["retrieval_type"] = "dense"
            results.append(meta)
        return results

    def search_sparse(self, query: str, k: int) -> List[dict]:
        if not self.bm25 or not self.tokenized_corpus:
            return []

        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort and take top k
        top_n = np.argsort(scores)[::-1][:k]
        
        results = []
        for rank, idx in enumerate(top_n, start=1):
            if scores[idx] <= 0: continue
            meta = self.id_to_meta.get(int(idx), {}).copy()
            meta["score"] = float(scores[idx])
            meta["rank"] = rank
            meta["retrieval_type"] = "sparse"
            results.append(meta)
        return results

    def save(self) -> None:
        if self.index is None: return
        Path(_INDEX_DIR).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, _INDEX_FILE)
        
        meta_payload = {
            "dimension": self.dimension,
            "id_to_meta": {str(k): v for k, v in self.id_to_meta.items()},
            "chunk_id_to_row": self.chunk_id_to_row,
        }
        with open(_META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2)
            
        with open(_SPARSE_FILE, "wb") as f:
            pickle.dump(self.tokenized_corpus, f)
            
        logger.info(f"VectorStore saved to {_INDEX_DIR}")

    def load(self) -> bool:
        if not os.path.exists(_INDEX_FILE) or not os.path.exists(_META_FILE):
            return False
        try:
            self.index = faiss.read_index(_INDEX_FILE)
            with open(_META_FILE, "r", encoding="utf-8") as f:
                meta_payload = json.load(f)
            self.dimension = meta_payload["dimension"]
            self.id_to_meta = {int(k): v for k, v in meta_payload["id_to_meta"].items()}
            self.chunk_id_to_row = meta_payload["chunk_id_to_row"]
            
            if os.path.exists(_SPARSE_FILE):
                with open(_SPARSE_FILE, "rb") as f:
                    self.tokenized_corpus = pickle.load(f)
                self._update_bm25()
                
            return True
        except Exception as exc:
            logger.error(f"Load failed: {exc}")
            return False
    @property
    def total_vectors(self) -> int:
        """Return total number of vectors stored in the index."""
        return self.index.ntotal if self.index else 0


_store_instance: Optional[FAISSVectorStore] = None


def get_store() -> FAISSVectorStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = FAISSVectorStore()
        _store_instance.load()
    return _store_instance
