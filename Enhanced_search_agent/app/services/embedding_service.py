"""
Sentence embedding service for hybrid BM25 + embedding reranking.

Loads a ``sentence-transformers`` model lazily (first use per instance), encodes
text batches, and provides a small in-process LRU cache keyed by content hash.

If ``sentence-transformers`` is missing or the model fails to load, callers
should catch exceptions and skip hybrid reranking.
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import List, Optional

import numpy as np


class EmbeddingService:
    """
    Thin wrapper around ``SentenceTransformer`` with bounded LRU caching.

    Cache keys are SHA-256 of UTF-8 text; values are L2-normalized embedding
    vectors so cosine similarity is a dot product between rows.
    """

    def __init__(self, model_name: str, cache_max_entries: int = 2048):
        self.model_name = model_name
        self._cache_max = max(32, int(cache_max_entries))
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._st_model = None
        self._loaded_name: Optional[str] = None

    def _get_model(self):
        with self._model_lock:
            if self._st_model is not None and self._loaded_name == self.model_name:
                return self._st_model
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError(
                    "sentence-transformers is not installed; "
                    "pip install -r requirements.txt or set ENABLE_HYBRID_RERANK=false."
                ) from e
            self._st_model = SentenceTransformer(self.model_name)
            self._loaded_name = self.model_name
            return self._st_model

    @staticmethod
    def _cache_key(text: str) -> str:
        h = hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()
        return h[:24]

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Return float32 array shape ``(len(texts), dim)`` with L2-normalized rows.
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        model = self._get_model()
        dim = model.get_sentence_embedding_dimension()
        out = np.zeros((len(texts), dim), dtype=np.float32)
        to_encode_idx: List[int] = []
        to_encode_texts: List[str] = []

        for i, t in enumerate(texts):
            key = self._cache_key(t or "")
            with self._cache_lock:
                vec = self._cache.get(key)
            if vec is not None:
                with self._cache_lock:
                    self._cache.move_to_end(key)
                out[i] = vec
            else:
                to_encode_idx.append(i)
                to_encode_texts.append(t or "")

        if to_encode_idx:
            emb = model.encode(
                to_encode_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            if not isinstance(emb, np.ndarray):
                emb = np.asarray(emb, dtype=np.float32)
            emb = emb.astype(np.float32, copy=False)
            for j, row_i in enumerate(to_encode_idx):
                row = emb[j]
                key = self._cache_key(to_encode_texts[j])
                with self._cache_lock:
                    self._cache[key] = row.copy()
                    self._cache.move_to_end(key)
                    while len(self._cache) > self._cache_max:
                        self._cache.popitem(last=False)
                out[row_i] = row

        return out


def get_embedding_service(
    model_name: str,
    cache_max_entries: int = 2048,
) -> EmbeddingService:
    """One ``EmbeddingService`` per (model name, cache size) for the process lifetime."""
    if not hasattr(get_embedding_service, "_instances"):
        get_embedding_service._instances = {}  # type: ignore[attr-defined]
    inst: dict = get_embedding_service._instances  # type: ignore[attr-defined]
    key = (model_name, int(cache_max_entries))
    if key not in inst:
        inst[key] = EmbeddingService(model_name, int(cache_max_entries))
    return inst[key]
