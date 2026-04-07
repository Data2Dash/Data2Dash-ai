"""
tests/test_vector_store.py
Unit tests for app/knowledge_graph/store/vector_store.py

embed_texts is patched with a deterministic stub so no model download
is required during testing.
"""
from __future__ import annotations

import math
import random
from typing import List
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Stub embedding helpers (same logic as conftest but self-contained)
# ---------------------------------------------------------------------------

class _FakeEmb:
    def __init__(self, values: List[float]):
        self.values = values


def _stub_embed(texts: List[str]) -> List[_FakeEmb]:
    """Deterministic unit-vector embeddings based on text hash."""
    out = []
    for text in texts:
        rng = random.Random(hash(text) & 0xFFFFFFFF)
        vec = [rng.gauss(0, 1) for _ in range(32)]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        out.append(_FakeEmb([v / norm for v in vec]))
    return out


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def vstore():
    """Return a patched InMemoryVectorStore with 3 seeded entries."""
    with patch("app.knowledge_graph.store.vector_store.embed_texts", side_effect=_stub_embed):
        from app.knowledge_graph.store.vector_store import InMemoryVectorStore
        store = InMemoryVectorStore()
        store.add_texts(
            ids=["c1", "c2", "c3"],
            texts=[
                "The Transformer model uses self-attention mechanisms.",
                "LSTM networks handle long-range dependencies with memory cells.",
                "BERT is pre-trained on masked language modeling objectives.",
            ],
        )
        yield store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInMemoryVectorStore:

    def test_add_texts_populates_items(self, vstore):
        assert len(vstore.items) == 3

    def test_search_returns_results(self, vstore):
        with patch("app.knowledge_graph.store.vector_store.embed_texts", side_effect=_stub_embed):
            results = vstore.search("attention mechanism", top_k=2)
        assert len(results) <= 2
        assert len(results) > 0

    def test_search_result_structure(self, vstore):
        """Each result should be a (id, text, score) tuple."""
        with patch("app.knowledge_graph.store.vector_store.embed_texts", side_effect=_stub_embed):
            results = vstore.search("BERT language model", top_k=3)
        for item in results:
            cid, text, score = item
            assert isinstance(cid, str)
            assert isinstance(text, str)
            assert isinstance(score, float)

    def test_search_scores_between_neg1_and_1(self, vstore):
        """Cosine similarity should be in [-1, 1]."""
        with patch("app.knowledge_graph.store.vector_store.embed_texts", side_effect=_stub_embed):
            results = vstore.search("memory network", top_k=3)
        for _, _, score in results:
            assert -1.01 <= score <= 1.01

    def test_search_sorted_descending(self, vstore):
        """Results must be sorted by score descending."""
        with patch("app.knowledge_graph.store.vector_store.embed_texts", side_effect=_stub_embed):
            results = vstore.search("language model", top_k=3)
        scores = [s for _, _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self, vstore):
        with patch("app.knowledge_graph.store.vector_store.embed_texts", side_effect=_stub_embed):
            results = vstore.search("anything", top_k=1)
        assert len(results) == 1

    def test_top_k_larger_than_items(self, vstore):
        """Asking for more results than items should return all items."""
        with patch("app.knowledge_graph.store.vector_store.embed_texts", side_effect=_stub_embed):
            results = vstore.search("anything", top_k=100)
        assert len(results) == 3    # only 3 items in store

    def test_empty_store_returns_empty(self):
        with patch("app.knowledge_graph.store.vector_store.embed_texts", side_effect=_stub_embed):
            from app.knowledge_graph.store.vector_store import InMemoryVectorStore
            empty = InMemoryVectorStore()
            results = empty.search("query", top_k=5)
        assert results == []

    def test_ids_match_expected(self, vstore):
        with patch("app.knowledge_graph.store.vector_store.embed_texts", side_effect=_stub_embed):
            results = vstore.search("anything", top_k=3)
        returned_ids = {r[0] for r in results}
        assert returned_ids == {"c1", "c2", "c3"}
