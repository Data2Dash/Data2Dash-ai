"""
conftest.py — shared fixtures for the Knowledge Graph test suite.

All fixtures are pure Python: no LLM calls, no network, no model downloads.
The embed_texts function is monkey-patched with a deterministic stub wherever
InMemoryVectorStore is exercised.
"""
from __future__ import annotations

import math
import random
from typing import List, Tuple
from unittest.mock import patch

import pytest

from app.knowledge_graph.extraction.schema import Entity, Relation


# ---------------------------------------------------------------------------
# Deterministic stub embedding (32-dim random unit vectors seeded by text)
# ---------------------------------------------------------------------------

class _StubEmbedding:
    def __init__(self, values: List[float]):
        self.values = values


def _stub_embed(texts: List[str]) -> List[_StubEmbedding]:
    """Return a deterministic fake embedding for each text."""
    out = []
    for text in texts:
        rng = random.Random(hash(text) & 0xFFFFFFFF)
        vec = [rng.gauss(0, 1) for _ in range(32)]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        out.append(_StubEmbedding([v / norm for v in vec]))
    return out


# ---------------------------------------------------------------------------
# Entity / Relation fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_entities() -> List[Entity]:
    """Mix of abbreviation + full-form pairs and normal entities."""
    return [
        Entity(name="Recurrent Neural Network", type="Model"),
        Entity(name="RNN", type="Model"),                          # alias
        Entity(name="Long Short-Term Memory", type="Model"),
        Entity(name="LSTM", type="Model"),                         # alias
        Entity(name="Transformer", type="Model"),
        Entity(name="BLEU", type="Metric"),
        Entity(name="", type="Concept"),                           # empty → should drop
        Entity(name="28.4", type="Metric"),                        # numeric → should drop
    ]


@pytest.fixture
def sample_relations() -> List[Relation]:
    """Relations whose head/tail include abbreviation forms."""
    return [
        Relation("RNN", "Model", "USES", "LSTM", "Model", "RNN uses LSTM cells."),
        Relation("RNN", "Model", "USES", "LSTM", "Model", "Duplicate."),   # duplicate
        Relation("Transformer", "Model", "IMPROVES", "BLEU", "Metric", "Transformer improves BLEU."),
        Relation("", "Model", "RELATED_TO", "LSTM", "Model", None),        # empty head → drop
    ]


# ---------------------------------------------------------------------------
# Patch helper (used as a context manager in tests)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=False)
def patch_embeddings():
    """Patch embed_texts everywhere it is imported for the duration of a test."""
    with patch(
        "app.knowledge_graph.store.vector_store.embed_texts",
        side_effect=_stub_embed,
    ) as mock:
        yield mock
