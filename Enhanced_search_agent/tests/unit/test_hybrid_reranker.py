"""Unit tests for BM25 + embedding hybrid reranker (no real embedding model)."""
import sys
import os

import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests.conftest import make_paper
from app.services.hybrid_reranker import (
    OkapiBM25,
    apply_hybrid_rerank_to_papers,
    build_paper_text,
    compute_hybrid_scores,
    min_max_normalize,
    tokenize,
)


def test_min_max_normalize():
    assert min_max_normalize([]) == []
    assert min_max_normalize([5.0, 5.0, 5.0]) == [0.5, 0.5, 0.5]
    n = min_max_normalize([0.0, 10.0])
    assert abs(n[0] - 0.0) < 1e-6 and abs(n[1] - 1.0) < 1e-6


def test_tokenize_drops_short_tokens():
    assert "ab" not in tokenize("ab cd efgh")


def test_bm25_prefers_query_terms():
    corpus = [
        tokenize("transformer attention neural network"),
        tokenize("cooking pasta recipes"),
    ]
    bm25 = OkapiBM25(corpus)
    scores = bm25.score_query(tokenize("transformer attention"))
    assert scores[0] > scores[1]


def test_build_paper_text_includes_keywords_and_tags():
    p = make_paper(
        "My Title",
        abstract="Abstract body.",
        keywords=["k1", "k2"],
        topic_tags=["t1"],
    )
    t = build_paper_text(p)
    assert "My Title" in t and "Abstract" in t and "k1" in t and "t1" in t


class _MockEmbedder:
    """Deterministic L2-normalized vectors: higher dim[0] if 'quantum' in text."""

    dim = 8

    def encode(self, texts, batch_size=32):
        n = len(texts)
        out = np.zeros((n, self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            v = np.zeros(self.dim, dtype=np.float32)
            if "quantum" in (text or "").lower():
                v[0] = 1.0
            else:
                v[1] = 1.0
            nrm = float(np.linalg.norm(v)) + 1e-9
            out[i] = v / nrm
        return out


def test_compute_hybrid_scores_fuses_weights():
    papers = [
        make_paper("Alpha", abstract="quantum field theory"),
        make_paper("Beta", abstract="cooking pasta"),
    ]
    bm25_r, bm25_n, emb_r, emb_n, hybrid = compute_hybrid_scores(
        "quantum physics",
        papers,
        0.55,
        0.45,
        _MockEmbedder(),
    )
    assert len(hybrid) == 2
    assert hybrid[0] > hybrid[1]


def test_apply_hybrid_sets_paper_fields():
    papers = [
        make_paper("A", abstract="quantum"),
        make_paper("B", abstract="recipes"),
    ]
    meta = apply_hybrid_rerank_to_papers(
        "quantum",
        papers,
        0.5,
        0.5,
        _MockEmbedder(),
    )
    assert meta.get("hybrid_rerank_status") == "ok"
    assert papers[0].hybrid_relevance_score >= papers[1].hybrid_relevance_score
    assert 0.0 <= papers[0].bm25_score <= 1.0
    assert 0.0 <= papers[0].embedding_score <= 1.0


def test_apply_hybrid_skips_empty_query():
    papers = [make_paper("X", abstract="y")]
    meta = apply_hybrid_rerank_to_papers("", papers, 0.5, 0.5, _MockEmbedder())
    assert meta.get("hybrid_rerank_status") == "skipped_empty"


def test_apply_hybrid_handles_embedder_error():
    class BadEmb:
        def encode(self, texts, batch_size=32):
            raise RuntimeError("no model")

    papers = [make_paper("T", abstract="body")]
    meta = apply_hybrid_rerank_to_papers("query", papers, 0.5, 0.5, BadEmb())
    assert "error" in (meta.get("hybrid_rerank_status") or "")
    assert papers[0].hybrid_relevance_score == 0.0
