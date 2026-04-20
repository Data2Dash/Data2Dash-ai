"""
Shared test fixtures and helpers for the DATA2DASH test suite.
"""
import sys
import os
import pytest

# Make the project root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas.paper import Paper


# ──────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_paper(
    title="Test Paper",
    abstract="This is an abstract about attention and transformers.",
    authors=None,
    published_date="2023-01-01",
    source="arxiv",
    citations=100,
    topic_tags=None,
    semantic_score=0.0,
    **kwargs,
) -> Paper:
    return Paper(
        id=kwargs.get("id", "test-001"),
        title=title,
        abstract=abstract,
        authors=authors or ["Alice Smith", "Bob Jones"],
        published_date=published_date,
        source=source,
        url=kwargs.get("url", "https://arxiv.org/abs/0000.00000"),
        doi=kwargs.get("doi", ""),
        arxiv_id=kwargs.get("arxiv_id", ""),
        openalex_work_id=kwargs.get("openalex_work_id", ""),
        citations=citations,
        influential_score=float(kwargs.get("influential_score", 0.0)),
        keywords=kwargs.get("keywords", []),
        institution_names=kwargs.get("institution_names", []),
        topic_tags=topic_tags or ["Machine Learning"],
        venue=kwargs.get("venue", ""),
        semantic_score=semantic_score,
        topic_relevance_score=float(kwargs.get("topic_relevance_score", 0.0)),
        inferred_topic_tags=kwargs.get("inferred_topic_tags", []),
        retrieval_path=kwargs.get("retrieval_path", ""),
        ranking_reasons=dict(kwargs.get("ranking_reasons") or {}),
        bm25_score=float(kwargs.get("bm25_score", 0.0)),
        embedding_score=float(kwargs.get("embedding_score", 0.0)),
        hybrid_relevance_score=float(kwargs.get("hybrid_relevance_score", 0.0)),
    )


@pytest.fixture
def sample_paper():
    return make_paper()


@pytest.fixture
def sample_papers():
    return [
        make_paper("Attention Is All You Need", citations=80000, published_date="2017-06-12",
                   source="arxiv", topic_tags=["NLP", "Transformers"],
                   abstract="We propose a new simple network architecture, the Transformer, based solely on attention mechanisms."),
        make_paper("BERT: Pre-training of Deep Bidirectional Transformers", citations=50000,
                   published_date="2018-10-11", source="semantic_scholar",
                   topic_tags=["NLP", "Language Model"],
                   abstract="We introduce BERT, a new language representation model."),
        make_paper("Denoising Diffusion Probabilistic Models", citations=15000,
                   published_date="2020-06-19", source="arxiv",
                   topic_tags=["Generative Models", "Diffusion"],
                   abstract="We present high quality image synthesis using diffusion models."),
        make_paper("Playing Atari with Deep Reinforcement Learning", citations=12000,
                   published_date="2013-12-19", source="openalex",
                   topic_tags=["Reinforcement Learning", "Deep Learning"],
                   abstract="We present a deep learning model to successfully learn control policies from raw pixels."),
        make_paper("Recent Paper 2024", citations=5, published_date="2024-03-01",
                   source="arxiv", topic_tags=["New Topic"],
                   abstract="A very recent paper about new topics."),
    ]
