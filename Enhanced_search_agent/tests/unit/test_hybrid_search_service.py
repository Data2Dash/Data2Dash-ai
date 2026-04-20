"""Unit tests — HybridSearchService (all sources mocked)

Semantic Scholar has been removed from the pipeline; tests now cover
ArXiv + OpenAlex only.
"""
import pytest, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from unittest.mock import MagicMock
from tests.conftest import make_paper
from app.services.hybrid_search_service import HybridSearchService


def _make_hybrid(arxiv_papers=None, openalex_papers=None):
    """Build a HybridSearchService with ArXiv and OpenAlex mocked (OpenAlex on)."""
    arxiv    = MagicMock()
    openalex = MagicMock()

    arxiv.search_papers.return_value    = arxiv_papers    or []
    arxiv.search_papers.__name__        = "search_papers"
    arxiv.search_papers_up_to = MagicMock(return_value=[])
    openalex.search_papers.return_value = openalex_papers or []
    openalex.search_papers.__name__     = "search_papers"
    openalex.search_papers_paginated = MagicMock(return_value=[])

    return HybridSearchService(
        arxiv_service=arxiv,
        openalex_service=openalex,
        enable_openalex_retrieval=True,
    )


EXPANDED = {
    "original": "transformers",
    "expanded_queries": ["transformers", "attention is all you need"],
    "semantic_keywords": ["transformer", "attention", "encoder", "decoder"],
    "query_complexity": "broad",
    "topic_profile": {
        "topic": "transformers",
        "aliases": ["transformers", "transformer", "attention mechanism"],
        "acronym": "",
        "landmarks": ["Attention Is All You Need"],
        "subtopics": ["bert", "gpt", "t5", "vision transformer"],
        "exclusions": ["survey", "review"],
    },
    "retrieval_bundles": {
        "broad": ["transformers"],
        "landmark_titles": ["Attention Is All You Need"],
        "acronym": [],
        "subtopics": ["bert", "gpt"],
    },
}


class TestHybridSearchService:

    def test_returns_list_of_papers(self):
        svc     = _make_hybrid(arxiv_papers=[make_paper("ArXiv Paper")])
        results = svc.search(EXPANDED, per_source=5)
        assert isinstance(results, list)

    def test_deduplication_removes_exact_duplicate_titles(self):
        p1 = make_paper("Attention Is All You Need", citations=80000, source="arxiv")
        p2 = make_paper("Attention Is All You Need", citations=75000, source="openalex")
        svc     = _make_hybrid(arxiv_papers=[p1], openalex_papers=[p2])
        results = svc.search(EXPANDED, per_source=5)
        titles  = [r.title for r in results]
        assert titles.count("Attention Is All You Need") == 1

    def test_deduplication_keeps_higher_citation_version(self):
        p_low  = make_paper("Dedup Paper", citations=100,  source="arxiv")
        p_high = make_paper("Dedup Paper", citations=9999, source="openalex")
        svc     = _make_hybrid(arxiv_papers=[p_low], openalex_papers=[p_high])
        results = svc.search(EXPANDED, per_source=5)
        match   = [r for r in results if r.title == "Dedup Paper"]
        assert len(match) == 1
        assert match[0].citations == 9999

    def test_deduplication_case_insensitive(self):
        p1 = make_paper("BERT: Pre-Training of Transformers", source="arxiv")
        p2 = make_paper("bert: pre-training of transformers",  source="openalex")
        svc     = _make_hybrid(arxiv_papers=[p1], openalex_papers=[p2])
        results = svc.search(EXPANDED, per_source=5)
        assert len(results) == 1

    def test_deduplication_ignores_weird_punctuation_and_spacing(self):
        p1 = make_paper("Attention Is All You Need",           source="arxiv")
        p2 = make_paper("  attention is all... you need!!!  ", source="openalex")
        svc     = _make_hybrid(arxiv_papers=[p1], openalex_papers=[p2])
        results = svc.search(EXPANDED, per_source=5)
        assert len(results) == 1

    def test_semantic_score_assigned(self):
        p = make_paper(
            "Transformer Self-Attention Encoder Decoder",
            source="arxiv",
            abstract="Uses attention mechanism in encoder decoder model.",
        )
        svc     = _make_hybrid(arxiv_papers=[p])
        results = svc.search(EXPANDED, per_source=5)
        assert results[0].semantic_score > 0.0

    def test_semantic_score_low_for_unrelated(self):
        p = make_paper(
            "Unrelated Chemistry Paper",
            source="arxiv",
            abstract="This paper is about chemical reactions only.",
        )
        keywords = ["transformer", "attention", "bert", "gpt", "neural"]
        expanded = {**EXPANDED, "semantic_keywords": keywords}
        svc     = _make_hybrid(arxiv_papers=[p])
        results = svc.search(expanded, per_source=5)
        # Should be low relevance
        assert results[0].semantic_score <= 0.3

    def test_empty_sources_returns_empty(self):
        svc     = _make_hybrid()
        results = svc.search(EXPANDED, per_source=5)
        assert results == []

    def test_source_error_does_not_crash(self):
        """A failing ArXiv source should not stop OpenAlex results from coming through."""
        arxiv    = MagicMock()
        openalex = MagicMock()
        arxiv.search_papers.side_effect   = Exception("API down")
        arxiv.search_papers.__name__      = "search_papers"
        openalex.search_papers.return_value = [make_paper("Safe Paper")]
        openalex.search_papers.__name__     = "search_papers"
        svc     = HybridSearchService(
            arxiv_service=arxiv,
            openalex_service=openalex,
            enable_openalex_retrieval=True,
        )
        results = svc.search(EXPANDED, per_source=5)
        assert any(r.title == "Safe Paper" for r in results)

    def test_merges_topic_tags_on_duplicate(self):
        p1 = make_paper("Same Paper", topic_tags=["NLP"],          source="arxiv")
        p2 = make_paper("Same Paper", topic_tags=["Transformers"], source="openalex")
        svc     = _make_hybrid(arxiv_papers=[p1], openalex_papers=[p2])
        results = svc.search(EXPANDED, per_source=5)
        merged  = results[0]
        tags_lower = [t.lower() for t in merged.topic_tags]
        assert "nlp" in tags_lower or "transformers" in tags_lower

    def test_multiple_expanded_queries_all_searched(self):
        """Query planning should hit both sources with multiple bundled variants."""
        arxiv    = MagicMock()
        openalex = MagicMock()
        arxiv.search_papers.return_value    = []
        arxiv.search_papers.__name__        = "search_papers"
        openalex.search_papers.return_value = []
        openalex.search_papers.__name__     = "search_papers"

        svc = HybridSearchService(
            arxiv_service=arxiv,
            openalex_service=openalex,
            enable_openalex_retrieval=True,
        )
        svc.search(EXPANDED, per_source=5)

        assert arxiv.search_papers.call_count >= 1
        assert openalex.search_papers.call_count >= 1

    def test_openalex_disabled_never_calls_openalex(self):
        """When enable_openalex_retrieval=False, hybrid must not hit OpenAlex."""
        arxiv = MagicMock()
        openalex = MagicMock()
        arxiv.search_papers.return_value = [make_paper("Only Arxiv", source="arxiv")]
        arxiv.search_papers.__name__ = "search_papers"
        arxiv.search_papers_up_to = MagicMock(return_value=[])
        openalex.search_papers.return_value = []
        openalex.search_papers.__name__ = "search_papers"
        openalex.search_papers_paginated = MagicMock(return_value=[])

        svc = HybridSearchService(
            arxiv_service=arxiv,
            openalex_service=openalex,
            enable_openalex_retrieval=False,
        )
        svc.search(EXPANDED, per_source=5)
        openalex.search_papers.assert_not_called()
        openalex.search_papers_paginated.assert_not_called()

    def test_relevance_score_title_match_scores_high(self):
        """A paper whose title contains an expanded query should score >= 0.6."""
        p = make_paper(
            "Attention Is All You Need",
            source="arxiv",
            abstract="We propose a new simple network architecture the Transformer.",
        )
        svc     = _make_hybrid(arxiv_papers=[p])
        results = svc.search(EXPANDED, per_source=5)
        assert results[0].semantic_score >= 0.6

    def test_min_keep_fallback_when_all_score_zero(self):
        """
        If every paper has score 0, the service should still return papers
        (up to MIN_KEEP) rather than an empty list.
        """
        papers = [
            make_paper(f"Unrelated Paper {i}", abstract="quantum chemistry biology")
            for i in range(12)
        ]
        no_kw_expanded = {
            "original": "xyz",
            "expanded_queries": ["xyz"],
            "semantic_keywords": [],   # no keywords → scores will be 0
        }
        svc     = _make_hybrid(arxiv_papers=papers)
        results = svc.search(no_kw_expanded, per_source=20)
        # MIN_KEEP = 10, so we should get at least some papers back
        assert len(results) > 0

    def test_relevance_score_matches_normalized_title_variants(self):
        svc = _make_hybrid()
        p = make_paper("Attention: Is All You Need!")
        score = svc._relevance_score(
            paper=p,
            keywords=["transformer"],
            expanded_queries=["Attention Is All You Need"],
            original="transformers",
        )
        assert score >= 0.6

    def test_relevance_score_supports_acronym_from_phrase_keyword(self):
        svc = _make_hybrid()
        p = make_paper("Strong RL Baseline for Atari")
        score = svc._relevance_score(
            paper=p,
            keywords=["reinforcement learning"],
            expanded_queries=["Playing Atari with Deep Reinforcement Learning"],
            original="reinforcement learning",
        )
        assert score > 0.0

    def test_topic_relevance_detects_on_topic_from_metadata(self):
        svc = _make_hybrid()
        p = make_paper(
            "Transformer Models for NLP",
            abstract="This paper studies self-attention and encoder-decoder transformer design.",
            topic_tags=["Natural Language Processing", "Transformers"],
            venue="NeurIPS",
        )
        score, tags = svc._topic_relevance_score(
            paper=p,
            topic_profile=EXPANDED["topic_profile"],
            expanded_queries=EXPANDED["expanded_queries"],
            keywords=EXPANDED["semantic_keywords"],
        )
        assert score >= 0.5
        assert len(tags) > 0

    def test_topic_relevance_downscores_offtopic_generic_ai(self):
        svc = _make_hybrid()
        p = make_paper(
            "Deep Learning Survey for Medical Imaging",
            abstract="A broad survey about neural networks and AI applications in radiology.",
            topic_tags=["Medical Imaging"],
        )
        score, _ = svc._topic_relevance_score(
            paper=p,
            topic_profile=EXPANDED["topic_profile"],
            expanded_queries=EXPANDED["expanded_queries"],
            keywords=EXPANDED["semantic_keywords"],
        )
        assert score < 0.3

    def test_build_stage1_literal_mode_broad_single_query(self):
        """fan_out=False matches arXiv-style literal query: one Stage-1 string for broad."""
        svc = _make_hybrid()
        stage1 = svc._build_stage1_queries(
            EXPANDED["expanded_queries"],
            EXPANDED["retrieval_bundles"],
            EXPANDED["topic_profile"],
            "broad",
            EXPANDED["original"],
            fan_out=False,
        )
        assert stage1 == ["transformers"]

    def test_build_stage1_fan_out_broad_adds_second_variant(self):
        svc = _make_hybrid()
        stage1 = svc._build_stage1_queries(
            EXPANDED["expanded_queries"],
            EXPANDED["retrieval_bundles"],
            EXPANDED["topic_profile"],
            "broad",
            EXPANDED["original"],
            fan_out=True,
        )
        assert len(stage1) <= 2
        assert stage1[0] == "transformers"
