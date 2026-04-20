"""Unit tests — RankingService"""
import pytest
from unittest.mock import patch
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from tests.conftest import make_paper
from app.services.ranking_service import RankingService


@pytest.fixture
def ranker():
    return RankingService()


class TestRankingService:

    def test_returns_same_count(self, ranker, sample_papers):
        ranked = ranker.rank_papers(sample_papers)
        assert len(ranked) == len(sample_papers)

    def test_highly_cited_recent_paper_ranks_high(self, ranker):
        old_low = make_paper("Old Paper", citations=1, published_date="2000-01-01", semantic_score=0.0)
        new_high = make_paper("Hot Paper", citations=10000, published_date="2024-01-01", semantic_score=0.9)
        ranked = ranker.rank_papers([old_low, new_high])
        assert ranked[0].title == "Hot Paper"

    def test_semantic_score_influences_rank(self, ranker):
        low_sem = make_paper("Low Relevance", citations=500, published_date="2022-01-01", semantic_score=0.0)
        high_sem = make_paper("High Relevance", citations=100, published_date="2022-01-01", semantic_score=1.0)
        ranked = ranker.rank_papers([low_sem, high_sem])
        assert ranked[0].title == "High Relevance"

    def test_empty_list(self, ranker):
        assert ranker.rank_papers([]) == []

    def test_single_paper(self, ranker):
        p = make_paper()
        ranked = ranker.rank_papers([p])
        assert ranked[0].title == p.title

    def test_score_is_between_0_and_1(self, ranker):
        p = make_paper(citations=500, published_date="2022-06-01", semantic_score=0.5,
                       source="arxiv")
        score = ranker.score_paper(p)
        assert 0.0 <= score <= 1.0

    def test_missing_published_date_does_not_crash(self, ranker):
        p = make_paper(published_date="")
        score = ranker.score_paper(p)
        assert score >= 0.0

    def test_unknown_source_gets_default_confidence(self, ranker):
        p = make_paper(source="unknown_source", citations=0, published_date="2023-01-01",
                       semantic_score=0.0)
        score = ranker.score_paper(p)
        assert score >= 0.0

    def test_recency_score_newer_is_higher(self, ranker):
        old_score = ranker._recency_score("2010-01-01")
        new_score = ranker._recency_score("2024-01-01")
        assert new_score > old_score

    def test_recency_score_very_old_paper_is_near_zero(self, ranker):
        """Exponential decay never reaches exactly 0 but should be negligibly small."""
        score = ranker._recency_score("1900-01-01")
        assert score < 0.001   # effectively zero in practice

    def test_recency_score_invalid_date_format_handled(self, ranker):
        """Invalid date strings should be handled gracefully and return 0.0."""
        score = ranker._recency_score("hello-world")
        assert score == 0.0

    def test_zero_citations_handled(self, ranker):
        """Papers with zero citations should still produce a valid score."""
        p = make_paper(citations=0, published_date="2023-01-01", semantic_score=0.5)
        score = ranker.score_paper(p)
        assert 0.0 <= score <= 1.0

    def test_merged_source_confidence_prefers_best_source(self, ranker):
        merged = make_paper(
            "Merged Source Paper",
            source="arxiv,openalex",
            citations=100,
            published_date="2024-01-01",
            semantic_score=0.4,
        )
        unknown = make_paper(
            "Unknown Source Paper",
            source="unknown",
            citations=100,
            published_date="2024-01-01",
            semantic_score=0.4,
        )
        assert ranker.score_paper(merged) >= ranker.score_paper(unknown)

    def test_acronym_title_boost_works_for_phrase_keywords(self, ranker):
        paper = make_paper(
            "A New RL Method for Planning",
            citations=10,
            published_date="2023-01-01",
            semantic_score=0.1,
        )
        boosted = ranker.score_paper(
            paper,
            query_keywords=["reinforcement learning"],
        )
        baseline = ranker.score_paper(
            paper,
            query_keywords=["graph theory"],
        )
        assert boosted > baseline

    def test_ranking_reasons_populated(self, ranker):
        p = make_paper("Test", semantic_score=0.5)
        ranker.rank_papers([p], user_query="transformers")
        assert getattr(p, "ranking_reasons", None)
        assert "composite" in p.ranking_reasons
        assert "weights" in p.ranking_reasons

    def test_topic_relevance_dominates_generic_metrics(self, ranker):
        on_topic = make_paper(
            "Language Models are Few-Shot Learners",
            citations=50,
            published_date="2020-01-01",
            semantic_score=0.5,
        )
        off_topic = make_paper(
            "General AI Benchmarking Survey",
            citations=5000,
            published_date="2025-01-01",
            semantic_score=0.4,
        )
        object.__setattr__(on_topic, "topic_relevance_score", 0.95)
        object.__setattr__(off_topic, "topic_relevance_score", 0.10)
        ranked = ranker.rank_papers(
            [off_topic, on_topic],
            query_keywords=["large language models", "language models are few-shot learners"],
        )
        assert ranked[0].title == "Language Models are Few-Shot Learners"

    def test_exclusion_soft_penalty_updates_reasons(self, ranker):
        p1 = make_paper("Paper A", semantic_score=0.5)
        p2 = make_paper("Paper B", semantic_score=0.4)
        with patch(
            "app.services.ranking_service._exclusion_soft_penalty_factors",
            return_value=[(0.85, 0.6), (1.0, 0.0)],
        ):
            ranker.rank_papers(
                [p1, p2],
                exclusion_phrases=["domain to avoid"],
            )
        assert p1.ranking_reasons.get("exclusion_penalty_factor") == 0.85
        assert p1.ranking_reasons.get("max_exclusion_similarity") == 0.6
        assert p2.ranking_reasons.get("exclusion_penalty_factor") == 1.0


@pytest.fixture
def sample_papers():
    from tests.conftest import make_paper
    return [
        make_paper(citations=1000, published_date="2021-01-01", semantic_score=0.8),
        make_paper(citations=50,   published_date="2023-06-01", semantic_score=0.2),
        make_paper(citations=500,  published_date="2019-01-01", semantic_score=0.5),
    ]
