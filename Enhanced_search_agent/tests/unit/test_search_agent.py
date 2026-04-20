"""Unit tests — SearchAgent (all external calls mocked)"""
import pytest, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from dataclasses import replace
from unittest.mock import MagicMock, patch
from tests.conftest import make_paper


def _make_mock_agent(papers=None):
    """Build a SearchAgent with every external dependency mocked."""
    from app.services.search_agent import SearchAgent

    agent = SearchAgent.__new__(SearchAgent)

    # Mocked services — Semantic Scholar removed from pipeline
    agent.arxiv_service     = MagicMock()
    agent.openalex_service  = MagicMock()
    agent.arxiv_service.search_papers.return_value = []
    agent.openalex_service.search_papers.return_value = []
    agent.analytics_service = MagicMock()
    agent.ranking_service  = MagicMock()
    agent.query_expander   = MagicMock()
    agent.hybrid_service   = MagicMock()

    fake_papers = papers or [
        make_paper("Attention Is All You Need", citations=80000, source="arxiv", semantic_score=0.9),
        make_paper("BERT",                       citations=50000, source="semantic_scholar", semantic_score=0.8),
        make_paper("GPT-3",                      citations=30000, source="openalex", semantic_score=0.7),
    ]

    # query_expander returns a standard expansion
    agent.query_expander.expand.return_value = {
        "original": "transformers",
        "expanded_queries": ["transformers", "attention is all you need"],
        "semantic_keywords": ["transformer", "attention", "BERT"],
        "topic_profile": {
            "topic": "transformers",
            "aliases": ["transformers"],
            "acronym": "",
            "landmarks": [],
            "subtopics": [],
            "exclusions": [],
            "parent_topic": "deep learning",
            "child_topics": [],
            "datasets": [],
            "benchmarks": [],
            "methods": [],
            "model_families": [],
        },
        "retrieval_bundles": {
            "broad": ["transformers"],
            "landmark_titles": [],
            "acronym": [],
            "subtopics": [],
        },
        "query_complexity": "broad",
    }

    # hybrid_service returns fake papers
    agent.hybrid_service.search.return_value = fake_papers

    # ranking_service returns same list
    agent.ranking_service.rank_papers.return_value = fake_papers

    # analytics_service returns a fake summary
    from app.schemas.analytics import AnalyticsSummary
    agent.analytics_service.compute_summary.return_value = AnalyticsSummary(
        total_papers=3, papers_last_30_days=0
    )

    return agent


class TestSearchAgentMocked:

    def test_returns_dict_with_required_keys(self):
        agent = _make_mock_agent()
        result = agent.search("transformers", page=1, per_page=10)
        for key in (
            "query",
            "papers",
            "ranked_papers",
            "analytics",
            "expanded_queries",
            "semantic_keywords",
            "total_found",
        ):
            assert key in result, f"Missing key: {key}"
        assert len(result["ranked_papers"]) == result["total_found"]

    def test_query_preserved_in_result(self):
        agent = _make_mock_agent()
        result = agent.search("transformers")
        assert result["query"] == "transformers"

    def test_expanded_queries_in_result(self):
        agent = _make_mock_agent()
        result = agent.search("transformers")
        assert "attention is all you need" in result["expanded_queries"]

    def test_papers_list_not_empty(self):
        agent = _make_mock_agent()
        result = agent.search("transformers", per_page=10)
        assert len(result["papers"]) > 0

    def test_pagination_page_1(self):
        papers = [make_paper(f"Paper {i}", citations=i) for i in range(20)]
        agent = _make_mock_agent(papers=papers)
        agent.ranking_service.rank_papers.return_value = papers
        result = agent.search("transformers", page=1, per_page=5)
        assert len(result["papers"]) == 5

    def test_pagination_page_2(self):
        papers = [make_paper(f"Paper {i}", citations=i) for i in range(20)]
        agent = _make_mock_agent(papers=papers)
        agent.ranking_service.rank_papers.return_value = papers
        result = agent.search("transformers", page=2, per_page=5)
        assert len(result["papers"]) == 5
        # Page 2 should be different papers from page 1
        result_p1 = agent.search("transformers", page=1, per_page=5)
        p1_titles = {p.title for p in result_p1["papers"]}
        p2_titles = {p.title for p in result["papers"]}
        assert p1_titles != p2_titles

    def test_total_found_reflects_full_pool(self):
        papers = [make_paper(f"Paper {i}") for i in range(15)]
        agent = _make_mock_agent(papers=papers)
        agent.ranking_service.rank_papers.return_value = papers
        result = agent.search("transformers", page=1, per_page=5)
        assert result["total_found"] == 15

    def test_query_expander_called_once(self):
        agent = _make_mock_agent()
        agent.search("transformers")
        assert agent.query_expander.expand.call_count == 2  # pre-expand + PRF merge

    def test_hybrid_service_called_with_expanded(self):
        agent = _make_mock_agent()
        agent.search("transformers")
        agent.hybrid_service.search.assert_called_once()

    def test_ranking_called_after_hybrid_search(self):
        agent = _make_mock_agent()
        agent.search("transformers")
        agent.ranking_service.rank_papers.assert_called_once()

    def test_analytics_computed_over_full_pool(self):
        agent = _make_mock_agent()
        agent.search("transformers")
        # compute_summary is called once with the ranked papers + query keyword arg
        agent.analytics_service.compute_summary.assert_called_once()
        call_kwargs = agent.analytics_service.compute_summary.call_args
        # query= should have been passed
        assert call_kwargs.kwargs.get("query") == "transformers"

    def test_max_results_zero_returns_full_ranked_pool(self):
        from app.core.config import settings as real_settings

        papers = [make_paper(f"Paper {i}") for i in range(24)]
        agent = _make_mock_agent(papers=papers)
        agent.ranking_service.rank_papers.return_value = papers
        uncapped = replace(real_settings, MAX_RESULTS=0)
        with patch("app.services.search_agent.settings", uncapped):
            result = agent.search("transformers", page=1, per_page=10)
        assert len(result["ranked_papers"]) == 24
        assert result["total_found"] == 24
        assert result["result_accounting"]["rank_cap_applied"] is False
        assert result["result_accounting"]["max_results_setting"] == 0
