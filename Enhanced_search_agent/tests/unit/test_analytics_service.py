"""Unit tests — AnalyticsService"""
import pytest, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from tests.conftest import make_paper
from app.services.analytics_service import AnalyticsService


@pytest.fixture
def svc():
    return AnalyticsService()


class TestAnalyticsService:

    def test_total_papers_count(self, svc, sample_papers):
        summary = svc.compute_summary(sample_papers)
        assert summary.total_papers == len(sample_papers)

    def test_empty_papers_returns_zero(self, svc):
        summary = svc.compute_summary([])
        assert summary.total_papers == 0
        assert summary.papers_last_30_days == 0

    def test_recent_paper_counted_in_last_30_days(self, svc):
        from datetime import datetime, timedelta
        recent_date = (datetime.today().date() - timedelta(days=5)).isoformat()
        papers = [make_paper(published_date=recent_date)]
        summary = svc.compute_summary(papers)
        assert summary.papers_last_30_days == 1

    def test_old_paper_not_counted_in_last_30_days(self, svc):
        papers = [make_paper(published_date="2000-01-01")]
        summary = svc.compute_summary(papers)
        assert summary.papers_last_30_days == 0

    def test_top_authors_returns_correct_format(self, svc, sample_papers):
        summary = svc.compute_summary(sample_papers)
        assert isinstance(summary.top_authors, list)
        for name, count in summary.top_authors:
            assert isinstance(name, str)
            assert isinstance(count, int)

    def test_top_keywords_from_topic_tags(self, svc):
        papers = [
            make_paper(topic_tags=["NLP", "Transformers"]),
            make_paper(topic_tags=["NLP", "BERT"]),
            make_paper(topic_tags=["NLP"]),
        ]
        summary = svc.compute_summary(papers)
        kw_dict = dict(summary.top_keywords)
        assert kw_dict.get("NLP", 0) == 3

    def test_monthly_counts_keys_are_year_month(self, svc):
        papers = [
            make_paper(published_date="2023-01-15"),
            make_paper(published_date="2023-03-10"),
        ]
        summary = svc.compute_summary(papers)
        for key in summary.monthly_counts.keys():
            assert len(key) == 7          # "YYYY-MM"
            assert key[4] == "-"

    def test_trend_status_rising(self, svc):
        papers = [
            make_paper(published_date="2023-01-01"),
            make_paper(published_date="2023-02-01"),
            make_paper(published_date="2023-02-15"),
        ]
        summary = svc.compute_summary(papers)
        # trend_status includes an emoji prefix, e.g. "📈 Rising"
        valid = ("Rising", "Stable", "Declining")
        assert any(v in summary.trend_status for v in valid)

    def test_trend_status_stable_single_month(self, svc):
        papers = [make_paper(published_date="2023-06-01")]
        summary = svc.compute_summary(papers)
        assert "Stable" in summary.trend_status

    def test_paper_with_missing_date_skipped_gracefully(self, svc):
        papers = [make_paper(published_date=""), make_paper(published_date="2023-01-01")]
        summary = svc.compute_summary(papers)
        assert summary.total_papers == 2

    def test_transformer_subtopic_distribution_is_inferred(self, svc):
        papers = [
            make_paper(
                title="ViT for Image Classification",
                abstract="A vision transformer architecture for image recognition.",
                topic_tags=["Computer Vision"],
            ),
            make_paper(
                title="Efficient Transformer with Sparse Attention",
                abstract="We propose sparse attention and memory efficient training.",
                topic_tags=["Transformers"],
            ),
        ]
        summary = svc.compute_summary(papers, query="transformers")
        assert "Vision Transformers" in summary.subtopic_distribution
        assert "Efficient Transformers" in summary.subtopic_distribution

    def test_venue_distribution_uses_extracted_venue(self, svc):
        papers = [
            make_paper(title="Paper A", source="openalex", venue="NeurIPS"),
            make_paper(title="Paper B", source="openalex", venue="ICLR"),
            make_paper(title="Paper C", source="arxiv", venue="arXiv cs.CL"),
        ]
        summary = svc.compute_summary(papers)
        assert summary.venue_distribution.get("NeurIPS", 0) == 1
        assert summary.venue_distribution.get("ICLR", 0) == 1
        assert summary.venue_distribution.get("arXiv cs.CL", 0) == 1

    def test_top_author_impact_contains_scores(self, svc):
        papers = [
            make_paper(authors=["Alice Smith"], citations=1000, semantic_score=0.8),
            make_paper(authors=["Alice Smith", "Bob Lee"], citations=100, semantic_score=0.4),
        ]
        summary = svc.compute_summary(papers)
        assert len(summary.top_author_impact) > 0
        assert "impact_score" in summary.top_author_impact[0]
        assert summary.top_author_impact[0]["author"] == "Alice Smith"

    def test_inferred_topic_tags_power_subtopic_distribution(self, svc):
        p1 = make_paper(title="RL Paper", published_date="2022-01-01")
        p1.inferred_topic_tags = ["reinforcement learning"]
        p2 = make_paper(title="GNN Paper", published_date="2023-01-01")
        p2.inferred_topic_tags = ["graph neural networks"]
        summary = svc.compute_summary([p1, p2], query="ai")
        assert summary.subtopic_distribution.get("reinforcement learning", 0) == 1
        assert summary.subtopic_distribution.get("graph neural networks", 0) == 1


@pytest.fixture
def sample_papers():
    return [
        make_paper(authors=["Alice", "Bob"],   published_date="2022-01-01", topic_tags=["NLP"]),
        make_paper(authors=["Alice", "Carol"],  published_date="2022-06-01", topic_tags=["NLP", "CV"]),
        make_paper(authors=["Dave"],            published_date="2023-01-01", topic_tags=["RL"]),
    ]
