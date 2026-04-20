"""Unit tests — CS / CE retrieval field filter."""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests.conftest import make_paper
from app.services.cs_ce_field_filter import paper_matches_cs_ce_fields


def test_arxiv_cs_category_passes():
    p = make_paper(
        venue="arXiv cs.LG",
        topic_tags=[],
        source="arxiv",
    )
    assert paper_matches_cs_ce_fields(p) is True


def test_arxiv_qbio_rejected():
    p = make_paper(
        venue="arXiv q-bio.NC",
        topic_tags=["Machine Learning"],
        source="arxiv",
    )
    assert paper_matches_cs_ce_fields(p) is False


def test_arxiv_cs_with_biology_concept_rejected():
    p = make_paper(
        venue="arXiv cs.LG",
        topic_tags=["Computer science", "Cell biology"],
        source="arxiv,openalex",
    )
    assert paper_matches_cs_ce_fields(p) is False


def test_openalex_only_with_computer_science():
    p = make_paper(
        venue="",
        topic_tags=["Computer science"],
        source="openalex",
    )
    assert paper_matches_cs_ce_fields(p) is True


def test_openalex_only_biology_rejected():
    p = make_paper(
        venue="",
        topic_tags=["Genetics"],
        source="openalex",
    )
    assert paper_matches_cs_ce_fields(p) is False


def test_openalex_only_unknown_tag_rejected():
    p = make_paper(
        venue="",
        topic_tags=["Materials science"],
        source="openalex",
    )
    assert paper_matches_cs_ce_fields(p) is False


def test_transformers_tag_passes():
    p = make_paper(
        venue="",
        topic_tags=["Transformers"],
        source="openalex",
    )
    assert paper_matches_cs_ce_fields(p) is True
