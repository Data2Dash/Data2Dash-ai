"""Unit tests for multi-signal ``dedupe_papers``."""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests.conftest import make_paper
from app.services.deduplication import dedupe_papers


def test_dedupe_merges_same_doi_different_titles():
    a = make_paper("Title A", source="arxiv", doi="10.1234/zenodo.1", citations=10)
    b = make_paper("Title B", source="openalex", doi="https://doi.org/10.1234/Zenodo.1", citations=50)
    out = dedupe_papers([a, b])
    assert len(out) == 1
    assert out[0].citations == 50
    assert "arxiv" in out[0].source and "openalex" in out[0].source


def test_dedupe_merges_same_arxiv_id():
    a = make_paper("Foo", arxiv_id="1706.03762", source="arxiv", citations=5)
    b = make_paper("Bar", arxiv_id="1706.03762", source="openalex", citations=100)
    out = dedupe_papers([a, b])
    assert len(out) == 1


def test_dedupe_keeps_different_works():
    # Distinct URLs so default arXiv placeholder ids never collide.
    a = make_paper("Alpha", doi="10.1/a", url="https://example.org/a")
    b = make_paper("Beta", doi="10.1/b", url="https://example.org/b")
    out = dedupe_papers([a, b])
    assert len(out) == 2
