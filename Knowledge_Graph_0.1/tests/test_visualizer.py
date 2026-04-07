"""
tests/test_visualizer.py
Unit tests for app/knowledge_graph/visualization/pyvis_visualizer.py

Uses a tiny stub GraphDocument so no LLM, PDF, or network call is needed.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from langchain_community.graphs.graph_document import Node, Relationship
from langchain_experimental.graph_transformers.llm import GraphDocument
from langchain_core.documents import Document

from app.knowledge_graph.visualization.pyvis_visualizer import visualize_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph_doc(
    node_ids=("Transformer", "Attention", "BLEU"),
    edges=(("Transformer", "IMPROVES", "BLEU"), ("Transformer", "USES", "Attention")),
) -> GraphDocument:
    nodes = [Node(id=nid, type="Concept") for nid in node_ids]
    node_map = {n.id: n for n in nodes}
    rels = [
        Relationship(source=node_map[src], target=node_map[tgt], type=rel)
        for src, rel, tgt in edges
        if src in node_map and tgt in node_map
    ]
    return GraphDocument(
        nodes=nodes,
        relationships=rels,
        source=Document(page_content="test"),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVisualizeGraph:

    def test_returns_none_on_empty_input(self):
        result = visualize_graph([])
        assert result is None

    def test_creates_html_file(self, tmp_path):
        gd = _make_graph_doc()
        out = tmp_path / "kg.html"
        result = visualize_graph([gd], output_file=str(out))
        assert result is not None
        assert os.path.exists(result)

    def test_html_file_is_not_empty(self, tmp_path):
        gd = _make_graph_doc()
        out = tmp_path / "kg.html"
        visualize_graph([gd], output_file=str(out))
        assert out.stat().st_size > 0

    def test_html_contains_search_input(self, tmp_path):
        gd = _make_graph_doc()
        out = tmp_path / "kg.html"
        visualize_graph([gd], output_file=str(out))
        content = out.read_text(encoding="utf-8")
        assert 'id="kg-search-input"' in content

    def test_html_contains_search_wrapper(self, tmp_path):
        gd = _make_graph_doc()
        out = tmp_path / "kg.html"
        visualize_graph([gd], output_file=str(out))
        content = out.read_text(encoding="utf-8")
        assert 'id="kg-search-wrapper"' in content

    def test_html_contains_js_build_suggestions(self, tmp_path):
        gd = _make_graph_doc()
        out = tmp_path / "kg.html"
        visualize_graph([gd], output_file=str(out))
        content = out.read_text(encoding="utf-8")
        assert "buildSuggestions" in content

    def test_html_contains_js_highlight_nodes(self, tmp_path):
        gd = _make_graph_doc()
        out = tmp_path / "kg.html"
        visualize_graph([gd], output_file=str(out))
        content = out.read_text(encoding="utf-8")
        assert "highlightNodes" in content

    def test_html_contains_node_labels(self, tmp_path):
        gd = _make_graph_doc()
        out = tmp_path / "kg.html"
        visualize_graph([gd], output_file=str(out))
        content = out.read_text(encoding="utf-8")
        # Node ids should appear somewhere in the HTML (vis.js injects them)
        assert "Transformer" in content

    def test_html_ends_with_body_tag(self, tmp_path):
        """Search overlay is injected before </body> — ensure body tag is present."""
        gd = _make_graph_doc()
        out = tmp_path / "kg.html"
        visualize_graph([gd], output_file=str(out))
        content = out.read_text(encoding="utf-8")
        assert "</body>" in content

    def test_single_node_graph(self, tmp_path):
        """Single node with no edges should not crash the visualizer."""
        gd = GraphDocument(
            nodes=[Node(id="Lonely", type="Concept")],
            relationships=[],
            source=Document(page_content="test"),
        )
        out = tmp_path / "single.html"
        result = visualize_graph([gd], output_file=str(out))
        assert result is not None
        assert os.path.exists(result)
