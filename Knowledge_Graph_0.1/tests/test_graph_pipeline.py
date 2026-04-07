"""
tests/test_graph_pipeline.py
Unit tests for pure helper functions in app/pipelines/graph_pipeline.py

We test only the deterministic helpers (_should_drop_node_id,
_drop_isolated_nodes, _build_nodes_and_edges) without running the
full async pipeline or touching any LLM / PDF code.
"""
from __future__ import annotations

import pytest

from langchain_community.graphs.graph_document import Node, Relationship

from app.knowledge_graph.extraction.schema import Entity, Relation
from app.pipelines.graph_pipeline import (
    _should_drop_node_id,
    _drop_isolated_nodes,
    _build_nodes_and_edges,
)


# ---------------------------------------------------------------------------
# _should_drop_node_id
# ---------------------------------------------------------------------------

class TestShouldDropNodeId:
    def test_drops_numeric(self):
        assert _should_drop_node_id("28.4") is True

    def test_drops_integer(self):
        assert _should_drop_node_id("100") is True

    def test_drops_empty(self):
        assert _should_drop_node_id("") is True

    def test_drops_single_char(self):
        assert _should_drop_node_id("A") is True

    def test_keeps_model_name(self):
        assert _should_drop_node_id("Transformer") is False

    def test_keeps_mixed_alphanumeric(self):
        # "L2.4 regularization" contains letters → keep
        assert _should_drop_node_id("L2.4 regularization") is False

    def test_keeps_metric_with_dash(self):
        assert _should_drop_node_id("BLEU-4") is False

    def test_drops_whitespace_only(self):
        assert _should_drop_node_id("   ") is True


# ---------------------------------------------------------------------------
# _drop_isolated_nodes
# ---------------------------------------------------------------------------

class TestDropIsolatedNodes:
    def _make_node(self, nid: str) -> Node:
        return Node(id=nid, type="Concept")

    def _make_edge(self, src: str, tgt: str) -> Relationship:
        return Relationship(
            source=self._make_node(src),
            target=self._make_node(tgt),
            type="RELATED_TO",
        )

    def test_removes_orphan_node(self):
        nodes = [self._make_node("A"), self._make_node("B"), self._make_node("Orphan")]
        edges = [self._make_edge("A", "B")]
        kept, _ = _drop_isolated_nodes(nodes, edges)
        ids = [n.id for n in kept]
        assert "Orphan" not in ids
        assert "A" in ids
        assert "B" in ids

    def test_keeps_all_when_all_connected(self):
        nodes = [self._make_node("A"), self._make_node("B"), self._make_node("C")]
        edges = [self._make_edge("A", "B"), self._make_edge("B", "C")]
        kept, _ = _drop_isolated_nodes(nodes, edges)
        assert len(kept) == 3

    def test_empty_edges_drops_all_nodes(self):
        nodes = [self._make_node("A"), self._make_node("B")]
        kept, _ = _drop_isolated_nodes(nodes, [])
        assert len(kept) == 0

    def test_empty_inputs(self):
        kept, edges = _drop_isolated_nodes([], [])
        assert kept == []
        assert edges == []

    def test_case_insensitive_matching(self):
        """Node stored as 'Transformer', edge references 'transformer' (lower)."""
        n = Node(id="Transformer", type="Model")
        e = Relationship(
            source=Node(id="transformer", type="Model"),
            target=Node(id="BLEU", type="Metric"),
            type="IMPROVES",
        )
        kept, _ = _drop_isolated_nodes([n], [e])
        # "Transformer".lower() == "transformer" which is in connected set
        assert len(kept) == 1


# ---------------------------------------------------------------------------
# _build_nodes_and_edges
# ---------------------------------------------------------------------------

class TestBuildNodesAndEdges:
    def test_basic_node_and_edge_creation(self):
        entities = [Entity("A", "Concept"), Entity("B", "Concept")]
        relations = [Relation("A", "Concept", "USES", "B", "Concept", None)]
        nodes, edges = _build_nodes_and_edges(entities, relations)
        node_ids = [n.id for n in nodes]
        assert "A" in node_ids or any("a" in i.lower() for i in node_ids)
        assert len(edges) == 1

    def test_merges_abbreviation_into_canonical(self):
        """RNN and Recurrent Neural Network → single node after alias merge."""
        entities = [
            Entity("Recurrent Neural Network", "Model"),
            Entity("RNN", "Model"),
        ]
        relations = [
            Relation("RNN", "Model", "USES", "Recurrent Neural Network", "Model", None),
        ]
        nodes, edges = _build_nodes_and_edges(entities, relations)
        node_ids = [n.id.lower() for n in nodes]
        # Should not have two separate nodes
        rnn_ids = [i for i in node_ids if "recurrent" in i or i == "rnn"]
        assert len(rnn_ids) == 1

    def test_no_isolated_alias_node(self):
        """After merging, RNN abbreviation node should not appear alone."""
        entities = [
            Entity("Recurrent Neural Network", "Model"),
            Entity("RNN", "Model"),
            Entity("LSTM", "Model"),
        ]
        relations = [
            Relation("Recurrent Neural Network", "Model", "USES", "LSTM", "Model", None),
        ]
        nodes, edges = _build_nodes_and_edges(entities, relations)
        ids_lower = [n.id.lower() for n in nodes]
        # "rnn" standalone (not inside a longer name) should not be present
        assert "rnn" not in ids_lower

    def test_drops_numeric_node_ids(self):
        entities = [Entity("92.3", "Score"), Entity("BERT", "Model")]
        relations = [Relation("BERT", "Model", "SCORES", "92.3", "Score", None)]
        nodes, _ = _build_nodes_and_edges(entities, relations)
        ids = [n.id for n in nodes]
        assert "92.3" not in ids

    def test_self_loop_edge_allowed(self):
        """Self-loops (head == tail) are odd but shouldn't crash."""
        entities = [Entity("A", "Concept")]
        relations = [Relation("A", "Concept", "RELATED_TO", "A", "Concept", None)]
        nodes, edges = _build_nodes_and_edges(entities, relations)
        assert len(nodes) >= 1

    def test_empty_inputs(self):
        nodes, edges = _build_nodes_and_edges([], [])
        assert nodes == []
        assert edges == []

    def test_relation_creates_missing_endpoint_node(self):
        """A node referenced only in a relation (not in entities) is auto-created."""
        entities = [Entity("Transformer", "Model")]
        relations = [Relation("Transformer", "Model", "USES", "Attention", "Mechanism", None)]
        nodes, edges = _build_nodes_and_edges(entities, relations)
        ids = [n.id for n in nodes]
        assert "Attention" in ids
        assert len(edges) == 1
