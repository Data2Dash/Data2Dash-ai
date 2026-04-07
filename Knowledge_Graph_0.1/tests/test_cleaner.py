"""
tests/test_cleaner.py
Unit tests for app/knowledge_graph/postprocess/cleaner.py
"""
from __future__ import annotations

import pytest

from app.knowledge_graph.extraction.schema import Entity, Relation
from app.knowledge_graph.postprocess.cleaner import clean_entities_relations


class TestCleanEntitiesRelations:

    def test_merges_abbreviation_entities(self):
        """RNN + Recurrent Neural Network → single canonical entity."""
        entities = [
            Entity(name="Recurrent Neural Network", type="Model"),
            Entity(name="RNN", type="Model"),
        ]
        relations = []
        ents, rels = clean_entities_relations(entities, relations)
        names = [e.name for e in ents]
        assert len(ents) == 1
        assert "RNN" not in names
        assert "Recurrent Neural Network" in names

    def test_remaps_relation_endpoints_after_merge(self):
        """Edge pointing at abbreviation 'RNN' should be redirected to canonical name."""
        entities = [
            Entity(name="Recurrent Neural Network", type="Model"),
            Entity(name="RNN", type="Model"),
            Entity(name="LSTM", type="Model"),
            Entity(name="Long Short-Term Memory", type="Model"),
        ]
        relations = [
            Relation("RNN", "Model", "USES", "LSTM", "Model", "RNN uses LSTM."),
        ]
        ents, rels = clean_entities_relations(entities, relations)
        assert len(rels) == 1
        assert rels[0].head == "Recurrent Neural Network"
        assert rels[0].tail == "Long Short-Term Memory"

    def test_drops_empty_entities(self):
        entities = [Entity(name="", type="Concept"), Entity(name="Transformer", type="Model")]
        ents, _ = clean_entities_relations(entities, [])
        assert all(e.name for e in ents)
        assert len(ents) == 1

    def test_drops_numeric_entities(self):
        entities = [Entity(name="92.3", type="Score"), Entity(name="BERT", type="Model")]
        ents, _ = clean_entities_relations(entities, [])
        names = [e.name for e in ents]
        assert "92.3" not in names

    def test_dedupes_relations(self):
        """Duplicate (head, tail, type) should be reduced to one."""
        entities = [Entity(name="A", type="Concept"), Entity(name="B", type="Concept")]
        relations = [
            Relation("A", "Concept", "USES", "B", "Concept", "ev1"),
            Relation("A", "Concept", "USES", "B", "Concept", "ev2"),
        ]
        _, rels = clean_entities_relations(entities, relations)
        assert len(rels) == 1

    def test_empty_inputs(self):
        ents, rels = clean_entities_relations([], [])
        assert ents == []
        assert rels == []

    def test_drops_relation_with_empty_head(self):
        entities = [Entity(name="Transformer", type="Model")]
        relations = [Relation("", "Model", "USES", "Transformer", "Model", None)]
        _, rels = clean_entities_relations(entities, relations)
        assert len(rels) == 0

    def test_multiple_aliases_resolved(self, sample_entities, sample_relations):
        """Full fixture: RNN→full, LSTM→full, empties/numerics dropped."""
        ents, rels = clean_entities_relations(sample_entities, sample_relations)
        names = [e.name for e in ents]
        assert "RNN" not in names
        assert "LSTM" not in names
        assert "28.4" not in names
        assert "" not in names
        # Relations should reference canonical names
        for r in rels:
            assert r.head != "RNN"
            assert r.tail != "LSTM"
