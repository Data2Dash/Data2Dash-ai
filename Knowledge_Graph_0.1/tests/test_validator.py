"""
tests/test_validator.py
Unit tests for app/knowledge_graph/extraction/validator.py
"""
from __future__ import annotations

import pytest

from app.knowledge_graph.extraction.schema import Entity, Relation
from app.knowledge_graph.extraction.validator import (
    _acronym_of,
    _build_alias_map,
    dedupe_entities,
    dedupe_relations,
    remap_relation_endpoints,
    normalize_rel_type,
)


# ---------------------------------------------------------------------------
# _acronym_of
# ---------------------------------------------------------------------------

class TestAcronymOf:
    def test_rnn(self):
        assert _acronym_of("Recurrent Neural Network", "RNN") is True

    def test_lstm(self):
        assert _acronym_of("Long Short-Term Memory", "LSTM") is True

    def test_bert_clean(self):
        # "from" contributes its initial "F", making BERFT ≠ BERT.
        # Our algorithm does not filter prepositions — use a preposition-free form.
        assert _acronym_of("Bidirectional Encoder Representations Transformers", "BERT") is True

    def test_bert_with_preposition_does_not_match(self):
        # Known limitation: stop-words are included in initials extraction.
        assert _acronym_of("Bidirectional Encoder Representations from Transformers", "BERT") is False

    def test_false_mismatch(self):
        assert _acronym_of("Some Random Model", "XYZ") is False

    def test_single_char_acronym_rejected(self):
        # Single-char strings are not meaningful acronyms
        assert _acronym_of("Attention", "A") is False

    def test_case_insensitive_acronym(self):
        # acronym supplied in lower case — method upcases before comparing
        assert _acronym_of("Recurrent Neural Network", "rnn") is True


# ---------------------------------------------------------------------------
# _build_alias_map
# ---------------------------------------------------------------------------

class TestBuildAliasMap:
    def test_inline_parens_full_then_abbr(self):
        names = ["Recurrent Neural Network (RNN)"]
        m = _build_alias_map(names)
        assert m.get("rnn") == "Recurrent Neural Network"

    def test_inline_parens_abbr_then_full(self):
        names = ["RNN (Recurrent Neural Network)"]
        m = _build_alias_map(names)
        assert m.get("rnn") == "Recurrent Neural Network"

    def test_cross_pair_detection(self):
        names = ["RNN", "Recurrent Neural Network"]
        m = _build_alias_map(names)
        assert m.get("rnn") == "Recurrent Neural Network"

    def test_cross_pair_lstm(self):
        names = ["LSTM", "Long Short-Term Memory"]
        m = _build_alias_map(names)
        assert m.get("lstm") == "Long Short-Term Memory"

    def test_no_false_positives(self):
        # "ABC" should NOT match "Attention Based Clustering" with wrong initials
        names = ["GNN", "Graph Attention Network"]
        m = _build_alias_map(names)
        # GAN ≠ GNN, so no match expected
        assert m.get("gnn") is None or m.get("gnn") != "Graph Attention Network"

    def test_empty_input(self):
        assert _build_alias_map([]) == {}


# ---------------------------------------------------------------------------
# dedupe_entities
# ---------------------------------------------------------------------------

class TestDedupeEntities:
    def test_merges_abbreviation_with_full_form(self):
        entities = [
            Entity(name="Recurrent Neural Network", type="Model"),
            Entity(name="RNN", type="Model"),
        ]
        result = dedupe_entities(entities)
        names = [e.name for e in result]
        # Only one entity, the canonical full form
        assert len(result) == 1
        assert "Recurrent Neural Network" in names

    def test_drops_empty_name(self):
        entities = [Entity(name="", type="Concept"), Entity(name="Transformer", type="Model")]
        result = dedupe_entities(entities)
        assert all(e.name for e in result)
        assert len(result) == 1

    def test_drops_numeric_only(self):
        entities = [Entity(name="28.4", type="Metric"), Entity(name="BLEU", type="Metric")]
        result = dedupe_entities(entities)
        names = [e.name for e in result]
        assert "28.4" not in names
        assert "BLEU" in names

    def test_exact_duplicate_removed(self):
        entities = [
            Entity(name="Transformer", type="Model"),
            Entity(name="Transformer", type="Model"),
        ]
        result = dedupe_entities(entities)
        assert len(result) == 1

    def test_preserves_different_types(self):
        # Same name but different type → kept separately? 
        # Current design: key is (canonical_lower, type). So different types → both kept.
        entities = [
            Entity(name="BLEU", type="Metric"),
            Entity(name="BLEU", type="Score"),
        ]
        result = dedupe_entities(entities)
        assert len(result) == 2

    def test_multiple_merges(self, sample_entities):
        result = dedupe_entities(sample_entities)
        names = [e.name for e in result]
        # RNN and its full form → 1; LSTM and its full form → 1; Transformer, BLEU → 1 each
        assert "RNN" not in names        # alias replaced
        assert "LSTM" not in names       # alias replaced
        assert "28.4" not in names       # dropped
        assert "" not in names           # dropped


# ---------------------------------------------------------------------------
# dedupe_relations
# ---------------------------------------------------------------------------

class TestDedupeRelations:
    def test_removes_exact_duplicate(self):
        rels = [
            Relation("A", "Concept", "USES", "B", "Concept", "ev1"),
            Relation("A", "Concept", "USES", "B", "Concept", "ev2"),  # same triple
        ]
        result = dedupe_relations(rels)
        assert len(result) == 1

    def test_keeps_different_relations(self):
        rels = [
            Relation("A", "Concept", "USES", "B", "Concept", None),
            Relation("A", "Concept", "IMPROVES", "B", "Concept", None),
        ]
        result = dedupe_relations(rels)
        assert len(result) == 2

    def test_drops_empty_head(self):
        rels = [Relation("", "Concept", "USES", "B", "Concept", None)]
        result = dedupe_relations(rels)
        assert len(result) == 0

    def test_drops_empty_tail(self):
        rels = [Relation("A", "Concept", "USES", "", "Concept", None)]
        result = dedupe_relations(rels)
        assert len(result) == 0

    def test_normalises_relation_type(self):
        rels = [Relation("A", "Concept", "uses", "B", "Concept", None)]
        result = dedupe_relations(rels)
        assert result[0].relation == "USES"


# ---------------------------------------------------------------------------
# remap_relation_endpoints
# ---------------------------------------------------------------------------

class TestRemapRelationEndpoints:
    def test_remaps_head(self):
        alias = {"rnn": "Recurrent Neural Network"}
        rels = [Relation("RNN", "Model", "USES", "LSTM", "Model", None)]
        result = remap_relation_endpoints(rels, alias)
        assert result[0].head == "Recurrent Neural Network"

    def test_remaps_tail(self):
        alias = {"lstm": "Long Short-Term Memory"}
        rels = [Relation("RNN", "Model", "USES", "LSTM", "Model", None)]
        result = remap_relation_endpoints(rels, alias)
        assert result[0].tail == "Long Short-Term Memory"

    def test_no_change_when_no_alias(self):
        alias = {}
        rels = [Relation("Transformer", "Model", "IMPROVES", "BLEU", "Metric", None)]
        result = remap_relation_endpoints(rels, alias)
        assert result[0].head == "Transformer"
        assert result[0].tail == "BLEU"

    def test_empty_relations(self):
        result = remap_relation_endpoints([], {"rnn": "Recurrent Neural Network"})
        assert result == []


# ---------------------------------------------------------------------------
# normalize_rel_type
# ---------------------------------------------------------------------------

class TestNormalizeRelType:
    def test_uppercases(self):
        assert normalize_rel_type("uses") == "USES"

    def test_spaces_to_underscores(self):
        assert normalize_rel_type("is based on") == "IS_BASED_ON"

    def test_none_fallback(self):
        assert normalize_rel_type(None) == "RELATED_TO"

    def test_empty_fallback(self):
        assert normalize_rel_type("") == "RELATED_TO"
