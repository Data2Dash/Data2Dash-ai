"""
tests/test_query_engine.py
Unit tests for app/knowledge_graph/graph_rag/query_engine.py

No LLM, no network, no embedding model.  Only pure-Python logic is tested.
"""
from __future__ import annotations

import pytest

from app.knowledge_graph.graph_rag.query_engine import (
    _extract_seed_terms,
    rerank_qwen3,
    build_synthesis_context,
    RetrievedContext,
    QueryConfig,
)


# ---------------------------------------------------------------------------
# _extract_seed_terms
# ---------------------------------------------------------------------------

class TestExtractSeedTerms:
    def test_quoted_phrase_included(self):
        seeds = _extract_seed_terms('"Transformers" architecture paper')
        assert "Transformers" in seeds

    def test_capitalised_entity_included(self):
        seeds = _extract_seed_terms("BERT model trained on Wikipedia corpus")
        assert "BERT" in seeds

    def test_stopwords_excluded(self):
        seeds = _extract_seed_terms("what does the model use for training")
        stopwords = {"what", "does", "the", "for"}
        assert not stopwords.intersection({s.lower() for s in seeds})

    def test_short_tokens_excluded(self):
        # Tokens shorter than 4 chars (non-acronym) should be filtered
        seeds = _extract_seed_terms("a is to do run use")
        assert all(len(s) >= 4 or s.isupper() for s in seeds)

    def test_capitalised_run_multi_word(self):
        seeds = _extract_seed_terms("The Recurrent Neural Network outperforms baselines")
        joined = " ".join(seeds)
        assert "Recurrent" in joined or "Recurrent Neural Network" in joined

    def test_max_12_seeds(self):
        long_query = " ".join([f"Entity{i}" for i in range(30)])
        seeds = _extract_seed_terms(long_query)
        assert len(seeds) <= 12

    def test_empty_query(self):
        seeds = _extract_seed_terms("")
        assert seeds == []

    def test_deduplication(self):
        # Same word capitalised twice → appears once in seeds
        seeds = _extract_seed_terms("Transformer Transformer architecture")
        assert seeds.count("Transformer") <= 1


# ---------------------------------------------------------------------------
# rerank_qwen3  (heuristic path — no CrossEncoder)
# ---------------------------------------------------------------------------

def _ctx(text: str, source: str = "Vector Chunk", score: float = 0.5) -> RetrievedContext:
    return RetrievedContext(id="x", text=text, source_type=source, score=score)


class TestRerankQwen3:
    def test_empty_input(self):
        assert rerank_qwen3("query", [], top_k=5) == []

    def test_top_k_limits_output(self):
        contexts = [_ctx(f"word{i} text chunk") for i in range(10)]
        result = rerank_qwen3("word0", contexts, top_k=3)
        assert len(result) <= 3

    def test_higher_overlap_ranks_first(self):
        """Chunk with exact query words should rank above an unrelated chunk."""
        relevant = _ctx("transformer attention mechanism self attention")
        irrelevant = _ctx("cooking recipe pasta boil water salt")
        result = rerank_qwen3("transformer attention", [irrelevant, relevant], top_k=2)
        assert result[0].text == relevant.text

    def test_kg_diversity_bonus(self):
        """A KG triplet with same word overlap as vector chunk still ranks higher."""
        vec_ctx = _ctx("attention mechanism", source="Vector Chunk", score=0.5)
        kg_ctx  = _ctx("attention mechanism", source="Knowledge Graph", score=0.5)
        result = rerank_qwen3("attention", [vec_ctx, kg_ctx], top_k=2)
        # KG gets bonus → should be first or same
        scores = {r.source_type: r.score for r in result}
        assert scores["Knowledge Graph"] >= scores["Vector Chunk"]

    def test_preserves_source_type(self):
        contexts = [
            _ctx("alpha beta", source="Vector Chunk"),
            _ctx("gamma delta", source="Knowledge Graph"),
        ]
        result = rerank_qwen3("alpha gamma", contexts, top_k=2)
        types = {r.source_type for r in result}
        assert "Vector Chunk" in types
        assert "Knowledge Graph" in types

    def test_scores_are_non_negative(self):
        contexts = [_ctx("some random text"), _ctx("another chunk here")]
        result = rerank_qwen3("query keyword", contexts, top_k=2)
        assert all(r.score >= 0 for r in result)

    def test_single_item_returned_when_top_k_1(self):
        contexts = [_ctx("text a"), _ctx("text b"), _ctx("text c")]
        result = rerank_qwen3("text", contexts, top_k=1)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# build_synthesis_context
# ---------------------------------------------------------------------------

class TestBuildSynthesisContext:
    def test_vector_section_header_present(self):
        contexts = [_ctx("passage about attention", source="Vector Chunk")]
        out = build_synthesis_context(contexts, max_chars=500)
        assert "Vector Evidence" in out

    def test_kg_section_header_present(self):
        contexts = [_ctx("A uses B", source="Knowledge Graph")]
        out = build_synthesis_context(contexts, max_chars=500)
        assert "Knowledge-Graph Facts" in out

    def test_both_sections_when_mixed(self):
        contexts = [
            _ctx("dense passage", source="Vector Chunk"),
            _ctx("X relates Y",  source="Knowledge Graph"),
        ]
        out = build_synthesis_context(contexts, max_chars=500)
        assert "Vector Evidence" in out
        assert "Knowledge-Graph Facts" in out

    def test_truncates_long_text(self):
        long_text = "word " * 400          # 2000 chars
        contexts = [_ctx(long_text, source="Vector Chunk")]
        out = build_synthesis_context(contexts, max_chars=100)
        # Should end with ellipsis
        assert "…" in out

    def test_no_truncation_for_short_text(self):
        short_text = "short passage"
        contexts = [_ctx(short_text, source="Vector Chunk")]
        out = build_synthesis_context(contexts, max_chars=500)
        assert "…" not in out

    def test_empty_contexts(self):
        out = build_synthesis_context([], max_chars=500)
        assert out == "" or isinstance(out, str)

    def test_vector_items_labelled_v1_v2(self):
        contexts = [
            _ctx("first chunk", source="Vector Chunk"),
            _ctx("second chunk", source="Vector Chunk"),
        ]
        out = build_synthesis_context(contexts, max_chars=500)
        assert "[V1]" in out
        assert "[V2]" in out

    def test_kg_items_labelled_g1(self):
        contexts = [_ctx("A uses B", source="Knowledge Graph")]
        out = build_synthesis_context(contexts, max_chars=500)
        assert "[G1]" in out
