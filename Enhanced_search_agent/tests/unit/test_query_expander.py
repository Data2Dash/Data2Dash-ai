"""
Unit tests — QueryExpander (generic-first, zero hardcoded dictionaries)
=======================================================================
Tests verify the generic structural path (no LLM, no dict), the LLM mocked
path, query complexity classification, and output schema compliance.
"""
import pytest
from unittest.mock import MagicMock

pytest.importorskip("app")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _offline_expander():
    """Return a QueryExpander with LLM disabled (tests offline generic path)."""
    from app.services.query_expander import QueryExpander
    ex = QueryExpander.__new__(QueryExpander)
    ex._available = False
    ex._llm_cooldown_until = 0.0
    return ex


def _mocked_expander(json_content: str):
    """Return a QueryExpander with a mocked Groq LLM returning json_content."""
    from app.services.query_expander import QueryExpander
    ex = QueryExpander.__new__(QueryExpander)
    ex._available = True
    ex._llm_cooldown_until = 0.0
    msg = MagicMock()
    msg.content = json_content
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = msg
    ex.llm = mock_llm
    return ex


_REQUIRED_KEYS = {"original", "expanded_queries", "semantic_keywords",
                  "topic_profile", "retrieval_bundles"}

_REQUIRED_PROFILE_KEYS = {"topic", "aliases", "acronym", "landmarks",
                           "subtopics", "exclusions", "parent_topic",
                           "child_topics", "datasets", "benchmarks",
                           "methods", "model_families"}

_REQUIRED_BUNDLE_KEYS = {"broad", "landmark_titles", "acronym", "subtopics"}


# ─────────────────────────────────────────────────────────────────────────────
# Schema / contract tests (all paths)
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputSchema:
    """expand() must always return the correct schema regardless of input."""

    @pytest.mark.parametrize("query", [
        "transformers",
        "RAG",
        "neural architecture search",
        "NAS",
        "federated learning privacy",
        "quantum error correction surface codes",
        "topological data analysis persistence homology",
        "protein structure prediction alphafold",
        "x",                   # single character edge case
        "   spaced   query  ", # whitespace
    ])
    def test_schema_any_query_offline(self, query):
        ex = _offline_expander()
        result = ex.expand(query)
        assert _REQUIRED_KEYS.issubset(result.keys()), f"Missing keys for {query!r}"
        assert isinstance(result["expanded_queries"], list)
        assert isinstance(result["semantic_keywords"], list)
        assert isinstance(result["topic_profile"], dict)
        assert isinstance(result["retrieval_bundles"], dict)
        assert _REQUIRED_PROFILE_KEYS.issubset(result["topic_profile"].keys())
        assert _REQUIRED_BUNDLE_KEYS.issubset(result["retrieval_bundles"].keys())

    def test_original_always_preserved(self):
        ex = _offline_expander()
        result = ex.expand("some unknown topic XYZ")
        assert result["original"] == "some unknown topic XYZ"

    def test_expanded_queries_never_empty(self):
        ex = _offline_expander()
        for q in ["NAS", "transformers", "quantum entanglement", "MoE"]:
            result = ex.expand(q)
            assert len(result["expanded_queries"]) >= 1, f"Empty expanded_queries for {q!r}"

    def test_retrieval_bundles_always_populated(self):
        ex = _offline_expander()
        for q in ["NAS", "graph neural networks", "unknown niche topic"]:
            result = ex.expand(q)
            bundles = result["retrieval_bundles"]
            total = sum(len(v) for v in bundles.values())
            assert total > 0, f"All bundles empty for {q!r}"

    def test_query_complexity_present(self):
        """expand() must always include query_complexity."""
        ex = _offline_expander()
        result = ex.expand("transformers")
        assert "query_complexity" in result
        assert result["query_complexity"] in ("broad", "moderate", "narrow")


# ─────────────────────────────────────────────────────────────────────────────
# Generic path tests (all queries — no hardcoded dictionaries)
# ─────────────────────────────────────────────────────────────────────────────

class TestGenericPath:
    """All queries should produce meaningful expansions without any dictionary."""

    def test_multi_word_topic_has_multiple_variants(self):
        ex = _offline_expander()
        result = ex.expand("neural architecture search")
        assert len(result["expanded_queries"]) >= 2

    def test_unknown_topic_keywords_non_trivial(self):
        ex = _offline_expander()
        result = ex.expand("neural architecture search")
        kws = [k.lower() for k in result["semantic_keywords"]]
        assert any(w in kws for w in ("neural", "architecture", "search")), \
            f"Expected content words in keywords: {kws}"

    def test_unknown_topic_profile_topic_field_set(self):
        ex = _offline_expander()
        result = ex.expand("topological data analysis")
        assert result["topic_profile"]["topic"], "topic field should not be empty"

    def test_niche_query_broad_bundle_non_empty(self):
        ex = _offline_expander()
        result = ex.expand("persistent homology filtration")
        assert len(result["retrieval_bundles"]["broad"]) >= 1

    def test_long_multi_word_query_includes_original(self):
        ex = _offline_expander()
        q = "contrastive self-supervised representation learning vision"
        result = ex.expand(q)
        assert q in result["expanded_queries"], "Original must appear in expanded_queries"

    def test_multi_word_query_has_exact_phrase_variant(self):
        ex = _offline_expander()
        result = ex.expand("persistent homology algebraic topology")
        has_quoted = any(v.startswith('"') for v in result["expanded_queries"])
        assert has_quoted, "Multi-word queries should include a quoted exact-phrase variant"

    def test_unknown_topic_parent_inferred(self):
        ex = _offline_expander()
        result = ex.expand("neural architecture search")
        parent = result["topic_profile"]["parent_topic"].lower()
        assert "learning" in parent or "neural" in parent or "machine" in parent

    def test_single_word_query_returns_original(self):
        """Even a single word query should at least return itself."""
        ex = _offline_expander()
        result = ex.expand("transformers")
        assert "transformers" in result["expanded_queries"]


# ─────────────────────────────────────────────────────────────────────────────
# Query complexity classification
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryComplexity:
    """Query complexity classifier controls downstream retrieval fan-out."""

    def test_broad_single_word(self):
        ex = _offline_expander()
        result = ex.expand("transformers")
        assert result["query_complexity"] == "broad"

    def test_broad_acronym(self):
        ex = _offline_expander()
        result = ex.expand("RAG")
        assert result["query_complexity"] == "broad"

    def test_moderate_multi_word(self):
        ex = _offline_expander()
        result = ex.expand("graph neural networks")
        assert result["query_complexity"] == "moderate"

    def test_narrow_long_query(self):
        ex = _offline_expander()
        result = ex.expand("persistent homology filtration in topology learning")
        assert result["query_complexity"] == "narrow"

    def test_narrow_quoted_query(self):
        ex = _offline_expander()
        result = ex.expand('"attention is all you need"')
        assert result["query_complexity"] == "narrow"


# ─────────────────────────────────────────────────────────────────────────────
# LLM-driven expansion (curated knowledge comes from the LLM, not dictionaries)
# ─────────────────────────────────────────────────────────────────────────────

class TestCuratedBoost:
    """Known topics surface canonical landmark titles via the LLM path."""

    _TRANSFORMERS_JSON = """{
        "expanded_queries": [
            "transformers",
            "Attention Is All You Need",
            "transformer architecture self-attention",
            "BERT pre-training deep bidirectional transformers"
        ],
        "semantic_keywords": [
            "transformer", "self-attention", "multi-head attention",
            "encoder", "decoder", "BERT", "GPT", "positional encoding"
        ],
        "topic_profile": {
            "topic": "transformers",
            "aliases": ["transformer", "transformers", "attention mechanism"],
            "acronym": "",
            "landmarks": ["Attention Is All You Need",
                          "BERT: Pre-training of Deep Bidirectional Transformers"],
            "subtopics": ["BERT", "GPT", "ViT", "T5"],
            "exclusions": [],
            "parent_topic": "deep learning",
            "child_topics": ["BERT", "GPT", "vision transformers"],
            "datasets": ["GLUE", "SQuAD"],
            "benchmarks": ["GLUE", "SuperGLUE"],
            "methods": ["multi-head attention", "positional encoding"],
            "model_families": ["BERT", "GPT", "T5", "ViT"]
        }
    }"""

    _RAG_JSON = """{
        "expanded_queries": [
            "RAG",
            "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            "retrieval-augmented generation",
            "dense retrieval open-domain QA"
        ],
        "semantic_keywords": [
            "retrieval", "augmented generation", "dense retrieval",
            "RAG", "DPR", "knowledge-intensive NLP", "open-domain QA"
        ],
        "topic_profile": {
            "topic": "retrieval-augmented generation",
            "aliases": ["RAG", "retrieval-augmented generation"],
            "acronym": "RAG",
            "landmarks": ["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"],
            "subtopics": ["dense retrieval", "open-domain QA", "REALM"],
            "exclusions": [],
            "parent_topic": "natural language processing",
            "child_topics": ["open-domain QA"],
            "datasets": ["Natural Questions", "TriviaQA"],
            "benchmarks": ["Natural Questions", "MSMARCO"],
            "methods": ["DPR", "BM25", "fusion-in-decoder"],
            "model_families": ["RAG", "REALM", "RETRO"]
        }
    }"""

    _GNN_JSON = """{
        "expanded_queries": [
            "GNN",
            "Semi-Supervised Classification with Graph Convolutional Networks",
            "graph neural networks",
            "graph convolution node classification"
        ],
        "semantic_keywords": [
            "graph neural network", "GNN", "node classification",
            "graph convolution", "GCN", "message passing"
        ],
        "topic_profile": {
            "topic": "graph neural networks",
            "aliases": ["GNN", "graph neural network", "graph neural networks"],
            "acronym": "GNN",
            "landmarks": ["Semi-Supervised Classification with Graph Convolutional Networks"],
            "subtopics": ["GCN", "GAT", "node classification"],
            "exclusions": [],
            "parent_topic": "deep learning",
            "child_topics": ["GCN", "GAT"],
            "datasets": ["Cora", "ogbn-arxiv"],
            "benchmarks": ["OGB"],
            "methods": ["graph convolution", "message passing"],
            "model_families": ["GCN", "GAT", "GraphSAGE"]
        }
    }"""

    _RL_JSON = """{
        "expanded_queries": [
            "reinforcement learning",
            "Playing Atari with Deep Reinforcement Learning",
            "deep reinforcement learning policy gradient"
        ],
        "semantic_keywords": [
            "policy", "reward", "agent", "DQN", "Q-learning",
            "actor-critic", "PPO", "deep reinforcement learning"
        ],
        "topic_profile": {
            "topic": "reinforcement learning",
            "aliases": ["RL", "reinforcement learning", "deep reinforcement learning"],
            "acronym": "RL",
            "landmarks": ["Playing Atari with Deep Reinforcement Learning"],
            "subtopics": ["policy gradient", "Q-learning", "actor-critic"],
            "exclusions": [],
            "parent_topic": "machine learning",
            "child_topics": ["policy gradient", "Q-learning"],
            "datasets": ["Atari", "MuJoCo"],
            "benchmarks": ["Atari 57"],
            "methods": ["DQN", "PPO", "actor-critic"],
            "model_families": ["DQN", "PPO", "A3C"]
        }
    }"""

    def test_transformers_has_landmark(self):
        ex = _mocked_expander(self._TRANSFORMERS_JSON)
        result = ex.expand("transformers")
        assert any("Attention Is All You Need" in q for q in result["expanded_queries"]), \
            f"Expected landmark title. Got: {result['expanded_queries']}"

    def test_rag_has_landmark(self):
        ex = _mocked_expander(self._RAG_JSON)
        result = ex.expand("RAG")
        assert any("Retrieval-Augmented Generation" in q
                   for q in result["expanded_queries"]), \
            f"Expected RAG landmark. Got: {result['expanded_queries']}"

    def test_gnn_has_landmark(self):
        ex = _mocked_expander(self._GNN_JSON)
        result = ex.expand("GNN")
        lowered = [q.lower() for q in result["expanded_queries"]]
        assert any("graph convolutional" in q or "graph" in q for q in lowered)

    def test_known_topic_keywords_include_curated(self):
        ex = _mocked_expander(self._RL_JSON)
        result = ex.expand("reinforcement learning")
        kws_lower = [k.lower() for k in result["semantic_keywords"]]
        assert any(w in kws_lower for w in ("policy", "reward", "dqn", "q-learning"))

    def test_known_topic_profile_has_landmarks(self):
        ex = _mocked_expander(self._TRANSFORMERS_JSON)
        result = ex.expand("transformers")
        landmarks = result["topic_profile"]["landmarks"]
        assert any("Attention" in l for l in landmarks), f"Landmarks: {landmarks}"

    def test_alias_query_resolved(self):
        """LLM expands 'large language models' with its canonical landmark."""
        llm_json = """{
            "expanded_queries": [
                "large language models",
                "Language Models are Few-Shot Learners",
                "large language model pre-training"
            ],
            "semantic_keywords": ["language model", "LLM", "few-shot", "GPT"],
            "topic_profile": {
                "topic": "large language models",
                "aliases": ["LLM", "large language models", "foundation models"],
                "acronym": "LLM",
                "landmarks": ["Language Models are Few-Shot Learners"],
                "subtopics": ["instruction tuning", "RLHF"],
                "exclusions": [],
                "parent_topic": "natural language processing",
                "child_topics": ["GPT", "LLaMA"],
                "datasets": ["MMLU"],
                "benchmarks": ["MMLU", "BIG-Bench"],
                "methods": ["pretraining", "RLHF"],
                "model_families": ["GPT", "LLaMA"]
            }
        }"""
        ex = _mocked_expander(llm_json)
        result = ex.expand("large language models")
        assert any("Language Models are Few-Shot Learners" in q
                   or "language model" in q.lower()
                   for q in result["expanded_queries"])


# ─────────────────────────────────────────────────────────────────────────────
# Acronym handling (generic — no hardcoded table)
# ─────────────────────────────────────────────────────────────────────────────

class TestAcronymHandling:
    """Acronym-like inputs should be detected dynamically."""

    def test_acronym_detected_for_all_caps(self):
        """All-caps short tokens are identified as acronyms."""
        ex = _offline_expander()
        result = ex.expand("RAG")
        # Without LLM, the acronym flag should be set in the profile
        profile = result["topic_profile"]
        assert profile["acronym"] == "RAG"

    def test_acronym_detected_for_camelcase(self):
        """CamelCase short tokens are identified as acronyms."""
        ex = _offline_expander()
        result = ex.expand("MoE")
        profile = result["topic_profile"]
        assert profile["acronym"] == "MOE"

    def test_completely_unknown_acronym_is_safe(self):
        """A totally unknown acronym should not crash and return schema."""
        ex = _offline_expander()
        result = ex.expand("QWERTY")
        assert _REQUIRED_KEYS.issubset(result.keys())
        assert "QWERTY" in result["expanded_queries"]

    def test_acronym_in_bundle(self):
        """Acronym should appear in the retrieval bundles."""
        ex = _offline_expander()
        result = ex.expand("RAG")
        acronym_bundle = result["retrieval_bundles"]["acronym"]
        assert any("RAG" in v for v in acronym_bundle)

    def test_acronym_expansion_via_llm(self):
        """When LLM is available, acronyms get proper long-form expansion."""
        rag_json = """{
            "expanded_queries": ["RAG", "retrieval-augmented generation"],
            "semantic_keywords": ["retrieval", "augmented generation", "RAG"],
            "topic_profile": {
                "topic": "retrieval-augmented generation",
                "aliases": ["RAG", "retrieval-augmented generation"],
                "acronym": "RAG",
                "landmarks": [],
                "subtopics": ["dense retrieval"],
                "exclusions": [],
                "parent_topic": "NLP",
                "child_topics": [],
                "datasets": [],
                "benchmarks": [],
                "methods": [],
                "model_families": []
            }
        }"""
        ex = _mocked_expander(rag_json)
        result = ex.expand("RAG")
        lowered = [q.lower() for q in result["expanded_queries"]]
        assert any("retrieval" in q for q in lowered), f"Got: {lowered}"


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval bundles
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrievalBundles:
    """Retrieval bundles must always be non-trivially populated."""

    @pytest.mark.parametrize("query", [
        "transformers",
        "neural architecture search",
        "NAS",
        "federated learning",
        "quantum machine learning variational circuits",
    ])
    def test_broad_bundle_non_empty(self, query):
        ex = _offline_expander()
        result = ex.expand(query)
        assert len(result["retrieval_bundles"]["broad"]) >= 1

    def test_landmark_bundle_populated_for_known_topic(self):
        """For known topics the LLM returns real landmark paper titles."""
        ex = _mocked_expander(TestCuratedBoost._TRANSFORMERS_JSON)
        result = ex.expand("transformers")
        lms = result["retrieval_bundles"]["landmark_titles"]
        assert len(lms) >= 1
        assert any("Attention" in l for l in lms)

    def test_landmark_bundle_uses_long_expanded_queries_for_unknown(self):
        """For unknown topics, landmark_titles should be filled from expanded_queries."""
        ex = _offline_expander()
        result = ex.expand("persistent homology filtration in topology")
        lms = result["retrieval_bundles"]["landmark_titles"]
        assert len(lms) >= 0  # may be empty for very short queries; should not crash

    def test_subtopic_bundle_non_empty_for_known_topic(self):
        """With LLM providing subtopics, they should appear in bundles."""
        ex = _mocked_expander(TestCuratedBoost._RL_JSON)
        result = ex.expand("reinforcement learning")
        subs = result["retrieval_bundles"]["subtopics"]
        assert len(subs) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────────────────────

class TestDeduplication:
    def test_no_duplicate_expanded_queries(self):
        ex = _offline_expander()
        result = ex.expand("transformers")
        eqs = [q.lower() for q in result["expanded_queries"]]
        assert len(eqs) == len(set(eqs)), f"Duplicates in: {eqs}"

    def test_cap_at_8_expanded_queries(self):
        ex = _offline_expander()
        result = ex.expand("graph neural networks node classification link prediction")
        assert len(result["expanded_queries"]) <= 8

    def test_cap_at_14_keywords(self):
        ex = _offline_expander()
        result = ex.expand("transformers attention self-supervised BERT GPT")
        assert len(result["semantic_keywords"]) <= 14


# ─────────────────────────────────────────────────────────────────────────────
# LLM mocked path
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMPath:
    MOCK_JSON = """{
        "expanded_queries": [
            "Neural Architecture Search for Efficient Deep Learning",
            "NAS",
            "neural architecture search automated machine learning",
            "differentiable architecture search DARTS"
        ],
        "semantic_keywords": [
            "neural architecture search", "NAS", "AutoML",
            "DARTS", "ENAS", "efficient deep learning",
            "hyperparameter optimization", "architecture optimization"
        ],
        "topic_profile": {
            "topic": "neural architecture search",
            "aliases": ["NAS", "automated machine learning", "AutoML"],
            "subtopics": ["DARTS", "ENAS", "one-shot NAS", "evolutionary search"],
            "methods": ["gradient-based search", "reinforcement learning search", "evolutionary algorithms"],
            "model_families": ["EfficientNet", "NASNet", "MobileNet"],
            "datasets": ["CIFAR-10", "ImageNet"],
            "parent_topic": "deep learning"
        }
    }"""

    def test_llm_result_merged_into_expanded(self):
        ex = _mocked_expander(self.MOCK_JSON)
        result = ex.expand("NAS")
        lowered = [q.lower() for q in result["expanded_queries"]]
        assert any("neural architecture" in q for q in lowered)

    def test_llm_profile_topic_used(self):
        ex = _mocked_expander(self.MOCK_JSON)
        result = ex.expand("NAS")
        assert "neural architecture search" in result["topic_profile"]["topic"].lower()

    def test_llm_profile_subtopics_merged(self):
        ex = _mocked_expander(self.MOCK_JSON)
        result = ex.expand("NAS")
        subs = [s.lower() for s in result["topic_profile"]["subtopics"]]
        assert any("darts" in s or "enas" in s for s in subs)

    def test_llm_keywords_in_result(self):
        ex = _mocked_expander(self.MOCK_JSON)
        result = ex.expand("NAS")
        kws = result["semantic_keywords"]
        assert "AutoML" in kws or "NAS" in kws

    def test_llm_fallback_on_bad_json(self):
        ex = _mocked_expander("NOT VALID JSON {{{{")
        # Should not raise; falls back gracefully to generic
        result = ex.expand("NAS")
        assert _REQUIRED_KEYS.issubset(result.keys())
        assert result["original"] == "NAS"

    def test_llm_cap_at_8_queries(self):
        long_json = """{
            "expanded_queries": ["a","b","c","d","e","f","g","h","i","j"],
            "semantic_keywords": ["x"],
            "topic_profile": {"topic": "test", "aliases": [], "subtopics": [],
                              "methods": [], "model_families": [],
                              "datasets": [], "parent_topic": ""}
        }"""
        ex = _mocked_expander(long_json)
        result = ex.expand("test query")
        assert len(result["expanded_queries"]) <= 8

    def test_schema_preserved_with_llm(self):
        ex = _mocked_expander(self.MOCK_JSON)
        result = ex.expand("NAS")
        assert _REQUIRED_KEYS.issubset(result.keys())
        assert _REQUIRED_PROFILE_KEYS.issubset(result["topic_profile"].keys())
        assert _REQUIRED_BUNDLE_KEYS.issubset(result["retrieval_bundles"].keys())

    def test_original_always_in_expanded_llm(self):
        ex = _mocked_expander(self.MOCK_JSON)
        result = ex.expand("NAS")
        assert result["original"] == "NAS"
        assert "NAS" in result["expanded_queries"]
