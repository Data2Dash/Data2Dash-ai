import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.services.semantic_rerank import tfidf_cosine_similarities, build_document_text


def test_tfidf_prefers_related_doc():
    q = "transformer attention mechanism"
    docs = [
        build_document_text("Cooking recipes", "pasta and sauce"),
        build_document_text("Attention Is All You Need", "self-attention transformer model"),
    ]
    sims = tfidf_cosine_similarities(q, docs)
    assert sims[1] > sims[0]
