"""
Lexical TF–IDF similarity reranker (no extra model weights on disk).

Treats the user query vs each candidate document (title + truncated abstract)
as a sparse bag-of-words vector, applies TF–IDF weighting, and returns cosine
similarities in ``[0, 1]``.  Intended as a **light semantic-ish** signal on
top-N candidates after the main composite score — configurable via settings.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Sequence, Dict


_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return _TOKEN.findall((text or "").lower())


def tfidf_cosine_similarities(query: str, documents: Sequence[str]) -> List[float]:
    """
    Return one similarity score per document (same order), each in [0, 1].

    Empty query or all-empty documents yields zeros.
    """
    q_toks = _tokens(query)
    if not q_toks:
        return [0.0] * len(documents)

    doc_tokens: List[List[str]] = []
    for doc in documents:
        doc_tokens.append(_tokens(doc))

    df: Counter[str] = Counter()
    for toks in doc_tokens:
        df.update(set(toks))
    n_docs = max(len(documents), 1)

    def idf(term: str) -> float:
        c = df.get(term, 0)
        # Smooth idf
        return math.log((n_docs + 1.0) / (c + 1.0)) + 1.0

    q_vec: Dict[str, float] = {}
    for t in set(q_toks):
        tf = q_toks.count(t) / max(len(q_toks), 1)
        q_vec[t] = tf * idf(t)

    def norm(counter: Dict[str, float]) -> float:
        return math.sqrt(sum(v * v for v in counter.values()))

    q_norm = norm(q_vec)
    if q_norm <= 0:
        return [0.0] * len(documents)

    scores: List[float] = []
    for toks in doc_tokens:
        if not toks:
            scores.append(0.0)
            continue
        d_vec: Dict[str, float] = {}
        for t in set(toks):
            tf = toks.count(t) / max(len(toks), 1)
            d_vec[t] = tf * idf(t)
        dot = sum(q_vec[t] * d_vec.get(t, 0.0) for t in q_vec)
        dn = norm(d_vec)
        sim = dot / (q_norm * dn) if dn > 0 else 0.0
        scores.append(max(0.0, min(1.0, sim)))
    return scores


def build_document_text(title: str, abstract: str, max_abstract_chars: int = 480) -> str:
    ab = (abstract or "").replace("\n", " ")
    if len(ab) > max_abstract_chars:
        ab = ab[:max_abstract_chars]
    return f"{title or ''} {ab}".strip()
