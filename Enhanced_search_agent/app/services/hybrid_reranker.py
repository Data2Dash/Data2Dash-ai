"""
Hybrid BM25 + embedding reranking for academic paper candidates.

Pipeline position
-----------------
Runs **after** hybrid retrieval + deduplication (candidate ``Paper`` list) and
**before** the final composite sort inside :class:`app.services.ranking_service.RankingService`
when ``settings.ENABLE_HYBRID_RERANK`` is true. It does **not** replace arXiv/OpenAlex
fetching.

Scoring
-------
1. Build ``paper_text`` per paper: title, abstract, keywords, topic_tags (see
   :func:`build_paper_text`).
2. **BM25** (Okapi) scores the query against the rerank corpus (top-N candidates only).
3. **Embedding** cosine similarity: L2-normalized query and document vectors from
   ``EmbeddingService`` → dot product in ``[0, 1]`` (non-negative for normalized positives).
4. Min–max normalize BM25 and embedding columns to ``[0, 1]`` within the pool.
5. ``hybrid_relevance = w_bm25 * bm25_n + w_emb * emb_n`` (weights renormalized to sum 1).

Explainability fields are written on each reranked ``Paper``:
``bm25_score``, ``embedding_score``, ``hybrid_relevance_score``, plus
``ranking_reasons`` keys from the caller.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from app.schemas.paper import Paper

_TOKEN = re.compile(r"[a-z0-9]+", re.I)


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens; drop very short tokens (noise for BM25)."""
    return [t.lower() for t in _TOKEN.findall(text or "") if len(t) > 2]


def build_paper_text(paper: Paper, max_chars: int = 6000) -> str:
    """
    Concatenate fields for lexical + dense retrieval (title first, then abstract).

    Format: ``"{title}. {abstract}. {keywords}. {topic_tags}"`` (truncated).
    """
    parts = [
        (paper.title or "").strip(),
        (paper.abstract or "").strip(),
        " ".join(paper.keywords or []),
        " ".join(paper.topic_tags or []),
    ]
    s = ". ".join(p for p in parts if p)
    if len(s) > max_chars:
        s = s[:max_chars]
    return s


def min_max_normalize(scores: Sequence[float], eps: float = 1e-9) -> List[float]:
    """Map scores to [0, 1]; constant input → 0.5 each."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < eps:
        return [0.5 for _ in scores]
    return [(float(s) - lo) / (hi - lo + eps) for s in scores]


class OkapiBM25:
    """
    Okapi BM25 over a tokenized corpus (query-time scoring only).

    Classic parameters ``k1``, ``b``; IDF uses Robertson–Spark Jones smoothing.
    """

    def __init__(self, tokenized_corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = tokenized_corpus
        self.N = len(tokenized_corpus)
        self.doc_freqs: Counter[str] = Counter()
        self.doc_lens: List[int] = []
        self.term_freqs: List[Counter[str]] = []

        for toks in tokenized_corpus:
            self.doc_lens.append(len(toks))
            tf = Counter(toks)
            self.term_freqs.append(tf)
            self.doc_freqs.update(tf.keys())

        self.avgdl = sum(self.doc_lens) / self.N if self.N else 0.0

    def _idf(self, term: str) -> float:
        n = self.doc_freqs.get(term, 0)
        # Robertson–Spark Jones IDF
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)

    def score_document(self, query_tokens: Sequence[str], doc_index: int) -> float:
        tf = self.term_freqs[doc_index]
        dl = self.doc_lens[doc_index]
        score = 0.0
        for term in query_tokens:
            if term not in tf:
                continue
            f = tf[term]
            idf = self._idf(term)
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl + 1e-9))
            score += idf * (f * (self.k1 + 1)) / (denom + 1e-9)
        return score

    def score_query(self, query_tokens: Sequence[str]) -> List[float]:
        if not self.N:
            return []
        qt = list(query_tokens)
        return [self.score_document(qt, i) for i in range(self.N)]


def compute_hybrid_scores(
    query: str,
    papers: List[Paper],
    bm25_weight: float,
    embedding_weight: float,
    embedder: Any,
) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    """
    Returns ``(bm25_raw, bm25_norm, emb_raw, emb_norm, hybrid)`` aligned with ``papers``.

    ``embedder`` must implement ``encode(texts: List[str]) -> ndarray`` with normalized rows.
    """
    if not papers:
        return [], [], [], [], []

    w1, w2 = float(bm25_weight), float(embedding_weight)
    s = w1 + w2
    if s <= 0:
        w1, w2 = 0.55, 0.45
    else:
        w1, w2 = w1 / s, w2 / s

    corpus_tokens = [tokenize(build_paper_text(p)) for p in papers]
    q_tokens = tokenize(query)
    if not q_tokens:
        q_tokens = tokenize(query.replace(".", " "))

    bm25 = OkapiBM25(corpus_tokens)
    bm25_raw = bm25.score_query(q_tokens) if corpus_tokens else [0.0] * len(papers)
    bm25_norm = min_max_normalize(bm25_raw)

    texts = [build_paper_text(p) for p in papers]
    doc_emb = embedder.encode(texts)
    q_emb = embedder.encode([query])
    # Cosine similarity = dot product when rows are L2-normalized
    emb_raw = (doc_emb @ q_emb[0]).tolist() if doc_emb.size else [0.0] * len(papers)
    emb_raw = [max(0.0, min(1.0, float(x))) for x in emb_raw]
    emb_norm = min_max_normalize(emb_raw)

    hybrid = [w1 * b + w2 * e for b, e in zip(bm25_norm, emb_norm)]
    return bm25_raw, bm25_norm, emb_raw, emb_norm, hybrid


def apply_hybrid_rerank_to_papers(
    query: str,
    papers: List[Paper],
    bm25_weight: float,
    embedding_weight: float,
    embedder: Any,
) -> Dict[str, Any]:
    """
    Mutates each paper in ``papers`` with lexical/embedding/hybrid debug scores.

    Returns a small metadata dict for ``ranking_reasons`` (pool size, weights).
    """
    meta: Dict[str, Any] = {
        "hybrid_rerank_pool": len(papers),
        "hybrid_bm25_weight": bm25_weight,
        "hybrid_embedding_weight": embedding_weight,
    }
    if not papers or not (query or "").strip():
        for p in papers:
            object.__setattr__(p, "bm25_score", 0.0)
            object.__setattr__(p, "embedding_score", 0.0)
            object.__setattr__(p, "hybrid_relevance_score", 0.0)
        meta["hybrid_rerank_status"] = "skipped_empty"
        return meta

    try:
        bm25_raw, bm25_n, emb_raw, emb_n, hybrid = compute_hybrid_scores(
            query, papers, bm25_weight, embedding_weight, embedder
        )
    except Exception as e:
        meta["hybrid_rerank_status"] = f"error:{type(e).__name__}"
        for p in papers:
            object.__setattr__(p, "bm25_score", 0.0)
            object.__setattr__(p, "embedding_score", 0.0)
            object.__setattr__(p, "hybrid_relevance_score", 0.0)
        return meta

    for i, p in enumerate(papers):
        # Expose min–max normalized components in [0, 1] plus fused hybrid score.
        object.__setattr__(p, "bm25_score", round(float(bm25_n[i]), 6))
        object.__setattr__(p, "embedding_score", round(float(emb_n[i]), 6))
        object.__setattr__(p, "hybrid_relevance_score", round(float(hybrid[i]), 6))

    meta["hybrid_rerank_status"] = "ok"
    meta["bm25_norm_mean"] = round(float(sum(bm25_n) / max(len(bm25_n), 1)), 4)
    return meta
