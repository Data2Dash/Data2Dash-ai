"""
Multi-signal academic paper deduplication.

Merges records from arXiv, OpenAlex, and local landmarks using DOI, arXiv id,
OpenAlex work id, normalized title, and (conservatively) high title similarity
with author + year agreement — reducing false merges vs title-only dedup.

Public API: ``dedupe_papers`` returns a list of merged |Paper| instances with
``source`` and ``retrieval_path`` ancestry preserved (same semantics as prior
``HybridSearchService._deduplicate``).
"""
from __future__ import annotations

import dataclasses
import re
from typing import Dict, List, Set, Tuple

from app.schemas.paper import Paper
from app.services.identifier_utils import (
    extract_arxiv_id_from_url,
    normalize_doi,
    normalize_openalex_work_id,
    publication_year,
)


def _norm_title(title: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (title or "").lower())).strip()


def _title_tokens(title: str) -> Set[str]:
    return {t for t in _norm_title(title).split() if len(t) > 1}


def _title_jaccard(a: str, b: str) -> float:
    ta, tb = _title_tokens(a), _title_tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def _author_surnames(authors: List[str], max_n: int = 4) -> Set[str]:
    out: Set[str] = set()
    for name in (authors or [])[:max_n]:
        parts = (name or "").strip().lower().split()
        if not parts:
            continue
        out.add(parts[-1])
    return out


def _merge_paths(p1: str, p2: str) -> str:
    vals: List[str] = []
    for v in (p1 or "").split(",") + (p2 or "").split(","):
        v = v.strip()
        if v and v not in vals:
            vals.append(v)
    return ",".join(vals)


def _merge_authors(a1: List[str], a2: List[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for name in (a1 or []) + (a2 or []):
        n = " ".join((name or "").split()).strip()
        key = n.lower()
        if len(n) < 3 or key in seen:
            continue
        seen.add(key)
        out.append(n)
    return out


def _normalise_tag(tag: str) -> str:
    t = _norm_title(tag)
    if not t:
        return ""
    mapping = {
        "gnn": "Graph Neural Networks",
        "gcn": "Graph Convolutional Networks",
        "llm": "Large Language Models",
        "rag": "Retrieval-Augmented Generation",
        "rl": "Reinforcement Learning",
    }
    return mapping.get(t, tag.strip())


def _merge_two_papers(winner: Paper, loser: Paper) -> Paper:
    """Return a new Paper merging ``loser`` into ``winner``.  Winner has >= citations."""
    merged_tags: List[str] = []
    for tag in winner.topic_tags + loser.topic_tags:
        nt = _normalise_tag(tag)
        if nt and nt not in merged_tags:
            merged_tags.append(nt)

    combined_sources = [s.strip() for s in winner.source.split(",") if s.strip()]
    for s in loser.source.split(","):
        s = s.strip()
        if s and s not in combined_sources:
            combined_sources.append(s)

    merged_abstract = (
        loser.abstract if len(loser.abstract) > len(winner.abstract) else winner.abstract
    )

    return dataclasses.replace(
        winner,
        topic_tags=merged_tags,
        source=",".join(combined_sources),
        authors=_merge_authors(winner.authors, loser.authors),
        venue=winner.venue or loser.venue,
        retrieval_path=_merge_paths(winner.retrieval_path, loser.retrieval_path),
        doi=winner.doi or loser.doi,
        arxiv_id=winner.arxiv_id or loser.arxiv_id,
        openalex_work_id=winner.openalex_work_id or loser.openalex_work_id,
        citations=max(winner.citations, loser.citations),
        abstract=merged_abstract,
    )


def _same_work(a: Paper, b: Paper) -> bool:
    """Return True if two records describe the same work (merge-safe)."""
    da, db = normalize_doi(a.doi), normalize_doi(b.doi)
    if da and db and da == db:
        return True

    axa, axb = (a.arxiv_id or "").lower(), (b.arxiv_id or "").lower()
    if axa and axb and axa == axb:
        return True
    # Recover arXiv id from URL if field empty
    if not axa:
        axa = extract_arxiv_id_from_url(a.url).lower()
    if not axb:
        axb = extract_arxiv_id_from_url(b.url).lower()
    if axa and axb and axa == axb:
        return True

    def _openalex_key(p: Paper) -> str:
        if (p.openalex_work_id or "").strip():
            return normalize_openalex_work_id(p.openalex_work_id)
        # Do not treat generic ``id`` as an OpenAlex work id unless the record is from OpenAlex.
        if "openalex" in (p.source or "").lower():
            return normalize_openalex_work_id(p.id)
        return ""

    oa, ob = _openalex_key(a), _openalex_key(b)
    if oa and ob and oa.upper() == ob.upper():
        return True

    na, nb = _norm_title(a.title), _norm_title(b.title)
    if na and nb and na == nb:
        return True

    # Conservative fuzzy merge: very high title overlap + year + author signal
    j = _title_jaccard(a.title, b.title)
    if j < 0.88:
        return False
    ya, yb = publication_year(a.published_date), publication_year(b.published_date)
    if ya is not None and yb is not None and abs(ya - yb) > 1:
        return False
    sa, sb = _author_surnames(a.authors), _author_surnames(b.authors)
    if sa and sb and not (sa & sb):
        return False
    return True


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[rj] = ri


def dedupe_papers(papers: List[Paper]) -> List[Paper]:
    """
    Deduplicate and merge a pool of papers.

    Preserves merged ``source`` (comma-separated) and ``retrieval_path`` like
    the previous title-only implementation, but clusters by stronger ids first.
    """
    if not papers:
        return []
    n = len(papers)
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if _same_work(papers[i], papers[j]):
                uf.union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        r = uf.find(i)
        groups.setdefault(r, []).append(i)

    merged: List[Paper] = []
    for _root, indices in groups.items():
        # Highest-citation record becomes the canonical merged object.
        indices_sorted = sorted(
            indices,
            key=lambda idx: (papers[idx].citations, len(papers[idx].abstract or "")),
            reverse=True,
        )
        base = papers[indices_sorted[0]]
        for idx in indices_sorted[1:]:
            base = _merge_two_papers(base, papers[idx])
        merged.append(base)
    return merged
