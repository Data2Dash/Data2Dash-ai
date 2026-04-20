"""
Restrict retrieval to computer science and computer-engineering–adjacent work.

Uses arXiv primary categories (from ``Paper.venue``) when present, OpenAlex-style
``topic_tags`` / ``inferred_topic_tags`` when present, and conservative substring
rules so obvious life-science hits (e.g. biology) drop out.
"""
from __future__ import annotations

import re
from typing import Iterable, List

from app.schemas.paper import Paper

_ARXIV_CAT = re.compile(r"arxiv\s+([\w.-]+)", re.I)

# arXiv subject classes aligned with CS + CE-style retrieval (tweak via venue only).
_ALLOWED_ARXIV_PREFIXES: tuple[str, ...] = (
    "cs.",
    "eess.",  # Electrical Eng. & Systems Science (signals, systems — CE-adjacent)
    "stat.ml",
)
# Explicit rejects (life / physical / econ) even if tags look noisy.
_BLOCKED_ARXIV_PREFIXES: tuple[str, ...] = (
    "q-bio.",
    "physics.",
    "cond-mat.",
    "quant-ph",
    "astro-ph",
    "nucl-th",
    "nucl-ex",
    "hep-",
    "gr-qc",
    "nlin.",
    "econ.",
    "q-fin.",
    "math-ph",
)

# OpenAlex / merged concept display names (lowercased substring match).
_TAG_BLOCKLIST: tuple[str, ...] = (
    "biology",
    "biological sciences",
    "biochemistry",
    "genetics",
    "genomics",
    "proteomics",
    "neuroscience",
    "ecology",
    "zoology",
    "botany",
    "medicine",
    "clinical",
    "immunology",
    "pharmacology",
    "pathology",
    "oncology",
    "cardiology",
    "surgery",
    "nursing",
    "psychiatry",
    "agronomy",
    "forestry",
    "marine biology",
    "virology",
    "microbiology",
    "cell biology",
    "molecular biology",
    "physiology",
    "anatomy",
    "dermatology",
    "astrophysics",
    "geology",
    "atmospheric science",
    "soil science",
    "veterinary",
)

_TAG_ALLOWLIST: tuple[str, ...] = (
    "computer science",
    "computing",
    "transformer",  # matches "Transformers", "Transformer …"
    "deep learning",
    "reinforcement learning",
    "neural network",
    "graph neural",
    "computer engineering",
    "software engineering",
    "electrical engineering",
    "electronic engineering",
    "telecommunications",
    "signal processing",
    "control system",
    "systems engineering",
    "information technology",
    "human computer interaction",
    "human–computer interaction",
    "theoretical computer science",
    "machine learning",
    "artificial intelligence",
    "data science",
    "world wide web",
    "internet",
    "computer network",
    "database",
    "operating systems",
    "computer programming",
    "programming language",
    "computer graphics",
    "computer vision",
    "natural language processing",
    "robotics",
    "embedded systems",
    "parallel computing",
    "distributed computing",
    "cloud computing",
    "high performance computing",
    "computer security",
    "cryptography",
    "algorithms",
    "software systems",
)


def _arxiv_primary_category(paper: Paper) -> str:
    v = (paper.venue or "").strip()
    m = _ARXIV_CAT.search(v)
    return (m.group(1) or "").strip().lower() if m else ""


def _sources(paper: Paper) -> List[str]:
    return [s.strip().lower() for s in (paper.source or "").split(",") if s.strip()]


def _tag_texts(paper: Paper) -> List[str]:
    out: List[str] = []
    for t in (paper.topic_tags or []):
        s = str(t).strip().lower()
        if s:
            out.append(s)
    for t in (getattr(paper, "inferred_topic_tags", None) or []):
        s = str(t).strip().lower()
        if s and s not in out:
            out.append(s)
    return out


def _any_substr(haystack: Iterable[str], needles: tuple[str, ...]) -> bool:
    for h in haystack:
        for n in needles:
            if n in h:
                return True
    return False


def paper_matches_cs_ce_fields(paper: Paper) -> bool:
    """
    True if the paper is treated as CS / computer-engineering scope for retrieval.

    * arXiv rows: primary category must match an allowed prefix and not a blocked prefix.
    * Concept tags: any blocklist phrase in a tag → reject; any allowlist phrase → helps pass.
    * arXiv-backed rows without tags rely on category; OpenAlex-only rows rely on tags.
    """
    srcs = _sources(paper)
    cat = _arxiv_primary_category(paper)

    if cat:
        if any(cat.startswith(b) for b in _BLOCKED_ARXIV_PREFIXES):
            return False
        if any(cat.startswith(a) for a in _ALLOWED_ARXIV_PREFIXES):
            tag_list = _tag_texts(paper)
            if tag_list and _any_substr(tag_list, _TAG_BLOCKLIST):
                return False
            return True

    tags = _tag_texts(paper)
    if tags:
        if _any_substr(tags, _TAG_BLOCKLIST):
            return False
        if _any_substr(tags, _TAG_ALLOWLIST):
            return True
        # OpenAlex-only rows must show an explicit CS/CE-ish concept; unknown tags are dropped.
        if "openalex" in srcs and "arxiv" not in srcs:
            return False
        return "arxiv" in srcs

    # No category and no tags (e.g. thin arXiv metadata): keep arXiv-sourced rows; drop unknown OA-only.
    if "arxiv" in srcs:
        return True
    if "openalex" in srcs:
        return False
    if "local_landmark" in srcs:
        return True
    return True
