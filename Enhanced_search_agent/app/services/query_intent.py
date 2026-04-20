"""
Lightweight query-intent classification (no extra LLM call).

Drives ranking boosts/penalties: surveys, seminal papers, recency, benchmarks,
implementation-heavy work, theory vs applications.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class QueryIntent:
    wants_survey: bool = False
    wants_seminal: bool = False
    wants_recent: bool = False
    wants_implementation: bool = False
    wants_benchmark: bool = False
    wants_theory: bool = False
    wants_application: bool = False


_SURVEY = re.compile(
    r"\b(survey|review\s+paper|literature\s+review|overview\s+of|state\s+of\s+the\s+art)\b",
    re.I,
)
_SEMINAL = re.compile(
    r"\b(classic|seminal|foundational|landmark|canonical|must-read|influential\s+paper)\b",
    re.I,
)
_RECENT = re.compile(
    r"\b(recent|latest|new|newest|sota|state-of-the-art|2023|2024|2025|2026)\b",
    re.I,
)
_IMPL = re.compile(
    r"\b(code|implementation|github|pytorch|tensorflow|jax|open\s*source|reproduce)\b",
    re.I,
)
_BENCH = re.compile(
    r"\b(benchmarks?|dataset|leaderboard|evaluation|comparison|versus|vs\.?)\b",
    re.I,
)
_THEORY = re.compile(
    r"\b(theory|theorem|proof|convergence|bound|optimization\s+landscape)\b",
    re.I,
)
_APP = re.compile(
    r"\b(application|real-world|deployment|clinical|industry|production)\b",
    re.I,
)


def classify_query_intent(query: str, extra_phrases: Optional[Iterable[str]] = None) -> QueryIntent:
    """
    Rule-based intent from the user query plus optional expansion strings.

    ``extra_phrases`` is typically ``semantic_keywords`` from the expander.
    """
    parts: List[str] = [query or ""]
    if extra_phrases:
        parts.extend(str(p) for p in extra_phrases)
    blob = " \n ".join(parts)

    return QueryIntent(
        wants_survey=bool(_SURVEY.search(blob)),
        wants_seminal=bool(_SEMINAL.search(blob)),
        wants_recent=bool(_RECENT.search(blob)),
        wants_implementation=bool(_IMPL.search(blob)),
        wants_benchmark=bool(_BENCH.search(blob)),
        wants_theory=bool(_THEORY.search(blob)),
        wants_application=bool(_APP.search(blob)),
    )
