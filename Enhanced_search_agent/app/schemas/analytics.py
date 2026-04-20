"""
Structured analytics summary schema for DATA2DASH.

Improvements in this version:
- stronger typing for nested records
- validation for numeric consistency
- immutable dataclasses for safer reuse
- helper properties for common checks
- safer ``from_dict`` / ``to_dict`` round-tripping
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Tuple


@dataclass(frozen=True)
class TopAuthorImpact:
    author: str
    papers: int = 0
    citations: int = 0
    avg_citations: float = 0.0

    def __post_init__(self) -> None:
        if self.papers < 0:
            raise ValueError("papers must be non-negative")
        if self.citations < 0:
            raise ValueError("citations must be non-negative")
        if self.avg_citations < 0:
            raise ValueError("avg_citations must be non-negative")


@dataclass(frozen=True)
class TopCitedPaper:
    title: str
    citations: int = 0
    url: str = ""
    year: int | None = None

    def __post_init__(self) -> None:
        if self.citations < 0:
            raise ValueError("citations must be non-negative")
        if self.year is not None and not (1000 <= self.year <= 9999):
            raise ValueError("year must be a 4-digit integer when provided")





def _clean_count_dict(data: Mapping[str, int]) -> Dict[str, int]:
    cleaned: Dict[str, int] = {}
    for key, value in data.items():
        name = str(key).strip()
        count = int(value)
        if name and count >= 0:
            cleaned[name] = count
    return dict(sorted(cleaned.items(), key=lambda x: (-x[1], x[0].lower())))





@dataclass(frozen=True)
class AnalyticsSummary:
    total_papers: int
    papers_last_30_days: int
    top_authors: List[Tuple[str, int]] = field(default_factory=list)
    top_keywords: List[Tuple[str, int]] = field(default_factory=list)
    monthly_counts: Dict[str, int] = field(default_factory=dict)
    trend_status: str = "Stable"

    # Extended metrics
    avg_citations: float = 0.0
    max_citations: int = 0
    year_distribution: Dict[str, int] = field(default_factory=dict)
    field_distribution: Dict[str, int] = field(default_factory=dict)
    subtopic_distribution: Dict[str, int] = field(default_factory=dict)
    venue_distribution: Dict[str, int] = field(default_factory=dict)
    source_distribution: Dict[str, int] = field(default_factory=dict)
    year_subtopic_trends: Dict[str, Dict[str, int]] = field(default_factory=dict)
    top_subtopics: List[Tuple[str, int]] = field(default_factory=list)
    top_author_impact: List[TopAuthorImpact] = field(default_factory=list)
    top_cited_papers: List[TopCitedPaper] = field(default_factory=list)
    llm_insight: str = ""

    def __post_init__(self) -> None:
        if self.total_papers < 0:
            raise ValueError("total_papers must be non-negative")
        if self.papers_last_30_days < 0:
            raise ValueError("papers_last_30_days must be non-negative")
        if self.papers_last_30_days > self.total_papers:
            raise ValueError("papers_last_30_days cannot exceed total_papers")
        if self.avg_citations < 0:
            raise ValueError("avg_citations must be non-negative")
        if self.max_citations < 0:
            raise ValueError("max_citations must be non-negative")
        if not isinstance(self.trend_status, str) or not self.trend_status.strip():
            raise ValueError("trend_status must be a non-empty string")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
