from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional


def _clean_str_list(values: List[str], lowercase: bool = False) -> List[str]:
    seen = set()
    cleaned = []
    for value in values or []:
        item = str(value).strip()
        if not item:
            continue
        key = item.lower() if lowercase else item
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item.lower() if lowercase else item)
    return cleaned


@dataclass(frozen=True)
class Paper:
    id: str
    title: str
    abstract: str
    authors: List[str]
    published_date: str
    source: str
    url: str

    # Cross-source deduplication keys
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    openalex_work_id: Optional[str] = None

    citations: int = 0
    influential_score: float = 0.0
    keywords: List[str] = field(default_factory=list)
    institution_names: List[str] = field(default_factory=list)
    topic_tags: List[str] = field(default_factory=list)
    venue: Optional[str] = None

    # Hybrid search signals
    semantic_score: float = 0.0
    topic_relevance_score: float = 0.0
    inferred_topic_tags: List[str] = field(default_factory=list)
    retrieval_path: Optional[str] = None

    # Ranking/debug
    ranking_reasons: Dict[str, Any] = field(default_factory=dict)

    # Hybrid rerank scores
    bm25_score: float = 0.0
    embedding_score: float = 0.0
    hybrid_relevance_score: float = 0.0

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("id must not be empty")
        if not self.title.strip():
            raise ValueError("title must not be empty")
        if not self.source.strip():
            raise ValueError("source must not be empty")
        if self.citations < 0:
            raise ValueError("citations must be non-negative")
        if self.influential_score < 0:
            raise ValueError("influential_score must be non-negative")
        if not (0.0 <= self.semantic_score <= 1.0):
            raise ValueError("semantic_score must be between 0 and 1")
        if not (0.0 <= self.topic_relevance_score <= 1.0):
            raise ValueError("topic_relevance_score must be between 0 and 1")
        if self.bm25_score < 0:
            raise ValueError("bm25_score must be non-negative")
        if self.embedding_score < 0:
            raise ValueError("embedding_score must be non-negative")
        if not (0.0 <= self.hybrid_relevance_score <= 1.0):
            raise ValueError("hybrid_relevance_score must be between 0 and 1")

        if self.published_date:
            try:
                datetime.strptime(self.published_date[:10], "%Y-%m-%d")
            except ValueError:
                raise ValueError("published_date must start with YYYY-MM-DD")

        object.__setattr__(self, "authors", _clean_str_list(self.authors, lowercase=False))
        object.__setattr__(self, "keywords", _clean_str_list(self.keywords, lowercase=True))
        object.__setattr__(self, "institution_names", _clean_str_list(self.institution_names, lowercase=False))
        object.__setattr__(self, "topic_tags", _clean_str_list(self.topic_tags, lowercase=True))
        object.__setattr__(self, "inferred_topic_tags", _clean_str_list(self.inferred_topic_tags, lowercase=True))

        if self.doi is not None:
            object.__setattr__(self, "doi", self.doi.strip() or None)
        if self.arxiv_id is not None:
            object.__setattr__(self, "arxiv_id", self.arxiv_id.strip() or None)
        if self.openalex_work_id is not None:
            object.__setattr__(self, "openalex_work_id", self.openalex_work_id.strip() or None)
        if self.venue is not None:
            object.__setattr__(self, "venue", self.venue.strip() or None)
        if self.retrieval_path is not None:
            object.__setattr__(self, "retrieval_path", self.retrieval_path.strip() or None)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)