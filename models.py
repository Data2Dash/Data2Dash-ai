"""
models.py - Shared Data Models
===============================
Single source of truth for all dataclasses used across the project.
Pure data containers only.
"""

from __future__ import annotations

import time
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# PDF PROCESSING DATA CLASSES
# -----------------------------------------------------------------------------

@dataclass
class ProcessedEquation:
    equation_id: str
    global_number: int
    text: str
    latex: Optional[str]
    page_number: int
    bbox: Tuple[float, float, float, float]
    section: str = ""
    context: str = ""
    confidence: float = 0.9
    raw_text: str = ""
    description: str = ""
    equation_type: str = "display"
    normalized_latex: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessedTable:
    table_id: str
    global_number: int
    page_number: int
    bbox: Tuple[float, float, float, float]
    markdown: str
    html_table: str = ""
    raw_text: str = ""
    caption: str = ""
    headers: List[str] = field(default_factory=list)
    parsed_data: Optional[Any] = None
    confidence: float = 0.9
    section: str = ""
    description: str = ""
    table_image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if "parsed_data" in d:
            d.pop("parsed_data")
            d["has_parsed_data"] = self.parsed_data is not None
        return d


@dataclass
class ProcessedFigure:
    figure_id: str
    global_number: int
    page_number: int
    bbox: Tuple[float, float, float, float]
    image_path: str = ""
    caption: str = ""
    raw_text: str = ""
    description: str = ""
    confidence: float = 0.9
    section: str = ""
    saved_path: Optional[str] = None
    page_width: float = 0.0
    page_height: float = 0.0
    visual_content_score: float = 1.0

    def __post_init__(self):
        if not self.saved_path and self.image_path:
            self.saved_path = self.image_path
        if not self.image_path and self.saved_path:
            self.image_path = self.saved_path

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessedSection:
    section_id: str
    title: str
    page_number: int
    content: str
    level: int = 1
    subsections: List[str] = field(default_factory=list)
    equations: List[int] = field(default_factory=list)
    tables: List[int] = field(default_factory=list)
    figures: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessedDocument:
    doc_id: str
    filename: str
    num_pages: int
    page_texts: List[str]
    enriched_page_texts: List[str]
    sections: List[ProcessedSection]
    equations: List[ProcessedEquation]
    tables: List[ProcessedTable]
    figures: List[ProcessedFigure]
    equation_registry: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    table_registry: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    figure_registry: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    title: str = "Unknown"
    authors: List[str] = field(default_factory=list)
    affiliations: List[str] = field(default_factory=list)
    abstract: str = ""
    year: str = ""
    date: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.equation_registry:
            self.equation_registry = {
                eq.global_number: eq.to_dict() for eq in self.equations
            }
        if not self.table_registry:
            self.table_registry = {
                tb.global_number: tb.to_dict() for tb in self.tables
            }
        if not self.figure_registry:
            self.figure_registry = {
                fig.global_number: fig.to_dict() for fig in self.figures
            }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "num_pages": self.num_pages,
            "title": self.title,
            "authors": self.authors,
            "affiliations": self.affiliations,
            "abstract": self.abstract,
            "year": self.year,
            "date": self.date,
            "sections": [s.to_dict() for s in self.sections],
            "equations": [e.to_dict() for e in self.equations],
            "tables": [t.to_dict() for t in self.tables],
            "figures": [f.to_dict() for f in self.figures],
            "metadata": self.metadata,
        }


# -----------------------------------------------------------------------------
# CHUNK / RETRIEVAL DATA CLASSES
# -----------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    doc_id: str
    page_number: int
    chunk_type: str
    metadata: Dict[str, Any]
    image_path: Optional[str] = None


@dataclass
class MultimodalChunk:
    chunk_id: str
    text: str
    doc_id: str
    page_num: int
    chunk_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding_text: str = ""
    section: str = ""
    global_number: Optional[int] = None
    image_path: Optional[str] = None

    @property
    def page_number(self) -> int:
        return self.page_num

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResult:
    chunk: MultimodalChunk
    similarity_score: float
    rank: int

    def __iter__(self):
        yield self.chunk
        yield self.similarity_score


# -----------------------------------------------------------------------------
# GLOBAL ELEMENT REGISTRY
# -----------------------------------------------------------------------------

class GlobalElementRegistry:
    def __init__(self):
        self._registry: Dict[str, Dict[str, Dict[int, str]]] = {}
        self._versions: Dict[str, List[Dict[str, Any]]] = {}
        self._docs: Dict[str, Dict[str, Any]] = {}

    def register(self, doc_id: str, chunk: MultimodalChunk) -> None:
        self._docs.setdefault(doc_id, {}).setdefault("chunks", []).append(chunk)
        num = chunk.metadata.get("global_number", chunk.global_number)
        if num is None:
            return
        self._registry.setdefault(doc_id, {}).setdefault(chunk.chunk_type, {})[
            int(num)
        ] = chunk.chunk_id

    def get(self, doc_id: str) -> Dict[str, Any]:
        return self._docs.get(doc_id, {})

    def lookup(
        self, element_type: str, number: int, doc_id: Optional[str] = None
    ) -> Optional[str]:
        if doc_id:
            return self._registry.get(doc_id, {}).get(element_type, {}).get(number)
        for doc_registry in self._registry.values():
            cid = doc_registry.get(element_type, {}).get(number)
            if cid:
                return cid
        return None

    def register_version(self, doc_id: str, metadata: Dict[str, Any]) -> str:
        version_id = f"v{len(self._versions.get(doc_id, [])) + 1}_{int(time.time())}"
        self._versions.setdefault(doc_id, []).append(
            {
                "version_id": version_id,
                "timestamp": time.time(),
                "metadata": metadata,
            }
        )
        logger.info("Registered version %s for doc %s", version_id, doc_id)
        return version_id

    def get_versions(self, doc_id: str) -> List[Dict[str, Any]]:
        return self._versions.get(doc_id, [])

    def clear(self, doc_id: Optional[str] = None) -> None:
        if doc_id:
            self._registry.pop(doc_id, None)
            self._versions.pop(doc_id, None)
            self._docs.pop(doc_id, None)
        else:
            self._registry.clear()
            self._versions.clear()
            self._docs.clear()

    def stats(self) -> Dict[str, Any]:
        total = sum(
            sum(len(nums) for nums in types.values())
            for types in self._registry.values()
        )
        return {"total_registered": total, "documents": list(self._registry.keys())}
