"""
Normalize and extract bibliographic identifiers for cross-source deduplication.

Used by arXiv/OpenAlex normalizers and the deduplication layer. All functions
are pure string logic (no network I/O).
"""
from __future__ import annotations

import re
from typing import Optional


_ARXIV_ABS = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/|arXiv:)\s*([0-9]{4}\.[0-9]{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Za-z-]+)?/\d{7}(?:v\d+)?)",
    re.I,
)


def normalize_doi(raw: str | None) -> str:
    """Return lowercase DOI without URL prefix, or empty string."""
    if not raw:
        return ""
    s = str(raw).strip().lower()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s)
    s = s.replace("doi:", "").strip()
    return s


def extract_arxiv_id_from_url(url: str | None) -> str:
    """Extract arXiv id (e.g. 1706.03762) from a URL or arxiv: string."""
    if not url:
        return ""
    m = _ARXIV_ABS.search(url.replace(" ", ""))
    if not m:
        return ""
    aid = m.group(1).lower()
    # Strip version suffix for stable matching
    aid = re.sub(r"v\d+$", "", aid)
    return aid


def extract_arxiv_id_from_entry_id(entry_id: str | None) -> str:
    """Parse arXiv Python client's entry_id (often a URL)."""
    return extract_arxiv_id_from_url(entry_id or "")


def normalize_openalex_work_id(raw: str | None) -> str:
    """
    OpenAlex URLs look like https://openalex.org/W2741809807 — return the W… id.
    """
    if not raw:
        return ""
    s = str(raw).strip()
    if "/" in s:
        s = s.rsplit("/", 1)[-1]
    s = s.split("?")[0].strip()
    if re.match(r"^W\d+$", s, re.I):
        return s.upper() if s.startswith("W") else s
    return s


def publication_year(published_date: str | None) -> Optional[int]:
    """Parse ISO date or year string; return None if unknown."""
    if not published_date:
        return None
    s = published_date.strip()
    if len(s) >= 4 and s[:4].isdigit():
        y = int(s[:4])
        return y if 1000 <= y <= 3000 else None
    return None
