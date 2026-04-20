"""
OpenAlex Service
  - search_papers(): returns normalised Paper objects from OpenAlex concept/keyword search
  - search_works(): legacy raw dict search (unchanged)

Resilience additions
────────────────────
* Bounded in-process LRU cache (256 entries) — eliminates duplicate round-trips
  when the same expanded query variant is searched by multiple threads.
* Exponential back-off with full jitter on connection errors / server errors.
* Explicit User-Agent header (OpenAlex requests it per their polite-pool docs).
* Configurable timeout raised to 25 s per call.
"""
import math
import random
import sys
import time
import threading
import json
from pathlib import Path
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import requests
from app.schemas.paper import Paper
from app.core.config import settings
from app.services.identifier_utils import (
    extract_arxiv_id_from_url,
    normalize_doi,
    normalize_openalex_work_id,
)

# ── Polite-pool User-Agent (OpenAlex recommends this) ─────────────────────────
_USER_AGENT = "DATA2DASH-SearchAgent/1.0 (research tool; contact=admin@data2dash.app)"


class _LRUCache:
    """Thread-safe bounded LRU cache."""
    def __init__(self, maxsize: int = 256):
        self._maxsize = maxsize
        self._data: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def set(self, key, value):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            if len(self._data) > self._maxsize:
                self._data.popitem(last=False)


class OpenAlexService:
    BASE_URL = "https://api.openalex.org"

    _cache = _LRUCache(maxsize=256)
    _request_lock = threading.Lock()

    # Back-off parameters
    _BASE_DELAY: float  = 2.0
    _MAX_DELAY:  float  = 30.0
    _MAX_RETRIES: int   = 2
    _cooldown_until: float = 0.0
    _cooldown_lock = threading.Lock()
    _disk_cache_path = Path(".cache/openalex_cache.json")
    _disk_fail_path = Path(".cache/openalex_failures.json")
    _min_interval_s: float = 1.2
    _last_request_s: float = 0.0

    def __init__(self):
        self.last_call = {
            "query": "",
            "per_page": 0,
            "attempted": 0,
            "network_calls": 0,
            "memory_cache_hit": 0,
            "disk_cache_hit": 0,
            "rate_limited": 0,
            "returned": 0,
            "total_matches_reported": None,
            "status": "unknown",  # ok | cached | rate_limited | error | empty
        }

    # ------------------------------------------------------------------
    # Search  →  list[Paper]
    # ------------------------------------------------------------------

    def search_papers(self, query: str, per_page: int = 10, *, offset: int = 0) -> list:
        """Full search: returns normalised Paper objects from OpenAlex.

        ``offset`` chooses the 1-based result page via OpenAlex's ``page`` param
        (``page = offset // per_page + 1``). Unaligned offsets share a page with
        the previous block.
        """
        query = self._normalise_query(query)
        per_page = max(1, min(int(per_page), 200))
        off = max(0, int(offset))
        page = min(500, (off // per_page) + 1)
        self.last_call = {
            "query": query,
            "per_page": per_page,
            "offset": off,
            "page": page,
            "attempted": 1,
            "network_calls": 0,
            "memory_cache_hit": 0,
            "disk_cache_hit": 0,
            "rate_limited": 0,
            "returned": 0,
            "total_matches_reported": None,
            "status": "unknown",
        }
        raw = self._fetch_works(query, per_page, page)
        papers = []
        for item in raw:
            p = self._normalize(item)
            if p:
                papers.append(p)
        self.last_call["returned"] = len(papers)
        if self.last_call["status"] == "unknown":
            self.last_call["status"] = "empty" if not papers else "ok"
        return papers

    def search_papers_paginated(
        self,
        query: str,
        *,
        per_page: int = 200,
        max_total: int = 2500,
        max_pages: int = 40,
    ) -> List[Paper]:
        """
        Cursor-based paging until ``max_total`` works, ``max_pages`` requests, or API ends.
        Does not use the single-page LRU/disk cache (each page is distinct).
        """
        query = self._normalise_query(query)
        per_page = max(1, min(int(per_page), 200))
        max_total = max(1, int(max_total))
        max_pages = max(1, int(max_pages))

        self.last_call = {
            "query": query,
            "per_page": per_page,
            "attempted": 1,
            "network_calls": 0,
            "memory_cache_hit": 0,
            "disk_cache_hit": 0,
            "rate_limited": 0,
            "returned": 0,
            "total_matches_reported": None,
            "status": "unknown",
            "paginated": True,
            "pages_fetched": 0,
        }

        if self._is_in_cooldown():
            self.last_call["rate_limited"] = 1
            self.last_call["status"] = "rate_limited"
            return []

        all_raw: List[dict] = []
        meta_count: Optional[int] = None
        cursor: Optional[str] = "*"
        pages = 0

        while pages < max_pages and len(all_raw) < max_total and cursor:
            payload = self._request_works_json(query, per_page, cursor)
            if payload is None:
                break
            if meta_count is None:
                meta_count = (payload.get("meta") or {}).get("count")
            batch = payload.get("results") or []
            if not batch:
                break
            all_raw.extend(batch)
            pages += 1
            self.last_call["pages_fetched"] = pages
            next_c = (payload.get("meta") or {}).get("next_cursor")
            if not next_c:
                break
            cursor = next_c
            if len(all_raw) >= max_total:
                break

        all_raw = all_raw[:max_total]
        papers: List[Paper] = []
        for item in all_raw:
            p = self._normalize(item)
            if p:
                papers.append(p)

        self.last_call["returned"] = len(papers)
        self.last_call["total_matches_reported"] = meta_count
        self.last_call["status"] = "ok" if papers else "empty"
        return papers

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request_works_json(self, query: str, per_page: int, cursor: str) -> Optional[Dict[str, Any]]:
        """Single OpenAlex /works request (optionally cursor-paged). No cache."""
        query = self._normalise_query(query)
        if self._is_in_cooldown():
            return None

        url = f"{self.BASE_URL}/works"
        params: Dict[str, Any] = {
            "search": query,
            "per-page": per_page,
            "select": (
                "id,title,abstract_inverted_index,authorships,"
                "publication_year,cited_by_count,primary_location,concepts,doi"
            ),
            "cursor": cursor,
        }
        if settings.OPENALEX_MAILTO:
            params["mailto"] = settings.OPENALEX_MAILTO
        headers = {"User-Agent": _USER_AGENT}

        for attempt in range(self._MAX_RETRIES):
            try:
                with self._request_lock:
                    now = time.time()
                    elapsed = now - self.__class__._last_request_s
                    if elapsed < self._min_interval_s:
                        time.sleep(self._min_interval_s - elapsed)
                    self.last_call["network_calls"] += 1
                    response = requests.get(url, params=params, headers=headers, timeout=25)
                    self.__class__._last_request_s = time.time()
                if response.status_code == 429:
                    retry_after = self._retry_after_seconds(response)
                    self._set_cooldown(retry_after or 20.0)
                    self.last_call["rate_limited"] = 1
                    self.last_call["status"] = "rate_limited"
                    if attempt < self._MAX_RETRIES - 1 and (retry_after and retry_after <= 3):
                        time.sleep(retry_after)
                        continue
                    return None
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt < self._MAX_RETRIES - 1:
                    delay = self._backoff_delay(attempt)
                    print(
                        f"[OpenAlexService] Paginated page failed: {e}. Retrying in {delay:.1f}s …",
                        file=sys.stderr,
                    )
                    time.sleep(delay)
                else:
                    print(f"[OpenAlexService] Paginated fetch failed for {query!r}: {e}", file=sys.stderr)
                    return None
        return None

    def _fetch_works(self, query: str, per_page: int, page: int = 1) -> list:
        query = self._normalise_query(query)
        page = max(1, min(int(page), 500))
        if self._is_in_cooldown():
            self.last_call["rate_limited"] = 1
            self.last_call["status"] = "rate_limited"
            stale = self._disk_cache_get(query, per_page, page)
            if stale is not None:
                self.last_call["disk_cache_hit"] = 1
                self.last_call["status"] = "cached"
                return stale
            return []

        cache_key = (query, per_page, page)
        cached = self._cache.get(cache_key)
        if cached is not None:
            print(f"[OpenAlexService] Cache hit for {query!r}", file=sys.stderr)
            self.last_call["memory_cache_hit"] = 1
            self.last_call["status"] = "cached"
            return cached.get("results", cached) if isinstance(cached, dict) else cached
        stale = self._disk_cache_get(query, per_page, page)
        if stale is not None:
            self._cache.set(cache_key, stale)
            self.last_call["disk_cache_hit"] = 1
            self.last_call["status"] = "cached"
            if isinstance(stale, dict):
                self.last_call["total_matches_reported"] = stale.get("meta_count")
                return stale.get("results", [])
            return stale

        # Failure cache: if this query was rate-limited recently, don't hit OpenAlex again.
        if self._failure_cache_rate_limited(query, per_page, page):
            self.last_call["rate_limited"] = 1
            self.last_call["status"] = "rate_limited"
            return []

        url = f"{self.BASE_URL}/works"
        params = {
            "search":   query,
            "per-page": per_page,
            "page":     page,
            "select":   (
                "id,title,abstract_inverted_index,authorships,"
                "publication_year,cited_by_count,primary_location,concepts,doi"
            ),
        }
        if settings.OPENALEX_MAILTO:
            params["mailto"] = settings.OPENALEX_MAILTO
        headers = {"User-Agent": _USER_AGENT}

        for attempt in range(self._MAX_RETRIES):
            try:
                with self._request_lock:
                    # Global spacing to be kind to OpenAlex + reduce 429s.
                    now = time.time()
                    elapsed = now - self.__class__._last_request_s
                    if elapsed < self._min_interval_s:
                        time.sleep(self._min_interval_s - elapsed)
                    self.last_call["network_calls"] += 1
                    response = requests.get(url, params=params, headers=headers, timeout=25)
                    self.__class__._last_request_s = time.time()
                if response.status_code == 429:
                    retry_after = self._retry_after_seconds(response)
                    self._set_cooldown(retry_after or 20.0)
                    self._failure_cache_set_rate_limited(query, per_page, page, retry_after or 20.0)
                    self.last_call["rate_limited"] = 1
                    self.last_call["status"] = "rate_limited"
                    # Retry once if Retry-After is short; otherwise fail fast.
                    if attempt < self._MAX_RETRIES - 1 and (retry_after and retry_after <= 3):
                        time.sleep(retry_after)
                        continue
                    raise requests.HTTPError("429 Too Many Requests")
                response.raise_for_status()
                payload = response.json()
                data = payload.get("results", [])
                meta_count = (payload.get("meta") or {}).get("count")
                bundle = {"results": data, "meta_count": meta_count}
                self.last_call["total_matches_reported"] = meta_count
                self._cache.set(cache_key, bundle)
                self._disk_cache_set(query, per_page, page, bundle)
                self.last_call["status"] = "ok"
                return data

            except Exception as e:
                if attempt < self._MAX_RETRIES - 1:
                    delay = self._backoff_delay(attempt)
                    print(
                        f"[OpenAlexService] Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s …",
                        file=sys.stderr,
                    )
                    time.sleep(delay)
                else:
                    print(
                        f"[OpenAlexService] All {self._MAX_RETRIES} attempts failed "
                        f"for query {query!r}: {e}",
                        file=sys.stderr,
                    )
                    stale = self._disk_cache_get(query, per_page, page)
                    if stale is not None:
                        self.last_call["disk_cache_hit"] = 1
                        self.last_call["status"] = "cached"
                        return stale
                    if self.last_call["status"] == "unknown":
                        self.last_call["status"] = "error"
                    return []

        return []   # unreachable, but keeps mypy happy

    @classmethod
    def _backoff_delay(cls, attempt: int) -> float:
        """Full-jitter exponential back-off."""
        ceiling = min(cls._MAX_DELAY, cls._BASE_DELAY * math.pow(2, attempt))
        return random.uniform(cls._BASE_DELAY, ceiling)

    @staticmethod
    def _retry_after_seconds(response) -> float:
        try:
            val = response.headers.get("Retry-After")
            return float(val) if val else 0.0
        except Exception:
            return 0.0

    @classmethod
    def _set_cooldown(cls, seconds: float) -> None:
        with cls._cooldown_lock:
            cls._cooldown_until = max(cls._cooldown_until, time.time() + min(max(seconds, 10.0), 120.0))

    @classmethod
    def _is_in_cooldown(cls) -> bool:
        with cls._cooldown_lock:
            return time.time() < cls._cooldown_until

    @classmethod
    def _disk_cache_ttl(cls) -> float:
        try:
            return float(getattr(settings, "DISK_CACHE_TTL_SECONDS", 86400) or 86400)
        except Exception:
            return 86400.0

    @classmethod
    def _disk_cache_get(cls, query: str, per_page: int, page: int = 1):
        try:
            path = cls._disk_cache_path
            if not path.exists():
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            key = f"{query.strip().lower()}::{per_page}::{page}"
            payload = data.get(key)
            ttl = cls._disk_cache_ttl()
            if isinstance(payload, dict) and payload.get("v") == 2:
                if time.time() - float(payload.get("ts", 0)) > ttl:
                    return None
                return payload.get("bundle")
            return payload
        except Exception:
            return None

    @classmethod
    def _disk_cache_set(cls, query: str, per_page: int, page: int, rows) -> None:
        try:
            path = cls._disk_cache_path
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
            key = f"{query.strip().lower()}::{per_page}::{page}"
            data[key] = {"v": 2, "ts": time.time(), "bundle": rows}
            # keep cache bounded
            if len(data) > 200:
                items = list(data.items())[-200:]
                data = {k: v for k, v in items}
            path.write_text(json.dumps(data), encoding="utf-8")
        except Exception:
            return

    @staticmethod
    def _normalise_query(query: str) -> str:
        q = " ".join((query or "").split())
        return q.strip()

    @classmethod
    def _failure_cache_rate_limited(cls, query: str, per_page: int, page: int = 1) -> bool:
        try:
            path = cls._disk_fail_path
            if not path.exists():
                return False
            data = json.loads(path.read_text(encoding="utf-8"))
            key = f"{query.strip().lower()}::{per_page}::{page}"
            until = float(data.get(key, 0.0) or 0.0)
            return time.time() < until
        except Exception:
            return False

    @classmethod
    def _failure_cache_set_rate_limited(cls, query: str, per_page: int, page: int, seconds: float) -> None:
        try:
            path = cls._disk_fail_path
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
            key = f"{query.strip().lower()}::{per_page}::{page}"
            data[key] = time.time() + min(max(seconds, 10.0), 180.0)
            if len(data) > 500:
                items = list(data.items())[-500:]
                data = {k: v for k, v in items}
            path.write_text(json.dumps(data), encoding="utf-8")
        except Exception:
            return

    def _normalize(self, item: dict) -> "Paper | None":
        title = (item.get("title") or "").strip()
        if not title:
            return None

        paper_id = item.get("id", "").split("/")[-1]
        abstract = self._reconstruct_abstract(item.get("abstract_inverted_index"))
        authors = [
            a.get("author", {}).get("display_name", "")
            for a in (item.get("authorships") or [])
        ]
        year = item.get("publication_year")
        published_date = f"{year}-01-01" if year else ""
        citations = item.get("cited_by_count") or 0

        # topics / concepts
        concepts = [c.get("display_name", "") for c in (item.get("concepts") or [])[:8]]

        # DOI → URL (canonical DOI is resolved again from ``ids`` below)
        location = item.get("primary_location") or {}
        venue = (location.get("source") or {}).get("display_name", "") if isinstance(location, dict) else ""

        ids = item.get("ids") or {}
        doi = normalize_doi(ids.get("doi") or item.get("doi"))
        arxiv_id = extract_arxiv_id_from_url(ids.get("arxiv") or "")
        oa_wid = normalize_openalex_work_id(ids.get("openalex") or item.get("id") or "")

        landing = (location.get("landing_page_url") or "") if isinstance(location, dict) else ""
        if landing:
            url = landing
        elif doi:
            url = f"https://doi.org/{doi}"
        else:
            url = f"https://openalex.org/{paper_id}"

        return Paper(
            id=paper_id,
            title=title,
            abstract=abstract,
            authors=authors,
            published_date=published_date,
            source="openalex",
            url=url,
            doi=doi,
            arxiv_id=arxiv_id,
            openalex_work_id=oa_wid or paper_id,
            citations=citations,
            influential_score=0.0,
            keywords=[],
            institution_names=[],
            topic_tags=concepts,
            venue=venue,
        )

    @staticmethod
    def _reconstruct_abstract(inverted_index: dict | None) -> str:
        """OpenAlex stores abstracts as an inverted index {word: [positions]}."""
        if not inverted_index:
            return ""
        try:
            word_positions = []
            for word, positions in inverted_index.items():
                for pos in positions:
                    word_positions.append((pos, word))
            word_positions.sort(key=lambda x: x[0])
            return " ".join(w for _, w in word_positions)
        except Exception:
            return ""