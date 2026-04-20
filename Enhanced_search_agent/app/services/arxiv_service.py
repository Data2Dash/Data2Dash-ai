# query -> arXiv API -> raw results -> normalize -> return list[Paper]
"""
ArXiv Service
Handles all communication with the arXiv API.

Key resilience features
───────────────────────
* Global inter-request spacing (≥3 s) — held via a class-level lock so that
  *concurrent threads* do not hammer the API simultaneously.
* Exponential back-off with full jitter on HTTP 429 / connection errors.
  Delays: ~4 s, ~8 s, ~16 s, ~32 s, ~64 s (capped) before giving up.
* Bounded in-process LRU cache (max 256 entries) keyed on
  (query, page, per_page, sort_by).  TTL is session-scoped (process lifetime).
  This eliminates duplicate network calls when the same expanded variant is
  searched by multiple threads.
"""
import itertools
import math
import random
import sys
import threading
import time
import json
from pathlib import Path
from collections import OrderedDict

import arxiv
from app.schemas.paper import Paper
from app.core.config import settings
from app.services.identifier_utils import extract_arxiv_id_from_entry_id


# ---------------------------------------------------------------------------
# Tiny bounded LRU cache — avoids importing functools.lru_cache which is not
# thread-safe for mutable return values without extra copying.
# ---------------------------------------------------------------------------
class _LRUCache:
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


class ArxivService:
    # ── Class-level singletons shared across all instances ──────────────────
    _cache = _LRUCache(maxsize=256)
    _rate_limit_lock = threading.Lock()
    _last_request_time: float = 0.0

    # Minimum gap between consecutive arXiv API calls (seconds)
    _MIN_INTERVAL: float = 3.5

    # Back-off parameters
    _BASE_DELAY: float = 4.0
    _MAX_DELAY: float = 64.0
    _MAX_RETRIES: int = 2
    _cooldown_until: float = 0.0
    _cooldown_lock = threading.Lock()
    _disk_cache_path = Path(".cache/arxiv_cache.json")

    def __init__(self):
        # arxiv.Client's built-in delay/retry is kept as a last-resort safety
        # net, but we manage timing ourselves for more predictable behaviour.
        self.client = arxiv.Client(
            page_size=10,
            delay_seconds=0.0,
            num_retries=1,          # let our own retry loop take charge
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_papers(
        self,
        query: str,
        page: int = 1,
        per_page: int = 10,
        sort_by: str = "relevance",
    ) -> list:
        query = " ".join((query or "").split())
        cache_key = (query, page, per_page, sort_by)
        if self._is_in_cooldown():
            stale = self._disk_cache_get(*cache_key)
            if stale is not None:
                return stale
            return []
        cached = self._cache.get(cache_key)
        if cached is not None:
            print(
                f"[ArxivService] Cache hit for {query!r} (page={page})",
                file=sys.stderr,
            )
            return cached
        stale = self._disk_cache_get(*cache_key)
        if stale is not None:
            self._cache.set(cache_key, stale)
            return stale

        offset = (page - 1) * per_page
        sort_mapping = {
            "relevance":       arxiv.SortCriterion.Relevance,
            "submittedDate":   arxiv.SortCriterion.SubmittedDate,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
        }

        search = arxiv.Search(
            query=query,
            max_results=offset + per_page,
            sort_by=sort_mapping.get(sort_by, arxiv.SortCriterion.Relevance),
        )

        results = []
        for attempt in range(self._MAX_RETRIES):
            try:
                self._throttle()                        # enforce global spacing
                generator = self.client.results(search)
                for result in itertools.islice(generator, offset, offset + per_page):
                    results.append(self._normalize_result(result))
                break  # success

            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "Too Many Requests" in err_str
                if is_rate_limit:
                    self._set_cooldown(30.0)

                if attempt < self._MAX_RETRIES - 1:
                    delay = self._backoff_delay(attempt, is_rate_limit)
                    print(
                        f"[ArxivService] Attempt {attempt + 1} failed "
                        f"({'429 rate-limit' if is_rate_limit else 'error'}): {e}. "
                        f"Retrying in {delay:.1f}s …",
                        file=sys.stderr,
                    )
                    time.sleep(delay)
                else:
                    print(
                        f"[ArxivService] All {self._MAX_RETRIES} attempts failed "
                        f"for query {query!r}: {e}",
                        file=sys.stderr,
                    )

        self._cache.set(cache_key, results)
        self._disk_cache_set(*cache_key, results)
        return results

    def search_papers_up_to(
        self,
        query: str,
        *,
        per_page: int = 50,
        max_total: int = 800,
        max_pages: int = 40,
        sort_by: str = "relevance",
    ) -> list:
        """
        Page through arXiv until ``max_total`` papers, ``max_pages`` requests, or a short page.
        Reuses ``search_papers`` (cache + throttle) per page.
        """
        query = " ".join((query or "").split())
        per_page = max(1, int(per_page))
        max_total = max(1, int(max_total))
        max_pages = max(1, int(max_pages))
        out: list = []
        page = 1
        while len(out) < max_total and page <= max_pages:
            batch = self.search_papers(query, page, per_page, sort_by)
            if not batch:
                break
            out.extend(batch)
            if len(batch) < per_page:
                break
            page += 1
        return out[:max_total]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _throttle(self):
        """Block the calling thread until the global inter-request gap is satisfied."""
        with self.__class__._rate_limit_lock:
            now = time.monotonic()
            elapsed = now - self.__class__._last_request_time
            if elapsed < self._MIN_INTERVAL:
                time.sleep(self._MIN_INTERVAL - elapsed)
            self.__class__._last_request_time = time.monotonic()

    @classmethod
    def _backoff_delay(cls, attempt: int, is_rate_limit: bool) -> float:
        """
        Full-jitter exponential back-off.
        On a 429 the cap is doubled to give the server extra breathing room.
        delay = random(0, min(cap, BASE * 2^attempt))
        """
        cap = cls._MAX_DELAY * (2 if is_rate_limit else 1)
        ceiling = min(cap, cls._BASE_DELAY * math.pow(2, attempt))
        return random.uniform(cls._BASE_DELAY, ceiling)   # always at least BASE

    def _normalize_result(self, result) -> Paper:
        category = getattr(result, "primary_category", "") or ""
        venue = f"arXiv {category}" if category else "arXiv"
        aid = extract_arxiv_id_from_entry_id(result.entry_id)
        return Paper(
            id=result.entry_id.split("/")[-1],
            title=result.title,
            abstract=result.summary.replace("\n", " "),
            authors=[author.name for author in result.authors],
            published_date=result.published.date().isoformat(),
            source="arxiv",
            url=result.pdf_url or result.entry_id,
            doi="",
            arxiv_id=aid,
            openalex_work_id="",
            citations=0,
            influential_score=0.0,
            keywords=[],
            institution_names=[],
            topic_tags=[],
            venue=venue,
        )

    @classmethod
    def _set_cooldown(cls, seconds: float) -> None:
        with cls._cooldown_lock:
            cls._cooldown_until = max(cls._cooldown_until, time.time() + min(max(seconds, 10.0), 180.0))

    @classmethod
    def _is_in_cooldown(cls) -> bool:
        with cls._cooldown_lock:
            return time.time() < cls._cooldown_until

    @classmethod
    def _disk_cache_ttl(cls) -> float:
        try:
            ax = int(getattr(settings, "ARXIV_DISK_CACHE_TTL_SECONDS", 0) or 0)
            if ax > 0:
                return float(ax)
            return float(getattr(settings, "DISK_CACHE_TTL_SECONDS", 86400) or 86400)
        except Exception:
            return 3600.0

    @classmethod
    def _disk_cache_get(cls, query: str, page: int, per_page: int, sort_by: str):
        try:
            path = cls._disk_cache_path
            if not path.exists():
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            key = f"{query.strip().lower()}::{page}::{per_page}::{sort_by}"
            payload = data.get(key)
            ttl = cls._disk_cache_ttl()
            if isinstance(payload, dict) and payload.get("v") == 2:
                if time.time() - float(payload.get("ts", 0)) > ttl:
                    return None
                rows = payload.get("rows")
            elif isinstance(payload, list):
                rows = payload
            else:
                return None
            if not isinstance(rows, list):
                return None
            return [Paper(**row) for row in rows if isinstance(row, dict)]
        except Exception:
            return None

    @classmethod
    def _disk_cache_set(cls, query: str, page: int, per_page: int, sort_by: str, rows: list) -> None:
        try:
            path = cls._disk_cache_path
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
            key = f"{query.strip().lower()}::{page}::{per_page}::{sort_by}"
            serial = [p.to_dict() if hasattr(p, "to_dict") else p for p in rows]
            data[key] = {"v": 2, "ts": time.time(), "rows": serial}
            if len(data) > 200:
                items = list(data.items())[-200:]
                data = {k: v for k, v in items}
            path.write_text(json.dumps(data), encoding="utf-8")
        except Exception:
            return