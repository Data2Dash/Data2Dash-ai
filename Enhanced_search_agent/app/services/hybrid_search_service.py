"""
Hybrid Search Service — Anti-Gravity Retrieval
Combines ArXiv (keyword) and OpenAlex (concept/keyword) results with staged,
demand-driven retrieval that starts light and only expands when needed.

Anti-Gravity Retrieval Strategy
───────────────────────────────
* Stage 1 (Light):   1–2 query variants only. For broad queries, cap to 1.
                     Stop expansion if pool already has enough good candidates.
* Stage 2 (Landmark): Only fire if Stage 1 pool < quality threshold AND
                      landmark titles exist in retrieval bundles.
* Stage 3 (Progressive): Only fire if pool still below soft target.
* Recovery:          If arXiv returned 429 or zero results, do one small retry
                     then prefer OpenAlex continuation.

Provider-Aware Throttling
─────────────────────────
* If arXiv returns 429 during Stage 1, immediately suppress all further arXiv
  calls for this request and prefer cached/OpenAlex results.
* Subtopic fan-out is demand-gated: only fires when current pool is thin.

No Hardcoded Dictionaries
─────────────────────────
* Removed canonical_map from _topic_relevance_score
* Removed _normalise_tag hardcoded mapping
* All topic normalization is now query-driven
"""
import re
import sys
import threading
import time
import os
import concurrent.futures
from functools import partial
from typing import Dict, List, Optional, Callable

from app.schemas.paper import Paper
from app.core.config import settings
from app.services.cs_ce_field_filter import paper_matches_cs_ce_fields
from app.services.deduplication import dedupe_papers
from app.services.landmark_title_match import landmark_phrase_anchor_strength, normalize_landmark_title
from app.services.local_landmarks import fallback_landmarks_for_topic


class HybridSearchService:
    """
    Anti-gravity retrieval: start light, expand only when needed.

    Strategy
    --------
    1. Stage 1 (Light): Fire 1–2 query variants against ArXiv + OpenAlex.
       For broad queries (1–2 tokens), use only 1 variant.
    2. Check pool quality. If sufficient (≥ threshold), skip heavier stages.
    3. Stage 2 (Landmark injection): Only if pool < quality gate AND
       landmark titles exist with no match in the pool.
    4. Stage 3 (Progressive multi-page): Only if pool still insufficient.
    5. Score each paper with multi-tier relevance algorithm.
    6. Return deduplicated, relevance-annotated list.
    """

    # Quality thresholds for stage gating
    _STAGE2_POOL_THRESHOLD = 15   # skip landmark injection if pool ≥ this
    _STAGE3_POOL_THRESHOLD = 30   # minimum pool size before skipping progressive (see progressive gate below)

    # Noise words that should not count toward token-overlap hits
    _STOPWORDS = frozenset({
        "a", "an", "the", "of", "in", "on", "at", "to", "for",
        "and", "or", "with", "by", "is", "are", "was", "be",
        "via", "from", "using", "based", "towards", "over",
    })

    def __init__(
        self,
        arxiv_service,
        openalex_service,
        *,
        enable_openalex_retrieval: Optional[bool] = None,
    ):
        self.arxiv    = arxiv_service
        self.openalex = openalex_service
        self.last_accounting: dict = {}
        if enable_openalex_retrieval is not None:
            self._enable_openalex = bool(enable_openalex_retrieval)
        else:
            self._enable_openalex = bool(getattr(settings, "ENABLE_OPENALEX_RETRIEVAL", True))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        expanded: dict,       # output of QueryExpander.expand()
        per_source: int = 20, # results per source per query variant
        llm_future=None,      # async LLM expansion to merge in before progressive fetch
    ) -> List[Paper]:
        """
        Returns a deduplicated, relevance-scored list of Paper objects.
        Each paper has its `semantic_score` attribute set (0-1 float).
        """
        queries  = expanded.get("expanded_queries", [expanded.get("original", "")])
        keywords = [kw.lower() for kw in expanded.get("semantic_keywords", [])]
        original = expanded.get("original", queries[0] if queries else "")
        bundles = expanded.get("retrieval_bundles", {})
        topic_profile = expanded.get("topic_profile", {})
        complexity = expanded.get("query_complexity", "moderate")
        allow_local_fallback = bool(expanded.get("enable_offline_landmark_fallback", False))

        if not self._enable_openalex:
            print(
                "[HybridSearch] OpenAlex disabled — arXiv-only retrieval "
                "(unset ENABLE_OPENALEX_RETRIEVAL or set it to true to use OpenAlex).",
                file=sys.stderr,
            )

        pool: List[Paper] = []
        lock = threading.Lock()
        source_health = {"arxiv": False, "openalex": False}
        # Track if arXiv returned 429 during this request
        arxiv_throttled = {"value": False}
        metrics = {
            "openalex": {"attempted": 0, "cache": 0, "rate_limited": 0, "returned": 0},
            "arxiv": {"attempted": 0, "returned": 0},
            "local_injected": 0,
        }

        def _run(source_name: str, fn, *args):
            try:
                if source_name == "arxiv" and arxiv_throttled["value"]:
                    return  # Skip if arXiv already 429'd this request
                if source_name == "openalex":
                    metrics["openalex"]["attempted"] += 1
                if source_name == "arxiv":
                    metrics["arxiv"]["attempted"] += 1
                results = fn(*args)
                with lock:
                    if source_name == "openalex" and hasattr(self.openalex, "last_call"):
                        status = self.openalex.last_call.get("status", "unknown")
                        tag = "openalex_live" if status == "ok" else ("openalex_cache" if status == "cached" else "openalex_unknown")
                        for p in results:
                            object.__setattr__(p, "retrieval_path", tag)
                    if source_name == "arxiv":
                        for p in results:
                            object.__setattr__(p, "retrieval_path", "arxiv")
                    pool.extend(results)
                    if results:
                        source_health[source_name] = True
                    if source_name == "openalex" and hasattr(self.openalex, "last_call"):
                        lc = self.openalex.last_call
                        metrics["openalex"]["cache"] += int(bool(lc.get("memory_cache_hit") or lc.get("disk_cache_hit")))
                        metrics["openalex"]["rate_limited"] += int(bool(lc.get("rate_limited")))
                        metrics["openalex"]["returned"] += int(lc.get("returned") or 0)
                    if source_name == "arxiv":
                        metrics["arxiv"]["returned"] += len(results)
            except Exception as e:
                err_str = str(e)
                # Detect arXiv 429 and throttle immediately
                if source_name == "arxiv" and ("429" in err_str or "Too Many Requests" in err_str):
                    arxiv_throttled["value"] = True
                    print(f"[HybridSearch] arXiv 429 detected — throttling all further arXiv calls this request", file=sys.stderr)
                _fn = fn.func.__name__ if isinstance(fn, partial) else getattr(fn, "__name__", "unknown")
                print(f"[HybridSearch] source error ({_fn}): {e}", file=sys.stderr)

        # Per-call page sizes (broad: each stage uses at most this per variant; raw pool
        # is also hard-capped per API source immediately before dedup — see Fix 3).
        per_source_limit = 50 if complexity == "broad" else per_source
        arxiv_budget = min(per_source_limit, max(1, int(settings.ARXIV_PER_QUERY_CAP)))
        openalex_budget = (
            min(per_source_limit, max(1, int(settings.OPENALEX_PER_QUERY_CAP)))
            if self._enable_openalex
            else 0
        )

        ax_sort = getattr(settings, "ARXIV_SEARCH_SORT_BY", "relevance")
        arxiv_sp = partial(self.arxiv.search_papers, sort_by=ax_sort)

        # ── STAGE 1: Light retrieval ─────────────────────────────────────
        # Build stage 1 queries based on complexity
        fan_out = bool(getattr(settings, "FAN_OUT_ALL_VARIANTS", False))
        stage1_queries = self._build_stage1_queries(
            queries, bundles, topic_profile, complexity, original, fan_out
        )
        if not fan_out:
            print(
                f"[HybridSearch] Literal-query mode: stage1 uses up to {len(stage1_queries)} "
                f"variant(s), arXiv sort={ax_sort!r} (set FAN_OUT_ALL_VARIANTS=true for wider fan-out)",
                file=sys.stderr,
            )

        print(
            f"[HybridSearch] Stage 1: {len(stage1_queries)} variants "
            f"(complexity={complexity}, fan_out={fan_out})",
            file=sys.stderr,
        )

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        futures = []

        for q in stage1_queries:
            futures.append(executor.submit(_run, "arxiv", arxiv_sp, q, 1, arxiv_budget))
            if self._enable_openalex and openalex_budget > 0:
                futures.append(executor.submit(_run, "openalex", self.openalex.search_papers, q, openalex_budget))

        # Wait for Stage 1
        concurrent.futures.wait(futures, timeout=60)

        print(
            f"[HybridSearch] Stage 1 pool: {len(pool)} papers "
            f"(arxiv_throttled={arxiv_throttled['value']})",
            file=sys.stderr,
        )

        # ── STAGE 2: Landmark injection (demand-gated) ───────────────────
        # Only fire if pool is thin AND we have landmark titles to search
        landmark_titles = bundles.get("landmark_titles", [])
        if (
            len(pool) < self._STAGE2_POOL_THRESHOLD
            and landmark_titles
            and not self._provider_in_cooldown("arxiv")
            and not arxiv_throttled["value"]
        ):
            stage2_futures = []
            for q in landmark_titles[:2]:
                if len(q.split()) >= 4:  # Only search actual paper-title-like queries
                    stage2_futures.append(executor.submit(_run, "arxiv", arxiv_sp, q, 1, 5))
                    if self._enable_openalex:
                        stage2_futures.append(executor.submit(_run, "openalex", self.openalex.search_papers, q, 5))
            if stage2_futures:
                concurrent.futures.wait(stage2_futures, timeout=30)
                print(f"[HybridSearch] Stage 2 landmark injection: pool now {len(pool)}", file=sys.stderr)
        elif len(pool) < self._STAGE2_POOL_THRESHOLD and landmark_titles and self._enable_openalex:
            # arXiv throttled — try OpenAlex-only landmark injection
            stage2_futures = []
            for q in landmark_titles[:2]:
                if len(q.split()) >= 4:
                    stage2_futures.append(executor.submit(_run, "openalex", self.openalex.search_papers, q, 5))
            if stage2_futures:
                concurrent.futures.wait(stage2_futures, timeout=30)
                print(f"[HybridSearch] Stage 2 (OpenAlex-only): pool now {len(pool)}", file=sys.stderr)

        # ── Subtopic fan-out (demand-gated, max 2) ────────────────────────
        subtopics = bundles.get("subtopics", [])
        if len(pool) < self._STAGE2_POOL_THRESHOLD and subtopics:
            sub_futures = []
            for st in subtopics[:2]:
                if not arxiv_throttled["value"] and not self._provider_in_cooldown("arxiv"):
                    sub_futures.append(executor.submit(_run, "arxiv", arxiv_sp, st, 1, arxiv_budget))
                if self._enable_openalex and openalex_budget > 0:
                    sub_futures.append(executor.submit(_run, "openalex", self.openalex.search_papers, st, openalex_budget))
            if sub_futures:
                concurrent.futures.wait(sub_futures, timeout=30)
                print(f"[HybridSearch] Subtopic fan-out: pool now {len(pool)}", file=sys.stderr)

        # ── Recovery pass ─────────────────────────────────────────────────
        # If arXiv returned nothing (e.g., rate-limited), do one small retry
        if not source_health["arxiv"] and not arxiv_throttled["value"]:
            recovery_q = (topic_profile.get("aliases", [])[:1] or [original])
            for q in recovery_q[:1]:
                try:
                    retry = arxiv_sp(q, 1, 8)
                    if retry:
                        pool.extend(retry)
                        source_health["arxiv"] = True
                        break
                except Exception as e:
                    if "429" in str(e):
                        arxiv_throttled["value"] = True
                    print(f"[HybridSearch] arXiv recovery failed: {e}", file=sys.stderr)

        # ── Resolve delayed LLM expansion before progressive ─────────────
        if llm_future:
            try:
                expanded = llm_future.result(timeout=15.0)
                queries  = expanded.get("expanded_queries", [expanded.get("original", "")])
                keywords = [kw.lower() for kw in expanded.get("semantic_keywords", [])]
                bundles  = expanded.get("retrieval_bundles", {})
                topic_profile = expanded.get("topic_profile", {})
                print(f"[HybridSearch] LLM expansion injected before progressive fetch", file=sys.stderr)
            except Exception as e:
                print(f"[HybridSearch] LLM future failed or timed out: {e}", file=sys.stderr)

        # ── Relevance Feedback ────────────────────────────────────────────
        # Extract topic-specific keywords from high-confidence papers only.
        # Filter out generic academic noise terms.
        if getattr(settings, "ENABLE_RELEVANCE_FEEDBACK", True) and pool:
            from collections import Counter
            _GENERIC_NOISE = frozenset({
                "computer science", "artificial intelligence", "ai",
                "machine learning", "deep learning", "engineering",
                "mathematics", "science", "research", "study",
                "analysis", "review", "survey", "method", "approach",
                "algorithm", "model", "system", "framework", "network",
                "data", "information", "technology", "computing",
                "applied sciences", "natural sciences", "social sciences",
                "electrical engineering", "computer engineering",
            })
            # Use only top high-confidence papers (sorted by semantic_score)
            scored_pool = sorted(
                pool, key=lambda p: getattr(p, "semantic_score", 0.0), reverse=True
            )[:15]
            term_freq = Counter()
            for p in scored_pool:
                tags = getattr(p, "topic_tags", []) or []
                kws = getattr(p, "keywords", []) or []
                for term in tags + kws:
                    t = term.strip().lower()
                    if (
                        t
                        and len(t.split()) <= 3
                        and t not in _GENERIC_NOISE
                        and len(t) >= 3
                    ):
                        term_freq[t] += 1
            if term_freq:
                # Only keep terms that appear ≥2 times (reduces noise)
                discovered = [
                    t for t, count in term_freq.most_common(8)
                    if count >= 2 and t not in keywords
                ][:5]
                if discovered:
                    keywords.extend(discovered)
                    print(f"[HybridSearch] Relevance Feedback discovered terms: {discovered}", file=sys.stderr)

        # ── STAGE 3: Progressive fetch (demand-gated) ────────────────────
        # Do not use a tiny fixed ceiling (e.g. 30): Stage-1 can already return
        # ARXIV_PER_QUERY_CAP (+ OpenAlex) papers, which would skip progressive
        # forever and strand the pool at ~100. Expand until the pool reaches
        # at least one full first-pass budget above the quality floor, up to
        # RETRIEVAL_SOFT_CAP (progressive sub-caps still apply inside).
        ax_cap_cfg = max(1, int(getattr(settings, "ARXIV_PER_QUERY_CAP", 75)))
        oa_cap_cfg = (
            max(1, int(getattr(settings, "OPENALEX_PER_QUERY_CAP", 150)))
            if self._enable_openalex
            else 0
        )
        soft_cap = max(1, int(getattr(settings, "RETRIEVAL_SOFT_CAP", 5000) or 5000))
        progressive_need_pool = min(
            soft_cap,
            max(self._STAGE3_POOL_THRESHOLD, ax_cap_cfg + oa_cap_cfg + 1),
        )

        if (
            getattr(settings, "ENABLE_PROGRESSIVE_SOURCE_FETCH", True)
            and len(pool) < progressive_need_pool
        ):
            pool = self._progressive_extend_pool(
                pool,
                metrics,
                source_health,
                original,
                arxiv_throttled["value"],
                broad_per_source_cap=(per_source_limit if complexity == "broad" else None),
                arxiv_sort_by=ax_sort,
            )
        elif getattr(settings, "ENABLE_PROGRESSIVE_SOURCE_FETCH", True) and len(pool) >= progressive_need_pool:
            print(
                f"[HybridSearch] Skipping progressive fetch — pool already has {len(pool)} papers "
                f"(threshold={progressive_need_pool})",
                file=sys.stderr,
            )

        executor.shutdown(wait=False)

        print(
            f"[HybridSearch] Raw pool (staged + recovery + progressive): {len(pool)} papers",
            file=sys.stderr,
        )

        # ── Landmark injection (safety net, demand-gated) ─────────────────
        # Only inject if we have very few papers and providers aren't throttled
        provider_gate = (not self._provider_in_cooldown("arxiv")) or (
            self._enable_openalex and not self._provider_in_cooldown("openalex")
        )
        if (
            pool
            and len(pool) < self._STAGE2_POOL_THRESHOLD
            and provider_gate
            and not arxiv_throttled["value"]
        ):
            pool = self._inject_missing_landmarks(
                pool,
                bundles.get("landmark_titles", []),
                per_source_limit,
                arxiv_sp,
                self._enable_openalex,
            )

        need_local = not source_health["arxiv"] or (self._enable_openalex and not source_health["openalex"])
        if allow_local_fallback and need_local:
            before = len(pool)
            pool = self._inject_local_landmarks(pool, topic_profile)
            metrics["local_injected"] += max(len(pool) - before, 0)

        if complexity == "broad":
            pool = self._cap_raw_pool_per_source_broad(pool, cap=50)
            print(
                "[HybridSearch] Broad query cap applied: 50 per source (complexity=broad)",
                file=sys.stderr,
            )

        # ── Deduplicate ───────────────────────────────────────────────────
        unique = dedupe_papers(pool)
        after_dedup_n = len(unique)
        print(f"[HybridSearch] After dedup: {after_dedup_n} papers", file=sys.stderr)

        # ── Score relevance ───────────────────────────────────────────────
        for paper in unique:
            lexical_score = self._relevance_score(
                paper, keywords, queries + bundles.get("landmark_titles", []), original
            )
            topic_rel, inferred_tags = self._topic_relevance_score(
                paper=paper,
                topic_profile=topic_profile,
                expanded_queries=queries + bundles.get("landmark_titles", []),
                keywords=keywords,
            )
            object.__setattr__(paper, "topic_relevance_score", topic_rel)
            object.__setattr__(paper, "inferred_topic_tags", inferred_tags)
            # Blend metadata-aware relevance with lexical query-match score.
            object.__setattr__(paper, "semantic_score", round(min(1.0, 0.72 * topic_rel + 0.28 * lexical_score), 4))
            self._apply_source_quality_adjustment(paper, source_health)

        if bool(getattr(settings, "ENABLE_RETRIEVAL_CS_CE_FIELD_FILTER", True)):
            before_cs = len(unique)
            unique = [p for p in unique if paper_matches_cs_ce_fields(p)]
            dropped_cs = before_cs - len(unique)
            if dropped_cs:
                print(
                    f"[HybridSearch] CS/CE field filter removed {dropped_cs} papers "
                    f"(kept {len(unique)} of {before_cs}).",
                    file=sys.stderr,
                )

        # Sort by score descending
        unique.sort(key=lambda p: p.semantic_score, reverse=True)

        # Keep papers with a meaningful relevance signal
        MIN_KEEP = max(1, int(getattr(settings, "HYBRID_FILTER_KEEP_MIN", 50) or 50))
        MIN_SCORE = float(getattr(settings, "HYBRID_FILTER_MIN_SEMANTIC", 0.02) or 0.0)
        MIN_TOPIC = float(getattr(settings, "HYBRID_FILTER_MIN_TOPIC", 0.02) or 0.0)
        filtered = [
            p for p in unique
            if p.semantic_score > MIN_SCORE and getattr(p, "topic_relevance_score", 0.0) >= MIN_TOPIC
        ]
        removed_by_filter = len(unique) - len(filtered)
        if removed_by_filter > 0:
            print(
                f"[HybridSearch] Relevance filter removed {removed_by_filter} papers "
                f"(min_semantic={MIN_SCORE}, min_topic={MIN_TOPIC}). "
                f"Kept {len(filtered)} / {len(unique)} unique papers.",
                file=sys.stderr,
            )
        if len(filtered) < MIN_KEEP:
            extras = [p for p in unique if p not in filtered]
            extras.sort(key=lambda p: len(p.abstract), reverse=True)
            added = extras[: MIN_KEEP - len(filtered)]
            filtered += added
            print(
                f"[HybridSearch] MIN_KEEP fallback: added {len(added)} papers to reach {len(filtered)} total.",
                file=sys.stderr,
            )

        # Preserve multi-source spread when OpenAlex retrieval is enabled
        min_sources = 2 if self._enable_openalex else 1
        filtered = self._ensure_source_spread(filtered, unique, min_sources=min_sources)

        if not filtered:
            if unique:
                filtered = unique[:MIN_KEEP]
            elif allow_local_fallback:
                filtered = self._inject_local_landmarks([], topic_profile)

        # Clear, explicit per-search retrieval summary.
        print(
            "[HybridSearch] Retrieval summary | "
            f"arXiv: attempted={metrics['arxiv']['attempted']} returned={metrics['arxiv']['returned']} | "
            f"OpenAlex: attempted={metrics['openalex']['attempted']} returned={metrics['openalex']['returned']} "
            f"cache_hits={metrics['openalex']['cache']} rate_limited={metrics['openalex']['rate_limited']} | "
            f"local_injected={metrics['local_injected']} | "
            f"arxiv_throttled={arxiv_throttled['value']}",
            file=sys.stderr,
        )

        # Persist accounting for UI/API response.
        openalex_total = None
        openalex_status = "unknown"
        if hasattr(self.openalex, "last_call"):
            openalex_total = self.openalex.last_call.get("total_matches_reported")
            openalex_status = self.openalex.last_call.get("status", "unknown")

        def _kept_count(src_name: str) -> int:
            return sum(
                1 for p in filtered
                if any(s.strip() == src_name for s in (p.source or "").split(","))
            )

        def _count_with_source(papers: List[Paper], src: str) -> int:
            return sum(1 for p in papers if any(s.strip() == src for s in (p.source or "").split(",")))

        openalex_raw = _count_with_source(pool, "openalex")
        openalex_after_dedup = _count_with_source(unique, "openalex")
        openalex_after_filter = _count_with_source(filtered, "openalex")

        oa_pages = 0
        if hasattr(self.openalex, "last_call") and isinstance(self.openalex.last_call, dict):
            oa_pages = int(self.openalex.last_call.get("pages_fetched") or 0)

        self.last_accounting = {
            "progressive": {
                "enabled": bool(getattr(settings, "ENABLE_PROGRESSIVE_SOURCE_FETCH", True)),
                "openalex_retrieval_enabled": self._enable_openalex,
                "retrieval_soft_cap": int(getattr(settings, "RETRIEVAL_SOFT_CAP", 5000) or 5000),
                "arxiv_progressive_added": int(metrics.get("arxiv", {}).get("progressive_added", 0) or 0),
                "openalex_progressive_added": int(metrics.get("openalex", {}).get("progressive_added", 0) or 0),
                "openalex_pages_fetched": oa_pages,
            },
            "sources": {
                "arxiv": {
                    "source_total_matches": None,
                    "source_fetched_count": metrics["arxiv"]["returned"],
                    "source_kept_count": _kept_count("arxiv"),
                    "status": "ok" if source_health["arxiv"] else ("throttled" if arxiv_throttled["value"] else "error"),
                },
                "openalex": {
                    "source_total_matches": openalex_total,
                    "source_fetched_count": metrics["openalex"]["returned"],
                    "source_kept_count": _kept_count("openalex"),
                    "status": openalex_status,
                    "cache_hits": metrics["openalex"]["cache"],
                    "rate_limited": metrics["openalex"]["rate_limited"],
                    "raw_pool_count": openalex_raw,
                    "after_dedup_count": openalex_after_dedup,
                    "after_filter_count": openalex_after_filter,
                    "dedup_removed_est": max(openalex_raw - openalex_after_dedup, 0),
                    "filter_removed_est": max(openalex_after_dedup - openalex_after_filter, 0),
                },
                "local_landmark": {
                    "source_total_matches": None,
                    "source_fetched_count": metrics["local_injected"],
                    "source_kept_count": _kept_count("local_landmark"),
                    "status": "injected" if metrics["local_injected"] else "none",
                },
            },
            "pipeline": {
                "cs_ce_field_filter_enabled": bool(
                    getattr(settings, "ENABLE_RETRIEVAL_CS_CE_FIELD_FILTER", True)
                ),
                "raw_pool_count": len(pool),
                "after_dedup_count": after_dedup_n,
                "after_cs_ce_field_filter_count": len(unique),
                "cs_ce_field_filter_removed": max(
                    0,
                    after_dedup_n - len(unique),
                ),
                "after_filter_count": len(filtered),
                "final_ranked_count": len(filtered),
                "filter_removed_count": max(len(unique) - len(filtered), 0),
                "filter_thresholds": {
                    "min_semantic_score": MIN_SCORE,
                    "min_topic_relevance": MIN_TOPIC,
                    "min_keep_fallback": MIN_KEEP,
                },
                "query_complexity": complexity,
                "stage1_query_count": len(stage1_queries),
                "arxiv_throttled": arxiv_throttled["value"],
                "fan_out_all_variants": fan_out,
                "arxiv_search_sort_by": ax_sort,
            },
        }

        print(
            f"[HybridSearch] Final accounting | "
            f"raw_pool={len(pool)} → dedup={len(unique)} → filter={len(filtered)} "
            f"(removed_by_filter={removed_by_filter})",
            file=sys.stderr,
        )

        return filtered

    # ------------------------------------------------------------------
    # Stage 1 query builder (anti-gravity)
    # ------------------------------------------------------------------

    def _build_stage1_queries(
        self,
        expanded_queries: List[str],
        bundles: dict,
        topic_profile: dict,
        complexity: str,
        original: str,
        fan_out: bool,
    ) -> List[str]:
        """
        Build Stage 1 query list based on query complexity.

        When ``fan_out`` is False (default), behave closer to arXiv.org search:
        prioritize the user's literal ``original`` query and cap how many LLM
        variants are used so one expansion path cannot dominate retrieval.

        Broad queries  → max 1-2 variants (original + maybe one alias)
        Moderate       → max 2-4 variants
        Narrow         → max 3-5 variants
        """
        orig = (original or (expanded_queries[0] if expanded_queries else "") or "").strip()
        canonical = (bundles.get("broad", [])[:1] or [topic_profile.get("topic", "")] or [])[:1]
        seed = orig or (str(canonical[0]).strip() if canonical and canonical[0] else "")

        # Complexity-based caps
        if complexity == "broad":
            max_variants = 2
        elif complexity == "moderate":
            max_variants = 4
        else:  # narrow
            max_variants = 5

        if not fan_out:
            if complexity == "broad":
                max_variants = 1
            elif complexity == "moderate":
                max_variants = min(max_variants, 2)
            else:
                max_variants = min(max_variants, 3)

        candidates: List[str] = []
        if seed:
            candidates.append(seed)
        for q in expanded_queries:
            q = (q or "").strip()
            if not q or q.lower() in {c.lower() for c in candidates}:
                continue
            candidates.append(q)
            if len(candidates) >= max_variants:
                break

        # Deduplicate by normalized form
        seen_norm = set()
        stage1: List[str] = []
        for q in candidates:
            nq = self._normalise_title(q)
            if nq and nq not in seen_norm:
                seen_norm.add(nq)
                stage1.append(q)

        return stage1[:max_variants]

    # ------------------------------------------------------------------
    # Progressive fetch (demand-gated)
    # ------------------------------------------------------------------

    def _progressive_extend_pool(
        self,
        pool: List[Paper],
        metrics: dict,
        source_health: dict,
        primary_query: str,
        arxiv_throttled: bool,
        broad_per_source_cap: Optional[int] = None,
        arxiv_sort_by: str = "relevance",
    ) -> List[Paper]:
        """Add arXiv pages + OpenAlex cursor pages until configurable caps."""
        if not primary_query:
            return pool

        soft = max(1, int(getattr(settings, "RETRIEVAL_SOFT_CAP", 5000) or 5000))
        room = max(0, soft - len(pool))
        if room <= 0:
            return pool

        # When arXiv is throttled, skip arXiv progressive entirely
        ax_cap = 0 if arxiv_throttled else min(
            int(getattr(settings, "ARXIV_PROGRESSIVE_MAX_TOTAL", 800) or 800),
            room,
        )
        oa_cap = min(
            int(getattr(settings, "OPENALEX_PROGRESSIVE_MAX_TOTAL", 3000) or 3000),
            room,
        )
        if not self._enable_openalex:
            oa_cap = 0
        if broad_per_source_cap is not None:
            cap = max(1, int(broad_per_source_cap))
            ax_cap = min(ax_cap, cap)
            oa_cap = min(oa_cap, cap)

        def fetch_arxiv():
            if ax_cap <= 0 or self._provider_in_cooldown("arxiv"):
                return []
            try:
                out = self.arxiv.search_papers_up_to(
                    primary_query,
                    per_page=max(1, int(getattr(settings, "ARXIV_PROGRESSIVE_PAGE_SIZE", 50) or 50)),
                    max_total=ax_cap,
                    max_pages=max(1, int(getattr(settings, "ARXIV_PROGRESSIVE_MAX_PAGES", 40) or 40)),
                    sort_by=arxiv_sort_by,
                )
                for p in out:
                    object.__setattr__(p, "retrieval_path", "arxiv")
                return out
            except Exception as e:
                print(f"[HybridSearch] Progressive arXiv failed: {e}", file=sys.stderr)
                return []

        def fetch_openalex():
            if oa_cap <= 0 or self._provider_in_cooldown("openalex"):
                return []
            try:
                out = self.openalex.search_papers_paginated(
                    primary_query,
                    per_page=max(1, min(200, int(getattr(settings, "OPENALEX_PROGRESSIVE_PER_PAGE", 200) or 200))),
                    max_total=oa_cap,
                    max_pages=max(1, int(getattr(settings, "OPENALEX_PROGRESSIVE_MAX_PAGES", 40) or 40)),
                )
                for p in out:
                    if not getattr(p, "retrieval_path", ""):
                        object.__setattr__(p, "retrieval_path", "openalex_live")
                return out
            except Exception as e:
                print(f"[HybridSearch] Progressive OpenAlex failed: {e}", file=sys.stderr)
                return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as prog_executor:
            fut_ax = prog_executor.submit(fetch_arxiv)
            fut_oa = (
                prog_executor.submit(fetch_openalex)
                if self._enable_openalex and oa_cap > 0
                else None
            )

            try:
                extra_ax = fut_ax.result(timeout=120)
            except Exception as e:
                print(f"[HybridSearch] Progressive arXiv timed out: {e}", file=sys.stderr)
                extra_ax = []

            extra_oa: List[Paper] = []
            if fut_oa is not None:
                try:
                    extra_oa = fut_oa.result(timeout=120)
                except Exception as e:
                    print(f"[HybridSearch] Progressive OpenAlex timed out: {e}", file=sys.stderr)
                    extra_oa = []

        if extra_ax:
            pool = pool + extra_ax
            source_health["arxiv"] = True
            metrics["arxiv"]["returned"] = metrics["arxiv"].get("returned", 0) + len(extra_ax)
            metrics["arxiv"]["progressive_added"] = len(extra_ax)

        if extra_oa:
            pool = pool + extra_oa
            source_health["openalex"] = True
            metrics["openalex"]["returned"] = metrics["openalex"].get("returned", 0) + len(extra_oa)
            metrics["openalex"]["progressive_added"] = len(extra_oa)

        print(
            f"[HybridSearch] After progressive fetch: raw pool size {len(pool)} (soft cap {soft})",
            file=sys.stderr,
        )
        return pool

    def _ensure_source_spread(self, filtered: List[Paper], unique: List[Paper], min_sources: int = 2) -> List[Paper]:
        sources = {s.strip() for p in filtered for s in p.source.split(",") if s.strip()}
        if len(sources) >= min_sources:
            return filtered
        missing_candidates = [
            p for p in unique
            if any(s.strip() not in sources for s in p.source.split(","))
        ]
        missing_candidates.sort(
            key=lambda p: (
                getattr(p, "topic_relevance_score", 0.0),
                getattr(p, "semantic_score", 0.0),
            ),
            reverse=True,
        )
        for cand in missing_candidates:
            if cand not in filtered:
                filtered.append(cand)
                sources = {s.strip() for p in filtered for s in p.source.split(",") if s.strip()}
                if len(sources) >= min_sources:
                    break
        return filtered

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _inject_missing_landmarks(
        self,
        pool: List[Paper],
        queries: List[str],
        per_source: int,
        arxiv_fetch: Callable[..., List],
        enable_openalex: bool,
    ) -> List[Paper]:
        """
        For each expanded query that looks like a paper title (≥ 4 words) and
        has no substring match in the current pool, fire targeted arXiv +
        (optionally) OpenAlex fetches to guarantee it appears in the candidate set.
        """
        if not pool:
            return pool

        norm_pool_titles = {self._normalise_title(p.title) for p in pool}

        injected: List[Paper] = []
        for q in (queries or [])[:2]:
            words = q.split()
            if len(words) < 4:
                continue

            norm_q = self._normalise_title(q)
            already_present = any(norm_q in t for t in norm_pool_titles)
            if already_present:
                continue

            print(
                f"[HybridSearch] Landmark missing from pool: {q!r} — injecting …",
                file=sys.stderr,
            )
            fetches: List[Callable[..., List[Paper]]] = [lambda q=q: arxiv_fetch(q, 1, 3)]
            if enable_openalex:
                fetches.append(lambda q=q: self.openalex.search_papers(q, 3))
            for fetch in fetches:
                try:
                    results = fetch()
                    injected.extend(results)
                    for p in results:
                        norm_pool_titles.add(self._normalise_title(p.title))
                except Exception as e:
                    print(f"[HybridSearch] Injection fetch failed: {e}", file=sys.stderr)

        if injected:
            print(
                f"[HybridSearch] Injected {len(injected)} landmark papers",
                file=sys.stderr,
            )
            pool = pool + injected

        return pool

    def _inject_local_landmarks(self, pool: List[Paper], topic_profile: dict) -> List[Paper]:
        topic = topic_profile.get("topic", "")
        local = fallback_landmarks_for_topic(topic)
        if not local:
            return pool
        existing = {self._normalise_title(p.title) for p in pool}
        for p in local:
            if self._normalise_title(p.title) not in existing:
                object.__setattr__(p, "retrieval_path", "local_landmark")
                pool.append(p)
        return pool

    @staticmethod
    def _normalise_title(title: str) -> str:
        """Lowercase, strip punctuation — used as dedup key."""
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", title.lower())).strip()

    @staticmethod
    def _cap_raw_pool_per_source_broad(pool: List[Paper], cap: int = 50) -> List[Paper]:
        """
        Keep insertion order; retain at most ``cap`` papers per API source (arxiv,
        openalex). Other sources (e.g. local_landmark) pass through uncapped.
        """
        counts: Dict[str, int] = {"arxiv": 0, "openalex": 0}
        out: List[Paper] = []
        for p in pool:
            srcs = [s.strip().lower() for s in (p.source or "").split(",") if s.strip()]
            matched = False
            for name in ("arxiv", "openalex"):
                if name in srcs:
                    matched = True
                    if counts[name] < cap:
                        counts[name] += 1
                        out.append(p)
                    break
            if not matched:
                out.append(p)
        return out

    @staticmethod
    def _apply_source_quality_adjustment(paper: Paper, source_health: dict) -> None:
        """
        Adjust semantic score based on source quality and degraded-source scenarios.
        """
        sources = [s.strip() for s in paper.source.split(",") if s.strip()]
        if not sources:
            return

        if len(sources) > 1:
            object.__setattr__(paper, "semantic_score", min(1.0, paper.semantic_score + 0.05))

        failed_sources = [s for s in sources if not source_health.get(s, True)]
        if failed_sources and len(failed_sources) == len(sources):
            object.__setattr__(paper, "semantic_score", max(0.0, paper.semantic_score - 0.04))

    def _provider_in_cooldown(self, provider: str) -> bool:
        try:
            if provider == "arxiv" and hasattr(self.arxiv, "_is_in_cooldown"):
                return bool(self.arxiv._is_in_cooldown())
            if provider == "openalex" and hasattr(self.openalex, "_is_in_cooldown"):
                return bool(self.openalex._is_in_cooldown())
        except Exception:
            return False
        return False

    def _topic_relevance_score(
        self,
        paper: Paper,
        topic_profile: dict,
        expanded_queries: List[str],
        keywords: List[str],
    ) -> tuple[float, List[str]]:
        """
        Metadata-aware topic classifier.
        Uses title + abstract + tags + venue + aliases/landmarks to decide if
        the paper is truly about the searched topic.
        No hardcoded canonical maps — all normalization is dynamic.
        """
        title = (paper.title or "").lower()
        abstract = (paper.abstract or "").lower()
        tags_text = " ".join(paper.topic_tags or []).lower()
        venue = (getattr(paper, "venue", "") or "").lower()
        text = f"{title} {abstract} {tags_text} {venue}"

        aliases = [a.lower() for a in topic_profile.get("aliases", [])]
        acronym = (topic_profile.get("acronym", "") or "").lower()
        landmarks = [l.lower() for l in topic_profile.get("landmarks", [])]
        subtopics = [s.lower() for s in topic_profile.get("subtopics", [])]
        exclusions = [e.lower() for e in topic_profile.get("exclusions", [])]
        ontology_terms: List[str] = []
        for key in (
            "parent_topic",
            "child_topics",
            "datasets",
            "benchmarks",
            "methods",
            "model_families",
        ):
            val = topic_profile.get(key)
            if isinstance(val, str) and val.strip():
                ontology_terms.append(val.lower())
            elif isinstance(val, (list, tuple)):
                ontology_terms.extend(str(x).lower() for x in val if str(x).strip())

        title_signal = 0.0
        abstract_signal = 0.0
        tag_signal = 0.0
        meta_signal = 0.0
        landmark_signal = 0.0
        acronym_signal = 0.0
        inferred_tags: List[str] = []

        for alias in aliases:
            if alias and alias in title:
                title_signal = max(title_signal, 1.0)
                inferred_tags.append(alias)
            if alias and alias in abstract:
                abstract_signal = max(abstract_signal, 1.0)
            if alias and alias in tags_text:
                tag_signal = max(tag_signal, 1.0)

        title_n = normalize_landmark_title(paper.title or "")
        for q in expanded_queries:
            nq = (q or "").lower()
            if not nq:
                continue
            qn = normalize_landmark_title(q)
            st = 0.0
            if len(qn.split()) >= 4:
                st = landmark_phrase_anchor_strength(title_n, qn)
                if st >= 0.95:
                    landmark_signal = max(landmark_signal, 1.0)
                    title_signal = max(title_signal, 0.95)
                    if "landmark-match" not in inferred_tags:
                        inferred_tags.append("landmark-match")
                elif st >= 0.4:
                    landmark_signal = max(landmark_signal, 0.52)
                    title_signal = max(title_signal, 0.62)
                    inferred_tags.append("landmark-derivative")
                elif st > 0:
                    landmark_signal = max(landmark_signal, 0.28)
                    title_signal = max(title_signal, 0.48)
                    inferred_tags.append("landmark-derivative")
            if len(nq.split()) >= 3 and nq in abstract and st < 0.42:
                landmark_signal = max(landmark_signal, 0.38)

        for kw in keywords + subtopics + ontology_terms:
            k = (kw or "").lower()
            if not k:
                continue
            if len(k.split()) >= 2 and k in text:
                abstract_signal = max(abstract_signal, 1.0)
                inferred_tags.append(k)
            elif k in tags_text:
                tag_signal = max(tag_signal, 1.0)
                inferred_tags.append(k)
            elif len(k) >= 3 and k in title:
                title_signal = max(title_signal, 0.55)
                inferred_tags.append(k)

        if acronym and re.search(rf"\b{re.escape(acronym)}\b", text):
            acronym_signal = 1.0
            inferred_tags.append(acronym)

        # Venue/category hints
        if any(v in venue for v in ("cs.cl", "cs.lg", "cs.cv", "machine learning", "neurips", "iclr", "icml")):
            meta_signal = 0.4

        penalty = 0.0
        if exclusions:
            if any(ex in title for ex in exclusions):
                penalty += 0.15
            elif any(ex in abstract for ex in exclusions):
                penalty += 0.10
        # Generic AI papers without topic evidence are strongly down-scored
        weak_topic_evidence = (title_signal + abstract_signal + tag_signal + landmark_signal + acronym_signal) < 0.5
        if weak_topic_evidence and any(g in text for g in ("deep learning", "neural network", "artificial intelligence")):
            penalty += 0.22

        score = (
            0.30 * title_signal +
            0.30 * abstract_signal +
            0.20 * tag_signal +
            0.05 * meta_signal +
            0.10 * landmark_signal +
            0.05 * acronym_signal -
            penalty
        )
        score = max(0.0, score)

        # Dynamic tag normalization — no hardcoded canonical_map
        # Deduplicate and clean inferred tags
        normalized_tags = []
        for t in inferred_tags:
            label = t.strip()
            if label and label not in normalized_tags:
                normalized_tags.append(label)
        return round(score, 4), normalized_tags[:8]

    def _relevance_score(
        self,
        paper: Paper,
        keywords: List[str],
        expanded_queries: List[str],
        original: str,
    ) -> float:
        """
        Multi-tier relevance score (clamped to [0, 1]).

        Title-based tiers are corroborated by abstract/tag evidence.
        Papers that only match on title (parody/derivative) are penalised.

        Tier 0 (≤0.70): Landmark paper title in TITLE — requires abstract
                        corroboration or halved.
        Tier 1 (≤0.60): Any expanded query in TITLE — requires abstract
                        corroboration or halved.
        Tier 2 (≤0.40): Query phrase in ABSTRACT / TAGS.
        Tier 3 (≤0.35): Keyword token overlap in TITLE.
        Tier 4 (≤0.30): Keyword token overlap in ABSTRACT + TAGS.
        Bonus  (≤0.10): Multi-word keyword phrase verbatim in title.
        Title-only penalty: -0.15 when title matches but body has no evidence.
        """
        if not keywords and not expanded_queries:
            return 0.0

        title_lower = paper.title.lower()
        title_norm = normalize_landmark_title(paper.title or "")
        abstract_lower = paper.abstract.lower()
        tags_text = " ".join(paper.topic_tags).lower()
        full_text      = abstract_lower + " " + tags_text

        score = 0.0
        title_matched = False  # Track if any title-based tier fired

        # ── Pre-compute abstract keyword evidence ────────────────────────────
        # Count how many keywords appear in abstract+tags (used for corroboration)
        abstract_kw_hits = 0
        for kw in keywords:
            if kw.lower() in full_text:
                abstract_kw_hits += 1

        has_abstract_evidence = abstract_kw_hits >= 2

        # ── Tier 0: landmark paper title in TITLE (first expanded query) ─────
        canonical = expanded_queries[0] if expanded_queries else ""
        canonical_norm = normalize_landmark_title(canonical) if canonical else ""
        if canonical and len(canonical_norm.split()) >= 4:
            c_st = landmark_phrase_anchor_strength(title_norm, canonical_norm)
            if c_st >= 0.95:
                score += 0.70 if has_abstract_evidence else 0.35
                title_matched = True
            elif c_st >= 0.4:
                score += 0.48 if has_abstract_evidence else 0.24
                title_matched = True
            elif c_st > 0:
                score += 0.30 if has_abstract_evidence else 0.15
                title_matched = True

        # ── Tier 1: any expanded query phrase in TITLE ────────────────────────
        if score < 0.60:
            best_t1 = 0.0
            for q in expanded_queries:
                q_norm = normalize_landmark_title(q) if q else ""
                if not q or len(q_norm.split()) < 4:
                    continue
                st = landmark_phrase_anchor_strength(title_norm, q_norm)
                if st >= 0.95:
                    best_t1 = max(best_t1, 0.60 if has_abstract_evidence else 0.30)
                elif st >= 0.4:
                    best_t1 = max(best_t1, 0.42 if has_abstract_evidence else 0.21)
                elif st > 0:
                    best_t1 = max(best_t1, 0.28 if has_abstract_evidence else 0.14)
            if best_t1 > 0:
                score += best_t1
                title_matched = True

        # ── Tier 2: exact query phrase in ABSTRACT / TAGS ─────────────────────
        if score < 0.50:
            for q in expanded_queries:
                if q and q.lower() in full_text:
                    score += 0.40
                    break

        # ── Build clean token set (without stopwords) ─────────────────────────
        all_tokens: set = set()
        for kw in keywords:
            all_tokens.update(kw.lower().split())
            acronym_str = "".join(t[0] for t in kw.lower().split() if t and t not in self._STOPWORDS)
            if len(acronym_str) >= 3:
                all_tokens.add(acronym_str)
        all_tokens -= self._STOPWORDS

        if all_tokens:
            # ── Tier 3: token overlap in TITLE (reduced from 0.50) ───────────
            title_hits = sum(1 for w in all_tokens if w in title_lower)
            score += (title_hits / len(all_tokens)) * 0.35

            # ── Tier 4: token overlap in ABSTRACT + TAGS ─────────────────────
            full_hits = sum(1 for w in all_tokens if w in full_text)
            score += (full_hits / len(all_tokens)) * 0.30

        # ── Bonus: multi-word keyword phrase verbatim in title ────────────────
        for kw in keywords:
            if len(kw.split()) > 1 and kw.lower() in title_lower:
                score += 0.10
                break

        # ── Title-only penalty ────────────────────────────────────────────────
        # If the paper title matched but the abstract and tags contain zero
        # keyword evidence, this is likely a parody or derivative paper.
        if title_matched and abstract_kw_hits == 0:
            score -= 0.15

        return round(max(0.0, score), 4)
