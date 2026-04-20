from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.services.analytics_service import AnalyticsService
from app.services.arxiv_service import ArxivService
from app.services.hybrid_search_service import HybridSearchService
from app.services.openalex_service import OpenAlexService
from app.services.query_expander import QueryExpander
from app.services.query_intent import classify_query_intent
from app.services.ranking_service import RankingService


@dataclass(frozen=True)
class SearchRequest:
    query: str
    page: int = 1
    per_page: int = 25


class SearchAgent:
    """
    Generic orchestration layer for the academic hybrid search pipeline.

    Responsibilities:
      1) query expansion (single pass — no redundant async re-expansion)
      2) candidate retrieval (anti-gravity: start light, expand if needed)
      3) ranking
      4) pagination
      5) analytics
      6) result accounting
    """

    def __init__(
        self,
        arxiv_service: Optional[ArxivService] = None,
        openalex_service: Optional[OpenAlexService] = None,
        analytics_service: Optional[AnalyticsService] = None,
        ranking_service: Optional[RankingService] = None,
        hybrid_service: Optional[HybridSearchService] = None,
        query_expander: Optional[QueryExpander] = None,
    ) -> None:
        self.arxiv_service = arxiv_service or ArxivService()
        self.openalex_service = openalex_service or OpenAlexService()
        self.analytics_service = analytics_service or AnalyticsService()
        self.ranking_service = ranking_service or RankingService()
        self.query_expander = query_expander if query_expander is not None else QueryExpander()
        self.hybrid_service = hybrid_service or HybridSearchService(
            arxiv_service=self.arxiv_service,
            openalex_service=self.openalex_service,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, page: int = 1, per_page: int = 25) -> Dict[str, Any]:
        request = SearchRequest(
            query=(query or "").strip(),
            page=max(int(page or 1), 1),
            per_page=max(int(per_page or 25), 1),
        )
        return self._search(request)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def _search(self, request: SearchRequest) -> Dict[str, Any]:
        if not request.query:
            return self._empty_response(request)

        print(f"[SearchAgent] Expanding query: {request.query!r} (with PRF pass)", file=sys.stderr)

        # Landmarks come from an initial expansion pass (topic_profile); PRF seeds only
        # from those landmark queries — never from the Stage 1 retrieval pool.
        pre_expanded = self.query_expander.expand(request.query, prf_terms=[])
        landmarks = (pre_expanded.get("topic_profile") or {}).get("landmarks") or []
        prf_terms = self._generate_prf_terms(request.query, landmarks[:2])

        # Second expansion merges PRF terms into the LLM prompt when available.
        expanded = self.query_expander.expand(request.query, prf_terms=prf_terms)
        expanded["enable_offline_landmark_fallback"] = True

        complexity = expanded.get("query_complexity", "moderate")
        print(
            f"[SearchAgent] Expanded to {len(expanded.get('expanded_queries', []))} variants "
            f"(complexity={complexity})",
            file=sys.stderr,
        )

        max_results_cfg = max(int(getattr(settings, "MAX_RESULTS", 0) or 0), 0)
        rank_cap_applied = max_results_cfg > 0
        retrieval_soft_cap = max(int(getattr(settings, "RETRIEVAL_SOFT_CAP", 5000) or 5000), 1)

        fetch_per_source = max_results_cfg if rank_cap_applied else retrieval_soft_cap
        fetch_per_source = max(fetch_per_source, 200)

        all_papers = self.hybrid_service.search(
            expanded=expanded,
            per_source=fetch_per_source,
        )

        hybrid_accounting = self._get_hybrid_accounting(all_papers)
        pipeline = hybrid_accounting.get("pipeline", {})

        retrieved_count = pipeline.get("raw_pool_count", len(all_papers))
        deduplicated_count = pipeline.get("after_dedup_count", len(all_papers))
        filtered_count = pipeline.get("after_filter_count", len(all_papers))

        print(
            f"[SearchAgent] Pipeline | retrieved={retrieved_count} "
            f"dedup={deduplicated_count} filtered={filtered_count} "
            f"pool={len(all_papers)}",
            file=sys.stderr,
        )

        ranked = self._rank_papers(request.query, expanded, all_papers)

        if rank_cap_applied:
            ranked = ranked[:max_results_cfg]

        page_info = self._paginate(ranked, request.page, request.per_page)
        page_papers = page_info["papers"]

        accounting = self._build_accounting(
            hybrid_accounting=hybrid_accounting,
            rank_cap_applied=rank_cap_applied,
            max_results_cfg=max_results_cfg,
            retrieval_soft_cap=retrieval_soft_cap,
            retrieved_count=retrieved_count,
            deduplicated_count=deduplicated_count,
            filtered_count=filtered_count,
            ranked_count=len(ranked),
            page_result_count=len(page_papers),
            page=page_info["page"],
            per_page=page_info["per_page"],
            max_page=page_info["max_page"],
        )

        analytics = self.analytics_service.compute_summary(ranked, query=request.query)
        source_counts = self._compute_source_counts(ranked)

        print(
            f"[SearchAgent] Final | retrieved={retrieved_count} dedup={deduplicated_count} "
            f"filtered={filtered_count} ranked={len(ranked)} page_size={len(page_papers)}",
            file=sys.stderr,
        )

        return {
            "query": request.query,
            "expanded_queries": expanded.get("expanded_queries", []),
            "semantic_keywords": expanded.get("semantic_keywords", []),
            "topic_profile": expanded.get("topic_profile", {}),
            "retrieval_bundles": expanded.get("retrieval_bundles", {}),
            "total_found": len(ranked),
            "source_counts": source_counts,
            "result_accounting": accounting,
            "ranked_papers": ranked,
            "papers": page_papers,
            "analytics": analytics,
        }

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def _rank_papers(
        self,
        query: str,
        expanded: Dict[str, Any],
        all_papers: list,
    ) -> list:
        profile = expanded.get("topic_profile", {}) or {}

        query_keywords = (
            expanded.get("expanded_queries", [])
            + expanded.get("semantic_keywords", [])
            + profile.get("landmarks", [])
            + profile.get("subtopics", [])
            + profile.get("datasets", [])
            + profile.get("benchmarks", [])
            + profile.get("methods", [])
            + profile.get("model_families", [])
            + profile.get("child_topics", [])
        )

        intent = classify_query_intent(query, query_keywords)

        exclusions = profile.get("exclusions") or []

        return self.ranking_service.rank_papers(
            all_papers,
            query_keywords=query_keywords,
            user_query=query,
            query_intent=intent,
            exclusion_phrases=exclusions,
        )

    # ------------------------------------------------------------------
    # Pagination / accounting / summaries
    # ------------------------------------------------------------------

    def _paginate(self, ranked: list, page: int, per_page: int) -> Dict[str, Any]:
        safe_per_page = max(int(per_page or 25), 1)
        max_page = max(1, (len(ranked) + safe_per_page - 1) // safe_per_page)
        safe_page = min(max(int(page or 1), 1), max_page)

        start = (safe_page - 1) * safe_per_page
        papers = ranked[start : start + safe_per_page]

        return {
            "papers": papers,
            "page": safe_page,
            "per_page": safe_per_page,
            "max_page": max_page,
        }

    def _get_hybrid_accounting(self, all_papers: list) -> Dict[str, Any]:
        raw_acc = getattr(self.hybrid_service, "last_accounting", None)
        return {**raw_acc} if isinstance(raw_acc, dict) else {"pipeline": {
            "raw_pool_count": len(all_papers),
            "after_dedup_count": len(all_papers),
            "after_filter_count": len(all_papers),
        }}

    def _build_accounting(
        self,
        *,
        hybrid_accounting: Dict[str, Any],
        rank_cap_applied: bool,
        max_results_cfg: int,
        retrieval_soft_cap: int,
        retrieved_count: int,
        deduplicated_count: int,
        filtered_count: int,
        ranked_count: int,
        page_result_count: int,
        page: int,
        per_page: int,
        max_page: int,
    ) -> Dict[str, Any]:
        accounting = {**hybrid_accounting}
        accounting["rank_cap_applied"] = rank_cap_applied
        accounting["max_results_setting"] = max_results_cfg
        accounting["retrieval_soft_cap"] = retrieval_soft_cap
        accounting["retrieved_count"] = retrieved_count
        accounting["deduplicated_count"] = deduplicated_count
        accounting["filtered_count"] = filtered_count
        accounting["final_ranked_count"] = ranked_count
        accounting["final_result_count"] = ranked_count
        accounting["page_result_count"] = page_result_count
        accounting["displayed_count"] = page_result_count
        accounting["page"] = page
        accounting["per_page"] = per_page
        accounting["max_page"] = max_page
        return accounting

    def _compute_source_counts(self, ranked: list) -> Dict[str, int]:
        source_counts: Dict[str, int] = {}
        for paper in ranked:
            raw_source = getattr(paper, "source", "") or ""
            for src in str(raw_source).split(","):
                name = src.strip()
                if not name:
                    continue
                source_counts[name] = source_counts.get(name, 0) + 1
        return source_counts

    def _empty_response(self, request: SearchRequest) -> Dict[str, Any]:
        return {
            "query": request.query,
            "expanded_queries": [],
            "semantic_keywords": [],
            "topic_profile": {},
            "retrieval_bundles": {},
            "total_found": 0,
            "source_counts": {},
            "result_accounting": {
                "rank_cap_applied": False,
                "max_results_setting": 0,
                "retrieval_soft_cap": 0,
                "retrieved_count": 0,
                "deduplicated_count": 0,
                "filtered_count": 0,
                "final_ranked_count": 0,
                "final_result_count": 0,
                "page_result_count": 0,
                "displayed_count": 0,
                "page": request.page,
                "per_page": request.per_page,
                "max_page": 1,
            },
            "ranked_papers": [],
            "papers": [],
            "analytics": self.analytics_service.compute_summary([], query=request.query),
        }

    # ------------------------------------------------------------------
    # Pseudo-Relevance Feedback (PRF)
    # ------------------------------------------------------------------

    def _generate_prf_terms(self, query: str, landmark_queries: List[str]) -> list[str]:
        """
        Cheap retrieval seeded only by ``topic_profile["landmarks"][:2]`` search
        queries (paper-title-like strings). Uses OpenAlex by default; set
        ``ENABLE_OPENALEX_RETRIEVAL=false`` for arXiv-only PRF seeding. Extracts
        title/abstract terms. If ``landmark_queries`` is empty, uses the original
        user query only — never the Stage 1 hybrid pool.
        """
        seeds = [q.strip() for q in (landmark_queries or [])[:2] if q and str(q).strip()]
        if not seeds:
            seeds = [(query or "").strip()] if (query or "").strip() else []
        if not seeds:
            return []
        print(f"[PRF] Seeding from landmarks: {seeds}", file=sys.stderr)
        try:
            all_papers = []
            seen_ids: set[str] = set()
            use_openalex = bool(getattr(settings, "ENABLE_OPENALEX_RETRIEVAL", True))
            ax_sort = getattr(settings, "ARXIV_SEARCH_SORT_BY", "relevance")
            for sq in seeds:
                if use_openalex:
                    batch = self.openalex_service.search_papers(sq, per_page=10, offset=0)
                else:
                    batch = self.arxiv_service.search_papers(
                        sq, page=1, per_page=10, sort_by=ax_sort
                    )
                for p in batch:
                    if use_openalex:
                        pid = getattr(p, "openalex_work_id", None) or getattr(p, "id", "") or ""
                    else:
                        pid = getattr(p, "arxiv_id", None) or getattr(p, "id", "") or ""
                    dedupe_key = (pid or p.title or "").strip().lower()
                    if dedupe_key and dedupe_key not in seen_ids:
                        seen_ids.add(dedupe_key)
                        all_papers.append(p)
            candidates = all_papers
            if not candidates:
                return []

            corpus = " ".join([p.title + " " + p.abstract for p in candidates]).lower()
            
            import re
            from collections import Counter
            
            tokens = re.findall(r'[a-z]{4,}', corpus)
            # Common English + paper stopwords
            stopwords = {"this", "that", "with", "from", "which", "these", "paper", "propose", "method", "model", "we", "the", "and", "for", "are", "have", "been", "can", "our", "results", "based", "using", "also", "show", "used", "two", "one", "new", "such", "than", "more", "their", "they", "will", "both", "were", "well", "some"}
            
            filtered = [t for t in tokens if t not in stopwords]
            counts = Counter(filtered)

            terms = [t for t, _ in counts.most_common(15)]
            print(f"[PRF] Seed terms extracted: {terms}", file=sys.stderr)
            return terms
        except Exception as e:
            print(f"[SearchAgent] PRF pass failed: {e}", file=sys.stderr)
            return []