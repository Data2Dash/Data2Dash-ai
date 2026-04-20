"""
Ranking Service
Produces a composite score for each paper and returns them sorted descending.

The composite layer uses :class:`LearnableRankingWeights` (see ``app.core.ranking_weights``)
so weights can be tuned via ``RANKING_WEIGHTS_JSON`` without code edits.

Optional **hybrid BM25 + embedding** reranking (``ENABLE_HYBRID_RERANK``) runs on the
top-N candidate pool when enabled. Otherwise **TF–IDF** reranking (``ENABLE_SEMANTIC_RERANK``)
may blend a lightweight lexical signal. Retrieval from arXiv/OpenAlex is unchanged.

Each ranked paper receives ``ranking_reasons`` (dict) for explainability.
"""
from __future__ import annotations

import math
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.ranking_weights import LearnableRankingWeights, load_learnable_weights
from app.services.landmark_title_match import landmark_phrase_anchor_strength, normalize_landmark_title
from app.services.query_intent import QueryIntent, classify_query_intent
from app.services.semantic_rerank import build_document_text, tfidf_cosine_similarities


def _norm_title(title: str) -> str:
    """Normalise title to its SEMINAL_TITLES key format."""
    return re.sub(r"[^a-z0-9 ]", "", (title or "").lower()).strip()


def _token_set(text: str) -> set[str]:
    return {t for t in _norm_title(text).split() if len(t) >= 2}


def _exclusion_soft_penalty_factors(
    papers: list,
    exclusion_phrases: List[str],
) -> Optional[List[Tuple[float, float]]]:
    """
    For each paper, return (score multiplier, max_cosine_to_any_exclusion) for
    ``abstract`` vs exclusion text. Embeddings are L2-normalized (dot = cosine).
    """
    phrases = [str(p).strip() for p in exclusion_phrases if str(p).strip()]
    if not phrases or not papers:
        return None
    try:
        import numpy as np

        from app.services.embedding_service import get_embedding_service

        embedder = get_embedding_service(
            settings.EMBEDDING_MODEL_NAME,
            int(getattr(settings, "HYBRID_EMBEDDING_CACHE_MAX", 2048) or 2048),
        )
        ex_emb = embedder.encode(phrases)
        abs_texts = [((getattr(p, "abstract", None) or "") or "")[:12000] for p in papers]
        ab_emb = embedder.encode(abs_texts)
        if ex_emb.size == 0 or ab_emb.size == 0:
            return None
        sims = ab_emb @ ex_emb.T
        max_s = sims.max(axis=1)
        factors = 1.0 - 0.15 * max_s
        factors = np.clip(factors.astype(np.float64), 0.0, 1.0)
        out: List[Tuple[float, float]] = []
        for i in range(len(papers)):
            out.append((float(factors[i]), float(max_s[i])))
        return out
    except Exception:
        return None


class RankingService:
    """
    Composite ranking with configurable weights, query-intent modifiers,
    age-adjusted citations, optional hybrid BM25+embedding or TF–IDF rerank, and
    per-paper explanations.
    """

    def __init__(self, weights: Optional[LearnableRankingWeights] = None):
        self._fixed_weights = weights

    def _weights(self) -> LearnableRankingWeights:
        return self._fixed_weights or load_learnable_weights()

    @staticmethod
    def _source_confidence(source: str) -> float:
        if not source:
            return 0.5
        sources = [s.strip() for s in source.split(",") if s.strip()]
        if not sources:
            return 0.5
        scores = [settings.SOURCE_CONFIDENCE.get(s, 0.5) for s in sources]
        return max(scores)

    @staticmethod
    def _seminal_titles():
        """SEMINAL_TITLES no longer exported from query_expander — return empty set."""
        return frozenset()

    @staticmethod
    def _landmark_registry_boost(norm_title: str) -> Tuple[float, bool]:
        """Return (score boost 0..0.12, survey_flag) from structured local registry."""
        try:
            from app.services import local_landmarks as ll

            return ll.registry_boost_and_survey_flag(norm_title)
        except Exception:
            return 0.0, False

    def rank_papers(
        self,
        papers,
        query_keywords: list = None,
        user_query: str = "",
        query_intent: Optional[QueryIntent] = None,
        *,
        exclusion_phrases: Optional[List[str]] = None,
    ):
        w = self._weights()
        seminal = self._seminal_titles()
        intent = query_intent or classify_query_intent(
            user_query or "",
            query_keywords or [],
        )

        preliminary: List[Tuple[float, Dict[str, Any], Any]] = []
        for p in papers:
            score, reasons = self._score_paper_with_breakdown(
                p, query_keywords, seminal, intent, w, user_query
            )
            preliminary.append((score, reasons, p))

        preliminary.sort(key=lambda x: x[0], reverse=True)

        ex_list = [str(e).strip() for e in (exclusion_phrases or []) if str(e).strip()]
        if ex_list and preliminary:
            pen = _exclusion_soft_penalty_factors([x[2] for x in preliminary], ex_list)
            if pen and len(pen) == len(preliminary):
                reranked_ex: List[Tuple[float, Dict[str, Any], Any]] = []
                for triple, (fac, mx_sim) in zip(preliminary, pen):
                    score, reasons, paper = triple
                    new_score = max(0.0, score * fac)
                    nr = {
                        **reasons,
                        "exclusion_penalty_factor": round(fac, 4),
                        "max_exclusion_similarity": round(mx_sim, 4),
                    }
                    reranked_ex.append((new_score, nr, paper))
                preliminary = reranked_ex
                preliminary.sort(key=lambda x: x[0], reverse=True)
                print(
                    f"[Ranking] Exclusion embedding penalty active: n_phrases={len(ex_list)}",
                    file=sys.stderr,
                )

        qtext = (user_query or "").strip()
        hybrid_done = False

        # ── Hybrid BM25 + embedding rerank (preferred when enabled) ─────────
        if getattr(settings, "ENABLE_HYBRID_RERANK", False) and preliminary and qtext:
            h_blend = float(getattr(settings, "HYBRID_RERANK_COMPOSITE_BLEND", 0.28) or 0.0)
            pool = int(getattr(settings, "HYBRID_RERANK_TOP_N", 40) or 40)
            pool = min(max(pool, 1), len(preliminary))
            if h_blend > 0:
                try:
                    from app.services.embedding_service import get_embedding_service
                    from app.services.hybrid_reranker import apply_hybrid_rerank_to_papers

                    embedder = get_embedding_service(
                        settings.EMBEDDING_MODEL_NAME,
                        settings.HYBRID_EMBEDDING_CACHE_MAX,
                    )
                    top_papers = [triple[2] for triple in preliminary[:pool]]
                    meta = apply_hybrid_rerank_to_papers(
                        qtext,
                        top_papers,
                        settings.HYBRID_RERANK_BM25_WEIGHT,
                        settings.HYBRID_RERANK_EMBEDDING_WEIGHT,
                        embedder,
                    )
                    if meta.get("hybrid_rerank_status") == "ok":
                        adjusted: List[Tuple[float, Dict[str, Any], Any]] = []
                        for i, triple in enumerate(preliminary):
                            base_score, reasons, paper = triple
                            if i < pool:
                                h = float(getattr(paper, "hybrid_relevance_score", 0.0) or 0.0)
                                final = (1.0 - h_blend) * base_score + h_blend * h
                                reasons = {
                                    **reasons,
                                    "hybrid_relevance": round(h, 4),
                                    "bm25_norm": getattr(paper, "bm25_score", 0.0),
                                    "embedding_cos": getattr(paper, "embedding_score", 0.0),
                                    "hybrid_rerank_meta": meta,
                                    "hybrid_composite_blend": h_blend,
                                }
                            else:
                                final = base_score
                                reasons = {
                                    **reasons,
                                    "hybrid_relevance": 0.0,
                                    "hybrid_composite_blend": 0.0,
                                }
                            reasons["composite_pre_hybrid"] = round(base_score, 4)
                            reasons["composite"] = round(final, 4)
                            adjusted.append((final, reasons, paper))
                        adjusted.sort(key=lambda x: x[0], reverse=True)
                        preliminary = adjusted
                        hybrid_done = True
                except Exception:
                    for triple in preliminary[:pool]:
                        object.__setattr__(triple[2], "bm25_score", 0.0)
                        object.__setattr__(triple[2], "embedding_score", 0.0)
                        object.__setattr__(triple[2], "hybrid_relevance_score", 0.0)

        # ── Legacy TF–IDF rerank (if hybrid not applied) ─────────────────────
        if (
            not hybrid_done
            and getattr(settings, "ENABLE_SEMANTIC_RERANK", True)
            and preliminary
            and qtext
        ):
            blend = float(getattr(settings, "RERANK_BLEND", 0.25) or 0.0)
            pool = int(getattr(settings, "RERANK_POOL_SIZE", 200) or 200)
            if blend > 0 and pool > 0:
                pool = min(pool, len(preliminary))
                top_slice = preliminary[:pool]
                docs = [
                    build_document_text(x[2].title, x[2].abstract)
                    for x in top_slice
                ]
                sims = tfidf_cosine_similarities(qtext, docs)
                adjusted2: List[Tuple[float, Dict[str, Any], Any]] = []
                for i, triple in enumerate(preliminary):
                    base_score, reasons, paper = triple
                    if i < len(sims):
                        sim = sims[i]
                    else:
                        sim = 0.0
                    if i < pool:
                        final = (1.0 - blend) * base_score + blend * sim
                        reasons = {**reasons, "semantic_rerank_sim": round(sim, 4), "rerank_blend": blend}
                    else:
                        final = base_score
                        reasons = {**reasons, "semantic_rerank_sim": 0.0, "rerank_blend": 0.0}
                    reasons["composite_pre_rerank"] = round(base_score, 4)
                    reasons["composite"] = round(final, 4)
                    adjusted2.append((final, reasons, paper))
                adjusted2.sort(key=lambda x: x[0], reverse=True)
                preliminary = adjusted2

        out = []
        for _, reasons, paper in preliminary:
            object.__setattr__(paper, "ranking_reasons", reasons)
            out.append(paper)
        return out

    def score_paper(
        self,
        paper,
        query_keywords: list | None = None,
        seminal_titles=None,
        query_intent: Optional[QueryIntent] = None,
        user_query: str = "",
    ) -> float:
        """
        Score a single paper and attach `ranking_reasons`.
        Kept for backwards compatibility with unit tests and callers.
        """
        if seminal_titles is None:
            seminal_titles = self._seminal_titles()
        intent = query_intent or classify_query_intent(
            user_query or "",
            query_keywords or [],
        )
        score, reasons = self._score_paper_with_breakdown(
            paper,
            query_keywords,
            seminal_titles,
            intent,
            self._weights(),
            user_query,
        )
        object.__setattr__(paper, "ranking_reasons", reasons)
        return float(score)

    def _score_paper_with_breakdown(
        self,
        paper,
        query_keywords: list | None,
        seminal_titles,
        intent: QueryIntent,
        w: LearnableRankingWeights,
        user_query: str = "",
    ) -> Tuple[float, Dict[str, Any]]:
        recency_score = self._recency_score(paper.published_date)
        citation_blended, citation_raw, cpy = self._citation_score_blended(
            paper.citations, paper.published_date, w.citation_age_blend
        )
        source_score = self._source_confidence(paper.source)
        semantic_score = getattr(paper, "semantic_score", 0.0)
        topic_relevance = getattr(paper, "topic_relevance_score", semantic_score)

        title_lower = paper.title.lower() if paper.title else ""
        norm_title = _norm_title(paper.title) if paper.title else ""

        survey_penalty = 0.0
        if any(wd in title_lower for wd in ("survey", "review", "overview", "tutorial")):
            requested = bool(intent.wants_survey)
            if not requested and query_keywords:
                for kw in query_keywords:
                    if any(wd in kw.lower() for wd in ("survey", "review", "overview", "tutorial")):
                        requested = True
                        break
            if not requested:
                survey_penalty = 0.25

        reg_boost, reg_survey = self._landmark_registry_boost(norm_title)
        if reg_survey and not intent.wants_survey:
            survey_penalty = max(survey_penalty, 0.08)

        title_boost = 0.0
        if query_keywords:
            for kw in query_keywords:
                kw_lower = kw.lower().strip()
                if not kw_lower:
                    continue

                norm_kw = re.sub(r"[^a-z0-9 ]", "", kw_lower).strip()
                acronym = "".join(t[0] for t in norm_kw.split() if t)

                if kw_lower == title_lower or norm_kw == norm_title:
                    title_boost = max(title_boost, 0.45)
                elif len(kw_lower) >= 4 and (
                    title_lower.startswith(kw_lower + ":")
                    or title_lower.startswith(kw_lower + " -")
                    or title_lower.startswith(kw_lower + " —")
                ):
                    title_boost = max(title_boost, 0.35)
                elif len(kw_lower.split()) >= 3 and kw_lower in title_lower:
                    if len(title_lower.split()) > len(kw_lower.split()) + 3:
                        title_boost = max(title_boost, 0.10) # Parody penalty
                    else:
                        title_boost = max(title_boost, 0.30)
                elif len(kw_lower) >= 3 and kw_lower in title_lower:
                    if len(kw_lower) <= 5:
                        if f" {kw_lower} " in f" {title_lower} ":
                            title_boost = max(title_boost, 0.18)
                    else:
                        title_boost = max(title_boost, 0.15)
                elif len(acronym) >= 2 and re.search(rf"\b{re.escape(acronym)}\b", title_lower):
                    title_boost = max(title_boost, 0.12)
                elif len(norm_kw.split()) >= 4:
                    kw_tokens = _token_set(norm_kw)
                    title_tokens = _token_set(norm_title)
                    if kw_tokens and title_tokens:
                        overlap = len(kw_tokens & title_tokens) / max(len(kw_tokens), 1)
                        if overlap >= 0.85:
                            # Token overlap alone is NOT sufficient — capped low
                            # to prevent parody/derivative titles from dominating
                            title_boost = max(title_boost, 0.30)
                        elif overlap >= 0.65:
                            title_boost = max(title_boost, 0.20)

        seminal_boost = 0.35 if norm_title in seminal_titles else 0.0
        seminal_boost = min(0.45, seminal_boost + reg_boost)
        landmark_relevance = 0.0
        nt_full = normalize_landmark_title(paper.title or "")
        if query_keywords:
            for kw in query_keywords:
                nkw = normalize_landmark_title(kw)
                if len(nkw.split()) < 4:
                    continue
                st = landmark_phrase_anchor_strength(nt_full, nkw)
                if st >= 0.95:
                    landmark_relevance = max(landmark_relevance, 5.0)
                elif st >= 0.4:
                    landmark_relevance = max(landmark_relevance, 1.15)
                elif st > 0:
                    landmark_relevance = max(landmark_relevance, 0.45)

        concept_boost = 0.0
        query_kw_lower = ""
        user_q_lower = user_query.lower().strip()
        if len(user_q_lower) > 3:
            query_kw_lower = user_q_lower

        if query_kw_lower:
            # Concept vs Tool heuristic
            title_words = title_lower.split()
            if title_words and query_kw_lower in title_words[0]:
                # Strong signal: concept is the primary architectural subject
                concept_boost += 0.25
            
            # Application/Tool penalty
            if f"using {query_kw_lower}" in title_lower or f"with {query_kw_lower}" in title_lower or f"based on {query_kw_lower}" in title_lower:
                concept_boost -= 0.15
            elif f"for {query_kw_lower}" in title_lower:
                concept_boost -= 0.15

        # ── Gate ALL title-derived boosts by topic_relevance ──────────────
        # Title similarity should only help ranking when there is genuine
        # topical alignment.  Use topic_relevance as a continuous multiplier
        # so papers with strong topic evidence keep their boost, while
        # parody/derivative titles with weak evidence are heavily damped.
        #
        # topic_relevance  | effective title_boost multiplier
        # ≥ 0.5            | 1.0 (full boost)
        # 0.3 – 0.5        | 0.5 – 1.0 (linearly interpolated)
        # < 0.3            | 0.3 (floor — very damped)
        if topic_relevance >= 0.5:
            title_gate = 1.0
        elif topic_relevance >= 0.3:
            title_gate = 0.5 + (topic_relevance - 0.3) / 0.2 * 0.5
        else:
            title_gate = 0.3

        title_boost *= title_gate
        seminal_boost *= title_gate
        landmark_relevance *= title_gate

        # Citation credibility: genuinely important papers accumulate
        # citations.  Parody or trivially-derivative papers rarely do.
        citations = getattr(paper, "citations", 0) or 0
        if citations < 10 and title_boost > 0.1:
            title_boost *= 0.5

        intent_recency = recency_score
        if intent.wants_recent:
            intent_recency = min(1.0, intent_recency * 1.18)
        if intent.wants_seminal:
            intent_recency = min(1.0, intent_recency * 1.22)

        primary_relevance = (
            w.blend_topic_in_primary * topic_relevance
            + (1.0 - w.blend_topic_in_primary) * semantic_score
        )
        if intent.wants_seminal:
            primary_relevance = min(1.0, primary_relevance + 0.04)

        base_score = (
            w.w_primary * primary_relevance
            + w.w_citation * citation_blended
            + w.w_recency * intent_recency
            + w.w_source * source_score
        )

        intent_misc = 0.0
        ab = (paper.abstract or "").lower()
        if intent.wants_implementation and any(
            x in ab for x in ("github", "pytorch", "tensorflow", "implementation", "code")
        ):
            intent_misc += 0.02
        if intent.wants_benchmark and any(
            x in title_lower for x in ("benchmark", "dataset", "leaderboard")
        ):
            intent_misc += 0.025
        if intent.wants_theory and any(x in ab for x in ("theorem", "proof", "bound", "convergence")):
            intent_misc += 0.02
        if intent.wants_application and any(
            x in ab for x in ("real-world", "deployment", "clinical", "production")
        ):
            intent_misc += 0.02

        raw = (
            base_score
            + title_boost
            + seminal_boost
            + landmark_relevance
            + concept_boost
            + intent_misc
            - survey_penalty
        )
        final = max(0.0, raw)

        reasons: Dict[str, Any] = {
            "composite": round(final, 4),
            "primary_relevance": round(primary_relevance, 4),
            "topic_relevance": round(topic_relevance, 4),
            "semantic_score": round(semantic_score, 4),
            "citation_component": round(citation_blended, 4),
            "citation_log_raw": round(citation_raw, 4),
            "citations_per_year": round(cpy, 4),
            "recency_component": round(intent_recency, 4),
            "source_component": round(source_score, 4),
            "title_boost": round(title_boost, 4),
            "seminal_boost": round(seminal_boost, 4),
            "landmark_relevance": round(landmark_relevance, 4),
            "concept_boost": round(concept_boost, 4),
            "survey_penalty": round(survey_penalty, 4),
            "intent_misc": round(intent_misc, 4),
            "weights": {
                "blend_topic_in_primary": w.blend_topic_in_primary,
                "w_primary": w.w_primary,
                "w_citation": w.w_citation,
                "w_recency": w.w_recency,
                "w_source": w.w_source,
                "citation_age_blend": w.citation_age_blend,
            },
            "intent": {
                "wants_survey": intent.wants_survey,
                "wants_seminal": intent.wants_seminal,
                "wants_recent": intent.wants_recent,
                "wants_implementation": intent.wants_implementation,
                "wants_benchmark": intent.wants_benchmark,
                "wants_theory": intent.wants_theory,
                "wants_application": intent.wants_application,
            },
        }
        return final, reasons

    @staticmethod
    def _recency_score(published_date: str) -> float:
        if not published_date:
            return 0.0
        try:
            days_old = (
                datetime.today().date() - datetime.fromisoformat(published_date).date()
            ).days
            return math.exp(-0.000379 * max(days_old, 0))
        except Exception:
            return 0.0

    @staticmethod
    def _citation_score_blended(
        citations: int,
        published_date: str,
        age_blend: float,
    ) -> Tuple[float, float, float]:
        """
        Returns (blended_score, raw_log_score, cites_per_year).

        ``cites_per_year`` uses publication year vs today; unknown year falls
        back to raw log-only via ``age_blend`` gating inside the blend.
        """
        c = max(int(citations or 0), 0)
        raw = min(math.log1p(c) / math.log1p(1000), 1.0) if c > 0 else 0.0

        cpy = 0.0
        cpy_norm = 0.0
        try:
            if published_date and len(published_date) >= 4 and published_date[:4].isdigit():
                y = int(published_date[:4])
                cy = datetime.today().year
                years = max(cy - y, 0) + 0.5
                cpy = c / years
                cpy_norm = min(math.log1p(cpy) / math.log1p(500), 1.0) if c > 0 else 0.0
        except Exception:
            cpy_norm = 0.0

        eff_blend = age_blend if (published_date and len(published_date) >= 4) else 0.0
        blended = (1.0 - eff_blend) * raw + eff_blend * cpy_norm
        return blended, raw, cpy
