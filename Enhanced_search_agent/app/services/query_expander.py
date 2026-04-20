"""
QueryExpander — Generic, query-driven academic query expansion.

Design
------
**Zero hardcoded dictionaries.**  All expansion is derived from the user query
itself via structural parsing and optional LLM enrichment.

Structural parsing (instant, offline):
  • Detects acronyms dynamically (all-caps or CamelCase ≤7 chars)
  • Generates quoted exact-phrase, survey, and "deep …" variants
  • Infers parent topic from token-level keyword heuristics
  • Extracts content tokens as semantic keywords

LLM enrichment (optional, Groq):
  • Generates aliases, landmark titles, subtopics, methods, etc.
  • Merged on top of structural output; never required

Query complexity classification:
  • Short/broad queries (1–2 tokens) are flagged so downstream retrieval
    can cap fan-out and avoid latency explosion.

Public API
----------
``QueryExpander().expand(query: str) -> dict`` always returns a dict with:
    original            – the raw query string
    expanded_queries    – list[str], ≤ 8, deduplicated, original always first
    semantic_keywords   – list[str], ≤ 14, deduplicated
    topic_profile       – dict (topic, aliases, acronym, landmarks, subtopics,
                          exclusions, parent_topic, child_topics, datasets,
                          benchmarks, methods, model_families)
    retrieval_bundles   – dict (broad, landmark_titles, acronym, subtopics)
    query_complexity    – "broad" | "moderate" | "narrow"
"""
from __future__ import annotations

import json
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Parent-topic inference (lightweight keyword scan — no topic dictionaries)
# ---------------------------------------------------------------------------

_PARENT_RULES: List[Tuple[List[str], str]] = [
    (["neural", "deep", "convolutional", "recurrent", "transformer", "attention", "lstm"], "deep learning"),
    (["language model", "nlp", "text", "translation", "summarization", "tokeniz"], "natural language processing"),
    (["reinforcement", "policy", "reward", "agent", "markov", "q-learning", "actor"], "reinforcement learning"),
    (["image", "vision", "visual", "segmentation", "detection", "recognition"], "computer vision"),
    (["graph", "node", "edge", "link prediction"], "graph machine learning"),
    (["generative", "diffusion", "synthesis", "sampling"], "generative models"),
    (["quantum", "qubit", "circuit", "entanglement"], "quantum computing"),
    (["federated", "privacy", "differential privacy", "distributed learning"], "federated learning"),
    (["continual", "catastrophic", "forgetting", "lifelong"], "continual learning"),
    (["meta", "few-shot", "zero-shot", "learning to learn"], "meta-learning"),
    (["bayesian", "uncertainty", "probabilistic", "posterior"], "probabilistic machine learning"),
    (["topological", "homology", "manifold", "geometric", "persistent"], "topological/geometric learning"),
    (["protein", "drug", "genomics", "molecular", "biology"], "computational biology"),
    (["fairness", "bias", "ethical", "explainability", "interpretability"], "responsible AI"),
    (["retrieval", "search", "information retrieval", "dense retrieval"], "information retrieval"),
]


def _infer_parent_topic(text: str) -> str:
    t = text.lower()
    for keywords, parent in _PARENT_RULES:
        if any(kw in t for kw in keywords):
            return parent
    return "machine learning"


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_EXPANSION_PROMPT = """\
You are an academic search query expansion expert with deep knowledge of \
research literature across all scientific domains.

Given a user query{prf_notice}, generate a comprehensive expansion that will help retrieve \
the most relevant academic papers from arXiv and OpenAlex.

Query: {query}
{prf_context}
Return ONLY a valid JSON object (no markdown fences, no explanations) with \
exactly these keys:

{{
  "expanded_queries": [
    // 5-8 query strings that a researcher would search for.
    // MUST include the original query as the first item.
    // Include full landmark paper titles (e.g. "Attention Is All You Need"),
    // canonical long-form names, key subtopic phrases, and alias variants.
  ],
  "semantic_keywords": [
    // 8-14 short terms/phrases capturing the topic's core vocabulary.
    // Include acronyms, full forms, related methods, and evaluation terms.
  ],
  "topic_profile": {{
    "topic":         "canonical name for this research topic",
    "aliases":       ["list of alternative names and abbreviations"],
    "acronym":       "primary acronym if one exists, else empty string",
    "landmarks":     [
      // 2-5 exact titles of the most seminal/foundational papers in this area.
      // These should be real, well-known paper titles.
    ],
    "subtopics":     ["3-6 direct sub-areas or variants"],
    "exclusions":    ["topics to exclude if query is ambiguous (can be empty)"],
    // For BROAD one- or two-word queries (e.g. "Transformers", "RAG"): you MUST output
    // 3-6 short phrases naming concrete adjacent domains users often do NOT mean —
    // e.g. "medical image segmentation", "speech separation", "time series forecasting" —
    // so ranking can deprioritize off-domain hits. Never leave exclusions empty for such queries
    // unless the query is already narrowly scoped.
    "parent_topic":  "the broader field this belongs to",
    "child_topics":  ["2-4 more specific child topics"],
    "datasets":      ["2-4 benchmark datasets commonly used"],
    "benchmarks":    ["2-4 evaluation benchmarks or leaderboards"],
    "methods":       ["3-5 key algorithms or techniques"],
    "model_families":["2-4 notable model families or architectures"]
  }}
}}"""


# ---------------------------------------------------------------------------
# QueryExpander
# ---------------------------------------------------------------------------

class QueryExpander:
    """
    Generic, query-driven academic query expander.

    All expansion is derived from the query itself via structural parsing.
    An optional Groq LLM call enriches the output when available.
    No hardcoded acronym tables, topic dictionaries, or landmark lists.
    """

    _LLM_COOLDOWN_SECS: float = 90.0

    def __init__(self) -> None:
        self._available = False
        self._llm_cooldown_until: float = 0.0
        self.llm = None

        try:
            from app.core.config import settings  # type: ignore
            api_key = getattr(settings, "GROQ_API_KEY", "")
            if api_key:
                from langchain_groq import ChatGroq  # type: ignore
                self.llm = ChatGroq(
                    api_key=api_key,
                    model=getattr(settings, "FAST_MODEL", "llama-3.1-8b-instant"),
                    temperature=0.1,
                    max_tokens=1024,
                )
                self._available = True
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand(self, query: str, prf_terms: List[str] = None) -> Dict[str, Any]:
        """Return a fully populated expansion dict for any query."""
        query = (query or "").strip()
        prf_terms = prf_terms or []

        # Classify query complexity for downstream retrieval control
        complexity = self._classify_complexity(query)

        # Always build the structural generic expansion first (instant, no LLM)
        generic = self._generic_expand(query)

        llm_res: Dict[str, Any] = {}
        llm_prof: Dict[str, Any] = {}

        if self._available and not self._llm_in_cooldown():
            try:
                llm_res, llm_prof = self._llm_expand(query, prf_terms)
            except Exception as exc:
                print(f"[QueryExpander] LLM expansion failed: {exc}", file=sys.stderr)
                self._llm_cooldown_until = time.time() + self._LLM_COOLDOWN_SECS

        # Merge structural + LLM output
        merged = self._merge_all(query, generic, llm_res)

        # Build profile (LLM data takes priority over heuristics)
        profile = self._build_generic_topic_profile(query, merged, llm_prof)

        # Finalise and cap
        merged["expanded_queries"] = self._dedupe_similar_queries(
            merged.get("expanded_queries", [])[:12], min_jaccard=0.85
        )[:8]
        merged["semantic_keywords"] = self._dedupe(
            merged.get("semantic_keywords", [])
        )[:14]
        merged["topic_profile"] = profile
        merged["retrieval_bundles"] = self._build_retrieval_bundles(profile, merged)
        merged["original"] = query
        merged["query_complexity"] = complexity
        return merged

    # ------------------------------------------------------------------
    # Query complexity classification
    # ------------------------------------------------------------------

    def _classify_complexity(self, query: str) -> str:
        """
        Classify query as broad/moderate/narrow for retrieval control.

        Broad:    1-2 content tokens (e.g. "transformers", "RAG")
        Moderate: 3-4 content tokens
        Narrow:   5+ content tokens or contains quoted phrases
        """
        q = query.strip()
        if '"' in q:
            return "narrow"

        _stop = {
            "a", "an", "the", "of", "in", "on", "at", "to", "for",
            "and", "or", "with", "by", "is", "are", "via", "using",
        }
        tokens = [
            t for t in re.split(r"[\s\-_]+", q.lower())
            if t and t not in _stop and len(t) >= 2
        ]
        if len(tokens) <= 2:
            return "broad"
        elif len(tokens) <= 4:
            return "moderate"
        return "narrow"

    # ------------------------------------------------------------------
    # LLM expansion
    # ------------------------------------------------------------------

    def _llm_expand(self, query: str, prf_terms: List[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Call Groq and parse the structured JSON response.
        Raises on any failure so the caller can engage the fallback.
        """
        prf_notice = " and a list of frequently occurring terms from a first-pass retrieval" if prf_terms else ""
        prf_ctx = f"\nFrequent terms found in recent literature (use these to ground your expansion):\n{', '.join(prf_terms)}\n" if prf_terms else ""
        
        # Intent classification before expansion: check if query is a raw architecture/concept
        q_lower = query.lower().strip()
        complexity = self._classify_complexity(query)
        broad_rules = ""
        if complexity == "broad":
            broad_rules = (
                "\nBROAD QUERY: The query is only 1-2 words. You MUST fill topic_profile[\"exclusions\"] "
                "with 3-6 specific adjacent domains (e.g. medical imaging, speech/audio, finance, time series) "
                "that are commonly confused with this topic, so the ranker can down-rank them.\n"
            )
        intent_rules = ""
        if len(q_lower.split()) <= 2 and not any(w in q_lower for w in ["for", "in", "predict", "detect", "segment", "diagnose", "time series", "vision", "speech", "application"]):
            intent_rules = (
                "\nIMPORTANT INTENT CLASSIFICATION: The user is asking about a foundational architecture or concept. "
                "You MUST suppress application-domain variants (e.g. do not include 'transformers for speech'). "
                "Focus strictly on the architecture, pre-training, and fundamental variants.\n"
                "You MUST populate topic_profile[\"exclusions\"] with 3–6 short phrases naming application domains or adjacent fields to deprioritize "
                "(e.g. speech and audio separation, finance and econometrics, drug discovery, unrelated survey papers) when they are not what the user asked for.\n"
            )

        prompt = _EXPANSION_PROMPT.format(
            query=query, prf_notice=prf_notice, prf_context=prf_ctx + broad_rules + intent_rules
        )
        response = self.llm.invoke(prompt)
        raw = (response.content or "").strip()

        # Strip markdown fences if the model wrapped the JSON anyway
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

        data: Dict[str, Any] = json.loads(raw)

        llm_res: Dict[str, Any] = {
            "expanded_queries": [
                str(q).strip() for q in data.get("expanded_queries", []) if str(q).strip()
            ],
            "semantic_keywords": [
                str(k).strip() for k in data.get("semantic_keywords", []) if str(k).strip()
            ],
        }
        llm_prof: Dict[str, Any] = dict(data.get("topic_profile", {}))
        return llm_res, llm_prof

    # ------------------------------------------------------------------
    # Structural generic expansion (no LLM, works offline)
    # ------------------------------------------------------------------

    def _generic_expand(self, query: str) -> Dict[str, Any]:
        """
        Pure heuristic expansion — no topic knowledge, works for any input.

        Covers:
          • Dynamic acronym detection (all-caps / CamelCase ≤7 chars)
          • Quoted exact-phrase variant for multi-word queries
          • "survey" / "deep …" / "review" token variants
          • Keyword extraction from content tokens
        """
        q = query.strip()
        q_lower = q.lower()
        words = q.split()
        _stop = {
            "a", "an", "the", "of", "in", "on", "at", "to", "for",
            "and", "or", "with", "by", "is", "are", "via", "using",
        }
        tokens = [
            t for t in re.split(r"[\s\-_]+", q_lower)
            if t and t not in _stop and len(t) >= 2
        ]

        expanded: List[str] = [q]  # original is always first

        # Dynamic acronym detection: single short token, all-caps or CamelCase
        is_acronym = (
            len(words) == 1
            and len(q) <= 7
            and (
                q.upper() == q
                or re.match(r"^[A-Z][a-zA-Z]{1,5}$", q)
            )
        )

        # Multi-word: quoted exact-phrase variant
        if len(words) >= 3:
            quoted = f'"{q}"'
            if quoted.lower() not in [e.lower() for e in expanded]:
                expanded.append(quoted)

        if len(words) >= 2:
            # "deep {query}" for neural-adjacent queries
            neural = ("learning", "network", "model", "neural", "architecture")
            if any(t in q_lower for t in neural):
                v = f"deep {q_lower}"
                if v not in [e.lower() for e in expanded]:
                    expanded.append(v)
            # survey variant
            sv = f"{q} survey"
            if sv.lower() not in [e.lower() for e in expanded]:
                expanded.append(sv)
            # review variant for longer queries
            if len(words) >= 4:
                rv = f"{q} review"
                if rv.lower() not in [e.lower() for e in expanded]:
                    expanded.append(rv)

        # Keywords from content tokens
        keywords: List[str] = list(tokens)

        return {
            "original": q,
            "expanded_queries": expanded,
            "semantic_keywords": keywords,
            # Internal flags used by profile builder
            "_is_acronym": is_acronym,
            "_long_form": "",
            "_inferred_parent": _infer_parent_topic(q_lower),
        }

    # ------------------------------------------------------------------
    # Merge generic + LLM output
    # ------------------------------------------------------------------

    def _merge_all(
        self,
        query: str,
        generic: Dict[str, Any],
        llm_res: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Merge the structural generic dict with LLM output.
        LLM queries are appended after the structural ones; dedup is applied
        later so original always remains first.
        """
        merged = dict(generic)

        all_queries = list(generic.get("expanded_queries", []))
        for qv in llm_res.get("expanded_queries", []):
            if qv.lower() not in [e.lower() for e in all_queries]:
                all_queries.append(qv)

        all_kws = list(generic.get("semantic_keywords", []))
        for kw in llm_res.get("semantic_keywords", []):
            if kw.lower() not in [k.lower() for k in all_kws]:
                all_kws.append(kw)

        merged["expanded_queries"] = all_queries
        merged["semantic_keywords"] = all_kws
        return merged

    # ------------------------------------------------------------------
    # Topic profile
    # ------------------------------------------------------------------

    def _build_generic_topic_profile(
        self,
        query: str,
        merged: Dict[str, Any],
        llm_prof: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build topic_profile, preferring LLM data over structural heuristics.
        All 12 required keys are always present.
        """
        is_acronym: bool = merged.get("_is_acronym", False)
        long_form: str = merged.get("_long_form", "")
        inferred_parent: str = merged.get("_inferred_parent", "machine learning")

        def _lget(key: str, default: Any = None) -> Any:
            """Get from LLM profile; fall back to default."""
            v = llm_prof.get(key)
            if v is None or v == "" or v == []:
                return default() if callable(default) else default
            return v

        # Topic name: prefer LLM, fall back to long-form or query
        topic: str = _lget("topic") or long_form or query

        # Acronym: prefer LLM; if not provided and query looks like acronym, use it
        acronym: str = _lget("acronym", "")
        if not acronym and is_acronym:
            acronym = query.upper()

        # Aliases: LLM list, always include original query
        aliases: List[str] = list(_lget("aliases", []))
        if query.lower() not in [a.lower() for a in aliases]:
            aliases = [query] + aliases

        # Landmarks: LLM list; fallback = long multi-word expanded queries
        landmarks: List[str] = list(_lget("landmarks", []))
        if not landmarks:
            landmarks = [
                qv for qv in merged.get("expanded_queries", [])
                if len(qv.split()) >= 4 and not qv.startswith('"')
            ][:3]

        # Subtopics
        subtopics: List[str] = list(_lget("subtopics", []))

        # Parent topic: LLM value or heuristic
        parent_raw = _lget("parent_topic") or inferred_parent
        if isinstance(parent_raw, list):
            parent_raw = parent_raw[0] if parent_raw else inferred_parent
        parent_topic: str = str(parent_raw) or inferred_parent

        return {
            "topic":          topic,
            "aliases":        aliases,
            "acronym":        acronym or "",
            "landmarks":      landmarks,
            "subtopics":      subtopics,
            "exclusions":     list(_lget("exclusions", [])),
            "parent_topic":   parent_topic,
            "child_topics":   list(_lget("child_topics", [])),
            "datasets":       list(_lget("datasets", [])),
            "benchmarks":     list(_lget("benchmarks", [])),
            "methods":        list(_lget("methods", [])),
            "model_families": list(_lget("model_families", [])),
        }

    # ------------------------------------------------------------------
    # Retrieval bundles
    # ------------------------------------------------------------------

    def _build_retrieval_bundles(
        self,
        profile: Dict[str, Any],
        merged: Dict[str, Any],
    ) -> Dict[str, Any]:
        all_expanded = merged.get("expanded_queries", [])

        # broad: aliases / topic name (for broad keyword searches)
        broad: List[str] = list(profile.get("aliases", []))[:3]
        if not broad:
            broad = [profile.get("topic") or merged.get("original", "")]

        # landmark_titles: seminal paper titles from the profile
        landmark_titles: List[str] = list(profile.get("landmarks", []))[:4]
        if not landmark_titles:
            # Fall back to long expanded queries that look like paper titles
            landmark_titles = [
                qv for qv in all_expanded
                if len(qv.split()) >= 4 and not qv.startswith('"')
            ][:2]

        # acronym bundle
        acronym: str = profile.get("acronym", "")
        acronym_bundle: List[str] = []
        if acronym:
            acronym_bundle.append(acronym)
        original = merged.get("original", "")
        if (
            original.upper() == original
            and 2 <= len(original) <= 7
            and original not in acronym_bundle
        ):
            acronym_bundle.append(original)

        # subtopics
        subtopics: List[str] = list(profile.get("subtopics", []))[:4]

        return {
            "broad":           broad,
            "landmark_titles": landmark_titles,
            "acronym":         acronym_bundle,
            "subtopics":       subtopics,
        }

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _llm_in_cooldown(self) -> bool:
        return time.time() < self._llm_cooldown_until

    def _dedupe(self, items: List[str]) -> List[str]:
        seen: set = set()
        out: List[str] = []
        for item in items:
            key = item.lower().strip()
            if key and key not in seen:
                seen.add(key)
                out.append(item)
        return out

    def _dedupe_similar_queries(
        self, queries: List[str], min_jaccard: float = 0.85
    ) -> List[str]:
        """Remove queries that are near-duplicates (Jaccard on word tokens).
        Quoted exact-phrase variants are preserved — they use different search
        semantics even when they share the same word tokens."""

        def _tok(s: str) -> set:
            return set(re.split(r"\W+", s.lower())) - {""}

        kept: List[str] = []
        for q in queries:
            tq = _tok(q)
            if not tq:
                continue
            # Always keep quoted exact-phrase variants — they are structurally
            # distinct search modifiers even if tokens overlap fully.
            is_quoted = q.strip().startswith('"') and q.strip().endswith('"')
            if is_quoted:
                kept.append(q)
                continue
            duplicate = False
            for k in kept:
                # Don't compare against quoted variants either
                if k.strip().startswith('"') and k.strip().endswith('"'):
                    continue
                tk = _tok(k)
                if tk:
                    inter = len(tq & tk)
                    union = len(tq | tk)
                    if union and inter / union >= min_jaccard:
                        duplicate = True
                        break
            if not duplicate:
                kept.append(q)
        return kept