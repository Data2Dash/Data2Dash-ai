# papers → counts/grouping/statistics → analytics summary
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from app.schemas.analytics import AnalyticsSummary


class AnalyticsService:
    _SUBTOPIC_RULES = {
        "NLP Transformers": (
            "nlp", "language model", "machine translation", "sequence to sequence",
            "tokenization", "encoder decoder", "bert", "gpt", "t5", "roberta",
        ),
        "Vision Transformers": (
            "vision transformer", "vit", "image classification", "segmentation",
            "object detection", "patch embedding", "detr", "deit", "swin",
        ),
        "Multimodal Transformers": (
            "multimodal", "vision-language", "vision language", "vlm",
            "clip", "flamingo", "blip", "llava", "cross-modal", "text-to-image",
        ),
        "Efficient Transformers": (
            "efficient transformer", "linear attention", "sparse attention",
            "longformer", "performer", "flashattention", "compressive", "reformer",
            "memory efficient", "low-rank",
        ),
        "Fine-tuning & Adaptation": (
            "fine-tuning", "instruction tuning", "parameter efficient",
            "peft", "lora", "adapter", "prompt tuning", "alignment", "rlhf",
        ),
        "LLM & Foundation Models": (
            "llm", "large language model", "foundation model", "instructgpt",
            "llama", "few-shot", "scaling law", "pretraining", "chatgpt",
        ),
    }

    _FIELD_RULES = {
        "NLP": ("nlp", "language", "translation", "question answering", "text"),
        "Computer Vision": ("image", "vision", "detection", "segmentation", "video"),
        "Multimodal AI": ("multimodal", "vision-language", "cross-modal", "audio-text"),
        "Optimization & Efficiency": ("efficient", "sparse", "latency", "throughput", "compression"),
        "LLMs": ("llm", "foundation model", "instruct", "prompt", "alignment"),
    }

    def compute_summary(self, papers, query: str = "") -> AnalyticsSummary:
        total_papers        = len(papers)
        papers_last_30_days = self._count_recent_papers(papers, 30)
        top_authors         = self._top_authors(papers)
        top_author_impact   = self._top_author_impact(papers)
        top_keywords        = self._top_keywords(papers)
        monthly_counts      = self._monthly_counts(papers)
        trend_status        = self._trend_status(monthly_counts)
        avg_citations       = self._avg_citations(papers)
        max_citations       = max((p.citations for p in papers), default=0)
        year_distribution   = self._year_distribution(papers)
        field_distribution  = self._field_distribution(papers, query)
        subtopic_distribution = self._subtopic_distribution(papers, query)
        top_subtopics = list(subtopic_distribution.items())[:10]
        venue_distribution = self._venue_distribution(papers)
        source_distribution = self._source_distribution(papers)
        year_subtopic_trends = self._year_subtopic_trends(papers, query)
        top_cited_papers    = self._top_cited_papers(papers, limit=5)
        llm_insight         = self._generate_llm_insight(
            papers, query, trend_status, top_keywords, avg_citations
        )

        return AnalyticsSummary(
            total_papers=total_papers,
            papers_last_30_days=papers_last_30_days,
            top_authors=top_authors,
            top_keywords=top_keywords,
            monthly_counts=monthly_counts,
            trend_status=trend_status,
            avg_citations=avg_citations,
            max_citations=max_citations,
            year_distribution=year_distribution,
            field_distribution=field_distribution,
            subtopic_distribution=subtopic_distribution,
            venue_distribution=venue_distribution,
            source_distribution=source_distribution,
            year_subtopic_trends=year_subtopic_trends,
            top_subtopics=top_subtopics,
            top_author_impact=top_author_impact,
            top_cited_papers=top_cited_papers,
            llm_insight=llm_insight,
        )

    # ------------------------------------------------------------------
    # Existing helpers (unchanged)
    # ------------------------------------------------------------------

    def _count_recent_papers(self, papers, days: int) -> int:
        cutoff = datetime.today().date() - timedelta(days=days)
        return sum(
            1 for p in papers
            if p.published_date and self._safe_date(p.published_date) >= cutoff
        )

    def _top_authors(self, papers, limit: int = 8):
        counter: Counter = Counter()
        for paper in papers:
            normalized = self._normalize_authors(paper.authors)
            counter.update(normalized)
        return counter.most_common(limit)

    def _top_author_impact(self, papers, limit: int = 8) -> list:
        by_author: dict = defaultdict(lambda: {"paper_count": 0, "citations": 0, "semantic": 0.0})
        for paper in papers:
            authors = self._normalize_authors(paper.authors)
            for author in authors:
                entry = by_author[author]
                entry["paper_count"] += 1
                entry["citations"] += max(paper.citations, 0)
                entry["semantic"] += max(getattr(paper, "semantic_score", 0.0), 0.0)

        ranked = []
        for author, stats in by_author.items():
            impact = (
                stats["paper_count"] * 3.0 +
                min(stats["citations"] / 200.0, 50.0) +
                stats["semantic"] * 5.0
            )
            ranked.append({
                "author": author,
                "paper_count": stats["paper_count"],
                "citations": stats["citations"],
                "impact_score": round(impact, 2),
            })
        ranked.sort(key=lambda x: (x["impact_score"], x["citations"]), reverse=True)
        return ranked[:limit]

    def _top_keywords(self, papers, limit: int = 10):
        counter: Counter = Counter()
        for paper in papers:
            counter.update(self._canonical_tag(t) for t in (paper.topic_tags or []) if t.strip())
        return counter.most_common(limit)

    def _monthly_counts(self, papers) -> dict:
        counts: dict = defaultdict(int)
        for paper in papers:
            if paper.published_date:
                try:
                    dt  = datetime.fromisoformat(paper.published_date)
                    key = dt.strftime("%Y-%m")
                    counts[key] += 1
                except Exception:
                    pass
        return dict(sorted(counts.items()))

    def _trend_status(self, monthly_counts: dict) -> str:
        values = list(monthly_counts.values())
        if len(values) < 2:
            return "Stable"
        if values[-1] > values[-2]:
            return "📈 Rising"
        if values[-1] < values[-2]:
            return "📉 Declining"
        return "➡️ Stable"

    # ------------------------------------------------------------------
    # New helpers
    # ------------------------------------------------------------------

    def _avg_citations(self, papers) -> float:
        if not papers:
            return 0.0
        return round(sum(p.citations for p in papers) / len(papers), 1)

    def _year_distribution(self, papers) -> dict:
        counts: dict = defaultdict(int)
        for paper in papers:
            if paper.published_date:
                try:
                    year = str(datetime.fromisoformat(paper.published_date).year)
                    counts[year] += 1
                except Exception:
                    pass
        return dict(sorted(counts.items()))

    def _field_distribution(self, papers, query: str = "") -> dict:
        counts: dict = Counter()
        for paper in papers:
            text = self._paper_text(paper)
            matched = False
            for field, keywords in self._FIELD_RULES.items():
                if any(kw in text for kw in keywords):
                    counts[field] += 1
                    matched = True
            for tag in paper.topic_tags:
                if tag.strip():
                    counts[self._canonical_tag(tag.strip())] += 1
            if not matched and query and "transform" in query.lower():
                counts["Transformers"] += 1
        # Return top 12 fields
        return dict(counts.most_common(12))

    def _subtopic_distribution(self, papers, query: str = "") -> dict:
        counts: Counter = Counter()
        for paper in papers:
            labels = list(getattr(paper, "inferred_topic_tags", []) or [])
            if not labels:
                labels = self._infer_subtopics(paper, query=query)
            if not labels:
                continue
            for label in labels:
                counts[label] += 1
        return dict(counts.most_common(12))

    def _infer_subtopics(self, paper, query: str = "") -> list[str]:
        text = self._paper_text(paper)
        labels = []
        for label, keywords in self._SUBTOPIC_RULES.items():
            if any(kw in text for kw in keywords):
                labels.append(label)
        if not labels and "transform" in text:
            labels.append("General Transformers")
        if not labels and query and "transform" in query.lower():
            labels.append("General Transformers")
        return labels

    def _venue_distribution(self, papers, limit: int = 12) -> dict:
        counts: Counter = Counter()
        for paper in papers:
            venue = (getattr(paper, "venue", "") or "").strip()
            if not venue:
                venue = "arXiv" if "arxiv" in paper.source else "Unknown Venue"
            counts[venue] += 1
        return dict(counts.most_common(limit))

    def _source_distribution(self, papers) -> dict:
        counts: Counter = Counter()
        for paper in papers:
            for src in (paper.source or "").split(","):
                s = src.strip()
                if s:
                    counts[s] += 1
        return dict(counts.most_common(8))

    def _year_subtopic_trends(self, papers, query: str = "", limit_subtopics: int = 5) -> dict:
        # Pick top subtopics first to keep dashboard compact.
        top_subtopics = [name for name, _ in self._subtopic_distribution(papers, query).items()][:limit_subtopics]
        if not top_subtopics:
            return {}
        trends: dict = {name: defaultdict(int) for name in top_subtopics}
        for paper in papers:
            if not paper.published_date:
                continue
            try:
                year = str(datetime.fromisoformat(paper.published_date).year)
            except Exception:
                continue
            labels = self._infer_subtopics(paper, query=query)
            if not labels:
                labels = list(getattr(paper, "inferred_topic_tags", []) or [])
            for label in labels:
                if label in trends:
                    trends[label][year] += 1
        return {k: dict(sorted(v.items())) for k, v in trends.items()}

    def _top_cited_papers(self, papers, limit: int = 5) -> list:
        sorted_papers = sorted(papers, key=lambda p: p.citations, reverse=True)
        result = []
        for p in sorted_papers[:limit]:
            inferred_subtopics = self._infer_subtopics(p)
            result.append({
                "title":     p.title,
                "citations": p.citations,
                "url":       p.url,
                "year":      p.published_date[:4] if p.published_date else "N/A",
                "authors":   p.authors[:2],
                "source":    p.source,
                "venue":     getattr(p, "venue", "") or "Unknown Venue",
                "subtopics": inferred_subtopics[:2],
            })
        return result

    def _generate_llm_insight(
        self,
        papers,
        query: str,
        trend: str,
        top_keywords,
        avg_citations: float,
    ) -> str:
        """Generate a 3-4 sentence research landscape summary using the LLM."""
        if not papers:
            return ""
        try:
            from app.core.config import settings
            from langchain_groq import ChatGroq

            top_kw = ", ".join(kw for kw, _ in top_keywords[:6]) or "N/A"
            years  = sorted(
                set(
                    p.published_date[:4]
                    for p in papers
                    if p.published_date and len(p.published_date) >= 4
                )
            )
            year_range = f"{years[0]}–{years[-1]}" if len(years) > 1 else (years[0] if years else "N/A")

            prompt = f"""You are a research trend analyst. Write a concise 3-sentence paragraph (max 80 words) summarising the research landscape for the topic: "{query}".

Facts to include:
- {len(papers)} papers retrieved, published across {year_range}
- Trend status: {trend}
- Average citations per paper: {avg_citations}
- Dominant research themes / fields: {top_kw}

Be specific, insightful, and academic in tone. Do NOT use bullet points. Write flowing prose only."""

            llm = ChatGroq(
                api_key=settings.GROQ_API_KEY,
                model=settings.FAST_MODEL,
                temperature=0.3,
                max_tokens=200,
            )
            response = llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"[AnalyticsService] LLM insight failed: {e}", file=sys.stderr)
            return ""

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_date(date_str: str):
        try:
            return datetime.fromisoformat(date_str).date()
        except Exception:
            return datetime.min.date()

    @staticmethod
    def _normalize_authors(authors: list[str]) -> list[str]:
        cleaned = []
        for author in authors or []:
            name = " ".join((author or "").split()).strip()
            if len(name) >= 3:
                cleaned.append(name)
        return cleaned

    def _paper_text(self, paper) -> str:
        fields = [
            paper.title or "",
            paper.abstract or "",
            " ".join(paper.topic_tags or []),
            " ".join(getattr(paper, "inferred_topic_tags", []) or []),
            " ".join(paper.keywords or []),
            getattr(paper, "venue", "") or "",
        ]
        return " ".join(fields).lower()

    @staticmethod
    def _canonical_tag(tag: str) -> str:
        tag_l = tag.lower()
        if "transform" in tag_l:
            return "Transformers"
        if "language" in tag_l or tag_l == "nlp":
            return "NLP"
        if "vision" in tag_l or "image" in tag_l:
            return "Computer Vision"
        if "multimodal" in tag_l:
            return "Multimodal AI"
        if "llm" in tag_l or "foundation" in tag_l:
            return "LLMs"
        return tag