"""
advanced_formatter.py
=====================
Response post-processing and UI-facing formatting helpers.

Goals:
- Keep answers concise and grounded.
- Prevent duplicated equation text in the answer body.
- Support targeted equation/table/figure selection.
- Preserve existing table/equation UI formatting by returning structured metadata.
- Return all requested tables / equations / figures when the user explicitly asks for all.
- Preserve equation rendering while improving explanation quality.
"""

from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AdvancedResponseFormatter:
    def __init__(self):
        self.eq_keywords = {
            "equation", "equations", "formula", "formulas", "math", "mathematical",
            "probability", "definition", "derive", "derivation", "mips", "encoder",
            "decoder", "latent", "marginalization", "explain equation", "show equation",
        }
        self.table_keywords = {
            "table", "tables", "result", "results", "score", "scores",
            "benchmark", "benchmarks", "performance", "dataset", "evaluation",
            "triviaqa", "nq", "natural questions", "webquestions", "wq",
            "fever", "msmarco", "ms marco", "jeopardy", "compare", "comparison",
            "baseline", "ablation",
        }
        self.figure_keywords = {
            "figure", "figures", "diagram", "diagrams", "architecture",
            "overview", "pipeline", "framework", "model", "system"
        }
        self.metadata_keywords = {
            "title", "author", "authors", "year", "date", "published",
            "abstract", "summary", "affiliation", "university", "institution", "arxiv"
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def format_response(
        self,
        query: str,
        answer_text: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        equations: Optional[List[Dict[str, Any]]] = None,
        tables: Optional[List[Dict[str, Any]]] = None,
        figures: Optional[List[Dict[str, Any]]] = None,
        document_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        query_l = (query or "").lower().strip()
        sources = sources or []
        equations = equations or []
        tables = tables or []
        figures = figures or []
        document_metadata = document_metadata or {}

        is_eq_list = self._is_all_equations_request(query_l)
        is_table_list = self._is_all_tables_request(query_l)
        is_figure_list = self._is_all_figures_request(query_l)
        is_specific_eq = self._is_specific_equation_request(query_l) and not is_eq_list
        is_table_query = self._is_table_query(query_l) and not is_table_list
        is_figure_query = self._is_figure_query(query_l) and not is_figure_list
        is_count_query = self._is_count_query(query_l)
        is_explanation_query = self._is_explanation_query(query_l)

        clean_answer = self._cleanup_answer_text(answer_text or "")

        if is_count_query:
            summary = self._format_count_only(query_l, equations, tables, figures, document_metadata)
            return {
                "summary_text": summary,
                "equations": [],
                "tables": [],
                "figures": [],
                "citations": self._collect_citations(sources),
                "mode": "count",
            }

        if is_eq_list:
            selected_equations = self._normalize_equations(equations)
            summary = clean_answer or self._build_all_equations_summary(selected_equations)
            summary = self._remove_equation_text_leak(summary, selected_equations)
            return {
                "summary_text": summary,
                "equations": selected_equations,
                "tables": [],
                "figures": [],
                "citations": self._collect_citations(sources, selected_equations),
                "mode": "all_equations",
            }

        if is_table_list:
            selected_tables = self._normalize_tables(tables)
            summary = clean_answer or self._build_all_tables_summary(selected_tables)
            summary = self._cleanup_non_equation_summary(summary)
            return {
                "summary_text": summary,
                "equations": [],
                "tables": selected_tables,
                "figures": [],
                "citations": self._collect_citations(sources, selected_tables),
                "mode": "all_tables",
            }

        if is_figure_list:
            selected_figures = self._normalize_figures(figures)
            summary = clean_answer or self._build_all_figures_summary(selected_figures)
            summary = self._cleanup_non_equation_summary(summary)
            return {
                "summary_text": summary,
                "equations": [],
                "tables": [],
                "figures": selected_figures,
                "citations": self._collect_citations(sources, selected_figures),
                "mode": "all_figures",
            }

        if is_specific_eq:
            selected_equation = self._pick_best_equation(query_l, equations)
            selected_equations = [selected_equation] if selected_equation else []
            summary = clean_answer or self._build_specific_equation_summary(query_l, selected_equation)
            summary = self._remove_equation_text_leak(summary, selected_equations)
            if selected_equation and (is_explanation_query or self._looks_too_generic(summary)):
                summary = self._augment_equation_explanation(summary, selected_equation, query_l)
            return {
                "summary_text": summary,
                "equations": selected_equations,
                "tables": [],
                "figures": [],
                "citations": self._collect_citations(sources, selected_equations),
                "mode": "specific_equation",
            }

        if is_table_query:
            best_table = self._pick_best_table(query_l, tables)
            selected_tables = [best_table] if best_table else []
            summary = clean_answer or self._build_table_summary(best_table)
            summary = self._cleanup_non_equation_summary(summary)
            return {
                "summary_text": summary,
                "equations": [],
                "tables": selected_tables,
                "figures": [],
                "citations": self._collect_citations(sources, selected_tables),
                "mode": "table",
            }

        if is_figure_query:
            best_figure = self._pick_best_figure(query_l, figures)
            selected_figures = [best_figure] if best_figure else []
            summary = clean_answer or self._build_figure_summary(best_figure)
            summary = self._cleanup_non_equation_summary(summary)
            return {
                "summary_text": summary,
                "equations": [],
                "tables": [],
                "figures": selected_figures,
                "citations": self._collect_citations(sources, selected_figures),
                "mode": "figure",
            }

        summary = self._cleanup_non_equation_summary(clean_answer)
        summary = self._ensure_citation_in_text(summary, sources)
        return {
            "summary_text": summary,
            "equations": [],
            "tables": [],
            "figures": [],
            "citations": self._collect_citations(sources),
            "mode": "general",
        }

    # ------------------------------------------------------------------
    # Query classification
    # ------------------------------------------------------------------

    def _is_all_equations_request(self, q: str) -> bool:
        patterns = [
            "show all equations", "show all equation", "list equations", "list all equations",
            "show equations", "what equations are", "all equations", "all equation",
            "equations mentioned", "extract all equations", "extract every mathematical formula",
            "every mathematical formula", "all mathematical formulas", "show me all equation",
            "show me all equations", "show every equation",
        ]
        return any(p in q for p in patterns)

    def _is_all_tables_request(self, q: str) -> bool:
        patterns = [
            "show all tables", "show all table", "list tables", "list all tables",
            "show tables", "all tables", "all table", "extract all tables", "show every table",
        ]
        return any(p in q for p in patterns)

    def _is_all_figures_request(self, q: str) -> bool:
        patterns = [
            "show all figures", "show all figure", "list figures", "list all figures",
            "show figures", "all figures", "all figure", "extract all figures", "show every figure",
        ]
        return any(p in q for p in patterns)

    def _is_specific_equation_request(self, q: str) -> bool:
        if any(k in q for k in self.eq_keywords):
            return True
        if re.search(r"\bequation\s+\d+\b", q):
            return True
        if re.search(r"p[_\s-]*eta|pη|p[_\s-]*theta|pθ|d\(z\)|q\(x\)|rag-token|rag token|rag sequence|rag-sequence", q):
            return True
        return False

    def _is_table_query(self, q: str) -> bool:
        return any(k in q for k in self.table_keywords)

    def _is_figure_query(self, q: str) -> bool:
        return any(k in q for k in self.figure_keywords)

    def _is_count_query(self, q: str) -> bool:
        count_words = {"how many", "number of", "count"}
        asset_words = {"equation", "equations", "table", "tables", "figure", "figures"}
        return any(w in q for w in count_words) and any(a in q for a in asset_words)

    def _is_explanation_query(self, q: str) -> bool:
        return any(k in q for k in ["explain", "why", "how", "what does", "interpret", "meaning", "defined", "definition"])

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup_answer_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"\bShort Summary\s+Short\b", "Short Summary", text, flags=re.I)
        text = re.sub(r"\bExplanation\s+Short\b", "Explanation", text, flags=re.I)
        text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.S)
        text = re.sub(r"```(?:latex|math|text)?\n.*?```", "", text, flags=re.S)
        text = text.replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def _cleanup_non_equation_summary(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"\bTechnical Details\b.*$", "", text, flags=re.I | re.S)
        text = re.sub(r"\\[A-Za-z]+", "", text)
        text = re.sub(r"[_^{}]", "", text)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        seen = set()
        kept = []
        for s in sentences:
            key = re.sub(r"\s+", " ", s.strip().lower())
            if key and key not in seen:
                seen.add(key)
                kept.append(s.strip())
        return " ".join(kept).strip()

    def _remove_equation_text_leak(self, text: str, equations: List[Dict[str, Any]]) -> str:
        if not text:
            return ""
        cleaned = text
        cleaned = re.sub(r"(?:\b[prdqxyRNAGT\-ηθ∑∏⊤≈∈\(\)\[\]\|:\.]+\s*){12,}", "", cleaned, flags=re.I)
        for eq in equations or []:
            raw = (eq.get("raw_text") or eq.get("text") or "").strip()
            if raw and len(raw) > 10:
                cleaned = cleaned.replace(raw, "")
            latex = (eq.get("normalized_latex") or eq.get("latex") or "").strip()
            if latex and len(latex) > 10:
                cleaned = cleaned.replace(latex, "")
        cleaned = re.sub(r"\b(Evidence from paper|Explanation|Short Summary)\s*[:\-]?\s*(?=\n|$)", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        return cleaned.strip()

    def _looks_too_generic(self, text: str) -> bool:
        if not text:
            return True
        generic_patterns = [
            r"most relevant equation",
            r"it is the most relevant formula",
            r"appears on page",
            r"document does not contain",
        ]
        return any(re.search(p, text, re.I) for p in generic_patterns)

    # ------------------------------------------------------------------
    # Equation helpers
    # ------------------------------------------------------------------

    def _normalize_equations(self, equations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        norm = []
        for eq in equations or []:
            item = dict(eq)
            page = item.get("page_number", item.get("page", None))
            item["page_number"] = page
            if not item.get("normalized_latex") and item.get("latex"):
                item["normalized_latex"] = item["latex"]
            if not item.get("raw_text") and item.get("text"):
                item["raw_text"] = item["text"]
            norm.append(item)

        def sort_key(x):
            n = x.get("global_number") or self._extract_number(x.get("label", ""))
            page = x.get("page_number")
            if page is None:
                page = 9999
            return (9999 if n is None else n, page)

        norm.sort(key=sort_key)
        return norm

    def _pick_best_equation(self, query: str, equations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        eqs = self._normalize_equations(equations)
        if not eqs:
            return None

        best_score = None
        best_eq = None
        for eq in eqs:
            text = " ".join([
                str(eq.get("label", "")),
                str(eq.get("raw_text", "")),
                str(eq.get("text", "")),
                str(eq.get("normalized_latex", "")),
                str(eq.get("description", "")),
            ]).lower()

            score = 0
            m = re.search(r"\bequation\s+(\d+)\b", query)
            if m:
                qn = int(m.group(1))
                en = eq.get("global_number") or self._extract_number(eq.get("label", ""))
                if en == qn:
                    score += 100

            if any(k in query for k in ["mips", "inner product", "d(z)", "q(x)", "document encoder", "query encoder"]):
                if any(k in text for k in ["d(z)", "q(x)", "exp", "⊤", "bert"]):
                    score += 45

            if any(k in query for k in ["p_eta", "pη", "retrieval probability", "probability formula"]):
                if any(k in text for k in ["pη", "p_eta", "exp"]):
                    score += 45

            if "rag-token" in query or "rag token" in query:
                if "rag-token" in text or "ragtoken" in text:
                    score += 50

            if "rag-sequence" in query or "rag sequence" in query:
                if "rag-sequence" in text or "ragsequence" in text:
                    score += 50

            query_terms = set(re.findall(r"[a-zA-Z0-9_\-\(\)]+", query))
            text_terms = set(re.findall(r"[a-zA-Z0-9_\-\(\)]+", text))
            score += len(query_terms & text_terms)

            if best_score is None or score > best_score:
                best_score = score
                best_eq = eq

        return best_eq

    def _build_all_equations_summary(self, equations: List[Dict[str, Any]]) -> str:
        if not equations:
            return "The document does not contain this information."
        lines = [f"Found {len(equations)} equations in the document.", ""]
        for eq in equations:
            label = eq.get("label") or f"Equation {eq.get('global_number', '?')}"
            page = eq.get("page_number", "?")
            lines.append(f"- {label} (Page {page})")
        return "\n".join(lines).strip()

    def _build_specific_equation_summary(self, query: str, eq: Optional[Dict[str, Any]]) -> str:
        if not eq:
            return "The document does not contain this information."
        label = eq.get("label") or f"Equation {eq.get('global_number', '?')}"
        page = eq.get("page_number", "?")
        desc = (eq.get("description") or "").strip()
        base = f"{label} appears on Page {page}."
        if desc:
            base += f" {desc}"
        return base

    def _augment_equation_explanation(self, summary: str, eq: Dict[str, Any], query: str) -> str:
        label = eq.get("label") or f"Equation {eq.get('global_number', '?')}"
        page = eq.get("page_number", "?")
        desc = self._infer_equation_explanation(eq)
        if summary and "does not contain" not in summary.lower() and not self._looks_too_generic(summary):
            return summary
        return f"{label} is the relevant formula on Page {page}. {desc}".strip()

    def _infer_equation_explanation(self, eq: Dict[str, Any]) -> str:
        text = " ".join([
            str(eq.get("label", "")),
            str(eq.get("raw_text", "")),
            str(eq.get("text", "")),
            str(eq.get("normalized_latex", "")),
            str(eq.get("description", "")),
        ]).lower()

        if "rag-sequence" in text or "ragsequence" in text:
            return (
                "It defines RAG-Sequence by marginalizing over the top-k retrieved documents and then generating the full output sequence conditioned on each retrieved document."
            )
        if "rag-token" in text or "ragtoken" in text:
            return (
                "It defines RAG-Token, where the model can attend to a different retrieved document at each decoding step, which gives it more flexibility during generation."
            )
        if any(k in text for k in ["d(z)", "q(x)", "bert", "exp"]):
            return (
                "It defines the retrieval probability using the similarity between a document embedding d(z) and a query embedding q(x), where both are produced by neural encoders."
            )
        desc = (eq.get("description") or "").strip()
        if desc:
            return desc
        return "It is one of the key mathematical definitions used in the paper."

    # ------------------------------------------------------------------
    # Table helpers
    # ------------------------------------------------------------------

    def _normalize_tables(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        norm = [dict(tb) for tb in tables or []]
        norm.sort(key=lambda x: ((x.get("global_number") or 9999), (x.get("page_number") or 9999)))
        return norm

    def _pick_best_table(self, query: str, tables: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not tables:
            return None
        best = None
        best_score = None
        for tb in tables:
            text = " ".join([
                str(tb.get("caption", "")),
                str(tb.get("markdown", "")),
                str(tb.get("raw_text", "")),
                str(tb.get("description", "")),
            ]).lower()
            score = 0

            dataset_terms = [
                "triviaqa", "nq", "natural questions", "webquestions", "wq",
                "fever", "msmarco", "ms marco", "jeopardy", "open-domain qa"
            ]
            metric_terms = [
                "score", "scores", "bleu", "rouge", "accuracy", "label acc",
                "f1", "em", "performance", "results", "benchmark", "evaluation",
                "ablation", "baseline"
            ]

            for term in dataset_terms:
                if term in query and term in text:
                    score += 20
            for term in metric_terms:
                if term in query and term in text:
                    score += 10

            m = re.search(r"\btable\s+(\d+)\b", query)
            if m:
                qn = int(m.group(1))
                tn = tb.get("global_number") or self._extract_number(tb.get("label", ""))
                if tn == qn:
                    score += 100

            q_terms = set(re.findall(r"[a-zA-Z0-9\-]+", query))
            t_terms = set(re.findall(r"[a-zA-Z0-9\-]+", text))
            score += len(q_terms & t_terms)

            if best_score is None or score > best_score:
                best_score = score
                best = tb
        return best

    def _build_all_tables_summary(self, tables: List[Dict[str, Any]]) -> str:
        if not tables:
            return "The document does not contain this information."
        lines = [f"Found {len(tables)} tables in the document.", ""]
        for tb in tables:
            label = tb.get("label") or f"Table {tb.get('global_number', '?')}"
            page = tb.get("page_number", "?")
            caption = (tb.get("caption") or "").strip()
            if caption and caption.lower() != str(label).lower():
                lines.append(f"- {label} (Page {page}): {caption}")
            else:
                lines.append(f"- {label} (Page {page})")
        return "\n".join(lines).strip()

    def _build_table_summary(self, table: Optional[Dict[str, Any]]) -> str:
        if not table:
            return "The document does not contain this information."
        caption = table.get("caption") or f"Table {table.get('global_number', '?')}"
        page = table.get("page_number", "?")
        return f"The most relevant result is in {caption} (Page {page})."

    # ------------------------------------------------------------------
    # Figure helpers
    # ------------------------------------------------------------------

    def _normalize_figures(self, figures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        norm = [dict(fig) for fig in figures or []]
        norm.sort(key=lambda x: ((x.get("global_number") or 9999), (x.get("page_number") or 9999)))
        return norm

    def _pick_best_figure(self, query: str, figures: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not figures:
            return None
        best = None
        best_score = None
        for fig in figures:
            text = " ".join([
                str(fig.get("caption", "")),
                str(fig.get("description", "")),
                str(fig.get("raw_text", "")),
            ]).lower()
            score = 0
            if any(k in query for k in ["architecture", "overview", "pipeline", "model", "system", "diagram"]):
                for kw in ["architecture", "overview", "pipeline", "model", "system", "framework"]:
                    if kw in text:
                        score += 15
            q_terms = set(re.findall(r"[a-zA-Z0-9\-]+", query))
            f_terms = set(re.findall(r"[a-zA-Z0-9\-]+", text))
            score += len(q_terms & f_terms)
            if best_score is None or score > best_score:
                best_score = score
                best = fig
        return best

    def _build_all_figures_summary(self, figures: List[Dict[str, Any]]) -> str:
        if not figures:
            return "The document does not contain this information."
        lines = [f"Found {len(figures)} figures in the document.", ""]
        for fig in figures:
            label = fig.get("label") or f"Figure {fig.get('global_number', '?')}"
            page = fig.get("page_number", "?")
            caption = (fig.get("caption") or "").strip()
            if caption and caption.lower() != str(label).lower():
                lines.append(f"- {label} (Page {page}): {caption}")
            else:
                lines.append(f"- {label} (Page {page})")
        return "\n".join(lines).strip()

    def _build_figure_summary(self, figure: Optional[Dict[str, Any]]) -> str:
        if not figure:
            return "The document does not contain this information."
        caption = figure.get("caption") or f"Figure {figure.get('global_number', '?')}"
        page = figure.get("page_number", "?")
        return f"The most relevant figure is {caption} (Page {page})."

    # ------------------------------------------------------------------
    # Counts / citations
    # ------------------------------------------------------------------

    def _format_count_only(
        self,
        query: str,
        equations: List[Dict[str, Any]],
        tables: List[Dict[str, Any]],
        figures: List[Dict[str, Any]],
        document_metadata: Dict[str, Any],
    ) -> str:
        if "equation" in query:
            count = document_metadata.get("display_equation_count", len(equations))
            return str(count)
        if "table" in query:
            count = document_metadata.get("table_count", len(tables))
            return str(count)
        if "figure" in query:
            count = document_metadata.get("figure_count", len(figures))
            return str(count)
        return "The document does not contain this information."

    def _collect_citations(
        self,
        sources: Optional[List[Dict[str, Any]]] = None,
        assets: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        citations: List[str] = []
        for src in sources or []:
            c = self._source_to_citation(src)
            if c and c not in citations:
                citations.append(c)
        for asset in assets or []:
            if not asset:
                continue
            if "caption" in asset and "table" in str(asset.get("caption", "")).lower():
                c = f"(Source: Table {asset.get('global_number', '?')})"
            elif "caption" in asset and "figure" in str(asset.get("caption", "")).lower():
                c = f"(Source: Figure {asset.get('global_number', '?')})"
            elif asset.get("global_number") is not None and ("latex" in asset or "normalized_latex" in asset or "raw_text" in asset):
                c = f"(Source: Equation {asset.get('global_number', '?')})"
            else:
                page = asset.get("page_number")
                c = f"(Source: Page {page})" if page is not None else None
            if c and c not in citations:
                citations.append(c)
        return citations

    def _source_to_citation(self, src: Dict[str, Any]) -> Optional[str]:
        if not src:
            return None
        if src.get("source_type") == "table":
            return f"(Source: Table {src.get('global_number', '?')})"
        if src.get("source_type") == "figure":
            return f"(Source: Figure {src.get('global_number', '?')})"
        if src.get("source_type") == "equation":
            return f"(Source: Equation {src.get('global_number', '?')})"
        page = src.get("page_number", src.get("page"))
        if page is not None:
            try:
                return f"(Source: Page {int(page)})"
            except Exception:
                return f"(Source: Page {page})"
        return None

    def _ensure_citation_in_text(self, text: str, sources: List[Dict[str, Any]]) -> str:
        if not text:
            return ""
        if "(Source:" in text:
            return text
        cits = self._collect_citations(sources)
        if cits:
            return f"{text}\n\n{cits[0]}"
        return text

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def _extract_number(self, text: str) -> Optional[int]:
        if not text:
            return None
        m = re.search(r"(\d+)", str(text))
        return int(m.group(1)) if m else None
