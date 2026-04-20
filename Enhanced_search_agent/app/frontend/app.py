import os
import sys
import datetime
import re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import pandas as pd
import plotly.express as px

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from app.core.config import settings
from app.services.search_agent import SearchAgent
from app.schemas.paper import Paper
from typing import List, Optional

st.set_page_config(page_title="DATA2DASH", page_icon="📚", layout="wide")




BASE_THEME_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #05070b;
    --surface: #0c1117;
    --surface-alt: #111821;
    --surface-soft: #151d27;
    --border: #1f2a36;
    --border-strong: #324255;
    --text: #e4ebf3;
    --text-muted: #9ca9b8;
    --headline: #f7fbff;
    --primary: #7aa2c8;
    --primary-strong: #547ca3;
    --primary-soft: #122031;
    --accent: #79a8a7;
    --accent-soft: #102223;
    --success: #68b38f;
    --warning: #c49a63;
    --danger: #d07d7d;
    --shadow: 0 20px 48px rgba(0, 0, 0, 0.42);
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background:
        radial-gradient(circle at top left, rgba(122, 162, 200, 0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(121, 168, 167, 0.08), transparent 22%),
        linear-gradient(180deg, #06090d 0%, var(--bg) 100%);
    color: var(--text);
    font-family: 'Manrope', sans-serif;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090d12 0%, #0d131a 100%);
    border-right: 1px solid var(--border);
}

[data-testid="stHeader"] {
    background: rgba(6, 9, 13, 0.82);
    border-bottom: 1px solid rgba(31, 42, 54, 0.8);
    backdrop-filter: blur(10px);
}

h1, h2, h3, h4, h5, h6 {
    color: var(--headline);
    letter-spacing: -0.02em;
}

p, label, .stMarkdown, .stText, .stCaption {
    color: var(--text);
}

a {
    color: var(--primary);
}

.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
    background: rgba(12, 17, 23, 0.96);
    border: 1px solid var(--border);
    border-radius: 14px;
    color: var(--text);
    box-shadow: none;
}

.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--primary);
    box-shadow: 0 0 0 1px rgba(122, 162, 200, 0.16);
}

.stButton > button {
    border-radius: 14px;
    border: 1px solid var(--primary);
    background: linear-gradient(180deg, #6f9bc5 0%, var(--primary-strong) 100%);
    color: #081018;
    font-weight: 700;
    letter-spacing: 0.01em;
    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.28);
}

.stButton > button:hover {
    border-color: var(--primary-strong);
    background: linear-gradient(180deg, #88b0d3 0%, var(--primary) 100%);
}

.stAlert, div[data-testid="stNotification"], .stInfo, .stWarning, .stSuccess, .stError {
    border-radius: 14px;
}

div[data-testid="stDataFrame"], div[data-testid="stPlotlyChart"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 0.35rem;
    box-shadow: var(--shadow);
}

.page-shell {
    padding: 0.5rem 0 0.25rem;
}

.page-kicker {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.38rem 0.8rem;
    border-radius: 999px;
    background: rgba(18, 32, 49, 0.96);
    border: 1px solid rgba(122, 162, 200, 0.16);
    color: var(--primary);
    font-size: 0.74rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.page-title {
    margin: 0.95rem 0 0.35rem;
    font-size: clamp(2rem, 3vw, 2.9rem);
    font-weight: 800;
    color: var(--headline);
}

.page-subtitle {
    max-width: 760px;
    margin: 0;
    color: var(--text-muted);
    font-size: 1rem;
    line-height: 1.7;
}

div[data-testid="stChatInput"] textarea {
    background: rgba(12, 17, 23, 0.96);
    color: var(--text);
}
"""


def apply_professional_theme(extra_css: str = "") -> None:
    cleaned_extra_css = extra_css.replace("<style>", "").replace("</style>", "").strip()
    merged_css = BASE_THEME_CSS
    if cleaned_extra_css:
        merged_css = f"{BASE_THEME_CSS}\n{cleaned_extra_css}"
    st.markdown(f"<style>\n{merged_css}\n</style>", unsafe_allow_html=True)


def render_page_header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="page-shell">
          <div class="page-kicker">{kicker}</div>
          <h1 class="page-title">{title}</h1>
          <p class="page-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

"""Pure helpers for filtering/sorting ranked papers in the Search UI (no Streamlit)."""







def paper_composite_score(p: Paper) -> float:
    rr = getattr(p, "ranking_reasons", None) or {}
    c = rr.get("composite")
    if c is not None:
        try:
            return float(c)
        except (TypeError, ValueError):
            pass
    return max(
        float(getattr(p, "semantic_score", 0.0) or 0.0),
        float(getattr(p, "topic_relevance_score", 0.0) or 0.0),
        float(getattr(p, "hybrid_relevance_score", 0.0) or 0.0),
    )


def _paper_year(p: Paper) -> Optional[int]:
    """Extract publication year as int, or None."""
    date = getattr(p, "published_date", "") or ""
    if len(date) >= 4 and date[:4].isdigit():
        return int(date[:4])
    return None


def _paper_sources(p: Paper) -> list[str]:
    """Return list of individual source labels for a paper (handles comma-merged)."""
    src = getattr(p, "source", "") or ""
    return [s.strip() for s in src.split(",") if s.strip()]


def build_search_view(
    ranked_full: List[Paper],
    *,
    view_filter: str,
    sort_mode: str,
    min_relevance: float,
    # ── New filter parameters ────────────────────────────────────────────────
    source_filter: str = "all",       # "all" | "arxiv" | "openalex"
    year_min: Optional[int] = None,   # inclusive lower bound on publication year
    year_max: Optional[int] = None,   # inclusive upper bound on publication year
    min_citations: int = 0,           # minimum citation count
    author_query: str = "",           # substring match against author list
) -> List[Paper]:
    """
    Filter (all / top-K / high relevance) then apply additional facet filters
    (source, year, citations, author), then sort — for the results list.

    All filters operate on the FULL ranked_papers list that was returned by
    SearchAgent.  Nothing is pre-trimmed before this function runs.
    """
    base = list(ranked_full)

    # ── Focus / top-K filter ─────────────────────────────────────────────────
    if view_filter == "top10":
        base = base[:10]
    elif view_filter == "top20":
        base = base[:20]
    elif view_filter == "top50":
        base = base[:50]
    elif view_filter == "top100":
        base = base[:100]
    elif view_filter == "high_relevance":
        thr = max(0.0, min(1.0, float(min_relevance)))
        base = [p for p in base if paper_composite_score(p) >= thr]

    # ── Source filter ────────────────────────────────────────────────────────
    if source_filter and source_filter != "all":
        base = [p for p in base if source_filter in _paper_sources(p)]

    # ── Year range filter ────────────────────────────────────────────────────
    if year_min is not None:
        base = [p for p in base if (_paper_year(p) or 0) >= year_min]
    if year_max is not None:
        base = [p for p in base if (_paper_year(p) or 9999) <= year_max]

    # ── Minimum citations filter ─────────────────────────────────────────────
    if min_citations > 0:
        base = [p for p in base if (getattr(p, "citations", 0) or 0) >= min_citations]

    # ── Author text search ───────────────────────────────────────────────────
    if author_query.strip():
        aq = author_query.strip().lower()
        base = [
            p for p in base
            if any(aq in (a or "").lower() for a in (getattr(p, "authors", []) or []))
        ]

    # ── Sort ─────────────────────────────────────────────────────────────────
    if sort_mode == "relevance":
        pass   # already in ranker order
    elif sort_mode == "citations":
        base.sort(
            key=lambda p: (getattr(p, "citations", 0) or 0, paper_composite_score(p)),
            reverse=True,
        )
    elif sort_mode == "newest":
        base.sort(
            key=lambda p: (getattr(p, "published_date", "") or "", paper_composite_score(p)),
            reverse=True,
        )
    elif sort_mode == "oldest":
        base.sort(
            key=lambda p: (getattr(p, "published_date", "") or "", paper_composite_score(p)),
            reverse=False,
        )

    return base
def render_home():
    
    
    
    
    
    
    
    apply_professional_theme(
        """
        <style>
        .home-card {
            background: rgba(12, 17, 23, 0.9);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 1.3rem 1.4rem;
            box-shadow: var(--shadow);
            margin-top: 1rem;
        }
        .home-list {
            margin: 0.6rem 0 0;
            padding-left: 1.15rem;
            color: var(--text-muted);
            line-height: 1.9;
        }
        .home-list strong {
            color: var(--headline);
        }
        </style>
        """
    )
    
    render_page_header(
        "Research Workspace",
        "DATA2DASH",
        "A streamlined research intelligence interface for discovering papers, reviewing trends, and turning results into clear answers.",
    )
    
    st.markdown(
        """
        <div class="home-card">
          <p>Use the sidebar to move through the workflow:</p>
          <ul class="home-list">
            <li><strong>Search</strong> to find and rank academic papers.</li>
            <li><strong>Insights</strong> to review analytics, trends, and field distribution.</li>
            <li><strong>Chat</strong> to ask focused questions about the retrieved literature.</li>
          </ul>
          <p style="margin-top:1rem;color:var(--text-muted)">
            DATA2DASH combines search, enrichment, and lightweight analysis in one interface designed for research exploration.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.info("Start from the Search page to retrieve papers and generate insights.")
    
def render_search():
    
    
    
    
    
    
    
    apply_professional_theme(
        """
        <style>
        /* ── Control row ─────────────────────────────────────────────────────── */
        [data-testid="stHorizontalBlock"] {
            align-items: center;
            gap: 0.75rem;
        }
        [data-testid="stTextInput"], [data-testid="stSelectbox"], [data-testid="stNumberInput"] {
            width: 100%;
            min-width: 0;
        }
        [data-testid="stNumberInput"] { overflow: visible; }
        .stTextInput input,
        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] > div {
            height: 44px !important;
            min-height: 44px !important;
            border-radius: 14px !important;
        }
        [data-testid="stNumberInput"] button {
            height: 44px !important;
            min-height: 44px !important;
            width: 44px !important;
            border-radius: 12px !important;
        }
        [data-testid="stNumberInput"] div[data-baseweb="input"] { min-width: 150px; }
        @media (max-width: 900px) {
            [data-testid="stNumberInput"] div[data-baseweb="input"] { min-width: 130px; }
        }
        @media (max-width: 700px) {
            [data-testid="stHorizontalBlock"] { gap: 0.55rem; }
        }
    
        .chip {
            display: inline-block;
            border-radius: 999px;
            padding: 4px 11px;
            font-size: 0.76rem;
            margin: 2px 3px;
            font-weight: 700;
        }
        .chip-blue   { background: var(--primary-soft); border: 1px solid rgba(122,162,200,.18); color: var(--primary); }
        .chip-purple { background: #181727; border: 1px solid rgba(145,136,196,.18); color: #b2abd8; }
        .chip-green  { background: var(--accent-soft); border: 1px solid rgba(121,168,167,.16); color: var(--accent); }
        .chip-gray   { background: var(--surface-soft); border: 1px solid var(--border); color: var(--text-muted); }
    
        .metric-strip {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin: 0.8rem 0 1rem;
        }
        .metric-pill {
            background: rgba(12,17,23,.88);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 6px 14px;
            font-size: 0.8rem;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .metric-pill .mpv { font-weight: 800; color: var(--headline); }
        .metric-pill .mpa { font-weight: 700; color: var(--accent); }
    
        /* Pipeline funnel strip */
        .funnel-strip {
            display: flex;
            align-items: center;
            gap: 6px;
            flex-wrap: wrap;
            margin: 0.5rem 0 1rem;
            font-size: 0.78rem;
            color: var(--text-muted);
        }
        .funnel-val {
            font-weight: 800;
            font-size: 0.9rem;
            color: var(--headline);
        }
        .funnel-arrow { color: var(--border-strong); font-size: 1rem; }
        .funnel-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: .05em; }
    
        .info-card {
            background: rgba(12,17,23,.9);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 14px 18px;
            margin-bottom: 1rem;
            box-shadow: var(--shadow);
        }
        .filter-card {
            background: rgba(12,17,23,.85);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 14px 18px 10px;
            margin-bottom: 1rem;
        }
        .paper-card {
            background: rgba(12,17,23,.92);
            border: 1px solid rgba(31,42,54,.95);
            border-radius: 20px;
            padding: 20px 24px;
            margin-bottom: 14px;
            transition: border-color .2s, box-shadow .2s, transform .15s;
            box-shadow: var(--shadow);
        }
        .paper-card:hover {
            border-color: rgba(122,162,200,.24);
            box-shadow: 0 20px 38px rgba(0,0,0,.3);
            transform: translateY(-1px);
        }
        .paper-title { font-size: 1.05rem; font-weight: 700; line-height: 1.45; margin-bottom: 6px; }
        .paper-title a { color: var(--primary); text-decoration: none; }
        .paper-title a:hover { text-decoration: underline; }
        .paper-meta {
            font-size: .78rem;
            color: var(--text-muted);
            margin: 3px 0;
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }
        .paper-micro { margin-top: 6px; display: flex; gap: 10px; flex-wrap: wrap; font-size: .75rem; color: var(--text-muted); }
        .sep { color: var(--border-strong); }
        .paper-abstract {
            font-size: .85rem;
            color: var(--text);
            margin-top: 10px;
            line-height: 1.65;
            border-left: 3px solid var(--border);
            padding-left: 12px;
        }
        .rel-wrap { margin-top: 12px; }
        .rel-label { font-size: .72rem; color: var(--text-muted); margin-bottom: 3px; display: flex; justify-content: space-between; }
        .rel-bg { background: var(--surface-alt); border-radius: 999px; height: 8px; overflow: hidden; }
        .rel-fill { height: 8px; border-radius: 999px; }
        .src-badge { display: inline-block; padding: 2px 9px; border-radius: 12px; font-size: .71rem; font-weight: 600; }
        .src-arxiv          { background: #102119; color: var(--success); border: 1px solid rgba(104,179,143,.18); }
        .src-openalex       { background: #181727; color: #b2abd8; border: 1px solid rgba(145,136,196,.16); }
        .src-local_landmark { background: rgba(30,20,10,.9); color: #d4a96a; border: 1px solid rgba(212,169,106,.18); }
        .empty-state { text-align: center; padding: 4.5rem 1rem; color: var(--text-muted); }
        .empty-state .big-icon { font-size: 3.5rem; margin-bottom: 1rem; }
        .empty-state h3 { color: var(--headline); font-size: 1.3rem; margin-bottom: .5rem; }
        </style>
        """
    )
    
    render_page_header(
        "Search",
        "Hybrid Paper Search",
        "Search across academic sources with AI-assisted query expansion.  "
        "All retrieved papers are preserved — filter and sort the full set below.",
    )
    
    # ── Search bar ────────────────────────────────────────────────────────────────
    col_q, col_btn = st.columns([8, 2])
    with col_q:
        query = st.text_input(
            "Research topic",
            placeholder='',
            label_visibility="collapsed",
        )
    with col_btn:
        search_clicked = st.button("Search Papers", width="stretch", type="primary")
    
    # ── Session-state initialisation ──────────────────────────────────────────────
    _defaults = {
        "search_results":       None,
        "search_ui_page":       1,
        "search_ui_per_page":   int(getattr(settings, "DEFAULT_PAGE_SIZE", 25) or 25),
        # filter state
        "sf_source":            "all",
        "sf_year_min":          1990,
        "sf_year_max":          2025,
        "sf_min_citations":     0,
        "sf_author":            "",
        "sf_view_filter":       "all",
        "sf_sort_mode":         "relevance",
        "sf_min_rel":           0.35,
    }
    for k, v in _defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    
    # ── Run search ────────────────────────────────────────────────────────────────
    if search_clicked:
        if not query.strip():
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Expanding query, searching sources, ranking — please wait…"):
                try:
                    agent = SearchAgent()
                    results = agent.search(
                        query=query.strip(),
                        page=1,
                        per_page=int(st.session_state.search_ui_per_page),
                    )
                    st.session_state.search_results = results
                    st.session_state.search_ui_page = 1
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    st.stop()
    
    results = st.session_state.search_results
    
    # ═════════════════════════════════════════════════════════════════════════════
    # RESULTS SECTION
    # ═════════════════════════════════════════════════════════════════════════════
    if results:
        ranked_full = results.get("ranked_papers") or results.get("papers", [])
        accounting  = results.get("result_accounting", {}) or {}
        _pipe       = (accounting.get("pipeline") or {})
    
        # ── AI expansion info card ────────────────────────────────────────────────
        exp_chips = " ".join(
            f'<span class="chip chip-blue">{q}</span>'
            for q in results.get("expanded_queries", [])
        )
        kw_chips = " ".join(
            f'<span class="chip chip-purple">{kw}</span>'
            for kw in results.get("semantic_keywords", [])
        )
        st.markdown(
            f"""
            <div class="info-card">
              <div style="font-size:.72rem;font-weight:800;color:var(--primary);letter-spacing:.06em;text-transform:uppercase;margin-bottom:6px">
                AI Query Expansion
              </div>
              <div style="margin-bottom:6px">
                <span style="font-size:.72rem;color:var(--text-muted);font-weight:700">SEARCH VARIANTS &nbsp;</span>
                {exp_chips}
              </div>
              <div>
                <span style="font-size:.72rem;color:var(--text-muted);font-weight:700">SEMANTIC KEYWORDS &nbsp;</span>
                {kw_chips}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
        # ── Pipeline funnel strip ─────────────────────────────────────────────────
        _retrieved    = accounting.get("retrieved_count",    _pipe.get("raw_pool_count",     "—"))
        _dedup        = accounting.get("deduplicated_count", _pipe.get("after_dedup_count",  "—"))
        _filtered     = accounting.get("filtered_count",     _pipe.get("after_filter_count", "—"))
        _ranked       = accounting.get("final_ranked_count", len(ranked_full))
        _rank_cap_on  = bool(accounting.get("rank_cap_applied", False))
        _mrs          = accounting.get("max_results_setting", 0)
        _cap_note     = f" (capped at {_mrs} by MAX_RESULTS)" if _rank_cap_on else ""
        st.markdown(
            f"""
            <div class="funnel-strip">
              <div><span class="funnel-label">Retrieved</span><br><span class="funnel-val">{_retrieved}</span></div>
              <span class="funnel-arrow">→</span>
              <div><span class="funnel-label">After dedup</span><br><span class="funnel-val">{_dedup}</span></div>
              <span class="funnel-arrow">→</span>
              <div><span class="funnel-label">After filter</span><br><span class="funnel-val">{_filtered}</span></div>
              <span class="funnel-arrow">→</span>
              <div><span class="funnel-label">Ranked{_cap_note}</span><br><span class="funnel-val">{_ranked}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
        # ═════════════════════════════════════════════════════════════════════════
        # FILTER PANEL  (applies to full ranked_papers, not just the current page)
        # ═════════════════════════════════════════════════════════════════════════
        with st.expander("🔧 Filter & Sort results", expanded=True):
            st.markdown('<div class="filter-card">', unsafe_allow_html=True)
    
            # Row 1: focus + sort + source
            fr1 = st.columns([2, 2, 2])
            with fr1[0]:
                vf_options = ("all", "top20", "top50", "top100", "high_relevance")
                vf_labels  = {
                    "all":           "All ranked (paginate)",
                    "top20":         "Top 20 by rank",
                    "top50":         "Top 50 by rank",
                    "top100":        "Top 100 by rank",
                    "high_relevance":"High relevance (min score)",
                }
                view_filter = st.selectbox(
                    "Focus",
                    options=vf_options,
                    format_func=lambda x: vf_labels[x],
                    index=vf_options.index(st.session_state.sf_view_filter)
                        if st.session_state.sf_view_filter in vf_options else 0,
                    key="sf_view_filter",
                )
            with fr1[1]:
                sm_options = ("relevance", "citations", "newest", "oldest")
                sm_labels  = {
                    "relevance": "Relevance (ranker order)",
                    "citations": "Most cited",
                    "newest":    "Newest first",
                    "oldest":    "Oldest first",
                }
                sort_mode = st.selectbox(
                    "Sort by",
                    options=sm_options,
                    format_func=lambda x: sm_labels[x],
                    index=sm_options.index(st.session_state.sf_sort_mode)
                        if st.session_state.sf_sort_mode in sm_options else 0,
                    key="sf_sort_mode",
                )
            with fr1[2]:
                src_options = ("all", "arxiv", "openalex", "local_landmark")
                src_labels  = {
                    "all":            "All sources",
                    "arxiv":          "arXiv only",
                    "openalex":       "OpenAlex only",
                    "local_landmark": "Local landmarks only",
                }
                source_filter = st.selectbox(
                    "Source",
                    options=src_options,
                    format_func=lambda x: src_labels[x],
                    index=src_options.index(st.session_state.sf_source)
                        if st.session_state.sf_source in src_options else 0,
                    key="sf_source",
                )
    
            # Row 2: year range + min citations + author search
            fr2 = st.columns([3, 2, 3])
            with fr2[0]:
                import datetime
                this_year = datetime.date.today().year
                year_range = st.slider(
                    "Publication year range",
                    min_value=1950,
                    max_value=this_year,
                    value=(
                        int(st.session_state.sf_year_min or 1950),
                        int(st.session_state.sf_year_max or this_year),
                    ),
                    step=1,
                    key="_sf_year_slider",
                    help="Include only papers published within this year range.",
                )
                st.session_state.sf_year_min = year_range[0]
                st.session_state.sf_year_max = year_range[1]
    
            with fr2[1]:
                min_citations = st.number_input(
                    "Min citations",
                    min_value=0,
                    max_value=100_000,
                    value=int(st.session_state.sf_min_citations or 0),
                    step=10,
                    key="sf_min_citations",
                    help="Show only papers cited ≥ this many times.",
                )
    
            with fr2[2]:
                author_query = st.text_input(
                    "Author search (substring)",
                    value=st.session_state.sf_author or "",
                    placeholder='e.g. "Vaswani" or "Hinton"',
                    key="sf_author",
                    help="Filter to papers where at least one author name contains this text.",
                )
    
            # Row 3: min relevance slider (only visible for high_relevance focus)
            min_rel = 0.35
            if view_filter == "high_relevance":
                min_rel = st.slider(
                    "Min composite score",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state.sf_min_rel or 0.35),
                    step=0.05,
                    key="sf_min_rel",
                    help="Keep papers whose composite ranking score is ≥ this value.",
                )
    
            st.markdown('</div>', unsafe_allow_html=True)
    
        # Reset to page 1 when filters change
        _filter_sig = (
            view_filter, sort_mode, source_filter,
            year_range[0], year_range[1],
            min_citations, author_query,
            round(min_rel, 2) if view_filter == "high_relevance" else None,
        )
        if st.session_state.get("_filter_sig") != _filter_sig:
            st.session_state.search_ui_page = 1
            st.session_state["_filter_sig"] = _filter_sig
    
        # Build filtered + sorted view from FULL ranked pool
        view_list = build_search_view(
            ranked_full,
            view_filter=view_filter,
            sort_mode=sort_mode,
            min_relevance=min_rel,
            source_filter=source_filter,
            year_min=year_range[0] if year_range[0] > 1950 else None,
            year_max=year_range[1] if year_range[1] < this_year else None,
            min_citations=int(min_citations or 0),
            author_query=author_query or "",
        )
    
        # ── Pagination controls ───────────────────────────────────────────────────
        _pp_opts = (10, 20, 50, 100)
        pager = st.columns([3, 1, 2, 1])
        with pager[0]:
            per_page = st.selectbox(
                "Results per page",
                options=list(_pp_opts),
                index=_pp_opts.index(st.session_state.search_ui_per_page)
                if st.session_state.search_ui_per_page in _pp_opts else 1,
                key="search_per_page_sel",
            )
        if per_page != st.session_state.search_ui_per_page:
            st.session_state.search_ui_per_page = per_page
            st.session_state.search_ui_page = 1
    
        n_view   = len(view_list)
        max_page = max(1, (n_view + per_page - 1) // per_page) if n_view else 1
        st.session_state.search_ui_page = max(1, min(int(st.session_state.search_ui_page), max_page))
    
        with pager[1]:
            if st.button("◀ Prev", key="search_prev_pg",
                         disabled=st.session_state.search_ui_page <= 1 or n_view == 0):
                st.session_state.search_ui_page = max(1, st.session_state.search_ui_page - 1)
                st.rerun()
        with pager[2]:
            st.markdown(
                f'<div style="text-align:center;padding-top:10px;font-size:.9rem;'
                f'color:var(--headline);font-weight:700">'
                f'Page {st.session_state.search_ui_page} / {max_page}</div>',
                unsafe_allow_html=True,
            )
        with pager[3]:
            if st.button("Next ▶", key="search_next_pg",
                         disabled=st.session_state.search_ui_page >= max_page or n_view == 0):
                st.session_state.search_ui_page = min(max_page, st.session_state.search_ui_page + 1)
                st.rerun()
    
        start      = (st.session_state.search_ui_page - 1) * per_page
        papers     = view_list[start: start + per_page] if n_view else []
        ui_hi      = start + len(papers)
    
        # ── Status line + metric pills ────────────────────────────────────────────
        srcs      = results.get("source_counts", {})
        n_vars    = len(results.get("expanded_queries", []))
        prog      = accounting.get("progressive", {}) or {}
        src_acc   = accounting.get("sources", {}) or {}
        oa_acc    = src_acc.get("openalex", {}) or {}
        ax_acc    = src_acc.get("arxiv", {}) or {}
        _pipe_acc = accounting.get("pipeline", {}) or {}
    
        if n_view == 0:
            st.caption(
                f"**0** papers match the current filters (full ranked pool: **{len(ranked_full)}**). "
                "Try removing some filters or choosing **All ranked**."
            )
            st.info(
                "No papers match the active filters. "
                "Try resetting source, year range, citations, or author filters."
            )
        else:
            st.caption(
                f"Showing **{start+1}–{ui_hi}** of **{n_view}** filtered results "
                f"(full ranked pool: **{len(ranked_full)}** papers). "
                "Use **Focus** for top-K or high-relevance subsets; other filters narrow the full set."
            )
    
        st.markdown(
            f"""
            <div class="metric-strip">
              <div class="metric-pill">Full ranked pool <span class="mpv">{len(ranked_full)}</span></div>
              <div class="metric-pill">This view <span class="mpv">{n_view}</span></div>
              <div class="metric-pill">This page <span class="mpv">{len(papers)}</span>
                <span style="font-size:.72rem">(pg {st.session_state.search_ui_page}/{max_page})</span></div>
              <div class="metric-pill">arXiv <span class="mpv">{srcs.get("arxiv", 0)}</span></div>
              <div class="metric-pill">OpenAlex <span class="mpv">{srcs.get("openalex", 0)}</span></div>
              <div class="metric-pill">Variants <span class="mpv">{n_vars}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
        # ── Detailed accounting expander ──────────────────────────────────────────
        with st.expander("📊 Result accounting (pipeline breakdown)", expanded=False):
            thr  = _pipe_acc.get("filter_thresholds") or {}
            st.markdown(
                f"""
    **Pipeline funnel (this search run)**
    | Stage | Count |
    |-------|-------|
    | Raw pool (before dedup) | `{accounting.get("retrieved_count", _pipe_acc.get("raw_pool_count", "—"))}` |
    | After deduplication | `{accounting.get("deduplicated_count", _pipe_acc.get("after_dedup_count", "—"))}` |
    | After relevance filter | `{accounting.get("filtered_count", _pipe_acc.get("after_filter_count", "—"))}` |
    | Final ranked list | `{accounting.get("final_ranked_count", len(ranked_full))}` |
    | Shown on this page | `{len(papers)}` |
    
    **Filter thresholds applied**
    - Min semantic score: `{thr.get("min_semantic_score", "—")}` &nbsp; Min topic relevance: `{thr.get("min_topic_relevance", "—")}`
    - Min-keep fallback: `{thr.get("min_keep_fallback", "—")}` (papers rescued when pool < this)
    - Rank cap (`MAX_RESULTS`): `{"ON — capped at " + str(accounting.get("max_results_setting")) if accounting.get("rank_cap_applied") else "OFF — full list returned"}`
    - Queries fanned out in stage 1: `{_pipe_acc.get("stage1_query_count", "—")}` (FAN_OUT_ALL_VARIANTS=`{_pipe_acc.get("fan_out_all_variants", "—")}`)
    
    **Progressive fetch (multi-page arXiv + cursor OpenAlex)**
    - Enabled: `{prog.get("enabled", "—")}`
    - Extra from arXiv / OpenAlex: `{prog.get("arxiv_progressive_added", "—")}` / `{prog.get("openalex_progressive_added", "—")}` (OpenAlex pages: `{prog.get("openalex_pages_fetched", "—")}`)
    - Soft cap: `{prog.get("retrieval_soft_cap", "—")}`
    
    **Per source (in final ranked set)**
    | Source | Fetched (HTTP) | In final set |
    |--------|---------------|--------------|
    | arXiv | `{ax_acc.get("source_fetched_count", "—")}` | `{srcs.get("arxiv", 0)}` |
    | OpenAlex | `{oa_acc.get("source_fetched_count", "—")}` | `{srcs.get("openalex", 0)}` |
    | Local landmark | `{(src_acc.get("local_landmark") or {}).get("source_fetched_count", 0)}` | `{srcs.get("local_landmark", 0)}` |
    
    OpenAlex reported catalog total for query: `{oa_acc.get("source_total_matches", "unknown")}`
    OpenAlex status: `{oa_acc.get("status", "—")}` — cache hits: `{oa_acc.get("cache_hits", 0)}`, rate-limited: `{oa_acc.get("rate_limited", 0)}`
                """,
            )
    
        # ── Paper cards ───────────────────────────────────────────────────────────
        for paper in papers:
            sem        = getattr(paper, "semantic_score", 0.0)
            topic_rel  = getattr(paper, "topic_relevance_score", sem)
            pct        = int(sem * 100)
            topic_pct  = int(topic_rel * 100)
    
            src_raw    = paper.source or ""
            src_labels_map = {
                "arxiv":          ("arXiv",          "src-arxiv"),
                "openalex":       ("OpenAlex",        "src-openalex"),
                "local_landmark": ("Local landmark",  "src-local_landmark"),
            }
            src_parts = [s.strip() for s in src_raw.split(",") if s.strip()]
            src_badge_html = " ".join(
                f'<span class="src-badge {src_labels_map.get(s, (s, "src-arxiv"))[1]}">'
                f'{src_labels_map.get(s, (s, ""))[0]}</span>'
                for s in src_parts
            )
    
            authors_str = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                authors_str += f" +{len(paper.authors) - 3} more"
    
            snippet = (paper.abstract[:450] + "…") if len(paper.abstract) > 450 else paper.abstract
            tags_html = " ".join(
                f'<span class="chip chip-green">{t}</span>'
                for t in (paper.topic_tags or [])[:5]
            )
            inferred_tags_html = " ".join(
                f'<span class="chip chip-gray">{t}</span>'
                for t in (getattr(paper, "inferred_topic_tags", []) or [])[:4]
            )
    
            if pct >= 70:
                bar_color = "linear-gradient(90deg,#2f7a5f,#58a184)"
            elif pct >= 40:
                bar_color = "linear-gradient(90deg,#1f4b78,#3f719f)"
            else:
                bar_color = "linear-gradient(90deg,#7d8ca2,#a6b2c2)"
    
            cite_str  = f"{paper.citations:,}" if paper.citations else "—"
            date_full = paper.published_date or "N/A"
    
            rr = getattr(paper, "ranking_reasons", None) or {}
            rank_expl_html = ""
            if isinstance(rr, dict) and rr:
                parts = []
                for k in ("composite","hybrid_relevance","bm25_norm","embedding_cos",
                          "title_boost","citation_component","recency_component","survey_penalty"):
                    if k in rr and rr[k] is not None:
                        parts.append(f"{k.replace('_',' ')}={rr[k]}")
                if parts:
                    rank_expl_html = (
                        '<div class="paper-micro" style="opacity:.88;margin-top:4px">Ranking: '
                        + " · ".join(parts)
                        + "</div>"
                    )
    
            st.markdown(
                f"""
                <div class="paper-card">
                  <div class="paper-title">
                    <a href="{paper.url}" target="_blank">{paper.title}</a>
                  </div>
                  <div class="paper-meta">
                    <span>{authors_str or 'Unknown'}</span>
                    <span class="sep">|</span>
                    <span>{date_full}</span>
                    <span class="sep">|</span>
                    <span>{cite_str} citations</span>
                    <span class="sep">|</span>
                    {src_badge_html}
                  </div>
                  <div class="paper-micro">
                    <span>Topic relevance: <b style="color:var(--headline)">{topic_pct}%</b></span>
                    <span>Semantic score: <b style="color:var(--headline)">{pct}%</b></span>
                  </div>
                  {rank_expl_html}
                  {('<div style="margin-top:7px">' + tags_html + '</div>') if tags_html else ''}
                  {('<div style="margin-top:6px">' + inferred_tags_html + '</div>') if inferred_tags_html else ''}
                  <div class="paper-abstract">{snippet or '<i>No abstract available.</i>'}</div>
                  <div class="rel-wrap">
                    <div class="rel-label">
                      <span>Semantic Relevance</span>
                      <span style="color:var(--headline);font-weight:700">{pct}%</span>
                    </div>
                    <div class="rel-bg">
                      <div class="rel-fill" style="width:{pct}%;background:{bar_color}"></div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
        st.markdown(
            '<p style="font-size:.75rem;color:var(--text-muted);text-align:center;margin-top:1rem">'
            "Sources: ArXiv | OpenAlex | Powered by Groq LLM</p>",
            unsafe_allow_html=True,
        )
    
    # ── Empty / pre-search state ──────────────────────────────────────────────────
    elif not search_clicked:
        st.markdown(
            """
            <div class="empty-state">
              <div class="big-icon">🔬</div>
              <h3>Discover academic papers with hybrid AI search</h3>
              <p>
                Type any concept or acronym and the app will broaden the query, retrieve all
                available results, rank them, and let you filter the full set.
              </p>
              <p>
                <span class="chip chip-blue">RAG</span>
                <span class="chip chip-blue">Transformers</span>
                <span class="chip chip-blue">GAN</span>
                <span class="chip chip-blue">BERT</span>
                <span class="chip chip-blue">diffusion models</span>
                <span class="chip chip-blue">GNN</span>
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
def render_insights():
    
    
    
    
    
    
    
    apply_professional_theme(
        """
        <style>
        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--headline);
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin: 1.6rem 0 0.6rem;
            padding-bottom: 4px;
            border-bottom: 1px solid var(--border);
        }
        .metric-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 1.2rem;
        }
        .metric-card {
            flex: 1;
            min-width: 130px;
            background: rgba(12, 17, 23, 0.9);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 14px 18px;
            text-align: center;
            transition: border-color 0.2s;
            box-shadow: var(--shadow);
        }
        .metric-card:hover {
            border-color: rgba(122, 162, 200, 0.22);
        }
        .metric-card .val {
            font-size: 1.7rem;
            font-weight: 800;
            color: var(--primary);
        }
        .metric-card .lbl {
            font-size: 0.72rem;
            color: var(--text-muted);
            margin-top: 3px;
        }
        .trend-rising {
            color: var(--success);
        }
        .trend-declining {
            color: var(--danger);
        }
        .trend-stable {
            color: var(--text-muted);
        }
        .insight-card {
            background: linear-gradient(135deg, #0f151d 0%, #121a24 100%);
            border: 1px solid rgba(122, 162, 200, 0.16);
            border-radius: 20px;
            padding: 20px 24px;
            margin-bottom: 1.2rem;
            line-height: 1.7;
            color: var(--text);
            font-size: 0.95rem;
            box-shadow: var(--shadow);
        }
        .insight-label {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--primary);
            margin-bottom: 8px;
        }
        .cited-row {
            display: flex;
            align-items: center;
            gap: 12px;
            background: rgba(12, 17, 23, 0.9);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 12px 16px;
            margin-bottom: 8px;
            transition: border-color 0.2s;
            box-shadow: var(--shadow);
        }
        .cited-row:hover {
            border-color: rgba(122, 162, 200, 0.24);
        }
        .cited-rank {
            font-size: 1.4rem;
            font-weight: 800;
            color: var(--border-strong);
            min-width: 30px;
            text-align: center;
        }
        .cited-title {
            font-size: 0.88rem;
            font-weight: 600;
            color: var(--headline);
        }
        .cited-meta {
            font-size: 0.76rem;
            color: var(--text-muted);
            margin-top: 3px;
        }
        .cited-badge {
            margin-left: auto;
            background: #102119;
            border: 1px solid rgba(104, 179, 143, 0.16);
            border-radius: 8px;
            padding: 4px 10px;
            font-size: 0.78rem;
            font-weight: 700;
            color: var(--success);
            white-space: nowrap;
        }
        </style>
        """
    )
    
    render_page_header(
        "Analytics",
        "Research Insights",
        "Review publication momentum, key authors, citation leaders, and field-level signals in a more polished analysis view.",
    )
    
    results = st.session_state.get("search_results", None)
    
    if not results:
        st.markdown(
            """
            <div style="text-align:center;padding:5rem 0;color:var(--text-muted)">
              <div style="font-size:3rem">📊</div>
              <h3 style="color:var(--headline);margin-top:1rem">No data yet</h3>
              <p>Run a search on the Search page first, then come back here for insights.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()
    
    analytics = results["analytics"]
    query = results.get("query", "")
    
    st.markdown('<div class="section-title">Summary</div>', unsafe_allow_html=True)
    
    trend_raw = analytics.trend_status
    trend_class = (
        "trend-rising" if "Rising" in trend_raw else
        "trend-declining" if "Declining" in trend_raw else
        "trend-stable"
    )
    
    st.markdown(
        f"""
        <div class="metric-grid">
          <div class="metric-card">
            <div class="val">{analytics.total_papers}</div>
            <div class="lbl">Total Papers</div>
          </div>
          <div class="metric-card">
            <div class="val">{analytics.papers_last_30_days}</div>
            <div class="lbl">Published Last 30 Days</div>
          </div>
          <div class="metric-card">
            <div class="val">{analytics.avg_citations:,.1f}</div>
            <div class="lbl">Avg Citations / Paper</div>
          </div>
          <div class="metric-card">
            <div class="val">{analytics.max_citations:,}</div>
            <div class="lbl">Most Cited Paper</div>
          </div>
          <div class="metric-card">
            <div class="val"><span class="{trend_class}">{trend_raw}</span></div>
            <div class="lbl">Publication Trend</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    if analytics.llm_insight:
        st.markdown('<div class="section-title">AI Research Landscape Summary</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="insight-card">
              <div class="insight-label">Generated for "{query}"</div>
              {analytics.llm_insight}
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    col_left, col_right = st.columns(2, gap="large")
    
    with col_left:
        st.markdown('<div class="section-title">Monthly Publication Volume</div>', unsafe_allow_html=True)
        if analytics.monthly_counts:
            df_trend = pd.DataFrame(list(analytics.monthly_counts.items()), columns=["Month", "Papers"])
            fig_trend = px.line(df_trend, x="Month", y="Papers", markers=True)
            fig_trend.update_traces(line_color="#1f4b78", marker_color="#1f4b78")
            fig_trend.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title=None,
            )
            st.plotly_chart(fig_trend, width="stretch")
        else:
            st.info("Not enough date data to show trend.")
    
        st.markdown('<div class="section-title">Papers by Publication Year</div>', unsafe_allow_html=True)
        if analytics.year_distribution:
            df_years = pd.DataFrame(list(analytics.year_distribution.items()), columns=["Year", "Papers"])
            fig_years = px.bar(df_years, x="Year", y="Papers")
            fig_years.update_traces(marker_color="#6b7f99")
            fig_years.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title=None,
            )
            st.plotly_chart(fig_years, width="stretch")
        else:
            st.info("No year data available.")
    
        st.markdown('<div class="section-title">🌐 Source Distribution</div>',
                    unsafe_allow_html=True)
        if analytics.source_distribution:
            df_sources = pd.DataFrame(
                list(analytics.source_distribution.items()),
                columns=["Source", "Count"]
            )
            fig_sources = px.pie(df_sources, names="Source", values="Count", hole=0.45)
            fig_sources.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_sources, width="stretch")
        else:
            st.info("No source distribution available.")
    
    with col_right:
        st.markdown('<div class="section-title">Top Transformer Subtopics</div>', unsafe_allow_html=True)
        if analytics.subtopic_distribution:
            df_subtopics = (
                pd.DataFrame(list(analytics.subtopic_distribution.items()), columns=["Subtopic", "Count"])
                .sort_values("Count", ascending=True)
                .tail(10)
            )
            fig_sub = px.bar(df_subtopics, x="Count", y="Subtopic", orientation="h")
            fig_sub.update_traces(marker_color="#1f4b78")
            fig_sub.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title=None,
            )
            st.plotly_chart(fig_sub, width="stretch")
        else:
            st.info("No subtopic distribution available.")
    
        st.markdown('<div class="section-title">Research Field Distribution</div>', unsafe_allow_html=True)
        if analytics.field_distribution:
            df_fields = (
                pd.DataFrame(list(analytics.field_distribution.items()), columns=["Field", "Count"])
                .sort_values("Count", ascending=True)
                .tail(10)
            )
            fig_fields = px.bar(df_fields, x="Count", y="Field", orientation="h")
            fig_fields.update_traces(marker_color="#2f6b73")
            fig_fields.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title=None,
            )
            st.plotly_chart(fig_fields, width="stretch")
        else:
            st.info("No field/topic data available.")
    
        st.markdown('<div class="section-title">Author Impact (count + citations)</div>', unsafe_allow_html=True)
        if analytics.top_author_impact:
            df_authors = pd.DataFrame(analytics.top_author_impact).sort_values("impact_score", ascending=True)
            fig_authors = px.bar(df_authors, x="impact_score", y="author", orientation="h")
            fig_authors.update_traces(marker_color="#8a673e")
            fig_authors.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title=None,
            )
            st.plotly_chart(fig_authors, width="stretch")
        else:
            st.info("No author data available.")
    
        st.markdown('<div class="section-title">Venue Distribution</div>', unsafe_allow_html=True)
        if analytics.venue_distribution:
            df_venues = (
                pd.DataFrame(list(analytics.venue_distribution.items()), columns=["Venue", "Count"])
                .sort_values("Count", ascending=True)
                .tail(10)
            )
            fig_venues = px.bar(df_venues, x="Count", y="Venue", orientation="h")
            fig_venues.update_traces(marker_color="#2f6b73")
            fig_venues.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title=None,
            )
            st.plotly_chart(fig_venues, width="stretch")
        else:
            st.info("No venue distribution available.")
    
    st.markdown('<div class="section-title">Most Cited Papers in Results</div>', unsafe_allow_html=True)
    
    if analytics.top_cited_papers:
        medal = ["🥇", "🥈", "🥉", "4", "5"]
        for i, p in enumerate(analytics.top_cited_papers):
            authors_str = ", ".join(p["authors"]) + (" et al." if len(p["authors"]) >= 2 else "")
            src_map = {
                "arxiv": "arXiv",
                "semantic_scholar": "Semantic Scholar",
                "openalex": "OpenAlex",
            }
            src_label = src_map.get(p.get("source", ""), p.get("source", ""))
            st.markdown(
                f"""
                <div class="cited-row">
                  <div class="cited-rank">{medal[i] if i < len(medal) else i + 1}</div>
                  <div style="flex:1;min-width:0">
                    <div class="cited-title">
                      <a href="{p['url']}" target="_blank" style="color:var(--headline);text-decoration:none">{p['title']}</a>
                    </div>
                    <div class="cited-meta">
                      {authors_str or 'Unknown'} | {p['year']} | {src_label} | {p.get('venue', 'Unknown venue')}
                    </div>
                  </div>
                  <div class="cited-badge">{p['citations']:,} citations</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No citation data available.")
    
    st.markdown('<div class="section-title">📈 Year Trends by Subtopic</div>',
                unsafe_allow_html=True)
    if analytics.year_subtopic_trends:
        rows = []
        for subtopic, year_map in analytics.year_subtopic_trends.items():
            for year, count in year_map.items():
                rows.append({"Subtopic": subtopic, "Year": year, "Papers": count})
        if rows:
            df_subtrend = pd.DataFrame(rows)
            fig_subtrend = px.line(
                df_subtrend,
                x="Year",
                y="Papers",
                color="Subtopic",
                markers=True
            )
            fig_subtrend.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title=None,
            )
            st.plotly_chart(fig_subtrend, width="stretch")
    else:
        st.info("No subtopic trend data available.")
    
    st.markdown('<div class="section-title">Top Research Keywords / Fields</div>', unsafe_allow_html=True)
    if analytics.top_keywords:
        df_kw = pd.DataFrame(analytics.top_keywords, columns=["Keyword / Field", "# Papers"])
        st.dataframe(
            df_kw,
            width="stretch",
            hide_index=True,
            column_config={
                "# Papers": st.column_config.ProgressColumn(
                    "# Papers",
                    min_value=0,
                    max_value=int(df_kw["# Papers"].max()),
                    format="%d",
                )
            },
        )
    else:
        st.info("No keyword data available.")
    
def render_chat():
    
    
    
    
    
    
    
    apply_professional_theme(
        """
        <style>
        .chat-note {
            background: rgba(12, 17, 23, 0.88);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
            color: var(--text-muted);
            box-shadow: var(--shadow);
        }
        div[data-testid="stChatMessage"] {
            border-radius: 14px;
            border: 1px solid var(--border);
            padding: 0.25rem 0.65rem;
            background: rgba(12, 17, 23, 0.82);
        }
        </style>
        """
    )
    
    render_page_header(
        "Conversation",
        "Chat With Your Results",
        "Ask grounded questions about the papers you retrieved and get concise answers anchored in the current result set.",
    )
    
    results = st.session_state.get("search_results", None)
    
    if not results:
        st.warning("No search results found. Please run a search first.")
    else:
        st.markdown(
            '<div class="chat-note">Ask questions about the papers you just searched.</div>',
            unsafe_allow_html=True,
        )
    
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
    
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
    
        user_question = st.chat_input("Ask about these papers...")
    
        if user_question:
            papers = results.get("ranked_papers") or results.get("papers", [])
            context_parts = []
            for p in papers[:8]:
                context_parts.append(
                    f"Title: {p.title}\n"
                    f"Authors: {', '.join(p.authors)}\n"
                    f"Published: {p.published_date}\n"
                    f"Citations: {p.citations}\n"
                    f"Abstract: {p.abstract[:600]}"
                )
            context = "\n\n---\n\n".join(context_parts)
    
            system_prompt = (
                "You are a helpful research assistant. "
                "Answer the user's questions based on the following academic papers:\n\n"
                f"{context}\n\n"
                "Be concise, accurate, and cite paper titles when relevant."
            )
    
            st.session_state.chat_history.append({"role": "user", "content": user_question})
    
            with st.chat_message("user"):
                st.write(user_question)
    
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        llm = ChatGroq(
                            api_key=settings.GROQ_API_KEY,
                            model=settings.CHAT_MODEL,
                        )
                        messages = [SystemMessage(content=system_prompt)]
                        for msg in st.session_state.chat_history[:-1]:
                            if msg["role"] == "user":
                                messages.append(HumanMessage(content=msg["content"]))
                            else:
                                messages.append(SystemMessage(content=f"Assistant: {msg['content']}"))
                        messages.append(HumanMessage(content=user_question))
    
                        response = llm.invoke(messages)
                        answer = response.content
                        st.write(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Error calling LLM: {e}")
    

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Search", "Insights", "Chat"])
    
    if page == "Home":
        render_home()
    elif page == "Search":
        render_search()
    elif page == "Insights":
        render_insights()
    elif page == "Chat":
        render_chat()

if __name__ == "__main__":
    main()