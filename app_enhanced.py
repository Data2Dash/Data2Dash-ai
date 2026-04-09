"""
app_enhanced.py
===============
Main Streamlit UI for the Multimodal RAG Research Assistant.

This version focuses on:
- stable upload/process flow
- clear document state handling
- focused rendering for equations / tables / figures
- image upload support
- pasted text support
- compatibility with EnhancedRAGSystem
- cleaner table rendering without affecting equation rendering
"""

from __future__ import annotations

import os
import re
import uuid
import tempfile
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from enhanced_rag_system import EnhancedRAGSystem, EnhancedRAGConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------

def init_session_state():
    defaults = [
        ("rag_system", None),
        ("system_initialized", False),
        ("document_loaded", False),
        ("doc_info", None),
        ("chat_history", []),
        ("pending_query", ""),
        ("processing_error", None),
        ("uploaded_pdf_name", None),
        ("uploaded_image_name", None),
        ("uploaded_image_path", None),
        ("use_uploaded_image_next", False),
        ("last_result", None),
        ("pasted_content_title", "Pasted Content"),
        ("pasted_content_text", ""),
    ]
    for key, value in defaults:
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def build_config() -> EnhancedRAGConfig:
    return EnhancedRAGConfig(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        groq_model="llama-3.3-70b-versatile",
        groq_vision_model="llama-3.2-11b-vision-preview",
        chunk_size=1200,
        chunk_overlap=150,
        top_k=6,
        use_multiquery=True,
        use_self_rag_validation=True,
        strict_grounding=True,
        temp_dir="temp_data",
        exports_dir="exports",
        debug=False,
    )


def initialize_system(api_key: str) -> bool:
    try:
        config = build_config()
        st.session_state.rag_system = EnhancedRAGSystem(config=config, groq_api_key=api_key.strip())
        st.session_state.system_initialized = True
        st.session_state.processing_error = None
        logger.info("✅ System initialized")
        return True
    except Exception as e:
        st.session_state.system_initialized = False
        st.session_state.rag_system = None
        st.session_state.processing_error = str(e)
        st.error(f"❌ Failed to initialize system: {e}")
        return False


def save_uploaded_file(uploaded_file, suffix: Optional[str] = None) -> Path:
    suffix = suffix or Path(uploaded_file.name).suffix
    tmp_dir = Path(tempfile.gettempdir()) / "multimodal_rag_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / f"{uuid.uuid4().hex}{suffix}"
    out_path.write_bytes(uploaded_file.getbuffer())
    return out_path


def process_document(pdf_file) -> Optional[Dict[str, Any]]:
    try:
        if not pdf_file:
            st.session_state.processing_error = "No PDF file provided."
            return None
        if not st.session_state.get("rag_system"):
            st.session_state.processing_error = "System not initialized."
            return None

        p = save_uploaded_file(pdf_file, suffix=".pdf")
        with st.spinner("🔄 Processing document..."):
            summary = st.session_state.rag_system.process_document(str(p))

        st.session_state.document_loaded = True
        st.session_state.doc_info = st.session_state.rag_system.get_document_info()
        st.session_state.processing_error = None
        logger.info("✅ Document processed successfully")
        return summary
    except Exception as e:
        st.session_state.document_loaded = False
        st.session_state.doc_info = None
        st.session_state.processing_error = str(e)
        logger.exception("❌ Document processing failed")
        st.error(f"❌ Document processing failed: {e}")
        return None


def process_uploaded_image(image_file) -> Optional[str]:
    try:
        if not image_file:
            return None
        p = save_uploaded_file(image_file, suffix=Path(image_file.name).suffix or ".png")
        st.session_state.uploaded_image_path = str(p)
        st.session_state.uploaded_image_name = image_file.name
        if st.session_state.get("rag_system"):
            st.session_state.rag_system.set_uploaded_image(str(p))
        return str(p)
    except Exception as e:
        st.error(f"❌ Failed to save image: {e}")
        return None


def process_pasted_content(title: str, text: str) -> bool:
    try:
        if not text.strip():
            st.warning("Please paste some text first.")
            return False

        st.session_state.document_loaded = True
        st.session_state.doc_info = {
            "title": title.strip() or "Pasted Content",
            "filename": title.strip() or "Pasted Content",
            "num_pages": 1,
            "display_equation_count": 0,
            "inline_math_count": 0,
            "table_count": 0,
            "figure_count": 0,
            "equations": [],
            "tables": [],
            "figures": [],
            "document_metadata": {
                "title": title.strip() or "Pasted Content",
                "abstract": text[:1000],
            },
        }
        st.session_state.pasted_content_title = title.strip() or "Pasted Content"
        st.session_state.pasted_content_text = text
        st.success("✅ Pasted content loaded.")
        return True
    except Exception as e:
        st.error(f"❌ Failed to process pasted content: {e}")
        return False


def answer_query(query: str, mode: str = "standard") -> Dict[str, Any]:
    rag_system = st.session_state.get("rag_system")
    if not rag_system:
        return {
            "answer": "System not initialized.",
            "sources": [],
            "equations": [],
            "tables": [],
            "figures": [],
            "validated": False,
            "mode": "error",
        }

    if st.session_state.get("pasted_content_text") and not getattr(rag_system, "current_document", None):
        text = st.session_state.pasted_content_text
        answer = f"{text[:1200].strip()}\n\n(Source: Page 1)" if text.strip() else "The document does not contain this information."
        return {
            "answer": answer,
            "sources": ["(Source: Page 1)"],
            "equations": [],
            "tables": [],
            "figures": [],
            "validated": True,
            "mode": "pasted",
        }

    try:
        use_image = bool(st.session_state.get("use_uploaded_image_next", False))
        result = rag_system._query_async(
            user_query=query,
            mode=mode,
            include_sources=True,
            image_mode=use_image,
        )
        st.session_state.use_uploaded_image_next = False
        return result
    except TypeError:
        result = rag_system.query(
            user_query=query,
            mode=mode,
            include_sources=True,
            image_mode=bool(st.session_state.get("use_uploaded_image_next", False)),
        )
        st.session_state.use_uploaded_image_next = False
        return result
    except Exception as e:
        logger.exception("❌ Query failed")
        return {
            "answer": f"⚠️ Error generating response: {e}",
            "sources": [],
            "equations": [],
            "tables": [],
            "figures": [],
            "validated": False,
            "mode": "error",
        }


# -----------------------------------------------------------------------------
# Rendering helpers
# -----------------------------------------------------------------------------

def render_document_info(doc_info: Dict[str, Any]):
    title = doc_info.get("title") or doc_info.get("filename", "Unknown document")
    pages = doc_info.get("num_pages", 0)
    display_eq = doc_info.get("display_equation_count", 0)
    tbl = doc_info.get("table_count", 0)
    fig = doc_info.get("figure_count", 0)
    arxiv_id = doc_info.get("arxiv_id") or (doc_info.get("document_metadata") or {}).get("arxiv_id", "")

    st.markdown(f"### 📄 {title}")
    meta_bits = [f"🗒️ {pages} pages"]
    if arxiv_id:
        meta_bits.append(f"arXiv: {arxiv_id}")
    st.markdown(" · ".join(meta_bits))
    c1, c2, c3 = st.columns(3)
    c1.metric("Eq", display_eq)
    c2.metric("Tbl", tbl)
    c3.metric("Fig", fig)


def _normalize_table_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if text in {"nan", "None"}:
        return ""
    return text


def _drop_probable_index_columns(rows: List[List[str]]) -> List[List[str]]:
    if not rows or len(rows) < 2:
        return rows
    work = [list(r) for r in rows]
    changed = True
    while changed and work and len(work[0]) > 1:
        changed = False
        col0 = [(_normalize_table_cell(r[0]) if r else "") for r in work]
        data_col0 = col0[1:]
        if data_col0 and sum(v.isdigit() for v in data_col0) >= max(2, int(len(data_col0) * 0.6)):
            work = [r[1:] for r in work]
            changed = True
            continue
        if col0[0] == "" and sum(1 for v in data_col0 if v == "") >= int(len(data_col0) * 0.7):
            work = [r[1:] for r in work]
            changed = True
    return work


def _clean_table_rows(rows: List[List[str]]) -> List[List[str]]:
    if not rows:
        return rows
    max_len = max(len(r) for r in rows)
    cleaned = []
    for r in rows:
        row = [_normalize_table_cell(c) for c in r]
        if len(row) < max_len:
            row += [""] * (max_len - len(row))
        cleaned.append(row)
    cleaned = _drop_probable_index_columns(cleaned)
    return cleaned


def _markdown_table_to_rows(markdown: str) -> Optional[List[List[str]]]:
    if not markdown or "|" not in markdown:
        return None

    raw_lines = [ln.strip() for ln in markdown.splitlines() if ln.strip() and "|" in ln]
    if len(raw_lines) < 2:
        return None

    rows: List[List[str]] = []
    for line in raw_lines:
        cells = [_normalize_table_cell(c) for c in line.strip().strip("|").split("|")]
        if not cells:
            continue
        if all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) for c in cells if c != ""):
            continue
        rows.append(cells)

    if len(rows) < 2:
        return None
    return _clean_table_rows(rows)


def _raw_text_table_to_rows(raw_text: str) -> Optional[List[List[str]]]:
    if not raw_text:
        return None
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None

    parsed: List[List[str]] = []
    for ln in lines:
        if "|" in ln:
            cells = [_normalize_table_cell(c) for c in ln.strip().strip("|").split("|")]
        else:
            cells = [_normalize_table_cell(c) for c in re.split(r"\s{2,}|\t+", ln) if _normalize_table_cell(c)]
        if len(cells) >= 2:
            parsed.append(cells)

    if len(parsed) < 2:
        return None
    return _clean_table_rows(parsed)


def _html_table_to_rows(html_table: str) -> Optional[List[List[str]]]:
    if not html_table or pd is None:
        return None
    try:
        dfs = pd.read_html(html_table)
        if not dfs:
            return None
        df = dfs[0].fillna("")
        header = [_normalize_table_cell(c) for c in df.columns.tolist()]
        rows = [header] + [[_normalize_table_cell(v) for v in row] for row in df.astype(str).values.tolist()]
        return _clean_table_rows(rows)
    except Exception:
        return None


def _rows_quality(rows: List[List[str]]) -> float:
    if not rows or len(rows) < 2:
        return 0.0
    header = rows[0]
    data_rows = rows[1:]
    non_empty_header = sum(1 for c in header if c)
    if non_empty_header == 0:
        return 0.0
    non_empty_cells = sum(1 for r in data_rows for c in r if c)
    total_cells = max(1, sum(len(r) for r in data_rows))
    density = non_empty_cells / total_cells
    header_quality = non_empty_header / max(1, len(header))
    return round((density * 0.6) + (header_quality * 0.4), 3)


def _rows_to_dataframe(rows: List[List[str]]):
    if pd is None or not rows or len(rows) < 2:
        return None
    header = [_normalize_table_cell(c) or f"Col {i + 1}" for i, c in enumerate(rows[0])]
    seen = {}
    final_header = []
    for col in header:
        count = seen.get(col, 0)
        seen[col] = count + 1
        final_header.append(col if count == 0 else f"{col}_{count + 1}")
    data_rows = rows[1:]
    try:
        return pd.DataFrame(data_rows, columns=final_header)
    except Exception:
        return None


def _render_best_table_representation(tb: Dict[str, Any]) -> bool:
    parsed_rows = None
    if tb.get("parsed_data"):
        parsed = tb.get("parsed_data")
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            if pd is not None:
                try:
                    df = pd.DataFrame(parsed)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    return True
                except Exception:
                    pass
        elif isinstance(parsed, list) and parsed and isinstance(parsed[0], list):
            parsed_rows = _clean_table_rows(parsed)

    if parsed_rows is None:
        parsed_rows = _html_table_to_rows(tb.get("html_table") or "")
    if parsed_rows is None:
        parsed_rows = _markdown_table_to_rows(tb.get("markdown") or "")
    if parsed_rows is None:
        parsed_rows = _raw_text_table_to_rows(tb.get("raw_text") or "")

    if parsed_rows:
        quality = _rows_quality(parsed_rows)
        df = _rows_to_dataframe(parsed_rows)
        if df is not None and quality >= 0.55:
            st.dataframe(df, use_container_width=True, hide_index=True)
            return True
        if df is not None and quality >= 0.35:
            st.table(df)
            return True

    markdown = (tb.get("markdown") or "").strip()
    raw_text = (tb.get("raw_text") or "").strip()
    if markdown:
        st.markdown(markdown)
        return True
    if raw_text:
        st.code(raw_text)
        return True
    return False


def render_equations(equations: List[Dict[str, Any]]):
    if not equations:
        return

    st.markdown("### 📐 Mathematical Equations")
    for eq in equations:
        label = eq.get("label") or f"Equation {eq.get('global_number', '?')}"
        page = eq.get("page_number", "?")
        latex = eq.get("normalized_latex") or eq.get("latex") or ""
        raw_text = eq.get("raw_text") or eq.get("text") or ""

        st.markdown(f"**📐 {label}**")
        st.caption(f"Page {page}")

        if latex:
            try:
                st.latex(latex)
            except Exception:
                if raw_text:
                    st.code(raw_text)
        elif raw_text:
            st.code(raw_text)

        if raw_text:
            with st.expander("Raw extracted equation"):
                st.code(raw_text)


def render_tables(tables: List[Dict[str, Any]]):
    if not tables:
        return

    st.markdown("### 📊 Tables")
    for tb in tables:
        label = tb.get("label") or f"Table {tb.get('global_number', '?')}"
        page = tb.get("page_number", "?")
        caption = tb.get("caption") or label
        raw_text = tb.get("raw_text") or ""
        markdown = tb.get("markdown") or ""

        st.markdown(f"**📊 {label} — {caption}**")
        st.caption(f"Page {page}")

        rendered = _render_best_table_representation(tb)
        if not rendered:
            if markdown:
                st.markdown(markdown)
            elif raw_text:
                st.code(raw_text)
            else:
                st.info("No structured table representation available.")

        if raw_text and raw_text.strip() and raw_text.strip() != markdown.strip():
            with st.expander("Raw extracted table"):
                st.code(raw_text)


def render_figures(figures: List[Dict[str, Any]]):
    if not figures:
        return

    st.markdown("### 🖼️ Figures")
    for fig in figures:
        label = fig.get("label") or f"Figure {fig.get('global_number', '?')}"
        page = fig.get("page_number", "?")
        caption = fig.get("caption") or label
        image_path = fig.get("image_path") or ""

        st.markdown(f"**🖼️ {label} — {caption}**")
        st.caption(f"Page {page}")

        if image_path and os.path.exists(image_path):
            st.image(image_path, use_container_width=True)
        else:
            st.markdown(caption)


def render_sources(sources: List[str]):
    if not sources:
        return
    unique_sources = []
    for src in sources:
        if src not in unique_sources:
            unique_sources.append(src)
    st.markdown("### 📚 Sources")
    st.markdown(" ".join(unique_sources))


def render_response(result: Dict[str, Any]):
    validated = result.get("validated", False)
    answer = result.get("answer", "").strip()
    equations = result.get("equations", []) or []
    tables = result.get("tables", []) or []
    figures = result.get("figures", []) or []
    sources = result.get("sources", []) or []

    if validated:
        st.markdown("✅ Self-RAG validation passed")
    else:
        st.markdown("⚠️ Validation unavailable or failed")

    if answer:
        st.markdown(answer)
    if equations:
        render_equations(equations)
    if tables:
        render_tables(tables)
    if figures:
        render_figures(figures)
    if sources:
        render_sources(sources)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

with st.sidebar:
    st.markdown("# 🔬 RAG Assistant")
    st.caption("Multimodal PDF Research System V9.0")

    st.markdown("## ⚙️ Configuration")
    api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))

    if st.button("Initialize System", use_container_width=True, type="primary"):
        if not api_key.strip():
            st.warning("Please enter your Groq API key.")
        else:
            if initialize_system(api_key):
                st.success("✅ System initialized.")

    st.markdown("---")
    st.markdown("## 📄 Document")

    pdf_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_uploader")

    if pdf_file is not None:
        if st.session_state.uploaded_pdf_name != pdf_file.name:
            st.session_state.uploaded_pdf_name = pdf_file.name
            st.session_state.document_loaded = False
            st.session_state.doc_info = None
            st.session_state.processing_error = None

    if pdf_file and st.session_state.get("rag_system"):
        st.caption(f"Selected: {pdf_file.name}")
        if st.button("Process Document", use_container_width=True, type="primary", key="process_pdf_btn"):
            result = process_document(pdf_file)
            if result:
                st.success("✅ Processed!")
                st.rerun()

    if st.session_state.get("processing_error"):
        st.error(f"❌ {st.session_state.processing_error}")

    st.markdown("---")
    st.markdown("## 🖼️ Upload Image")
    image_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"], key="image_uploader")

    if image_file is not None:
        if st.session_state.uploaded_image_name != image_file.name:
            image_path = process_uploaded_image(image_file)
            if image_path:
                st.success("✅ Image uploaded.")

    if st.session_state.get("uploaded_image_path"):
        st.checkbox("Use uploaded image in next question", key="use_uploaded_image_next")

    st.markdown("---")
    st.markdown("## 📋 Paste Input")
    pasted_title = st.text_input("Pasted content title", value=st.session_state.get("pasted_content_title", "Pasted Content"))
    pasted_text = st.text_area(
        "Paste text or PDF excerpt",
        value=st.session_state.get("pasted_content_text", ""),
        height=140,
        placeholder="Paste text here with Ctrl+V...",
    )

    if st.button("Load Pasted Content", use_container_width=True):
        process_pasted_content(pasted_title, pasted_text)

    st.markdown("---")
    if st.session_state.get("doc_info"):
        render_document_info(st.session_state.doc_info)


# -----------------------------------------------------------------------------
# Main area
# -----------------------------------------------------------------------------

st.title("💬 Conversation")

if not st.session_state.system_initialized:
    st.info("👈 Enter your API key and click **Initialize System**.")
    st.stop()

if not st.session_state.document_loaded:
    if st.session_state.get("processing_error"):
        st.warning("⚠️ The PDF was uploaded, but processing did not complete.")
    else:
        st.info("👈 Upload a PDF and click **Process Document**")
    st.stop()

quick_actions = st.columns(4)
with quick_actions[0]:
    if st.button("Show all equations", use_container_width=True):
        st.session_state.pending_query = "Show all equations"
with quick_actions[1]:
    if st.button("Show all tables", use_container_width=True):
        st.session_state.pending_query = "Show all tables"
with quick_actions[2]:
    if st.button("Show all figures", use_container_width=True):
        st.session_state.pending_query = "Show all figures"
with quick_actions[3]:
    if st.button("Summarize the main findings", use_container_width=True):
        st.session_state.pending_query = "Summarize the main findings"

for item in st.session_state.chat_history:
    with st.chat_message(item["role"]):
        if item["role"] == "assistant" and isinstance(item["content"], dict):
            render_response(item["content"])
        else:
            st.markdown(str(item["content"]))

prompt = st.chat_input("Ask a question about the document...")

if st.session_state.pending_query and not prompt:
    prompt = st.session_state.pending_query
    st.session_state.pending_query = ""

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            q = prompt.lower()
            if any(k in q for k in ["explain", "why", "how"]):
                mode = "explanation"
            elif any(k in q for k in ["analyze", "compare", "difference"]):
                mode = "analysis"
            else:
                mode = "standard"

            result = answer_query(prompt, mode=mode)
            render_response(result)
            st.session_state.last_result = result

    st.session_state.chat_history.append({"role": "assistant", "content": result})
