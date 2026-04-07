"""
app/ui/streamlit_ui.py — clean, simple UI.
"""
import logging
import streamlit as st

from app.core.config import get_settings
from app.core.exceptions import AppError, LLMServiceError, ReportGenerationError, PDFExtractionError
from app.services.pipeline_service import run_pipeline

logger = logging.getLogger(__name__)


def render_app():
    settings = get_settings()
    st.set_page_config(page_title=settings.app_title, page_icon="📄", layout="centered")

    st.title("📄 Research Paper Summarizer")
    st.caption("Upload a PDF — get a clean, section-by-section summary report.")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_…",
                                help="Free at https://console.groq.com")
        model = st.selectbox("Model", [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
        ], help="llama-3.3-70b = best quality · llama-3.1-8b = saves tokens")
        st.divider()
        st.markdown("**How it works:**")
        st.markdown("1. 📄 Parse PDF → structured text\n"
                    "2. 🧩 Extract each section\n"
                    "3. ✍️ Summarize each section\n"
                    "4. 📑 Build clean PDF report")

    # Upload
    uploaded = st.file_uploader("Upload research paper (PDF)", type=["pdf"])

    if not uploaded:
        st.info("⬆️ Upload a PDF to get started.")
        return

    if not api_key:
        st.warning("⚠️ Enter your Groq API key in the sidebar.")
        return

    if st.button("🚀 Summarize Paper", type="primary", use_container_width=True):
        with st.status("Analyzing paper...", expanded=True) as status:
            try:
                st.write("📄 Parsing PDF...")
                st.write("🧩 Extracting sections...")
                st.write("✍️ Generating summary...")

                result = run_pipeline(
                    pdf_bytes=uploaded.read(),
                    api_key=api_key,
                    model=model,
                )

                st.write("📑 Building report...")
                status.update(label="✅ Done!", state="complete")

            except LLMServiceError as exc:
                status.update(label="❌ Error", state="error")
                st.error(f"AI Error: {exc.message}")
                if exc.original:
                    with st.expander("Details"):
                        st.code(str(exc.original))
                return
            except PDFExtractionError as exc:
                status.update(label="❌ Error", state="error")
                st.error(f"PDF Error: {exc.message}")
                return
            except AppError as exc:
                status.update(label="❌ Error", state="error")
                st.error(f"Error: {exc.message}")
                if hasattr(exc, 'original') and exc.original:
                    with st.expander("Details"):
                        st.code(str(exc.original))
                return
            except Exception as exc:
                logger.error("Unhandled UI pipeline error: %s: %s", type(exc).__name__, exc, exc_info=True)
                status.update(label="❌ Error", state="error")
                st.error("Unexpected server error while processing the document. Please try again.")
                with st.expander("Details"):
                    st.code(str(exc))
                return

        # Results
        st.success(f"✅ **{result.sections.title}**")

        tab_summary, tab_download = st.tabs(["📋 Summary", "📥 Download Report"])

        with tab_summary:
            st.markdown(result.summary_markdown)

        with tab_download:
            st.download_button(
                label="⬇️ Download PDF Report",
                data=result.report_pdf_bytes,
                file_name=f"summary_{uploaded.name}",
                mime="application/pdf",
                use_container_width=True,
            )
            st.caption(f"_{result.sections.title}_  ·  {result.sections.authors}")
