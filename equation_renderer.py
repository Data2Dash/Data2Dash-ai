"""
equation_renderer.py - equation rendering helpers
===============================================
Provides:
- sanitize_for_latex()
- st.latex / matplotlib / MathJax / code fallbacks
- simple stats for smoke tests
"""

from __future__ import annotations

import base64
import io
import logging
import re
from typing import Optional

import streamlit as st

logger = logging.getLogger(__name__)


class EquationRenderer:
    def __init__(self):
        self.matplotlib_available = self._check_matplotlib()
        self.render_stats = {
            "latex_success": 0,
            "matplotlib_fallback": 0,
            "html_fallback": 0,
            "failures": 0,
        }

    def _check_matplotlib(self) -> bool:
        try:
            import matplotlib  # noqa: F401
            import matplotlib.pyplot as plt  # noqa: F401
            return True
        except Exception:
            logger.warning("Matplotlib not available - image rendering disabled")
            return False

    def sanitize_for_latex(self, latex_str: str) -> str:
        if not latex_str:
            return ""
        cleaned = str(latex_str).strip()
        cleaned = re.sub(r"^```latex\s*", "", cleaned)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = re.sub(r"^\$\$", "", cleaned)
        cleaned = re.sub(r"\$\$$", "", cleaned)
        cleaned = re.sub(r"^\$", "", cleaned)
        cleaned = re.sub(r"\$$", "", cleaned)
        cleaned = re.sub(r"\(\s*\d{1,3}[a-z]?\s*\)\s*$", "", cleaned)
        cleaned = cleaned.replace("−", "-").replace("–", "-").replace("—", "-")
        cleaned = cleaned.replace("∣", "|").replace("│", "|").replace("¦", "|")
        cleaned = cleaned.replace("⋅", r"\cdot ").replace("·", r"\cdot ")
        cleaned = cleaned.replace("⊤", r"^\top ")
        cleaned = cleaned.replace("\ufffd", "").replace("\x00", "").replace("\x01", "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    # Backward compatible alias used by some code paths.
    clean_latex = sanitize_for_latex

    def get_stats(self):
        return dict(self.render_stats)

    def reset_stats(self):
        for key in self.render_stats:
            self.render_stats[key] = 0

    def latex_to_image_base64(self, latex_str: str) -> Optional[str]:
        if not self.matplotlib_available:
            return None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            cleaned = self.sanitize_for_latex(latex_str)
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.axis("off")
            ax.text(0.5, 0.5, f"${cleaned}$", fontsize=16, ha="center", va="center", transform=ax.transAxes)
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=150, transparent=True)
            plt.close(fig)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")
        except Exception as exc:
            logger.warning("Matplotlib rendering failed: %s", exc)
            return None

    def create_mathjax_html(self, latex_str: str) -> str:
        cleaned = self.sanitize_for_latex(latex_str)
        return f"""
        <div class="equation-container" style="margin: 1rem 0; padding: 1rem; background-color: #f8f9fa; border-left: 3px solid #4CAF50; border-radius: 4px; overflow-x: auto;">
            <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
            <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
            <div style="font-size: 1.1em; text-align: center;">$$ {cleaned} $$</div>
        </div>
        """

    def render_equation(self, latex_str: str, equation_number: Optional[int] = None) -> bool:
        if not latex_str or not str(latex_str).strip():
            self.render_stats["failures"] += 1
            return False

        cleaned = self.sanitize_for_latex(latex_str)
        if equation_number is not None:
            st.markdown(f"**📐 Equation {equation_number}**")

        try:
            st.latex(cleaned)
            self.render_stats["latex_success"] += 1
            return True
        except Exception as exc:
            logger.warning("st.latex failed: %s", exc)

        img_b64 = self.latex_to_image_base64(cleaned)
        if img_b64:
            st.markdown(
                f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%; height:auto; display:block; margin:1rem auto;"/>',
                unsafe_allow_html=True,
            )
            self.render_stats["matplotlib_fallback"] += 1
            return True

        try:
            st.markdown(self.create_mathjax_html(cleaned), unsafe_allow_html=True)
            self.render_stats["html_fallback"] += 1
            return True
        except Exception as exc:
            logger.warning("MathJax rendering failed: %s", exc)

        st.code(cleaned, language="latex")
        self.render_stats["failures"] += 1
        return False

    def render_equation_list(self, equations: list) -> None:
        if not equations:
            st.info("No equations found in the document.")
            return
        st.markdown("### 📐 Mathematical Equations")
        for i, eq in enumerate(equations, 1):
            with st.expander(f"Equation {i}" + (f" (Page {eq.get('page', '?')})" if isinstance(eq, dict) and 'page' in eq else "")):
                latex_str = eq.get("latex", "") if isinstance(eq, dict) else str(eq)
                if latex_str:
                    self.render_equation(latex_str, equation_number=i)
                else:
                    st.warning(f"Equation {i} has no content")


_renderer: Optional[EquationRenderer] = None


def get_equation_renderer() -> EquationRenderer:
    global _renderer
    if _renderer is None:
        _renderer = EquationRenderer()
    return _renderer


def render_equation(latex_str: str, equation_number: Optional[int] = None) -> bool:
    return get_equation_renderer().render_equation(latex_str, equation_number=equation_number)


def render_equation_list(equations: list) -> None:
    get_equation_renderer().render_equation_list(equations)


def render_equation_card(latex_str: str, equation_number: Optional[int] = None, page: Optional[str] = None) -> bool:
    header = "📐 Equation"
    if equation_number is not None:
        header += f" {equation_number}"
    if page is not None:
        header += f" — Page {page}"
    st.markdown(f"**{header}**")
    return render_equation(latex_str, equation_number=None)


def render_inline_equation(latex_str: str) -> bool:
    return render_equation(latex_str, equation_number=None)


def clean_latex_for_rendering(latex_str: str) -> str:
    return get_equation_renderer().sanitize_for_latex(latex_str)
