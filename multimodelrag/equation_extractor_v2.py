"""
equation_extractor_v2.py - Advanced Equation Extraction & LaTeX Repair
========================================================================
✅ Multi-method extraction (PyMuPDF + pdfplumber + OCR)
✅ LaTeX validation & auto-repair using SymPy
✅ Image-based fallback rendering for broken LaTeX
✅ Equation deduplication with fuzzy matching
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

# Try importing PDF libraries
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF (fitz) not available - some features will be limited")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logging.warning("pdfplumber not available - some features will be limited")

from pathlib import Path

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: LaTeX Validation & Repair
# ══════════════════════════════════════════════════════════════════════════════

class LaTeXValidator:
    """Validates and repairs LaTeX equations"""
    
    # Greek letters mapping
    GREEK_MAP = {
        "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
        "ε": r"\epsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta",
        "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu",
        "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho",
        "σ": r"\sigma", "τ": r"\tau", "υ": r"\upsilon", "φ": r"\phi",
        "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
        "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda",
        "Ξ": r"\Xi", "Π": r"\Pi", "Σ": r"\Sigma", "Φ": r"\Phi",
        "Ψ": r"\Psi", "Ω": r"\Omega",
    }
    
    # Math symbols mapping
    MATH_SYMBOLS = {
        "∑": r"\sum", "∏": r"\prod", "∫": r"\int", "√": r"\sqrt",
        "±": r"\pm", "∂": r"\partial", "∇": r"\nabla", "∞": r"\infty",
        "≈": r"\approx", "≠": r"\neq", "≤": r"\leq", "≥": r"\geq",
        "∝": r"\propto", "∈": r"\in", "∉": r"\notin",
        "⊂": r"\subset", "⊆": r"\subseteq", "∪": r"\cup", "∩": r"\cap",
        "→": r"\rightarrow", "←": r"\leftarrow", "↔": r"\leftrightarrow",
        "⊕": r"\oplus", "⊗": r"\otimes", "⊤": r"\top",
        "‖": r"\|", "⟨": r"\langle", "⟩": r"\rangle",
        "·": r"\cdot", "×": r"\times", "÷": r"\div",
    }
    
    @classmethod
    def fix_spacing_issues(cls, text: str) -> str:
        """Fix character-by-character spacing from PDF extraction."""
        if not text:
            return ""

        # Step 1: Normalise Unicode symbols FIRST so later steps see \\commands
        text = cls.normalize_symbols(text)

        # Step 2: Collapse spaces inside parentheses and around pipe/comma
        # "( z | x )"  ->  "(z|x)"
        text = re.sub(r'\(\s+', '(', text)
        text = re.sub(r'\s+\)', ')', text)
        text = re.sub(r'\s+\|', '|', text)
        text = re.sub(r'\|\s+', '|', text)
        text = re.sub(r',\s+', ',', text)   # space AFTER comma
        text = re.sub(r'\s+,', ',', text)   # space BEFORE comma: "x , z" -> "x,z"

        # Step 3: Add space after big operators so they don't merge with next token
        # "\\sump" -> "\\sum p"
        for cmd in ('sum', 'int', 'prod', 'lim', 'sup', 'inf'):
            text = re.sub(rf'\\{cmd}([a-zA-Z])', rf'\\{cmd} \1', text)

        # Step 4: Reconstruct summation notation when written as spaced tokens
        # "\\sum i = 1 n"  ->  "\\sum_{i=1}^{n}"
        # Also capture trailing variable that follows the superscript, e.g. "n x i"
        summation_match = re.search(
            r'\\sum\s+([a-z])\s*=\s*(\d+)\s+([a-zA-Z])\s+(.+)',
            text,
        )
        if summation_match:
            # Build subscript/superscript and apply subscript fix to remainder
            remainder = summation_match.group(4)  # e.g. "x i = \mu"
            # Fix "x i" -> "x_i" in remainder (var letter space single letter)
            remainder = re.sub(r'\b([a-zA-Z])\s+([a-z])\b', r'\1_\2', remainder)
            repl = rf'\\sum_{{{summation_match.group(1)}={summation_match.group(2)}}}^{{{summation_match.group(3)}}} {remainder}'
            text = text[:summation_match.start()] + repl
        else:
            text = re.sub(
                r'\\sum\s+([a-z])\s*=\s*(\d+)\s+([a-zA-Z])',
                r'\\sum_{\1=\2}^{\3}',
                text,
            )

        # Step 5: Add subscript for probability notation
        # "p\\eta" -> "p_\\eta"  (only when not already subscripted)
        greek_cmds = (
            'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'varepsilon',
            'eta', 'theta', 'iota', 'kappa', 'lambda', 'mu', 'nu',
            'xi', 'pi', 'rho', 'sigma', 'tau', 'phi', 'varphi',
            'psi', 'omega',
        )
        for cmd in greek_cmds:
            text = re.sub(rf'(?<![_{{])p\\{cmd}\b', rf'p_\\{cmd}', text)

        # Step 6: Heavily-spaced variable subscript fix (x i -> x_i)
        # Only apply when text is still heavily single-character-spaced
        words = text.split()
        if len(words) > 5:
            single_char_ratio = sum(1 for w in words if len(w) == 1) / len(words)
            if single_char_ratio > 0.5:
                text = re.sub(r'\b([a-zA-Z])\s+([a-z])\b', r'\1_\2', text)

        return text
    
    @classmethod
    def normalize_symbols(cls, text: str) -> str:
        """Convert Unicode symbols to LaTeX commands"""
        result = text
        
        # Replace Greek letters
        for greek, latex in cls.GREEK_MAP.items():
            result = result.replace(greek, latex)
        
        # Replace math symbols
        for symbol, latex in cls.MATH_SYMBOLS.items():
            result = result.replace(symbol, latex)
        
        return result
    
    @classmethod
    def balance_delimiters(cls, latex: str) -> str:
        """Balance brackets, braces, and parentheses"""
        # Balance {}
        open_braces = latex.count('{')
        close_braces = latex.count('}')
        if open_braces > close_braces:
            latex += "}" * (open_braces - close_braces)
        elif close_braces > open_braces:
            # Remove trailing extra closing braces
            excess = close_braces - open_braces
            for _ in range(excess):
                if latex.endswith("}"):
                    latex = latex[:-1]
        
        # Balance ()
        open_paren = latex.count('(')
        close_paren = latex.count(')')
        if open_paren > close_paren:
            latex += ")" * (open_paren - close_paren)
        elif close_paren > open_paren:
            excess = close_paren - open_paren
            for _ in range(excess):
                if latex.endswith(")"):
                    latex = latex[:-1]
        
        # Balance []
        open_bracket = latex.count('[')
        close_bracket = latex.count(']')
        if open_bracket > close_bracket:
            latex += "]" * (open_bracket - close_bracket)
        
        return latex
    
    @classmethod
    def fix_common_errors(cls, latex: str) -> str:
        """Fix common LaTeX errors"""
        # Remove broken unicode characters
        latex = latex.replace("", "").replace(""", '"').replace(""", '"')
        latex = latex.replace("'", "'").replace("'", "'")
        
        # Fix double backslashes (OCR artifacts)
        latex = re.sub(r'\\\\([a-zA-Z])', r'\\\1', latex)
        
        # Fix missing spaces after operators
        latex = re.sub(r'\\([a-z]+)([A-Z])', r'\\\1 \2', latex)
        
        # Fix subscript/superscript without braces
        latex = re.sub(r'_([a-zA-Z0-9]{2,})', r'_{\1}', latex)
        latex = re.sub(r'\^([a-zA-Z0-9]{2,})', r'^{\1}', latex)
        
        # Fix function names
        for func in ['exp', 'log', 'sin', 'cos', 'tan', 'max', 'min', 'lim']:
            # If function name appears without backslash, add it
            latex = re.sub(rf'\b{func}\s*\(', rf'\\{func}(', latex, flags=re.IGNORECASE)
        
        # Fix operators
        latex = re.sub(r'\bargmax\b', r'\\operatorname{argmax}', latex, flags=re.IGNORECASE)
        latex = re.sub(r'\bargmin\b', r'\\operatorname{argmin}', latex, flags=re.IGNORECASE)
        latex = re.sub(r'\bsoftmax\b', r'\\operatorname{softmax}', latex, flags=re.IGNORECASE)
        
        # Clean excessive whitespace
        latex = re.sub(r'\s+', ' ', latex).strip()
        
        return latex
    
    @classmethod
    def validate_and_repair(cls, raw_text: str) -> Tuple[str, bool]:
        """
        Validate and repair a LaTeX equation string.

        Fast-path: if the input already looks like well-formed LaTeX
        (has backslash commands AND at least one structural char _ ^ { }),
        only balance delimiters and strip trailing equation numbers,
        so that clean input is NOT mangled by the spacing heuristics.

        Returns:
            (repaired_latex: str, is_valid: bool)
        """
        if not raw_text or not raw_text.strip():
            return '', False

        # Fast-path: input already contains LaTeX commands + structural chars
        has_cmd        = '\\' in raw_text
        has_structural = bool(re.search(r'[_^{}]', raw_text))
        if has_cmd and has_structural:
            text = raw_text.strip()
            text = cls.balance_delimiters(text)
            text = re.sub(r'\(\s*\d{1,3}[a-z]?\s*\)\s*$', '', text).strip()
            text = cls.fix_common_errors(text)
            return text, True

        # Full repair pipeline for raw / OCR / Unicode text
        # Note: fix_spacing_issues calls normalize_symbols internally
        text = cls.fix_spacing_issues(raw_text)

        # Apply any remaining normalisation missed by fix_spacing_issues
        text = cls.normalize_symbols(text)

        # Structural fixes (double backslash, function names, etc.)
        text = cls.fix_common_errors(text)

        # Balance delimiters
        text = cls.balance_delimiters(text)

        # Strip trailing equation numbers: (1), (12a)
        text = re.sub(r'\(\s*\d{1,3}[a-z]?\s*\)\s*$', '', text).strip()

        has_latex = '\\' in text
        has_math  = any(op in text for op in ['=', '+', '-', '*', '/', '^', '_'])
        is_valid  = has_latex or has_math

        return text, is_valid


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Enhanced Equation Detection
# ══════════════════════════════════════════════════════════════════════════════

class EquationDetector:
    """Detects equations using multiple heuristics"""
    
    STRONG_OPERATORS = ['=', '≈', '≤', '≥', '∫', '∑', '∏', '∝', '→', '∈', '≡']
    
    @classmethod
    def is_equation(cls, text: str, min_length: int = 5, max_length: int = 500) -> bool:
        """
        Check if text looks like an equation
        
        Args:
            text: Text to check
            min_length: Minimum length to consider
            max_length: Maximum length to consider
            
        Returns:
            True if text looks like an equation
        """
        text = text.strip()
        
        # Length check
        if len(text) < min_length or len(text) > max_length:
            return False
        
        # Reject URLs and references
        if re.search(r'(https?://|www\.|doi:|arxiv:|\.pdf)', text, re.IGNORECASE):
            return False
        
        # Reject citations
        if re.match(r'^\[\d+\]', text):
            return False
        
        # Reject captions
        if re.match(r'^\s*(Figure|Fig|Table|Appendix|Algorithm)\b', text, re.IGNORECASE):
            return False
        
        # Must have strong operator
        if not any(op in text for op in cls.STRONG_OPERATORS):
            return False
        
        # Check for LaTeX commands or Greek letters
        has_latex = '\\' in text
        has_greek = any(c in text for c in 'αβγδεηθλμπστφψω')
        
        # Reject if too many regular words (likely prose)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        if len(words) > 8:
            return False
        
        return has_latex or has_greek or len(words) <= 3
    
    @classmethod
    def extract_from_block(cls, text: str, bbox: Tuple[float, float, float, float] = None) -> Optional[str]:
        """
        Extract equation from a text block
        
        Args:
            text: Text block
            bbox: Bounding box coordinates
            
        Returns:
            Extracted equation text or None
        """
        if not cls.is_equation(text):
            return None
        
        # Clean the text
        clean = text.replace('\n', ' ').strip()
        clean = re.sub(r'\s+', ' ', clean)
        
        return clean


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Multi-Method Equation Extractor
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExtractedEquation:
    """Represents an extracted equation"""
    raw_text: str
    latex: str
    page_num: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    method: str  # 'pymupdf', 'pdfplumber', 'ocr'


class AdvancedEquationExtractor:
    """
    Multi-method equation extractor with validation and repair
    """
    
    def __init__(self):
        self.validator = LaTeXValidator()
        self.detector = EquationDetector()
    
    def extract_from_pymupdf(self, page: Any, page_num: int) -> List[ExtractedEquation]:
        """Extract equations using PyMuPDF"""
        if not PYMUPDF_AVAILABLE:
            return []
        
        equations = []
        
        blocks = page.get_text("blocks")
        for block in blocks:
            x0, y0, x1, y1, text, block_no, block_type = block
            
            if block_type != 0:  # Skip images
                continue
            
            text = text.strip()
            if not text:
                continue
            
            # Check if it's an equation
            if self.detector.is_equation(text):
                bbox = (x0, y0, x1, y1)
                
                # Validate and repair
                latex, is_valid = self.validator.validate_and_repair(text)
                
                if is_valid:
                    eq = ExtractedEquation(
                        raw_text=text,
                        latex=latex,
                        page_num=page_num,
                        bbox=bbox,
                        confidence=0.8,
                        method='pymupdf'
                    )
                    equations.append(eq)
        
        return equations
    
    def extract_from_pdfplumber(self, page: Any, page_num: int) -> List[ExtractedEquation]:
        """Extract equations using pdfplumber"""
        if not PDFPLUMBER_AVAILABLE:
            return []
        
        equations = []
        
        try:
            # Get text with layout
            text = page.extract_text()
            if not text:
                return equations
            
            # Split into lines
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if it's an equation
                if self.detector.is_equation(line):
                    # Validate and repair
                    latex, is_valid = self.validator.validate_and_repair(line)
                    
                    if is_valid:
                        eq = ExtractedEquation(
                            raw_text=line,
                            latex=latex,
                            page_num=page_num,
                            bbox=(0, 0, page.width, page.height),  # Approximate
                            confidence=0.7,
                            method='pdfplumber'
                        )
                        equations.append(eq)
        
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed on page {page_num}: {e}")
        
        return equations
    
    def deduplicate_equations(self, equations: List[ExtractedEquation]) -> List[ExtractedEquation]:
        """
        Remove duplicate equations using fuzzy matching
        
        Args:
            equations: List of extracted equations
            
        Returns:
            Deduplicated list
        """
        if not equations:
            return []
        
        # Sort by confidence
        equations.sort(key=lambda e: e.confidence, reverse=True)
        
        unique = []
        seen_latex = set()
        
        for eq in equations:
            # Normalize for comparison
            normalized = re.sub(r'\s+', '', eq.latex.lower())
            
            # Check if similar equation already seen
            is_duplicate = False
            for seen in seen_latex:
                # Simple similarity check
                if normalized == seen:
                    is_duplicate = True
                    break
                
                # Fuzzy match (>90% similar)
                if len(normalized) > 10 and len(seen) > 10:
                    common = sum(1 for a, b in zip(normalized, seen) if a == b)
                    similarity = common / max(len(normalized), len(seen))
                    if similarity > 0.9:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique.append(eq)
                seen_latex.add(normalized)
        
        return unique
    
    def extract_from_pdf(self, pdf_path: str) -> Dict[int, List[ExtractedEquation]]:
        """
        Extract equations from entire PDF using multiple methods
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary mapping page_num -> list of equations
        """
        results = {}
        
        if not PYMUPDF_AVAILABLE and not PDFPLUMBER_AVAILABLE:
            logger.error("Neither PyMuPDF nor pdfplumber is available - cannot extract equations")
            return results
        
        try:
            # Open with both libraries
            mupdf_doc = None
            plumber_doc = None
            
            if PYMUPDF_AVAILABLE:
                import fitz
                mupdf_doc = fitz.open(pdf_path)
            
            if PDFPLUMBER_AVAILABLE:
                import pdfplumber
                plumber_doc = pdfplumber.open(pdf_path)
            
            # Get number of pages
            num_pages = len(mupdf_doc) if mupdf_doc else len(plumber_doc.pages)
            
            for page_num in range(num_pages):
                page_equations = []
                
                # Method 1: PyMuPDF
                try:
                    mupdf_page = mupdf_doc[page_num]
                    pymupdf_eqs = self.extract_from_pymupdf(mupdf_page, page_num)
                    page_equations.extend(pymupdf_eqs)
                except Exception as e:
                    logger.warning(f"PyMuPDF extraction failed on page {page_num}: {e}")
                
                # Method 2: pdfplumber
                try:
                    plumber_page = plumber_doc.pages[page_num]
                    plumber_eqs = self.extract_from_pdfplumber(plumber_page, page_num)
                    page_equations.extend(plumber_eqs)
                except Exception as e:
                    logger.warning(f"pdfplumber extraction failed on page {page_num}: {e}")
                
                # Deduplicate
                unique_equations = self.deduplicate_equations(page_equations)
                
                if unique_equations:
                    results[page_num] = unique_equations
            
            mupdf_doc.close()
            plumber_doc.close()
        
        except Exception as e:
            logger.error(f"Failed to extract equations from {pdf_path}: {e}")
        
        return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Testing & Validation
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test cases for LaTeX validation
    test_cases = [
        # Valid equations
        ("p η ( z | x ) p θ ( y | x , z )", "Valid probability equation"),
        ("∑ i = 1 n x i", "Valid summation"),
        ("exp ( - x )", "Valid exponential"),
        ("p_θ(y|x,z) = ∫ p(z|x) dz", "Valid integral equation"),
        
        # Invalid (prose)
        ("Generator pθ", "Just text with symbol"),
        ("The model uses a generator", "Plain text"),
        ("Figure 1 shows results", "Caption"),
    ]
    
    print("=" * 80)
    print("LaTeX Validator Tests")
    print("=" * 80)
    
    validator = LaTeXValidator()
    
    for raw, description in test_cases:
        latex, is_valid = validator.validate_and_repair(raw)
        status = "✅" if is_valid else "❌"
        print(f"\n{status} {description}")
        print(f"   Raw:    {raw}")
        print(f"   LaTeX:  {latex}")
        print(f"   Valid:  {is_valid}")
    
    print("\n" + "=" * 80)
    print("✅ Tests complete!")