"""
Landmark title matching for seminal paper queries.

Substring matches are not equivalent: a long title that *contains* a famous
short phrase ("… attention is all you need …") is usually a derivative in
another domain, not the canonical paper.
"""
from __future__ import annotations

import re


def normalize_landmark_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace (aligned with hybrid/ranker)."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (title or "").lower())).strip()


def landmark_phrase_anchor_strength(norm_title: str, landmark_norm: str) -> float:
    """
    How strongly ``norm_title`` instantiates ``landmark_norm`` (already normalized).

    Returns
    -------
    float
        1.0 = exact title or tight prefix extension (≤2 extra words),
        ~0.4–0.55 = plausible extended subtitle (3–5 extra words),
        ~0.35 = embedded substring (parody / colon titles / different phrasing),
        0.0 = no match.
    """
    if not landmark_norm or len(landmark_norm.split()) < 4:
        return 0.0
    nt = (norm_title or "").strip()
    ph = landmark_norm.strip()
    if not nt:
        return 0.0
    if nt == ph:
        return 1.0
    prefix = ph + " "
    if nt.startswith(prefix):
        extra = len(nt.split()) - len(ph.split())
        if extra <= 2:
            return 1.0
        if extra <= 5:
            return 0.48
        return 0.22
    if ph in nt:
        return 0.35
    return 0.0
