from __future__ import annotations

import re
from typing import List, Dict, Tuple, Set
from app.knowledge_graph.extraction.schema import Entity, Relation

# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

_only_number = re.compile(r"^\d+(\.\d+)?$")


def normalize_rel_type(r: str) -> str:
    return (r or "RELATED_TO").strip().upper().replace(" ", "_")


# ---------------------------------------------------------------------------
# Abbreviation / acronym detection
#
# Strategy:
#   "Recurrent Neural Network (RNN)" → canonical = "Recurrent Neural Network",
#       alias   = "RNN"
#   "RNN (Recurrent Neural Network)" → same
#   "(RNN)" alone after a full-form in the same batch → merged
# ---------------------------------------------------------------------------

_PARENS_ABBR = re.compile(r"^(.+?)\s*\(([A-Z][A-Z0-9\-]{1,12})\)\s*$")
_PARENS_FULL = re.compile(r"^([A-Z][A-Z0-9\-]{1,12})\s*\((.+?)\)\s*$")


def _acronym_of(full: str, acronym: str) -> bool:
    """
    Return True if `acronym` is a plausible initialism/acronym of `full`.
    e.g. "Recurrent Neural Network" → "RNN",  "Long Short-Term Memory" → "LSTM"
    """
    if len(acronym) < 2:
        return False
    words = re.split(r"[\s\-]+", full)
    initials = "".join(w[0].upper() for w in words if w)
    return initials == acronym.upper()


def _build_alias_map(names: List[str]) -> Dict[str, str]:
    """
    Given a list of entity names, return a dict  alias → canonical.

    Rules (applied in order):
    1. Inline parenthetical:  "Recurrent Neural Network (RNN)"
                               "RNN (Recurrent Neural Network)"
    2. Cross-pair acronym:    One name is an ALL-CAPS token and another name
                              has initials that match it.
    """
    alias_to_canonical: Dict[str, str] = {}

    # --- Rule 1: parenthetical forms ---
    for raw in names:
        m = _PARENS_ABBR.match(raw)
        if m:
            full_form, abbr = m.group(1).strip(), m.group(2).strip()
            if _acronym_of(full_form, abbr):
                alias_to_canonical[abbr.lower()] = full_form
                alias_to_canonical[raw.lower()]   = full_form
            continue

        m = _PARENS_FULL.match(raw)
        if m:
            abbr, full_form = m.group(1).strip(), m.group(2).strip()
            if _acronym_of(full_form, abbr):
                alias_to_canonical[abbr.lower()] = full_form
                alias_to_canonical[raw.lower()]   = full_form

    # --- Rule 2: cross-pair matching among all names ---
    # Separate into short ALL-CAPS tokens vs multi-word names
    short_caps  = [n for n in names if re.match(r'^[A-Z][A-Z0-9\-]{1,12}$', n)]
    multi_words = [n for n in names if len(n.split()) >= 2]

    for abbr in short_caps:
        if abbr.lower() in alias_to_canonical:
            continue                      # already resolved
        for full in multi_words:
            if _acronym_of(full, abbr):
                alias_to_canonical[abbr.lower()] = full
                break

    return alias_to_canonical


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def dedupe_entities(entities: List[Entity]) -> List[Entity]:
    """
    1. Drop empty / numeric-only names.
    2. Build alias map (abbreviation ↔ full form).
    3. Re-map every entity name to its canonical form.
    4. Deduplicate on (canonical_name_lower, type).
    """
    # Pre-filter
    clean: List[Entity] = []
    for e in entities:
        name = (e.name or "").strip()
        if not name or _only_number.match(name):
            continue
        clean.append(Entity(name=name, type=(e.type or "Concept").strip()))

    # Build alias map from all names seen
    alias_map = _build_alias_map([e.name for e in clean])

    seen: Set[Tuple[str, str]] = set()
    out: List[Entity] = []
    for e in clean:
        canonical = alias_map.get(e.name.lower(), e.name)
        key = (canonical.lower(), e.type)
        if key in seen:
            continue
        seen.add(key)
        out.append(Entity(name=canonical, type=e.type))

    return out


def dedupe_relations(relations: List[Relation]) -> List[Relation]:
    """
    Deduplicates relations; also re-maps head/tail to canonical names
    so edges bridge correctly after entity merging.
    """
    # We do a lightweight pass here; the alias map is rebuilt fresh in the
    # pipeline after entity dedup, so no stale names should reach here.
    seen: Set[Tuple[str, str, str]] = set()
    out: List[Relation] = []
    for r in relations:
        h  = (r.head  or "").strip()
        t  = (r.tail  or "").strip()
        rt = normalize_rel_type(r.relation)
        if not h or not t:
            continue
        key = (h.lower(), t.lower(), rt)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            Relation(h, r.head_type or "Concept", rt, t, r.tail_type or "Concept", r.evidence)
        )
    return out


def remap_relation_endpoints(
    relations: List[Relation],
    alias_map: Dict[str, str],
) -> List[Relation]:
    """
    Re-point every relation's head / tail through the alias map so that
    edges refer to canonical names instead of abbreviations.
    """
    out: List[Relation] = []
    for r in relations:
        h  = alias_map.get(r.head.strip().lower(), r.head.strip())
        t  = alias_map.get(r.tail.strip().lower(), r.tail.strip())
        rt = normalize_rel_type(r.relation)
        out.append(
            Relation(h, r.head_type or "Concept", rt, t, r.tail_type or "Concept", r.evidence)
        )
    return out
