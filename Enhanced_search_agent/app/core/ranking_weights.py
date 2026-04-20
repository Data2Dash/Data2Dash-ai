"""
Central, tunable ranking weights for DATA2DASH.

Default values mirror the previous hard-coded ``RankingService`` behavior.
Override at runtime with either:

1) ``RANKING_WEIGHTS_JSON`` environment variable
2) ``RANKING_WEIGHTS_FILE`` path to a JSON file

Both accept a JSON object with any subset of fields.

Examples::

    RANKING_WEIGHTS_JSON='{"w_primary":0.72,"w_citation":0.12,"w_recency":0.10,"w_source":0.06}'

    RANKING_WEIGHTS_FILE=/path/to/ranking_weights.json
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, Tuple


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LearnableRankingWeights:
    """Composite ranking linear layer (before title / seminal bonuses)."""

    blend_topic_in_primary: float = 0.75  # weight on topic_relevance vs semantic in primary
    w_primary: float = 0.70
    w_citation: float = 0.14
    w_recency: float = 0.10
    w_source: float = 0.06
    # Citation sub-score: blend log(raw cites) with age-normalized cites/year
    citation_age_blend: float = 0.35  # 0=log only, 1=cites-per-year only

    def validate(self) -> "LearnableRankingWeights":
        """Validate ranges and non-negativity constraints."""
        if not (0.0 <= self.blend_topic_in_primary <= 1.0):
            raise ValueError("blend_topic_in_primary must be between 0 and 1")
        if not (0.0 <= self.citation_age_blend <= 1.0):
            raise ValueError("citation_age_blend must be between 0 and 1")

        for name in ("w_primary", "w_citation", "w_recency", "w_source"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

        total = self.w_primary + self.w_citation + self.w_recency + self.w_source
        if total <= 0:
            raise ValueError(
                "At least one of w_primary, w_citation, w_recency, or w_source must be positive"
            )
        return self

    def normalized(self) -> "LearnableRankingWeights":
        """Renormalize w_primary..w_source to sum to 1 if they drift."""
        s = self.w_primary + self.w_citation + self.w_recency + self.w_source
        if s <= 0:
            return self
        return replace(
            self,
            w_primary=self.w_primary / s,
            w_citation=self.w_citation / s,
            w_recency=self.w_recency / s,
            w_source=self.w_source / s,
        )

    def validated_normalized(self) -> "LearnableRankingWeights":
        """Validate first, then normalize the primary linear weights."""
        return self.validate().normalized()


@dataclass(frozen=True)
class LoadedRankingWeights:
    """Loaded ranking weights plus lightweight diagnostics."""

    weights: LearnableRankingWeights
    source: str
    normalized: bool
    ignored_keys: Tuple[str, ...] = ()
    invalid_keys: Tuple[str, ...] = ()


_DEFAULT = LearnableRankingWeights().validated_normalized()


def _allowed_field_names() -> set[str]:
    return {f.name for f in fields(LearnableRankingWeights)}


def _read_override_payload() -> tuple[str, str]:
    """
    Return the raw JSON payload and a source label.

    Priority:
    1) RANKING_WEIGHTS_JSON
    2) RANKING_WEIGHTS_FILE
    """
    raw_env = (os.getenv("RANKING_WEIGHTS_JSON") or "").strip()
    if raw_env:
        return raw_env, "env:RANKING_WEIGHTS_JSON"

    file_path = (os.getenv("RANKING_WEIGHTS_FILE") or "").strip()
    if file_path:
        try:
            raw_file = Path(file_path).read_text(encoding="utf-8").strip()
            if raw_file:
                return raw_file, f"file:{file_path}"
            logger.warning("RANKING_WEIGHTS_FILE is empty: %s", file_path)
        except OSError as exc:
            logger.warning("Could not read RANKING_WEIGHTS_FILE=%s: %s", file_path, exc)

    return "", "default"


def _parse_weight_overrides(raw: str) -> tuple[Dict[str, float], Tuple[str, ...], Tuple[str, ...]]:
    """Parse and sanitize user-provided weight overrides."""
    if not raw:
        return {}, (), ()

    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Invalid ranking weights JSON; using defaults. Error: %s", exc)
        return {}, (), ()

    if not isinstance(data, dict):
        logger.warning("Ranking weights payload must be a JSON object; using defaults.")
        return {}, (), ()

    allowed = _allowed_field_names()
    kwargs: Dict[str, float] = {}
    ignored_keys = []
    invalid_keys = []

    for key, val in data.items():
        if key not in allowed:
            ignored_keys.append(str(key))
            continue
        try:
            kwargs[key] = float(val)
        except (TypeError, ValueError):
            invalid_keys.append(str(key))

    if ignored_keys:
        logger.warning("Ignoring unknown ranking weight keys: %s", ", ".join(sorted(ignored_keys)))
    if invalid_keys:
        logger.warning("Ignoring non-numeric ranking weight values for keys: %s", ", ".join(sorted(invalid_keys)))

    return kwargs, tuple(sorted(ignored_keys)), tuple(sorted(invalid_keys))


def load_learnable_weights_with_meta() -> LoadedRankingWeights:
    """
    Load ranking weights from env/file, validate them, normalize them,
    and return metadata describing how they were loaded.
    """
    raw, source = _read_override_payload()
    if not raw:
        return LoadedRankingWeights(
            weights=_DEFAULT,
            source="default",
            normalized=False,
        )

    kwargs, ignored_keys, invalid_keys = _parse_weight_overrides(raw)
    if not kwargs:
        return LoadedRankingWeights(
            weights=_DEFAULT,
            source=f"{source} (fallback: no valid overrides)",
            normalized=False,
            ignored_keys=ignored_keys,
            invalid_keys=invalid_keys,
        )

    try:
        merged = replace(_DEFAULT, **kwargs)
        normalized_needed = abs(
            (merged.w_primary + merged.w_citation + merged.w_recency + merged.w_source) - 1.0
        ) > 1e-9
        validated = merged.validated_normalized()
    except ValueError as exc:
        logger.warning("Invalid ranking weights override from %s; using defaults. Error: %s", source, exc)
        return LoadedRankingWeights(
            weights=_DEFAULT,
            source=f"{source} (fallback: invalid override)",
            normalized=False,
            ignored_keys=ignored_keys,
            invalid_keys=invalid_keys,
        )

    return LoadedRankingWeights(
        weights=validated,
        source=source,
        normalized=normalized_needed,
        ignored_keys=ignored_keys,
        invalid_keys=invalid_keys,
    )


def load_learnable_weights() -> LearnableRankingWeights:
    """
    Backward-compatible loader that returns only the weights object.
    """
    return load_learnable_weights_with_meta().weights
