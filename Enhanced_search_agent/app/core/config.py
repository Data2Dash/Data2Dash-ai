import os
from dataclasses import dataclass
from typing import Mapping

from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()


TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off"}


def _parse_arxiv_search_sort_by() -> str:
    """Map env to arXiv client sort keys (relevance | submittedDate | lastUpdatedDate)."""
    raw = env_str("ARXIV_SEARCH_SORT_BY", "relevance").strip()
    k = "".join(c for c in raw.lower() if c.isalnum())
    return {
        "relevance": "relevance",
        "submitteddate": "submittedDate",
        "lastupdateddate": "lastUpdatedDate",
    }.get(k, "relevance")


def env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value is not None else default


def env_int(name: str, default: int, *, min_value: int | None = None, max_value: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid integer for {name}: {raw!r}") from exc

    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {value}")
    return value


def env_float(
    name: str,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        value = default
    else:
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid float for {name}: {raw!r}") from exc

    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {value}")
    return value


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default

    normalized = raw.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"Invalid boolean for {name}: {raw!r}")


@dataclass(frozen=True)
class Settings:
    # =========================
    # App Info
    # =========================
    APP_NAME: str
    APP_ENV: str

    # =========================
    # API Keys / External services
    # =========================
    GROQ_API_KEY: str
    SEMANTIC_SCHOLAR_API_KEY: str
    OPENALEX_MAILTO: str

    # =========================
    # Models
    # =========================
    CHAT_MODEL: str
    FAST_MODEL: str

    # =========================
    # Search Settings
    # =========================
    DEFAULT_PAGE_SIZE: int
    MAX_RESULTS: int
    RETRIEVAL_SOFT_CAP: int
    ENABLE_PROGRESSIVE_SOURCE_FETCH: bool
    ARXIV_PROGRESSIVE_MAX_TOTAL: int
    OPENALEX_PROGRESSIVE_MAX_TOTAL: int
    ARXIV_PROGRESSIVE_PAGE_SIZE: int
    OPENALEX_PROGRESSIVE_PER_PAGE: int
    ARXIV_PROGRESSIVE_MAX_PAGES: int
    OPENALEX_PROGRESSIVE_MAX_PAGES: int
    ARXIV_PER_QUERY_CAP: int
    OPENALEX_PER_QUERY_CAP: int
    ARXIV_PER_QUERY_CAP_DEEP: int
    OPENALEX_PER_QUERY_CAP_DEEP: int
    FAN_OUT_ALL_VARIANTS: bool
    ENABLE_OPENALEX_RETRIEVAL: bool
    ENABLE_RETRIEVAL_CS_CE_FIELD_FILTER: bool
    ARXIV_SEARCH_SORT_BY: str
    ARXIV_DISK_CACHE_TTL_SECONDS: int
    HYBRID_FILTER_MIN_SEMANTIC: float
    HYBRID_FILTER_MIN_TOPIC: float
    HYBRID_FILTER_KEEP_MIN: int

    # =========================
    # Semantic rerank (TF-IDF)
    # =========================
    ENABLE_SEMANTIC_RERANK: bool
    RERANK_BLEND: float
    RERANK_POOL_SIZE: int

    # =========================
    # Hybrid BM25 + embedding rerank
    # =========================
    ENABLE_HYBRID_RERANK: bool
    EMBEDDING_MODEL_NAME: str
    HYBRID_RERANK_TOP_N: int
    HYBRID_RERANK_BM25_WEIGHT: float
    HYBRID_RERANK_EMBEDDING_WEIGHT: float
    HYBRID_RERANK_COMPOSITE_BLEND: float
    HYBRID_EMBEDDING_CACHE_MAX: int

    # =========================
    # Cache
    # =========================
    DISK_CACHE_TTL_SECONDS: int

    # =========================
    # Ranking Weights
    # =========================
    WEIGHT_RELEVANCE: float
    WEIGHT_RECENCY: float
    WEIGHT_CITATIONS: float
    WEIGHT_SOURCE_CONFIDENCE: float

    # =========================
    # Source Confidence Scores
    # =========================
    SOURCE_CONFIDENCE: Mapping[str, float]

    @classmethod
    def from_env(cls) -> "Settings":
        source_confidence = {
            "arxiv": env_float("SOURCE_CONFIDENCE_ARXIV", 0.85, min_value=0.0, max_value=1.0),
            "semantic_scholar": env_float(
                "SOURCE_CONFIDENCE_SEMANTIC_SCHOLAR", 0.95, min_value=0.0, max_value=1.0
            ),
            "openalex": env_float("SOURCE_CONFIDENCE_OPENALEX", 0.90, min_value=0.0, max_value=1.0),
        }

        return cls(
            # App Info
            APP_NAME=env_str("APP_NAME", "DATA2DASH"),
            APP_ENV=env_str("APP_ENV", "development"),

            # API Keys / External services
            GROQ_API_KEY=env_str("GROQ_API_KEY", ""),
            SEMANTIC_SCHOLAR_API_KEY=env_str("SEMANTIC_SCHOLAR_API_KEY", ""),
            OPENALEX_MAILTO=env_str("OPENALEX_MAILTO", ""),

            # Models
            CHAT_MODEL=env_str("CHAT_MODEL", "llama-3.3-70b-versatile"),
            FAST_MODEL=env_str("FAST_MODEL", "llama-3.1-8b-instant"),

            # Search Settings
            DEFAULT_PAGE_SIZE=env_int("DEFAULT_PAGE_SIZE", 25, min_value=1),
            MAX_RESULTS=env_int("MAX_RESULTS", 0),
            RETRIEVAL_SOFT_CAP=env_int("RETRIEVAL_SOFT_CAP", 1200, min_value=500),
            ENABLE_PROGRESSIVE_SOURCE_FETCH=env_bool("ENABLE_PROGRESSIVE_SOURCE_FETCH", True),
            ARXIV_PROGRESSIVE_MAX_TOTAL=env_int("ARXIV_PROGRESSIVE_MAX_TOTAL", 400, min_value=1),
            OPENALEX_PROGRESSIVE_MAX_TOTAL=env_int("OPENALEX_PROGRESSIVE_MAX_TOTAL", 400, min_value=1),
            ARXIV_PROGRESSIVE_PAGE_SIZE=env_int("ARXIV_PROGRESSIVE_PAGE_SIZE", 50, min_value=1, max_value=200),
            OPENALEX_PROGRESSIVE_PER_PAGE=env_int("OPENALEX_PROGRESSIVE_PER_PAGE", 200, min_value=1, max_value=200),
            ARXIV_PROGRESSIVE_MAX_PAGES=env_int("ARXIV_PROGRESSIVE_MAX_PAGES", 40, min_value=1),
            OPENALEX_PROGRESSIVE_MAX_PAGES=env_int("OPENALEX_PROGRESSIVE_MAX_PAGES", 30, min_value=1),
            ARXIV_PER_QUERY_CAP=env_int("ARXIV_PER_QUERY_CAP", 100, min_value=1),
            OPENALEX_PER_QUERY_CAP=env_int("OPENALEX_PER_QUERY_CAP", 75, min_value=1),
            ARXIV_PER_QUERY_CAP_DEEP=env_int("ARXIV_PER_QUERY_CAP_DEEP", 100, min_value=1),
            OPENALEX_PER_QUERY_CAP_DEEP=env_int("OPENALEX_PER_QUERY_CAP_DEEP", 100, min_value=1),
            FAN_OUT_ALL_VARIANTS=env_bool("FAN_OUT_ALL_VARIANTS", False),
            # Default True: hybrid + PRF use OpenAlex unless explicitly disabled (arXiv-only).
            ENABLE_OPENALEX_RETRIEVAL=env_bool("ENABLE_OPENALEX_RETRIEVAL", True),
            ENABLE_RETRIEVAL_CS_CE_FIELD_FILTER=env_bool(
                "ENABLE_RETRIEVAL_CS_CE_FIELD_FILTER", True
            ),
            ARXIV_SEARCH_SORT_BY=_parse_arxiv_search_sort_by(),
            ARXIV_DISK_CACHE_TTL_SECONDS=env_int(
                "ARXIV_DISK_CACHE_TTL_SECONDS", 3600, min_value=60
            ),
            HYBRID_FILTER_MIN_SEMANTIC=env_float("HYBRID_FILTER_MIN_SEMANTIC", 0.06, min_value=0.0, max_value=1.0),
            HYBRID_FILTER_MIN_TOPIC=env_float("HYBRID_FILTER_MIN_TOPIC", 0.06, min_value=0.0, max_value=1.0),
            HYBRID_FILTER_KEEP_MIN=env_int("HYBRID_FILTER_KEEP_MIN", 40, min_value=1),

            # Semantic rerank (TF-IDF)
            ENABLE_SEMANTIC_RERANK=env_bool("ENABLE_SEMANTIC_RERANK", True),
            RERANK_BLEND=env_float("RERANK_BLEND", 0.25, min_value=0.0, max_value=1.0),
            RERANK_POOL_SIZE=env_int("RERANK_POOL_SIZE", 200, min_value=1),

            # Hybrid BM25 + embedding rerank
            ENABLE_HYBRID_RERANK=env_bool("ENABLE_HYBRID_RERANK", False),
            EMBEDDING_MODEL_NAME=env_str("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
            HYBRID_RERANK_TOP_N=env_int("HYBRID_RERANK_TOP_N", 200, min_value=1),
            HYBRID_RERANK_BM25_WEIGHT=env_float("HYBRID_RERANK_BM25_WEIGHT", 0.55, min_value=0.0, max_value=1.0),
            HYBRID_RERANK_EMBEDDING_WEIGHT=env_float(
                "HYBRID_RERANK_EMBEDDING_WEIGHT", 0.45, min_value=0.0, max_value=1.0
            ),
            HYBRID_RERANK_COMPOSITE_BLEND=env_float(
                "HYBRID_RERANK_COMPOSITE_BLEND", 0.28, min_value=0.0, max_value=1.0
            ),
            HYBRID_EMBEDDING_CACHE_MAX=env_int("HYBRID_EMBEDDING_CACHE_MAX", 2048, min_value=1),

            # Cache
            DISK_CACHE_TTL_SECONDS=env_int("DISK_CACHE_TTL_SECONDS", 86400, min_value=60),

            # Ranking Weights
            WEIGHT_RELEVANCE=env_float("WEIGHT_RELEVANCE", 0.65, min_value=0.0, max_value=1.0),
            WEIGHT_RECENCY=env_float("WEIGHT_RECENCY", 0.10, min_value=0.0, max_value=1.0),
            WEIGHT_CITATIONS=env_float("WEIGHT_CITATIONS", 0.20, min_value=0.0, max_value=1.0),
            WEIGHT_SOURCE_CONFIDENCE=env_float(
                "WEIGHT_SOURCE_CONFIDENCE", 0.05, min_value=0.0, max_value=1.0
            ),

            # Source Confidence Scores
            SOURCE_CONFIDENCE=source_confidence,
        )

    def validate(self) -> None:
        if not self.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is missing in the .env file")

        if self.MAX_RESULTS < 0:
            raise ValueError("MAX_RESULTS must be >= 0. Use 0 for no cap.")

        if self.HYBRID_FILTER_KEEP_MIN > self.RETRIEVAL_SOFT_CAP:
            raise ValueError("HYBRID_FILTER_KEEP_MIN cannot exceed RETRIEVAL_SOFT_CAP")

        if self.ARXIV_PER_QUERY_CAP_DEEP < self.ARXIV_PER_QUERY_CAP:
            raise ValueError("ARXIV_PER_QUERY_CAP_DEEP should be >= ARXIV_PER_QUERY_CAP")

        if self.OPENALEX_PER_QUERY_CAP_DEEP < self.OPENALEX_PER_QUERY_CAP:
            raise ValueError("OPENALEX_PER_QUERY_CAP_DEEP should be >= OPENALEX_PER_QUERY_CAP")

        if self.ARXIV_PROGRESSIVE_MAX_TOTAL > self.RETRIEVAL_SOFT_CAP:
            raise ValueError("ARXIV_PROGRESSIVE_MAX_TOTAL cannot exceed RETRIEVAL_SOFT_CAP")

        if self.OPENALEX_PROGRESSIVE_MAX_TOTAL > self.RETRIEVAL_SOFT_CAP:
            raise ValueError("OPENALEX_PROGRESSIVE_MAX_TOTAL cannot exceed RETRIEVAL_SOFT_CAP")

        if self.RERANK_POOL_SIZE < self.HYBRID_RERANK_TOP_N:
            raise ValueError("RERANK_POOL_SIZE should be >= HYBRID_RERANK_TOP_N")

        total_weight = (
            self.WEIGHT_RELEVANCE
            + self.WEIGHT_RECENCY
            + self.WEIGHT_CITATIONS
            + self.WEIGHT_SOURCE_CONFIDENCE
        )
        if total_weight <= 0:
            raise ValueError("Ranking weights must sum to a positive value")

        rerank_weight_sum = self.HYBRID_RERANK_BM25_WEIGHT + self.HYBRID_RERANK_EMBEDDING_WEIGHT
        if self.ENABLE_HYBRID_RERANK and rerank_weight_sum <= 0:
            raise ValueError("Hybrid rerank weights must sum to a positive value when enabled")

        if self.ARXIV_SEARCH_SORT_BY not in (
            "relevance",
            "submittedDate",
            "lastUpdatedDate",
        ):
            raise ValueError(
                "ARXIV_SEARCH_SORT_BY must be one of relevance, submittedDate, lastUpdatedDate "
                f"(got {self.ARXIV_SEARCH_SORT_BY!r})"
            )

        if not self.OPENALEX_MAILTO:
            print("Warning: OPENALEX_MAILTO is not set. OpenAlex may rate-limit more aggressively.")


# Create global settings instance
settings = Settings.from_env()

# Validate config at startup
settings.validate()
