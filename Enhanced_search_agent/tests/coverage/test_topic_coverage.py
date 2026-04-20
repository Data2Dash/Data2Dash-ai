"""
Topic Coverage & Recall Tests
==============================
These tests measure what PERCENTAGE of known landmark papers the hybrid search
agent successfully retrieves for a given topic query.

Metrics computed per topic
--------------------------
  must_recall   : % of must_find papers found            (target ≥ 80 %)
  should_recall : % of should_find papers found          (target ≥ 40 %)
  total_recall  : combined % across must + should        (target ≥ 50 %)
  source_spread : number of distinct sources in results  (target ≥ 2)

How paper matching works
------------------------
A retrieved paper is considered a MATCH if any ground-truth title keyword
(≥4 chars) appears as a substring in the retrieved paper's title (lowercased).
This handles:
  ✓ "Attention Is All You Need" matched by keyword "attention is all you need"
  ✓ "BERT: Pre-training ..."   matched by keyword "bert"
  ✓ Capitalisation / punctuation differences

Run modes
---------
  pytest tests/coverage/                          # ALL topics
  pytest tests/coverage/ -k "transformers"        # single topic
  pytest tests/coverage/ --tb=short -s            # verbose with live output
"""
import sys
import os
import re
import json
import pytest
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from tests.coverage.ground_truth import GROUND_TRUTH
from app.services.search_agent import SearchAgent


# ─────────────────────────────────────────────────────────────────────────────
# Thresholds (adjust if needed)
# ─────────────────────────────────────────────────────────────────────────────
MUST_RECALL_THRESHOLD    = 0.80   # 80 % of must_find papers required
SHOULD_RECALL_THRESHOLD  = 0.40   # 40 % of should_find papers required
SOURCE_SPREAD_MIN        = 2      # at least 2 distinct sources
PER_PAGE                 = 20     # results per search (wider net)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lowercase + strip punctuation for fuzzy matching."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _paper_matches_keyword(paper_title: str, keyword: str) -> bool:
    """
    True if the keyword (normalised) appears as a substring
    in the paper title (normalised).
    """
    return _normalise(keyword) in _normalise(paper_title)


def _find_matches(retrieved_titles: list, target_keywords: list) -> list:
    """
    Returns the subset of target_keywords that are matched by at least
    one retrieved paper title.
    """
    found = []
    for kw in target_keywords:
        if any(_paper_matches_keyword(title, kw) for title in retrieved_titles):
            found.append(kw)
    return found


def _compute_mrr(retrieved_titles: list, target_keywords: list) -> float:
    """Computes the Mean Reciprocal Rank (MRR) of hits for the target keywords."""
    if not target_keywords:
        return 0.0
    rr_sum = 0.0
    for kw in target_keywords:
        for rank, title in enumerate(retrieved_titles, start=1):
            if _paper_matches_keyword(title, kw):
                rr_sum += 1.0 / rank
                break
    return rr_sum / len(target_keywords)


def _run_search(topic: str, per_page: int = PER_PAGE) -> dict:
    """Run the hybrid search agent and return the raw results dict."""
    agent = SearchAgent()
    results = agent.search(query=topic, page=1, per_page=per_page)
    return results


def _build_report(topic_entry: dict, results: dict) -> dict:
    """
    Given a ground-truth topic entry and search results, compute all
    recall/coverage metrics and return as a structured report dict.
    """
    retrieved_papers = results.get("ranked_papers") or results.get("papers", [])
    retrieved_titles = [p.title for p in retrieved_papers]
    all_titles_str   = "\n  ".join(retrieved_titles) or "  (none)"

    must_kws    = topic_entry["must_find"]
    should_kws  = topic_entry["should_find"]

    must_found    = _find_matches(retrieved_titles, must_kws)
    should_found  = _find_matches(retrieved_titles, should_kws)

    must_recall   = len(must_found)   / len(must_kws)   if must_kws   else 1.0
    should_recall = len(should_found) / len(should_kws) if should_kws else 1.0
    total_gt      = len(must_kws) + len(should_kws)
    total_found   = len(must_found) + len(should_found)
    total_recall  = total_found / total_gt if total_gt else 1.0

    sources = list({src.strip() for p in retrieved_papers for src in p.source.split(",") if src.strip()})
    source_spread = len(sources)

    must_mrr = _compute_mrr(retrieved_titles, must_kws)
    should_mrr = _compute_mrr(retrieved_titles, should_kws)

    expanded_queries = results.get("expanded_queries", [])
    semantic_keywords = results.get("semantic_keywords", [])

    return {
        "topic":             topic_entry["topic"],
        "description":       topic_entry["description"],
        "total_retrieved":   len(retrieved_papers),
        "total_pool":        results.get("total_found", len(retrieved_papers)),
        "expanded_queries":  expanded_queries,
        "semantic_keywords": semantic_keywords,
        "sources":           sources,
        "source_spread":     source_spread,
        "must": {
            "targets":  must_kws,
            "found":    must_found,
            "missed":   [k for k in must_kws if k not in must_found],
            "recall_%": round(must_recall * 100, 1),
            "mrr":      round(must_mrr, 3),
        },
        "should": {
            "targets":  should_kws,
            "found":    should_found,
            "missed":   [k for k in should_kws if k not in should_found],
            "recall_%": round(should_recall * 100, 1),
            "mrr":      round(should_mrr, 3),
        },
        "total_recall_%": round(total_recall * 100, 1),
        "retrieved_titles": retrieved_titles,
    }


def _print_report(report: dict):
    """Pretty-print a coverage report to stdout."""
    sep = "-" * 70
    print(f"\n{sep}")
    print(f"  TOPIC : {report['topic'].upper()}")
    print(f"  DESC  : {report['description']}")
    print(sep)
    print(f"  Query variants  : {report['expanded_queries']}")
    print(f"  Semantic KWs    : {report['semantic_keywords']}")
    print(f"  Sources hit     : {report['sources']}  (spread={report['source_spread']})")
    print(f"  Total retrieved : {report['total_retrieved']}  /  pool={report['total_pool']}")
    print()
    m = report["must"]
    print(f"  [PASS] MUST-FIND recall  : {m['recall_%']}%  "
          f"({len(m['found'])}/{len(m['targets'])}) | MRR: {m['mrr']}")
    if m["missed"]:
        print(f"     [FAIL] Missed : {m['missed']}")
    if m["found"]:
        print(f"     [OK] Found  : {m['found']}")
    print()
    s = report["should"]
    print(f"  [INFO] SHOULD-FIND recall: {s['recall_%']}%  "
          f"({len(s['found'])}/{len(s['targets'])}) | MRR: {s['mrr']}")
    if s["missed"]:
        print(f"     [FAIL] Missed : {s['missed']}")
    if s["found"]:
        print(f"     [OK] Found  : {s['found']}")
    print()
    print(f"  [WIN] TOTAL RECALL      : {report['total_recall_%']}%")
    print(f"\n  Retrieved paper titles:")
    for t in report["retrieved_titles"]:
        print(f"    - {t}")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Parametrised coverage tests — one test per topic
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.coverage
@pytest.mark.integration
@pytest.mark.parametrize(
    "topic_entry",
    GROUND_TRUTH,
    ids=[t["topic"].replace(" ", "_") for t in GROUND_TRUTH],
)
class TestTopicCoverage:

    def test_must_find_recall(self, topic_entry):
        """The agent MUST retrieve at least 80 % of canonical landmark papers."""
        results = _run_search(topic_entry["topic"])
        report  = _build_report(topic_entry, results)
        _print_report(report)

        recall = report["must"]["recall_%"] / 100
        missed = report["must"]["missed"]

        assert recall >= MUST_RECALL_THRESHOLD, (
            f"[{topic_entry['topic']}] Must-find recall too low: "
            f"{report['must']['recall_%']}% < {MUST_RECALL_THRESHOLD*100}%\n"
            f"Missed papers: {missed}"
        )

    def test_should_find_recall(self, topic_entry):
        """The agent SHOULD retrieve at least 40 % of secondary landmark papers."""
        results = _run_search(topic_entry["topic"])
        report  = _build_report(topic_entry, results)

        recall = report["should"]["recall_%"] / 100
        missed = report["should"]["missed"]

        assert recall >= SHOULD_RECALL_THRESHOLD, (
            f"[{topic_entry['topic']}] Should-find recall too low: "
            f"{report['should']['recall_%']}% < {SHOULD_RECALL_THRESHOLD*100}%\n"
            f"Missed papers: {missed}"
        )

    def test_must_find_mrr(self, topic_entry):
        """The agent MUST rank canonical breakthrough papers high (MRR >= 0.1)."""
        results = _run_search(topic_entry["topic"])
        report = _build_report(topic_entry, results)
        
        # We skip if there are no must_find papers
        if not topic_entry.get("must_find"):
            return
            
        mrr = report["must"]["mrr"]
        
        # 0.1 MRR means it appears in the top 10 on average. Target is at least 0.1!
        assert mrr >= 0.1, (
            f"[{topic_entry['topic']}] Must-find MRR is too low: {mrr} < 0.1. "
            f"Result rankings are poor."
        )

    def test_multi_source_spread(self, topic_entry):
        """Results must come from at least 2 different academic sources."""
        results = _run_search(topic_entry["topic"])
        report  = _build_report(topic_entry, results)

        assert report["source_spread"] >= SOURCE_SPREAD_MIN, (
            f"[{topic_entry['topic']}] Only {report['source_spread']} source(s) found. "
            f"Expected at least {SOURCE_SPREAD_MIN}.\n"
            f"Sources: {report['sources']}"
        )

    def test_minimum_papers_returned(self, topic_entry):
        """At least 5 papers must be returned for any topic."""
        results = _run_search(topic_entry["topic"])
        report  = _build_report(topic_entry, results)

        assert report["total_retrieved"] >= 5, (
            f"[{topic_entry['topic']}] Only {report['total_retrieved']} papers returned."
        )

    def test_query_was_expanded(self, topic_entry):
        """The LLM must produce at least 2 expanded query variants."""
        results = _run_search(topic_entry["topic"])
        expanded = results.get("expanded_queries", [])

        assert len(expanded) >= 2, (
            f"[{topic_entry['topic']}] Only {len(expanded)} query variant(s). "
            f"Expected at least 2."
        )

    def test_semantic_keywords_generated(self, topic_entry):
        """At least 3 semantic keywords must be generated for any topic."""
        results = _run_search(topic_entry["topic"])
        kws = results.get("semantic_keywords", [])

        assert len(kws) >= 3, (
            f"[{topic_entry['topic']}] Only {len(kws)} semantic keyword(s) generated."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Summary report — generates a full JSON coverage report for all topics
# ─────────────────────────────────────────────────────────────────────────────

def pytest_sessionfinish_hook_coverage():
    """
    Call this manually (not auto-called by pytest) to generate a full
    JSON report across all topics:

        python -m tests.coverage.test_topic_coverage
    """
    print("\n\n" + "═" * 70)
    print("  DATA2DASH HYBRID SEARCH — FULL COVERAGE REPORT")
    print("═" * 70)

    overall_must   = []
    overall_should = []
    all_reports    = []

    for entry in GROUND_TRUTH:
        print(f"\n  Searching: {entry['topic']} ...", end="", flush=True)
        t0 = time.time()
        try:
            results = _run_search(entry["topic"])
            report  = _build_report(entry, results)
            _print_report(report)
            overall_must.append(report["must"]["recall_%"])
            overall_should.append(report["should"]["recall_%"])
            all_reports.append(report)
        except Exception as e:
            print(f"\n  [ERROR] for '{entry['topic']}': {e}")
        elapsed = time.time() - t0
        print(f"  [TIME] {elapsed:.1f}s")

    if overall_must:
        avg_must   = sum(overall_must)   / len(overall_must)
        avg_should = sum(overall_should) / len(overall_should)
        print("\n" + "=" * 70)
        print(f"  [STATS] AVERAGE MUST-FIND   RECALL : {avg_must:.1f}%")
        print(f"  [STATS] AVERAGE SHOULD-FIND RECALL : {avg_should:.1f}%")
        print("=" * 70)

        # Save JSON report
        report_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "coverage_report.json"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(all_reports, f, indent=2)
        print(f"\n  [SAVE] Full report saved to: {os.path.abspath(report_path)}")


if __name__ == "__main__":
    pytest_sessionfinish_hook_coverage()
