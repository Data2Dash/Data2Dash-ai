import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.services.query_intent import classify_query_intent


def test_survey_intent():
    i = classify_query_intent("Give me a survey of transformers", [])
    assert i.wants_survey is True


def test_seminal_intent():
    i = classify_query_intent("seminal papers on diffusion", [])
    assert i.wants_seminal is True


def test_recent_intent():
    i = classify_query_intent("latest LLM benchmarks 2025", [])
    assert i.wants_recent is True
    assert i.wants_benchmark is True
