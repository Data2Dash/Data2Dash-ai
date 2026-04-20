"""Unit tests — landmark phrase anchor strength."""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.services.landmark_title_match import landmark_phrase_anchor_strength, normalize_landmark_title


def test_exact_match_full_strength():
    lm = normalize_landmark_title("Attention Is All You Need")
    assert landmark_phrase_anchor_strength(lm, lm) == 1.0


def test_prefix_small_extension_full_strength():
    lm = normalize_landmark_title("Attention Is All You Need")
    tight = normalize_landmark_title("Attention Is All You Need Revisited")
    assert landmark_phrase_anchor_strength(tight, lm) == 1.0


def test_prefix_three_extra_words_partial():
    lm = normalize_landmark_title("Attention Is All You Need")
    speech = normalize_landmark_title("Attention Is All You Need In Speech Separation")
    st = landmark_phrase_anchor_strength(speech, lm)
    assert 0.4 <= st < 1.0


def test_embedded_substring_is_derivative():
    lm = normalize_landmark_title("Attention Is All You Need")
    parody = normalize_landmark_title("Not All Attention Is All You Need")
    assert landmark_phrase_anchor_strength(parody, lm) == 0.35
