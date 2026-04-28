"""
Test evaluation metrics — verify correctness on toy examples with known values.
"""

import pytest
import numpy as np

from src.paper4.eval.metrics import (
    compute_cbr,
    compute_vcs,
    compute_eaa,
    compute_equ,
    compute_all_metrics,
)
from src.paper4.eval.bootstrap import bootstrap_ci, bootstrap_metric


# ── CBR Tests ─────────────────────────────────────────────────────────────────

def test_cbr_all_blind():
    """All examples have contradictions but none are acknowledged → CBR = 1.0."""
    results = [
        {"postcheck": {"stance_coverage": {"evidence_has_both_stances": True, "response_mentions_disagreement": False}}},
        {"postcheck": {"stance_coverage": {"evidence_has_both_stances": True, "response_mentions_disagreement": False}}},
    ]
    out = compute_cbr(results)
    assert out["cbr"] == 1.0
    assert out["blind"] == 2


def test_cbr_none_blind():
    """All contradictions are acknowledged → CBR = 0.0."""
    results = [
        {"postcheck": {"stance_coverage": {"evidence_has_both_stances": True, "response_mentions_disagreement": True}}},
        {"postcheck": {"stance_coverage": {"evidence_has_both_stances": True, "response_mentions_disagreement": True}}},
    ]
    out = compute_cbr(results)
    assert out["cbr"] == 0.0


def test_cbr_no_contradictions():
    """No contradictions in evidence → CBR = 0 (vacuously)."""
    results = [
        {"postcheck": {"stance_coverage": {"evidence_has_both_stances": False, "response_mentions_disagreement": False}}},
    ]
    out = compute_cbr(results)
    assert out["cbr"] == 0.0
    assert out["total_contradictions"] == 0


def test_cbr_mixed():
    """Half blind → CBR = 0.5."""
    results = [
        {"postcheck": {"stance_coverage": {"evidence_has_both_stances": True, "response_mentions_disagreement": True}}},
        {"postcheck": {"stance_coverage": {"evidence_has_both_stances": True, "response_mentions_disagreement": False}}},
    ]
    out = compute_cbr(results)
    assert out["cbr"] == 0.5


# ── VCS Tests ─────────────────────────────────────────────────────────────────

def test_vcs_all_covered():
    """All responses mention both stances → VCS = 1.0."""
    results = [
        {"postcheck": {"stance_coverage": {"response_mentions_both": True}}},
        {"postcheck": {"stance_coverage": {"response_mentions_both": True}}},
    ]
    assert compute_vcs(results)["vcs"] == 1.0


def test_vcs_none_covered():
    results = [
        {"postcheck": {"stance_coverage": {"response_mentions_both": False}}},
    ]
    assert compute_vcs(results)["vcs"] == 0.0


# ── EAA Tests ─────────────────────────────────────────────────────────────────

def test_eaa_perfect():
    """All citations are valid → EAA = 1.0."""
    results = [
        {"postcheck": {"citations": {"citation_count": 3, "valid_citations": [1, 2, 3]}}},
    ]
    assert compute_eaa(results)["eaa"] == 1.0


def test_eaa_half():
    results = [
        {"postcheck": {"citations": {"citation_count": 4, "valid_citations": [1, 2]}}},
    ]
    assert compute_eaa(results)["eaa"] == 0.5


def test_eaa_no_citations():
    results = [
        {"postcheck": {"citations": {"citation_count": 0, "valid_citations": []}}},
    ]
    assert compute_eaa(results)["eaa"] == 0.0


# ── Bootstrap Tests ───────────────────────────────────────────────────────────

def test_bootstrap_ci_deterministic():
    """Same seed → same CI."""
    values = [0.8, 0.9, 0.7, 0.85, 0.75]
    ci1 = bootstrap_ci(values, n_resamples=500, seed=42)
    ci2 = bootstrap_ci(values, n_resamples=500, seed=42)
    assert ci1["ci_lower"] == ci2["ci_lower"]
    assert ci1["ci_upper"] == ci2["ci_upper"]


def test_bootstrap_ci_contains_mean():
    """95% CI should contain the sample mean."""
    values = [0.5] * 100
    ci = bootstrap_ci(values, n_resamples=100, seed=42)
    assert ci["ci_lower"] <= ci["point_estimate"] <= ci["ci_upper"]


def test_bootstrap_single_value():
    """Single value → degenerate CI."""
    ci = bootstrap_ci([0.5])
    assert ci["ci_lower"] == ci["ci_upper"] == 0.5
    assert ci["ci_width"] == 0.0
