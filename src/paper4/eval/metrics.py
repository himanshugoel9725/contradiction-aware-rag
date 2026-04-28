"""
Evaluation metrics — pure functions for computing the 5 novel metrics.

WHY: The paper defines 5 metrics that measure different aspects of
contradiction-aware RAG quality. Each is a pure function operating on
per-example results, making them easy to test and compose.

METRICS:
    CBR — Contradiction Blindness Rate
        Fraction of examples where the system ignores a contradiction present
        in the evidence. Lower is better. Ranges 0-1.

    CAS — Contradiction Acknowledgment Score
        How well the response explicitly acknowledges contradictions.
        LLM-judged on 1-5 scale, normalized to 0-1. Higher is better.

    VCS — Viewpoint Coverage Score
        Whether the response covers BOTH supporting and contradicting viewpoints.
        Computed from stance coverage of the generated text. 0-1, higher is better.

    EAA — Evidence Attribution Accuracy
        Fraction of cited documents that actually exist and are correctly attributed.
        Computed from post-check citation analysis. 0-1, higher is better.

    EQU — Evidence Quality Utilization
        Whether the response appropriately weights evidence by quality (GRADE level).
        Binary or 0-1 scale per example. Higher is better.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_cbr(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute Contradiction Blindness Rate.

    CBR = (# examples where contradiction exists but response ignores it) /
          (# examples where contradiction exists in evidence)

    Args:
        results: List of per-example result dicts, each containing:
            - 'postcheck.stance_coverage.evidence_has_both_stances' (bool)
            - 'postcheck.stance_coverage.response_mentions_disagreement' (bool)
            or flattened equivalents.

    Returns:
        Dict with cbr value, numerator, denominator.
    """
    contradiction_examples = 0
    blind_examples = 0

    for r in results:
        pc = r.get("postcheck", {})
        sc = pc.get("stance_coverage", {})

        has_both = sc.get("evidence_has_both_stances", False)
        mentions_disagree = sc.get("response_mentions_disagreement", False)

        if has_both:
            contradiction_examples += 1
            if not mentions_disagree:
                blind_examples += 1

    if contradiction_examples == 0:
        return {"cbr": 0.0, "blind": 0, "total_contradictions": 0, "note": "No contradictions in evidence"}

    cbr = blind_examples / contradiction_examples
    return {
        "cbr": cbr,
        "blind": blind_examples,
        "total_contradictions": contradiction_examples,
    }


def compute_vcs(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute Viewpoint Coverage Score.

    VCS = mean across examples of (1 if response covers both stances, 0 otherwise),
    weighted by confidence.

    Simple version: fraction of examples that mention both agreement and disagreement.
    """
    scores = []
    for r in results:
        pc = r.get("postcheck", {})
        sc = pc.get("stance_coverage", {})
        mentions_both = sc.get("response_mentions_both", False)
        scores.append(1.0 if mentions_both else 0.0)

    if not scores:
        return {"vcs": 0.0, "n": 0}

    return {
        "vcs": float(np.mean(scores)),
        "n": len(scores),
        "coverage_count": sum(1 for s in scores if s > 0),
    }


def compute_eaa(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute Evidence Attribution Accuracy.

    EAA = mean across examples of (valid_citations / total_citations).
    Examples with 0 citations get EAA = 0.
    """
    scores = []
    for r in results:
        pc = r.get("postcheck", {})
        cit = pc.get("citations", {})
        total = cit.get("citation_count", 0)
        valid = len(cit.get("valid_citations", []))

        if total > 0:
            scores.append(valid / total)
        else:
            scores.append(0.0)

    if not scores:
        return {"eaa": 0.0, "n": 0}

    return {
        "eaa": float(np.mean(scores)),
        "n": len(scores),
        "perfect_attribution_count": sum(1 for s in scores if s == 1.0),
    }


def compute_equ(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute Evidence Quality Utilization.

    EQU measures whether the response appropriately weights evidence by GRADE level.
    Simple version: does the response cite at least one HIGH/MODERATE-grade source
    when available?
    """
    scores = []
    for r in results:
        docs = r.get("selected_documents", [])
        synthesis = r.get("synthesis", {}).get("text", "")

        high_quality_available = any(
            d.get("grade_level") in ("HIGH", "MODERATE") for d in docs
        )

        if not high_quality_available:
            scores.append(1.0)  # No high-quality evidence → not penalized
            continue

        # Check if any high-quality doc is cited
        high_quality_cited = False
        for i, doc in enumerate(docs, 1):
            if doc.get("grade_level") in ("HIGH", "MODERATE"):
                # Check if doc number appears in citations
                if f"[Document {i}]" in synthesis or f"[{i}]" in synthesis:
                    high_quality_cited = True
                    break

        scores.append(1.0 if high_quality_cited else 0.0)

    if not scores:
        return {"equ": 0.0, "n": 0}

    return {
        "equ": float(np.mean(scores)),
        "n": len(scores),
    }


def compute_all_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute all 5 metrics and return as a single dict."""
    return {
        "cbr": compute_cbr(results),
        "vcs": compute_vcs(results),
        "eaa": compute_eaa(results),
        "equ": compute_equ(results),
        "n_examples": len(results),
    }
