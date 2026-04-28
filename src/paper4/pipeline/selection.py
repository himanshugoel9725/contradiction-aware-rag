"""
Contradiction-Aware Evidence Selection.

WHY: Standard retrieval returns the most relevant documents, but they tend to
all support the same conclusion (confirmation bias). For contradiction-aware
synthesis, we NEED to include documents from both sides — SUPPORT and CONTRADICT.

HOW: After stance classification, this module re-ranks the retrieved set to
ensure both stances are represented in the final selection. It enforces a
minimum quota of each stance type and uses score-weighted interleaving.
"""

from __future__ import annotations

from typing import Any


def select_evidence(
    documents: list[dict[str, Any]],
    stances: list[dict[str, Any]],
    top_k: int = 10,
    min_support: int = 2,
    min_contradict: int = 2,
) -> list[dict[str, Any]]:
    """
    Select an evidence set that covers both SUPPORT and CONTRADICT stances.

    Strategy:
        1. Separate documents by stance label.
        2. Guarantee min_support SUPPORT docs and min_contradict CONTRADICT docs.
        3. Fill remaining slots with highest-scoring docs regardless of stance.

    Args:
        documents: Retrieved docs (each with 'doc_id', 'text', 'score').
        stances: Stance results (each with 'doc_id', 'label', 'confidence').
        top_k: Total number of docs to select.
        min_support: Minimum SUPPORT docs to include.
        min_contradict: Minimum CONTRADICT docs to include.

    Returns:
        Selected documents annotated with stance labels, sorted by score.
    """
    # Build stance lookup
    stance_map: dict[str, dict] = {s["doc_id"]: s for s in stances}

    # Tag each document with its stance
    tagged: list[dict[str, Any]] = []
    for doc in documents:
        doc_id = doc["doc_id"]
        stance = stance_map.get(doc_id, {"label": "NOT_ENOUGH_INFO", "confidence": 0.0})
        tagged.append({
            **doc,
            "stance_label": stance["label"],
            "stance_confidence": stance.get("confidence", 0.0),
        })

    # Separate by stance
    support = [d for d in tagged if d["stance_label"] == "SUPPORT"]
    contradict = [d for d in tagged if d["stance_label"] == "CONTRADICT"]
    other = [d for d in tagged if d["stance_label"] == "NOT_ENOUGH_INFO"]

    # Sort each group by retrieval score (descending)
    support.sort(key=lambda x: x.get("score", 0), reverse=True)
    contradict.sort(key=lambda x: x.get("score", 0), reverse=True)
    other.sort(key=lambda x: x.get("score", 0), reverse=True)

    # Build selected set with minimum quotas
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    # 1. Guarantee minimums
    for doc in support[:min_support]:
        if doc["doc_id"] not in selected_ids:
            selected.append(doc)
            selected_ids.add(doc["doc_id"])

    for doc in contradict[:min_contradict]:
        if doc["doc_id"] not in selected_ids:
            selected.append(doc)
            selected_ids.add(doc["doc_id"])

    # 2. Fill remaining from all docs by score
    all_remaining = [d for d in tagged if d["doc_id"] not in selected_ids]
    all_remaining.sort(key=lambda x: x.get("score", 0), reverse=True)

    for doc in all_remaining:
        if len(selected) >= top_k:
            break
        if doc["doc_id"] not in selected_ids:
            selected.append(doc)
            selected_ids.add(doc["doc_id"])

    # Sort final set by score
    selected.sort(key=lambda x: x.get("score", 0), reverse=True)

    return selected


def compute_stance_coverage(
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute stance coverage statistics for a selected evidence set.

    Returns:
        Dict with counts and ratios for each stance type.
    """
    total = len(selected)
    counts = {"SUPPORT": 0, "CONTRADICT": 0, "NOT_ENOUGH_INFO": 0}

    for doc in selected:
        label = doc.get("stance_label", "NOT_ENOUGH_INFO")
        if label in counts:
            counts[label] += 1

    return {
        "total": total,
        "counts": counts,
        "has_support": counts["SUPPORT"] > 0,
        "has_contradict": counts["CONTRADICT"] > 0,
        "has_both": counts["SUPPORT"] > 0 and counts["CONTRADICT"] > 0,
        "support_ratio": counts["SUPPORT"] / total if total else 0,
        "contradict_ratio": counts["CONTRADICT"] / total if total else 0,
    }
