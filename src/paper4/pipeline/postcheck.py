"""
Post-generation verification — sanity checks on synthesized responses.

WHY: Even with structured prompting, the LLM may omit contradictions, fail
to cite sources, or hallucinate citations. Post-check catches these issues
before the response is recorded as a final result.

CHECKS:
    1. Stance coverage: Does the response mention both supporting and contradicting evidence?
    2. Citation attribution: Does each cited document actually exist in the retrieved set?
    3. Structure compliance: For Strategy C, does the output have all 4 required sections?
"""

from __future__ import annotations

import re
from typing import Any


def _effective_doc_label(doc: dict[str, Any]) -> str:
    """
    Return the effective stance label for a document.

    HealthContradict doc_b (doc_id ending in ``_doc1``) is always CONTRADICT
    regardless of what was stored in the 'label' field (which may be wrong due
    to a historic data-loader bug where pair_type was never set).
    """
    label = doc.get("label")
    if label and label not in ("SUPPORT", "UNKNOWN", None):
        return label
    # HealthContradict doc_b pattern: hc_<topic>_doc1 or hc_<topic>_<instance>_doc1
    doc_id = str(doc.get("doc_id", ""))
    if re.search(r"^hc_\S+_doc1$", doc_id):
        return "CONTRADICT"
    return label or "UNKNOWN"


_DISMISSAL_PHRASES = (
    "no contradictions",
    "no direct contradiction",
    "no significant contradiction",
    "there is no contradiction",
    "there are no contradictions",
    "no conflicting",
    "both documents agree",
    "documents are consistent",
    "no disagreement",
    "evidence does not contradict",
    "no opposing",
)


def _has_substantive_disagreement(text: str) -> bool:
    """
    Return True only when a ``## Disagreement`` / ``## Contradictory`` section
    contains substantive disagreement content — not a dismissal.

    This prevents Strategy C's mandatory ``## Disagreement`` header from
    trivially satisfying the disagreement check even when the body says
    "no contradictions were found."

    Args:
        text: Full response text from the synthesis generator.

    Returns:
        True if the section exists and contains non-trivial disagreement content.
    """
    header_pattern = re.compile(
        r"##\s*(Disagreement|Contradictory)[^\n]*\n(.*?)(?=\n##|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    match = header_pattern.search(text)
    if not match:
        return False

    section_body = match.group(2).strip()

    # Too short to be substantive
    if len(section_body) < 30:
        return False

    body_lower = section_body.lower()
    if any(phrase in body_lower for phrase in _DISMISSAL_PHRASES):
        return False

    return True


def check_stance_coverage(
    response_text: str,
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Check whether the response mentions both SUPPORT and CONTRADICT evidence.

    Heuristic: looks for explicit multi-viewpoint phrases rather than generic
    words that appear naturally in medical prose (e.g. "support", "consistent").
    """
    # Require explicit multi-viewpoint phrases to avoid false positives from
    # medical prose that naturally uses words like "support" or "contradict".
    agree_keywords = [
        "the evidence supports", "studies support", "research supports",
        "evidence confirms", "findings support", "data support",
        "consistent with the claim", "in support of",
    ]
    disagree_keywords = [
        "the evidence contradicts", "studies contradict", "research contradicts",
        "conflicting evidence", "conflicting results", "conflicting findings",
        "inconsistent evidence", "inconsistent results", "contradictory evidence",
        "contradictory findings", "the evidence is mixed", "mixed evidence",
        "studies disagree", "on the other hand", "however, other studies",
        "however, some studies", "in contrast,", "contrary to",
        # NOTE: "## disagreement" intentionally removed — Strategy C always emits
        # this header even when the content says "no contradictions found",
        # causing trivially inflated VCS. Use _has_substantive_disagreement() instead.
    ]

    text_lower = response_text.lower()
    has_agree = any(kw in text_lower for kw in agree_keywords)
    has_disagree = (
        any(kw in text_lower for kw in disagree_keywords)
        or _has_substantive_disagreement(response_text)
    )

    # Check if documents actually contain both stances using the ground-truth
    # dataset annotation ('label'), not the runtime stance classifier prediction
    # ('stance_label'). The classifier frequently outputs NOT_ENOUGH_INFO, which
    # would incorrectly suppress CBR for cases where the retrieved set genuinely
    # contains both SUPPORT and CONTRADICT documents.
    # _effective_doc_label also infers CONTRADICT for healthcontradict doc_b
    # documents (doc_id ending in _doc1) to handle a historic labelling bug.
    stance_labels = {_effective_doc_label(d) for d in documents}
    evidence_has_both = "SUPPORT" in stance_labels and "CONTRADICT" in stance_labels

    return {
        "response_mentions_agreement": has_agree,
        "response_mentions_disagreement": has_disagree,
        "response_mentions_both": has_agree and has_disagree,
        "evidence_has_both_stances": evidence_has_both,
        "contradiction_acknowledged": has_disagree if evidence_has_both else True,
        "pass": (has_disagree if evidence_has_both else True),
    }


def check_citations(
    response_text: str,
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Verify that cited document references actually exist in the retrieved set.

    Looks for patterns like [Document 1], [Doc 1], [1], etc.
    """
    # Extract citation references — match both bracketed [Document N] / [N]
    # and prose forms "Document N" / "document N" that the LLM may produce.
    citation_pattern = r"\[(?:Document |Doc )?(\d+)\]|(?:Document|document) (\d+)\b"
    matches = re.findall(citation_pattern, response_text)
    cited_numbers = set(int(g) for tup in matches for g in tup if g)

    valid_range = set(range(1, len(documents) + 1))
    valid_citations = cited_numbers & valid_range
    invalid_citations = cited_numbers - valid_range

    return {
        "cited_numbers": sorted(cited_numbers),
        "valid_citations": sorted(valid_citations),
        "invalid_citations": sorted(invalid_citations),
        "total_documents": len(documents),
        "citation_count": len(cited_numbers),
        "hallucinated_citations": len(invalid_citations),
        "pass": len(invalid_citations) == 0,
    }


def check_structure(
    response_text: str,
    strategy: str = "C",
) -> dict[str, Any]:
    """
    For Strategy C, verify the response has all 4 required sections.

    Expected sections: Agreement, Disagreement, Evidence Quality, Conclusion.
    """
    if strategy != "C":
        return {"pass": True, "strategy": strategy, "note": "Structure check only applies to strategy C"}

    required_sections = ["agreement", "disagreement", "evidence quality", "conclusion"]
    text_lower = response_text.lower()

    found = {}
    for section in required_sections:
        # Look for section headers (## Section or **Section** patterns)
        pattern = rf"(?:##\s*{section}|\*\*{section}\*\*|{section}:)"
        found[section] = bool(re.search(pattern, text_lower))

    missing = [s for s, present in found.items() if not present]

    return {
        "strategy": strategy,
        "sections_found": {k: v for k, v in found.items()},
        "missing_sections": missing,
        "pass": len(missing) == 0,
    }


def run_postchecks(
    response_text: str,
    documents: list[dict[str, Any]],
    strategy: str = "C",
) -> dict[str, Any]:
    """
    Run all post-generation checks and return combined results.

    Returns:
        Dict with results from each check and an overall pass/fail.
    """
    stance = check_stance_coverage(response_text, documents)
    citations = check_citations(response_text, documents)
    structure = check_structure(response_text, strategy)

    all_pass = stance["pass"] and citations["pass"] and structure["pass"]

    return {
        "stance_coverage": stance,
        "citations": citations,
        "structure": structure,
        "all_pass": all_pass,
    }
