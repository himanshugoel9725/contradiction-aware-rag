"""
Test postcheck verification module.
"""

import pytest

from src.paper4.pipeline.postcheck import (
    check_stance_coverage,
    check_citations,
    check_structure,
    run_postchecks,
)


def test_stance_coverage_both_mentioned():
    """Response mentioning both agreement and disagreement → pass."""
    text = "Studies support this claim. However, some evidence contradicts the finding."
    docs = [
        {"stance_label": "SUPPORT", "text": "..."},
        {"stance_label": "CONTRADICT", "text": "..."},
    ]
    result = check_stance_coverage(text, docs)
    assert result["response_mentions_both"] is True
    assert result["pass"] is True


def test_stance_coverage_missing_disagreement():
    """Evidence has both stances but response only mentions agreement → fail."""
    text = "All studies support this claim consistently."
    docs = [
        {"stance_label": "SUPPORT", "text": "..."},
        {"stance_label": "CONTRADICT", "text": "..."},
    ]
    result = check_stance_coverage(text, docs)
    assert result["contradiction_acknowledged"] is False
    assert result["pass"] is False


def test_citations_all_valid():
    """All cited document numbers are within range → pass."""
    text = "As shown in [Document 1] and [Document 2], the evidence supports..."
    docs = [{"text": "..."}, {"text": "..."}, {"text": "..."}]
    result = check_citations(text, docs)
    assert result["pass"] is True
    assert result["hallucinated_citations"] == 0


def test_citations_hallucinated():
    """Citing [Document 5] when only 3 docs exist → fail."""
    text = "According to [Document 5], the claim is true."
    docs = [{"text": "..."}, {"text": "..."}, {"text": "..."}]
    result = check_citations(text, docs)
    assert result["pass"] is False
    assert 5 in result["invalid_citations"]


def test_structure_strategy_c_complete():
    """Strategy C response with all 4 sections → pass."""
    text = """## Agreement
    Supporting studies show...
    ## Disagreement
    Contradicting studies show...
    ## Evidence Quality
    RCTs vs observational...
    ## Conclusion
    Overall, the evidence suggests..."""
    result = check_structure(text, strategy="C")
    assert result["pass"] is True
    assert result["missing_sections"] == []


def test_structure_strategy_c_missing():
    """Strategy C missing sections → fail."""
    text = "## Agreement\nStudies show...\n## Conclusion\nOverall..."
    result = check_structure(text, strategy="C")
    assert result["pass"] is False
    assert "disagreement" in result["missing_sections"]


def test_structure_non_c_strategy():
    """Non-C strategy always passes structure check."""
    result = check_structure("Any text", strategy="A")
    assert result["pass"] is True


def test_run_postchecks_combined():
    """Full postcheck pipeline produces combined result."""
    text = """## Agreement
    [Document 1] supports the claim.
    ## Disagreement
    [Document 2] contradicts the finding.
    ## Evidence Quality
    Both are RCTs.
    ## Conclusion
    Evidence is mixed."""
    docs = [
        {"stance_label": "SUPPORT", "text": "..."},
        {"stance_label": "CONTRADICT", "text": "..."},
    ]
    result = run_postchecks(text, docs, strategy="C")
    assert result["all_pass"] is True
