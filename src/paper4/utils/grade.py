"""
GRADE evidence quality classifier.

WHY: Not all evidence is equal. A randomized controlled trial (RCT) outweighs
a case report. The GRADE framework (Grading of Recommendations, Assessment,
Development and Evaluations) assigns quality levels to evidence.

HOW: Uses PubMed Entrez API to look up publication type for each cited PMID,
then maps to a GRADE quality tier. Results are cached in JSON to avoid
repeated API calls (Entrez has a rate limit of ~3 requests/sec).

GRADE LEVELS:
    HIGH        — Systematic reviews, meta-analyses, large RCTs
    MODERATE    — Controlled trials, cohort studies
    LOW         — Case-control studies, observational studies
    VERY_LOW    — Case reports, expert opinion, editorials
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


# Mapping from PubMed publication types to GRADE levels
_PUBTYPE_TO_GRADE: dict[str, str] = {
    "Meta-Analysis": "HIGH",
    "Systematic Review": "HIGH",
    "Randomized Controlled Trial": "HIGH",
    "Clinical Trial": "MODERATE",
    "Controlled Clinical Trial": "MODERATE",
    "Clinical Trial, Phase III": "MODERATE",
    "Clinical Trial, Phase IV": "MODERATE",
    "Cohort Studies": "MODERATE",
    "Comparative Study": "MODERATE",
    "Multicenter Study": "MODERATE",
    "Observational Study": "LOW",
    "Case-Control Studies": "LOW",
    "Cross-Sectional Studies": "LOW",
    "Case Reports": "VERY_LOW",
    "Editorial": "VERY_LOW",
    "Letter": "VERY_LOW",
    "Comment": "VERY_LOW",
    "Review": "LOW",  # Non-systematic reviews
    "Journal Article": "LOW",  # Default for unspecified
}


_HIGH_KEYWORDS = (
    "randomized controlled trial",
    " rct ",
    "(rct)",
    "meta-analysis",
    "meta analysis",
    "systematic review",
    "double-blind",
    "double blind",
    "placebo-controlled",
    "placebo controlled",
    "cochrane",
)

_MODERATE_KEYWORDS = (
    "clinical trial",
    "cohort study",
    "cohort studies",
    "prospective study",
    "prospective studies",
    "retrospective study",
    "retrospective studies",
    "peer-reviewed",
    "peer reviewed",
    "published in",
    "journal of",
    "et al.",
)

_GRADE_WEIGHTS: dict[str, float] = {"HIGH": 1.0, "MODERATE": 0.6, "LOW": 0.3}


def tag_evidence_quality(documents: list[dict]) -> list[dict]:
    """
    Assign GRADE-style quality levels to documents using text heuristics.

    WHY: HealthContradict documents are ClueWeb web pages with no PMIDs, so
    the PubMed-based GRADEClassifier cannot be used. This function infers
    evidence quality from keywords in the document text.

    Grade levels:
        HIGH     — text signals RCT, meta-analysis, systematic review, etc.
        MODERATE — text signals clinical trial, cohort study, peer-reviewed, etc.
        LOW      — default for web pages, forum posts, anecdotal content.

    Args:
        documents: List of document dicts, each with at least a ``text`` field.

    Returns:
        The same list with ``grade_level`` and ``grade_weight`` added in-place.
    """
    counts: dict[str, int] = {"HIGH": 0, "MODERATE": 0, "LOW": 0}

    for doc in documents:
        text_lower = doc.get("text", "").lower()

        if any(kw in text_lower for kw in _HIGH_KEYWORDS):
            level = "HIGH"
        elif any(kw in text_lower for kw in _MODERATE_KEYWORDS):
            level = "MODERATE"
        else:
            level = "LOW"

        doc["grade_level"] = level
        doc["grade_weight"] = _GRADE_WEIGHTS[level]
        counts[level] += 1

    print(
        f"[GRADE] Evidence quality: {counts['HIGH']} HIGH, "
        f"{counts['MODERATE']} MODERATE, {counts['LOW']} LOW"
    )
    return documents


class GRADEClassifier:
    """
    Assigns GRADE quality levels to documents via PubMed publication type lookup.

    Caches results to a JSON file so lookups are done at most once per PMID.
    Respects Entrez rate limits (~3 requests/sec, with 0.34s delay).
    """

    def __init__(
        self,
        cache_path: str | Path = "./cache/grade_cache.json",
        email: str = "researcher@example.com",
    ) -> None:
        """
        Args:
            cache_path: Path to JSON cache file for Entrez results.
            email: Required by NCBI Entrez for rate-limiting compliance.
        """
        self.cache_path = Path(cache_path)
        self.email = email
        self._cache: dict[str, dict] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load existing cache from disk."""
        if self.cache_path.exists():
            with open(self.cache_path, "r") as f:
                self._cache = json.load(f)

    def _save_cache(self) -> None:
        """Persist cache to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f, indent=2)

    def classify(self, pmid: str) -> dict[str, Any]:
        """
        Look up GRADE level for a PMID.

        Args:
            pmid: PubMed ID (numeric string).

        Returns:
            Dict with grade_level, publication_types, source.
        """
        pmid = str(pmid).strip()

        if pmid in self._cache:
            return self._cache[pmid]

        # Fetch from Entrez
        pub_types = self._fetch_pub_types(pmid)
        grade = self._map_to_grade(pub_types)

        result = {
            "pmid": pmid,
            "publication_types": pub_types,
            "grade_level": grade,
            "source": "entrez",
        }
        self._cache[pmid] = result
        self._save_cache()
        return result

    def _fetch_pub_types(self, pmid: str) -> list[str]:
        """Fetch publication types from PubMed Entrez API."""
        try:
            from Bio import Entrez

            Entrez.email = self.email
            time.sleep(0.34)  # Rate limit compliance

            handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml", retmode="xml")
            from Bio import Medline
            import xml.etree.ElementTree as ET

            tree = ET.parse(handle)
            handle.close()

            pub_types = []
            for pt in tree.findall(".//PublicationType"):
                if pt.text:
                    pub_types.append(pt.text)

            return pub_types if pub_types else ["Journal Article"]

        except Exception as e:
            return ["Unknown"]

    def _map_to_grade(self, pub_types: list[str]) -> str:
        """Map publication types to highest applicable GRADE level."""
        grade_order = {"HIGH": 0, "MODERATE": 1, "LOW": 2, "VERY_LOW": 3}
        best_grade = "VERY_LOW"

        for pt in pub_types:
            grade = _PUBTYPE_TO_GRADE.get(pt, "VERY_LOW")
            if grade_order.get(grade, 3) < grade_order.get(best_grade, 3):
                best_grade = grade

        return best_grade

    def classify_batch(self, pmids: list[str]) -> list[dict[str, Any]]:
        """Classify multiple PMIDs."""
        return [self.classify(pmid) for pmid in pmids]

    def assign_to_documents(
        self,
        documents: list[dict[str, Any]],
        pmid_key: str = "pmid",
    ) -> list[dict[str, Any]]:
        """
        Add GRADE info to documents that have PMIDs.

        Modifies documents in-place and returns them.
        """
        for doc in documents:
            pmid = doc.get(pmid_key) or doc.get("metadata", {}).get(pmid_key)
            if pmid:
                grade_info = self.classify(str(pmid))
                doc["grade_level"] = grade_info["grade_level"]
                doc["publication_types"] = grade_info["publication_types"]
            else:
                doc["grade_level"] = "UNKNOWN"
                doc["publication_types"] = []
        return documents
