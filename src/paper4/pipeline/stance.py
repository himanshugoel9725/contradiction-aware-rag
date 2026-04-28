"""
Stance Classification — classify each document's stance toward a claim.

WHY: Before we can detect contradictions, we need to know whether each document
SUPPORTS, CONTRADICTS, or provides NOT_ENOUGH_INFO relative to the claim.
This is the foundation of contradiction-aware retrieval.

HOW: Uses OpenAI Structured Outputs with a 3-class enum. Each (claim, document)
pair gets an independent classification. Results are cached for resumability.

SCHEMA (Structured Output):
    {
        "label": "SUPPORT" | "CONTRADICT" | "NOT_ENOUGH_INFO",
        "confidence": float (0-1),
        "reasoning": str
    }
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field


# ── Stance Schema ─────────────────────────────────────────────────────────────


class StanceResult(BaseModel):
    """Classification result for one (claim, document) pair."""

    label: str = Field(description="SUPPORT, CONTRADICT, or NOT_ENOUGH_INFO")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    reasoning: str = Field(default="", description="Brief explanation of the classification")


STANCE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "stance_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "enum": ["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"],
                },
                "confidence": {"type": "number"},
                "reasoning": {"type": "string"},
            },
            "required": ["label", "confidence", "reasoning"],
            "additionalProperties": False,
        },
    },
}

STANCE_SYSTEM_PROMPT = """You are a biomedical evidence analyst. Given a claim and a document (abstract or passage), classify the document's stance toward the claim.

Labels:
- SUPPORT: The document provides evidence that the claim is true.
- CONTRADICT: The document provides evidence that the claim is false or contradicts it.
- NOT_ENOUGH_INFO: The document is related but does not clearly support or contradict.

Consider the specific biomedical details — population, intervention, outcome measures, and study design. Two studies may reach different conclusions due to different populations, dosages, or endpoints.

Return structured JSON with label, confidence (0-1), and brief reasoning."""


def _clip_text(text: str, max_chars: int = 4000) -> str:
    """Trim long passages to control token cost and avoid rate-limit spikes."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED]"


class StanceClassifier:
    """
    Classifies document stance (SUPPORT/CONTRADICT/NOT_ENOUGH_INFO) toward claims.

    Each (claim, document) pair gets an independent Structured Output call.
    Results are cached via the CachedOpenAIClient.
    """

    def __init__(self, client: Any) -> None:
        """
        Args:
            client: CachedOpenAIClient instance.
        """
        self.client = client
        self._failures: list[dict] = []

    def classify(self, claim: str, document_text: str, doc_id: str = "") -> dict[str, Any]:
        """
        Classify one document's stance toward a claim.

        Args:
            claim: Biomedical claim text.
            document_text: Document abstract or passage.
            doc_id: For logging/debugging.

        Returns:
            Dict with label, confidence, reasoning.
        """
        messages = [
            {"role": "system", "content": STANCE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Claim: {claim}\n\nDocument: {_clip_text(document_text)}",
            },
        ]

        try:
            response = self.client.chat(
                "stance_classification",
                messages=messages,
                response_format=STANCE_RESPONSE_FORMAT,
                max_tokens=512,
            )
            parsed = self.client.extract_json(response)
            result = StanceResult.model_validate(parsed)
            return result.model_dump()

        except Exception as e:
            self._failures.append({"claim": claim, "doc_id": doc_id, "error": str(e)})
            return {
                "label": "NOT_ENOUGH_INFO",
                "confidence": 0.0,
                "reasoning": f"Classification failed: {e}",
                "_fallback": True,
            }

    def classify_batch(
        self,
        claim: str,
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Classify stance for multiple documents against a single claim.

        Args:
            claim: The claim text.
            documents: List of dicts with 'doc_id' and 'text' keys.

        Returns:
            List of stance dicts, one per document (same order).
        """
        results = []
        for doc in documents:
            result = self.classify(
                claim=claim,
                document_text=doc.get("text", ""),
                doc_id=doc.get("doc_id", ""),
            )
            result["doc_id"] = doc.get("doc_id", "")
            results.append(result)
        return results

    @property
    def failure_count(self) -> int:
        return len(self._failures)
