"""
PICO Extraction — structured extraction of Population, Intervention, Comparator, Outcome.

WHY: Decomposing a biomedical claim into PICO elements allows fine-grained
matching between claims and evidence. Two documents may agree on Population
but contradict on Outcome, which is valuable for contradiction detection.

HOW: Uses OpenAI Structured Outputs (JSON schema enforcement) to ensure the
output always has the right shape. Falls back to raw-claim NLI if extraction
fails after retries.

SCHEMA (Structured Output):
    {
        "population": str,
        "intervention": str,
        "comparator": str | null,
        "outcome": str,
        "confidence": float (0-1)
    }
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field


# ── PICO Schema ───────────────────────────────────────────────────────────────


class PICOElements(BaseModel):
    """Structured PICO extraction result."""

    population: str = Field(description="Study population or patient group")
    intervention: str = Field(description="Treatment, exposure, or test being evaluated")
    comparator: str | None = Field(default=None, description="Comparison group or control (null if not stated)")
    outcome: str = Field(description="Measured health outcome or endpoint")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Extraction confidence 0-1")


# OpenAI Structured Outputs response_format
PICO_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "pico_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "population": {"type": "string"},
                "intervention": {"type": "string"},
                "comparator": {"type": ["string", "null"]},
                "outcome": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["population", "intervention", "comparator", "outcome", "confidence"],
            "additionalProperties": False,
        },
    },
}

PICO_SYSTEM_PROMPT = """You are a biomedical information extraction expert.
Given a biomedical claim, extract the PICO elements:
- Population: the patient group or study subjects
- Intervention: the treatment, drug, exposure, or test evaluated
- Comparator: the control or comparison group (null if not stated)
- Outcome: the measured health outcome or endpoint

Return structured JSON. If an element is unclear, make your best inference and
set confidence lower. Never leave population, intervention, or outcome empty."""


class PICOExtractor:
    """
    Extracts PICO elements from biomedical claims using OpenAI Structured Outputs.

    Falls back to a simplified extraction (claim as-is) if the API call fails,
    so the pipeline never blocks on a single extraction failure.
    """

    def __init__(self, client: Any) -> None:
        """
        Args:
            client: CachedOpenAIClient instance.
        """
        self.client = client
        self._failures: list[dict] = []

    def extract(self, claim: str) -> dict[str, Any]:
        """
        Extract PICO elements from a single claim.

        Args:
            claim: Biomedical claim text.

        Returns:
            Dict with population, intervention, comparator, outcome, confidence.
            On failure, returns fallback with confidence=0.0.
        """
        messages = [
            {"role": "system", "content": PICO_SYSTEM_PROMPT},
            {"role": "user", "content": claim},
        ]

        try:
            response = self.client.chat(
                "pico_extraction",
                messages=messages,
                response_format=PICO_RESPONSE_FORMAT,
                max_tokens=512,
            )
            parsed = self.client.extract_json(response)

            # Validate with Pydantic
            pico = PICOElements.model_validate(parsed)
            return pico.model_dump()

        except Exception as e:
            # Fallback: use the raw claim as all fields
            self._failures.append({"claim": claim, "error": str(e)})
            return {
                "population": "unspecified",
                "intervention": claim,
                "comparator": None,
                "outcome": "unspecified",
                "confidence": 0.0,
                "_fallback": True,
                "_error": str(e),
            }

    def extract_batch(self, claims: list[str]) -> list[dict[str, Any]]:
        """Extract PICO for multiple claims."""
        return [self.extract(claim) for claim in claims]

    @property
    def failure_count(self) -> int:
        return len(self._failures)

    @property
    def failures(self) -> list[dict]:
        return list(self._failures)
