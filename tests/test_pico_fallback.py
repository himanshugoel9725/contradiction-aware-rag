"""
Test PICO extraction fallback behavior.

When the API call fails (no key, malformed response, etc.), the PICO extractor
should gracefully fall back to a default result rather than crashing the pipeline.
"""

import pytest
from unittest.mock import MagicMock

from src.paper4.pipeline.pico import PICOExtractor


class _MockClientFailure:
    """Mock client that always raises on chat()."""

    def chat(self, *args, **kwargs):
        raise RuntimeError("API key not set")


class _MockClientBadJson:
    """Mock client that returns invalid JSON."""

    def chat(self, *args, **kwargs):
        return {"choices": [{"message": {"content": "not json"}}]}

    @staticmethod
    def extract_json(response):
        import json
        content = response["choices"][0]["message"]["content"]
        return json.loads(content)  # Will raise


class _MockClientGood:
    """Mock client that returns valid PICO JSON."""

    def chat(self, *args, **kwargs):
        return {
            "choices": [{
                "message": {
                    "content": '{"population":"adults","intervention":"aspirin","comparator":"placebo","outcome":"heart attack risk","confidence":0.9}'
                }
            }]
        }

    @staticmethod
    def extract_json(response):
        import json
        return json.loads(response["choices"][0]["message"]["content"])


def test_fallback_on_api_failure():
    """API failure → fallback result with confidence=0.0."""
    extractor = PICOExtractor(_MockClientFailure())
    result = extractor.extract("Aspirin reduces heart attacks")

    assert result["confidence"] == 0.0
    assert result.get("_fallback") is True
    assert extractor.failure_count == 1


def test_fallback_on_bad_json():
    """Bad JSON → fallback result."""
    extractor = PICOExtractor(_MockClientBadJson())
    result = extractor.extract("Some claim")

    assert result["confidence"] == 0.0
    assert result.get("_fallback") is True


def test_successful_extraction():
    """Valid response → proper PICO elements."""
    extractor = PICOExtractor(_MockClientGood())
    result = extractor.extract("Aspirin reduces heart attacks in adults")

    assert result["population"] == "adults"
    assert result["intervention"] == "aspirin"
    assert result["comparator"] == "placebo"
    assert result["outcome"] == "heart attack risk"
    assert result["confidence"] == 0.9
    assert extractor.failure_count == 0


def test_batch_extraction():
    """Batch extraction processes all claims."""
    extractor = PICOExtractor(_MockClientGood())
    results = extractor.extract_batch(["Claim 1", "Claim 2", "Claim 3"])

    assert len(results) == 3
    assert all(r["confidence"] == 0.9 for r in results)
