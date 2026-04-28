"""
Test cache key stability — same input must produce identical SHA-256 keys.

This is critical: if cache keys drift between sessions, we re-pay for
identical API calls. Tests verify that key ordering, whitespace, and
JSON serialization don't affect the key.
"""

import json
import pytest
from src.paper4.llm.openai_client import _cache_key


def test_same_body_same_key():
    """Identical request bodies produce identical keys."""
    body = {"messages": [{"role": "user", "content": "hello"}], "model": "gpt-4o"}
    key1 = _cache_key("chat", "gpt-4o", body)
    key2 = _cache_key("chat", "gpt-4o", body)
    assert key1 == key2


def test_key_order_independent():
    """Dict key order shouldn't matter (JSON sort_keys=True)."""
    body1 = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}
    body2 = {"messages": [{"role": "user", "content": "hello"}], "model": "gpt-4o"}
    assert _cache_key("chat", "gpt-4o", body1) == _cache_key("chat", "gpt-4o", body2)


def test_different_content_different_key():
    """Different content must produce different keys."""
    body1 = {"messages": [{"role": "user", "content": "hello"}]}
    body2 = {"messages": [{"role": "user", "content": "world"}]}
    assert _cache_key("chat", "gpt-4o", body1) != _cache_key("chat", "gpt-4o", body2)


def test_different_endpoint_different_key():
    """Different endpoints produce different keys."""
    body = {"input": "some text"}
    assert _cache_key("chat", "gpt-4o", body) != _cache_key("embedding", "gpt-4o", body)


def test_different_model_different_key():
    """Different models produce different keys."""
    body = {"messages": [{"role": "user", "content": "hello"}]}
    assert _cache_key("chat", "gpt-4o", body) != _cache_key("chat", "gpt-4o-mini", body)


def test_key_is_hex_sha256():
    """Key should be a 64-char lowercase hex string (SHA-256)."""
    body = {"test": True}
    key = _cache_key("chat", "gpt-4o", body)
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)


def test_nested_dict_stability():
    """Nested structures should produce stable keys."""
    body = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is PICO?"},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "test", "strict": True},
        },
    }
    key1 = _cache_key("chat", "gpt-4o", body)
    key2 = _cache_key("chat", "gpt-4o", body)
    assert key1 == key2
