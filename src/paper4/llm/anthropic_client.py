"""
Cached Anthropic (Claude) client — drop-in replacement for CachedOpenAIClient.

WHY: Same caching, cost-tracking, and retry logic as the OpenAI client but
calls the Anthropic Messages API. All downstream code (judge, pico, synthesis,
stance) uses the same .chat() / .extract_json() / .extract_content() interface —
no changes needed anywhere else.

HOW:
    - All chat calls flow through CachedAnthropicClient.chat()
    - Responses are stored in SQLite (anthropic_cache.db) keyed by SHA-256
    - Structured outputs use Anthropic tool_use: the response_format JSON schema
      is converted to a tool definition; the tool_use content block is extracted
      and serialised to a JSON string, then returned in a fake OpenAI-shaped dict
      so extract_json() works unchanged
    - .embed() delegates to CachedOpenAIClient (Anthropic has no embeddings API)

BUDGET SAFETY:
    - Accepts max_budget_usd — raises BudgetExhaustedError when crossed
    - Catches Anthropic credit-exhaustion API errors and re-raises as
      BudgetExhaustedError with a clear user-facing message
    - Prints a warning at 50%, 75%, 90% of max_budget_usd

MODEL ROUTING TABLE:
    Task type             → Model
    ──────────────────────  ───────────────────────────────
    pico_extraction       → claude-3-5-haiku-20241022
    stance_classification → claude-3-5-haiku-20241022
    judge_simple          → claude-3-5-haiku-20241022
    cbr_baseline          → claude-3-5-haiku-20241022
    generation            → claude-3-5-sonnet-20241022
    judge_complex         → claude-3-5-sonnet-20241022
    longcontext           → claude-3-5-sonnet-20241022
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


# ── Exception ────────────────────────────────────────────────────────────────


class BudgetExhaustedError(RuntimeError):
    """Raised when the Anthropic credit balance or configured budget is exhausted."""
    pass


# ── Model routing ─────────────────────────────────────────────────────────────

ANTHROPIC_ROUTING: dict[str, str] = {
    "pico_extraction": "claude-haiku-4-5-20251001",
    "stance_classification": "claude-haiku-4-5-20251001",
    "judge_simple": "claude-haiku-4-5-20251001",
    "cbr_baseline": "claude-haiku-4-5-20251001",
    "generation": "claude-sonnet-4-5-20250929",
    "judge_complex": "claude-sonnet-4-5-20250929",
    "longcontext": "claude-sonnet-4-5-20250929",
}

# Pricing per 1M tokens (input, output)
ANTHROPIC_PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-sonnet-4-5-20250929": (3.00, 15.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-opus-4-20250514": (15.00, 75.00),
    # Legacy names kept for cache compatibility
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
}

# Budget warning thresholds (fractions of max_budget_usd)
_WARN_AT = (0.50, 0.75, 0.90)


# ── Cost tracker ──────────────────────────────────────────────────────────────


@dataclass
class AnthropicCostTracker:
    """Thread-safe cost accumulator for Anthropic API calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_calls: int = 0
    cache_hits: int = 0
    cost_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _warned_at: set = field(default_factory=set, repr=False)

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        cached: bool = False,
        max_budget: float | None = None,
    ) -> None:
        with self._lock:
            self.total_calls += 1
            if cached:
                self.cache_hits += 1
                return

            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

            in_rate, out_rate = ANTHROPIC_PRICING.get(model, (3.00, 15.00))
            call_cost = (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000
            self.cost_usd += call_cost

            if max_budget and max_budget > 0:
                ratio = self.cost_usd / max_budget
                for threshold in _WARN_AT:
                    if ratio >= threshold and threshold not in self._warned_at:
                        self._warned_at.add(threshold)
                        print(
                            f"\n⚠️  [Budget] ${self.cost_usd:.2f} spent of "
                            f"${max_budget:.2f} ({ratio:.0%})"
                        )

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_calls": self.total_calls,
                "cache_hits": self.cache_hits,
                "cache_hit_rate": (
                    f"{self.cache_hits / self.total_calls:.1%}"
                    if self.total_calls else "N/A"
                ),
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "estimated_cost_usd": f"${self.cost_usd:.4f}",
            }


# ── SQLite cache (shared schema with openai_client) ───────────────────────────

_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    endpoint TEXT NOT NULL,
    model TEXT NOT NULL,
    request_body TEXT NOT NULL,
    response_body TEXT NOT NULL,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);
"""


class _CacheDB:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._local = threading.local()
        conn = self._conn()
        conn.executescript(_CACHE_SCHEMA)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.commit()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path, timeout=30)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def get(self, key: str) -> dict | None:
        row = self._conn().execute(
            "SELECT response_body, prompt_tokens, completion_tokens FROM cache WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return {
            "response_body": row["response_body"],
            "prompt_tokens": row["prompt_tokens"],
            "completion_tokens": row["completion_tokens"],
        }

    def put(
        self,
        key: str,
        endpoint: str,
        model: str,
        request_body: str,
        response_body: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        self._conn().execute(
            """INSERT OR IGNORE INTO cache
               (key, endpoint, model, request_body, response_body,
                prompt_tokens, completion_tokens, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                key, endpoint, model, request_body, response_body,
                prompt_tokens, completion_tokens,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn().commit()


def _cache_key(endpoint: str, model: str, request_body: dict) -> str:
    canonical = json.dumps(request_body, sort_keys=True, separators=(",", ":"))
    payload = f"{endpoint}:{model}:{canonical}"
    return hashlib.sha256(payload.encode()).hexdigest()


# ── Schema conversion: OpenAI response_format → Anthropic tool ───────────────


def _response_format_to_tool(response_format: dict) -> tuple[list[dict], dict]:
    """
    Convert an OpenAI JSON-schema response_format to an Anthropic tool + tool_choice.

    The OpenAI format is:
        {"type": "json_schema", "json_schema": {"name": "...", "schema": {...}}}

    The Anthropic tool format is:
        tools = [{"name": "...", "description": "...", "input_schema": {...}}]
        tool_choice = {"type": "tool", "name": "..."}
    """
    schema_def = response_format.get("json_schema", response_format)
    name = schema_def.get("name", "structured_output")
    schema = schema_def.get("schema", schema_def)

    # Strip "strict" and "additionalProperties: false" — not needed for Anthropic
    clean_schema = {k: v for k, v in schema.items() if k != "additionalProperties"}

    tool = {
        "name": name,
        "description": f"Return a structured {name} response.",
        "input_schema": clean_schema,
    }
    tool_choice = {"type": "tool", "name": name}
    return [tool], tool_choice


# ── Main client ───────────────────────────────────────────────────────────────


class CachedAnthropicClient:
    """
    Cached Anthropic Messages API client with OpenAI-compatible interface.

    All methods match CachedOpenAIClient so downstream code is unchanged:
        .chat(task_type, messages, *, temperature, max_tokens, response_format, model_override)
        .extract_content(response)  — static
        .extract_json(response)     — static
        .embed(texts, ...)          — delegates to CachedOpenAIClient
        .print_cost_summary()

    Example:
        client = CachedAnthropicClient()
        resp = client.chat("pico_extraction", messages=[...])
        text = CachedAnthropicClient.extract_content(resp)
    """

    def __init__(
        self,
        cache_dir: str | Path = "./cache",
        api_key: str | None = None,
        max_budget_usd: float | None = None,
        openai_api_key: str | None = None,
    ) -> None:
        """
        Args:
            cache_dir: Directory for SQLite cache. Created if missing.
            api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
            max_budget_usd: If set, raises BudgetExhaustedError when exceeded.
            openai_api_key: OpenAI key for .embed() delegation. Falls back to
                            OPENAI_API_KEY env var.
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable."
            )

        self._client = anthropic.Anthropic(api_key=resolved_key)
        self._cache = _CacheDB(cache_dir / "anthropic_cache.db")
        self.cost = AnthropicCostTracker()
        self._max_budget = max_budget_usd
        self._openai_api_key = openai_api_key
        self._cache_dir = cache_dir

        # Lazy-loaded embedding / OpenAI-fallback clients
        self._embedding_client: Any = None
        self._openai_fallback: Any = None  # used for gpt-* model_override

    # ── Chat completions ──────────────────────────────────────────────────

    def chat(
        self,
        task_type: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: dict | None = None,
        model_override: str | None = None,
    ) -> dict[str, Any]:
        """
        Send a chat completion request with caching, retries, and cost tracking.

        The response is returned in OpenAI-compatible shape:
            {"choices": [{"message": {"content": "..."}}], "usage": {...}, "model": "..."}

        Structured output (response_format): the JSON schema is converted to an
        Anthropic tool call. The tool_use block's ``input`` dict is serialised to
        a JSON string placed in ``choices[0].message.content`` so extract_json()
        works unchanged.

        Raises:
            BudgetExhaustedError: If configured budget is exceeded or Anthropic
                reports insufficient credits.
        """
        # Budget pre-check
        if self._max_budget and self.cost.cost_usd >= self._max_budget:
            raise BudgetExhaustedError(
                f"⚠️  Budget cap of ${self._max_budget:.2f} reached "
                f"(spent ${self.cost.cost_usd:.2f}). Add funds and rerun with --resume."
            )

        model = model_override or ANTHROPIC_ROUTING.get(task_type, "claude-3-5-sonnet-20241022")

        # Delegate gpt-* models to CachedOpenAIClient
        if model.startswith("gpt-"):
            if self._openai_fallback is None:
                from src.paper4.llm.openai_client import CachedOpenAIClient
                self._openai_fallback = CachedOpenAIClient(
                    cache_dir=self._cache_dir,
                    api_key=self._openai_api_key,
                )
            return self._openai_fallback.chat(
                task_type,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                model_override=model,
            )

        # Separate system messages from user/assistant messages
        system_parts: list[str] = []
        user_messages: list[dict] = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                user_messages.append(msg)

        system_text = "\n\n".join(system_parts) if system_parts else None

        # Build canonical request body for cache key
        request_body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": user_messages,
        }
        if system_text:
            request_body["system"] = system_text
        if response_format:
            request_body["_response_format"] = response_format  # for cache key only

        key = _cache_key("chat", model, request_body)

        # Cache lookup
        cached = self._cache.get(key)
        if cached is not None:
            self.cost.record(model, cached["prompt_tokens"], cached["completion_tokens"], cached=True)
            return json.loads(cached["response_body"])

        # Live API call
        raw = self._call_api(
            model=model,
            system=system_text,
            messages=user_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
        )

        # Build OpenAI-shaped response dict
        response_dict = self._to_openai_shape(raw, response_format)

        input_tokens = raw.usage.input_tokens
        output_tokens = raw.usage.output_tokens

        response_json = json.dumps(response_dict, default=str)
        self._cache.put(
            key=key,
            endpoint="chat",
            model=model,
            request_body=json.dumps(request_body, sort_keys=True),
            response_body=response_json,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )

        self.cost.record(
            model, input_tokens, output_tokens,
            cached=False, max_budget=self._max_budget,
        )
        return response_dict

    @retry(
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APITimeoutError)),
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        reraise=True,
    )
    def _call_api(
        self,
        model: str,
        system: str | None,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        response_format: dict | None,
    ) -> Any:
        """Make the actual Anthropic API call. Handles tool_use for structured output."""
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        if response_format:
            tools, tool_choice = _response_format_to_tool(response_format)
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        try:
            return self._client.messages.create(**kwargs)
        except anthropic.PermissionDeniedError as e:
            if "credit" in str(e).lower() or "billing" in str(e).lower():
                raise BudgetExhaustedError(
                    "⚠️  Anthropic credits exhausted. Add funds at "
                    "https://console.anthropic.com/settings/billing then rerun with --resume."
                ) from e
            raise
        except anthropic.BadRequestError as e:
            msg = str(e).lower()
            if "credit" in msg or "billing" in msg or "balance" in msg:
                raise BudgetExhaustedError(
                    "⚠️  Anthropic credits exhausted. Add funds at "
                    "https://console.anthropic.com/settings/billing then rerun with --resume."
                ) from e
            raise
        except anthropic.APIStatusError as e:
            if e.status_code in (402, 403):
                raise BudgetExhaustedError(
                    "⚠️  Anthropic credits exhausted (status "
                    f"{e.status_code}). Add funds at "
                    "https://console.anthropic.com/settings/billing then rerun with --resume."
                ) from e
            raise

    @staticmethod
    def _to_openai_shape(raw: Any, response_format: dict | None) -> dict[str, Any]:
        """Convert Anthropic response to OpenAI-compatible dict."""
        if response_format:
            # Find the tool_use content block
            tool_block = next(
                (b for b in raw.content if b.type == "tool_use"), None
            )
            if tool_block is not None:
                content_str = json.dumps(tool_block.input)
            else:
                # Fallback: look for text block
                text_block = next(
                    (b for b in raw.content if b.type == "text"), None
                )
                content_str = text_block.text if text_block else "{}"
        else:
            text_block = next(
                (b for b in raw.content if b.type == "text"), None
            )
            content_str = text_block.text if text_block else ""

        return {
            "choices": [{"message": {"content": content_str, "role": "assistant"}}],
            "usage": {
                "prompt_tokens": raw.usage.input_tokens,
                "completion_tokens": raw.usage.output_tokens,
            },
            "model": raw.model,
        }

    # ── Convenience extractors (same interface as CachedOpenAIClient) ─────────

    @staticmethod
    def extract_content(response: dict[str, Any]) -> str:
        """Get the assistant's text from a chat completion response."""
        return response["choices"][0]["message"]["content"]

    @staticmethod
    def extract_json(response: dict[str, Any]) -> dict[str, Any]:
        """Parse JSON from the assistant's text (for structured outputs)."""
        content = response["choices"][0]["message"]["content"]
        return json.loads(content)

    # ── Embeddings (delegated to OpenAI) ─────────────────────────────────────

    def embed(
        self,
        texts: list[str],
        model: str = "text-embedding-3-large",
        *,
        batch_size: int = 100,
    ) -> list[list[float]]:
        """
        Embed texts via OpenAI (Anthropic has no embeddings API).

        Requires OPENAI_API_KEY to be set or openai_api_key passed at init.
        """
        if self._embedding_client is None:
            from src.paper4.llm.openai_client import CachedOpenAIClient
            self._embedding_client = CachedOpenAIClient(
                cache_dir=Path(self._cache.db_path).parent,
                api_key=self._openai_api_key,
            )
        return self._embedding_client.embed(texts, model=model, batch_size=batch_size)

    # ── Cost reporting ────────────────────────────────────────────────────────

    def print_cost_summary(self) -> None:
        """Print human-readable cost summary to stdout."""
        s = self.cost.summary()
        print("\n╔══════════════════════════════════════╗")
        print("║     Anthropic API Cost Summary       ║")
        print("╠══════════════════════════════════════╣")
        print(f"║  Total calls:       {s['total_calls']:>14}  ║")
        print(f"║  Cache hits:        {s['cache_hits']:>14}  ║")
        print(f"║  Cache hit rate:    {s['cache_hit_rate']:>14}  ║")
        print(f"║  Input tokens:      {s['input_tokens']:>14}  ║")
        print(f"║  Output tokens:     {s['output_tokens']:>14}  ║")
        print(f"║  Estimated cost:    {s['estimated_cost_usd']:>14}  ║")
        if self._max_budget:
            remaining = max(0.0, self._max_budget - self.cost.cost_usd)
            print(f"║  Budget remaining:  ${remaining:>13.4f}  ║")
        print("╚══════════════════════════════════════╝\n")

    def cache_stats(self) -> dict[str, int]:
        """Return cache size statistics."""
        conn = self._cache._conn()
        row = conn.execute(
            "SELECT COUNT(*) as cnt, COALESCE(SUM(prompt_tokens),0) as pt, "
            "COALESCE(SUM(completion_tokens),0) as ct FROM cache"
        ).fetchone()
        return {"rows": row["cnt"], "total_input_tokens": row["pt"], "total_output_tokens": row["ct"]}
