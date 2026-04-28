"""
Cached OpenAI client — single gateway for ALL LLM and embedding calls.

WHY: Every API call costs money. During development, identical prompts are sent
repeatedly (debug reruns, resumed experiments, etc.). A transparent SQLite cache
avoids re-paying for identical requests. It also captures exact model outputs for
reproducibility auditing.

HOW:
    1. All chat, structured-output, and embedding calls flow through CachedOpenAIClient.
    2. Before calling the API, we SHA-256 hash (endpoint + model + canonical JSON body).
       If a cache row exists, we return it immediately — zero cost.
    3. Retries use tenacity (exponential backoff, 8 attempts, 1-60 s).
    4. Every call's token counts are accumulated in a running cost tracker.
    5. Structured Outputs use openai's JSON-schema enforcement (response_format).

CACHE SCHEMA (SQLite):
    CREATE TABLE cache (
        key TEXT PRIMARY KEY,           -- SHA-256 hex
        endpoint TEXT NOT NULL,         -- 'chat' | 'embedding'
        model TEXT NOT NULL,
        request_body TEXT NOT NULL,     -- canonical JSON
        response_body TEXT NOT NULL,    -- raw JSON from API
        prompt_tokens INTEGER,
        completion_tokens INTEGER,
        created_at TEXT NOT NULL        -- ISO-8601
    );

MODEL ROUTING TABLE:
    Task type             → Model
    ──────────────────────  ──────────────
    pico_extraction       → gpt-4o-mini
    stance_classification → gpt-4o-mini
    generation            → gpt-4o
    judge_simple          → gpt-4o-mini
    judge_complex         → gpt-4o
    cbr_baseline          → gpt-4o-mini
    longcontext           → gpt-4o

USAGE:
    client = CachedOpenAIClient()
    response = client.chat("pico_extraction", messages=[...])
    embedding = client.embed(["some text"])
    client.print_cost_summary()
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

import tiktoken
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


# ── Model routing table ──────────────────────────────────────────────────────

MODEL_ROUTING: dict[str, str] = {
    "pico_extraction": "gpt-4o-mini",
    "stance_classification": "gpt-4o-mini",
    "generation": "gpt-4o",
    "judge_simple": "gpt-4o-mini",
    "judge_complex": "gpt-4o",
    "cbr_baseline": "gpt-4o-mini",
    "longcontext": "gpt-4o",
}

# Pricing per 1 M tokens (input, output) — update when OpenAI changes rates
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-large": (0.13, 0.0),  # embeddings have no output cost
}


# ── Cost tracker ──────────────────────────────────────────────────────────────


@dataclass
class CostTracker:
    """
    Accumulates token usage and dollar cost across all API calls in a session.

    Thread-safe: uses a lock because pipeline stages may overlap via asyncio/threads.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    total_calls: int = 0
    cache_hits: int = 0
    cost_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        cached: bool = False,
    ) -> None:
        """Record usage for one API call. If cached, cost is $0."""
        with self._lock:
            self.total_calls += 1
            if cached:
                self.cache_hits += 1
                return

            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens

            in_rate, out_rate = PRICING.get(model, (0.0, 0.0))
            self.cost_usd += (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000

    def record_embedding(self, model: str, token_count: int, *, cached: bool = False) -> None:
        with self._lock:
            self.total_calls += 1
            if cached:
                self.cache_hits += 1
                return

            self.embedding_tokens += token_count
            in_rate, _ = PRICING.get(model, (0.0, 0.0))
            self.cost_usd += (token_count * in_rate) / 1_000_000

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_calls": self.total_calls,
                "cache_hits": self.cache_hits,
                "cache_hit_rate": (
                    f"{self.cache_hits / self.total_calls:.1%}"
                    if self.total_calls
                    else "N/A"
                ),
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "embedding_tokens": self.embedding_tokens,
                "estimated_cost_usd": f"${self.cost_usd:.4f}",
            }


# ── SQLite cache layer ────────────────────────────────────────────────────────

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
    """
    Thread-safe SQLite wrapper for the response cache.

    Uses WAL mode for concurrent reads during multi-threaded pipeline stages.
    Connection-per-thread pattern avoids SQLite's threading restrictions.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._local = threading.local()

        # Initialize schema on first creation
        conn = self._conn()
        conn.executescript(_CACHE_SCHEMA)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.commit()

    def _conn(self) -> sqlite3.Connection:
        """Get or create a connection for the current thread."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path, timeout=30)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def get(self, key: str) -> dict | None:
        """Fetch a cached response by SHA-256 key. Returns None on miss."""
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
        """Insert a response into the cache. Ignores conflicts (idempotent)."""
        self._conn().execute(
            """INSERT OR IGNORE INTO cache
               (key, endpoint, model, request_body, response_body,
                prompt_tokens, completion_tokens, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                key,
                endpoint,
                model,
                request_body,
                response_body,
                prompt_tokens,
                completion_tokens,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn().commit()

    def stats(self) -> dict[str, int]:
        """Return row count and total stored tokens."""
        row = self._conn().execute(
            "SELECT COUNT(*) as cnt, COALESCE(SUM(prompt_tokens),0) as pt, "
            "COALESCE(SUM(completion_tokens),0) as ct FROM cache"
        ).fetchone()
        return {"rows": row["cnt"], "total_prompt_tokens": row["pt"], "total_completion_tokens": row["ct"]}


# ── Hashing utility ──────────────────────────────────────────────────────────


def _cache_key(endpoint: str, model: str, request_body: dict) -> str:
    """
    Deterministic SHA-256 cache key.

    Canonical form: endpoint + model + JSON with sorted keys (no whitespace).
    """
    canonical = json.dumps(request_body, sort_keys=True, separators=(",", ":"))
    payload = f"{endpoint}:{model}:{canonical}"
    return hashlib.sha256(payload.encode()).hexdigest()


# ── Token counting ────────────────────────────────────────────────────────────

_TOKEN_ENCODINGS: dict[str, Any] = {}


def _count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens using tiktoken. Caches encoding objects."""
    if model not in _TOKEN_ENCODINGS:
        try:
            _TOKEN_ENCODINGS[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            _TOKEN_ENCODINGS[model] = tiktoken.get_encoding("cl100k_base")
    return len(_TOKEN_ENCODINGS[model].encode(text))


def _truncate_for_embedding(text: str, model: str, max_tokens: int = 8192) -> str:
    """Trim embedding input to model token limit to avoid 400 errors."""
    if model not in _TOKEN_ENCODINGS:
        try:
            _TOKEN_ENCODINGS[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            _TOKEN_ENCODINGS[model] = tiktoken.get_encoding("cl100k_base")

    enc = _TOKEN_ENCODINGS[model]
    token_ids = enc.encode(text)
    if len(token_ids) <= max_tokens:
        return text
    return enc.decode(token_ids[:max_tokens])


# ── Main client ───────────────────────────────────────────────────────────────


class CachedOpenAIClient:
    """
    Single gateway for all OpenAI API calls with transparent SQLite caching,
    automatic retries, cost tracking, and structured output support.

    All pipeline modules should use this client rather than calling openai directly.

    Example:
        client = CachedOpenAIClient()
        resp = client.chat("pico_extraction", messages=[
            {"role": "system", "content": "Extract PICO elements."},
            {"role": "user", "content": claim_text},
        ])
        print(resp["choices"][0]["message"]["content"])
        client.print_cost_summary()
    """

    def __init__(
        self,
        cache_dir: str | Path = "./cache",
        api_key: str | None = None,
    ) -> None:
        """
        Args:
            cache_dir: Directory for SQLite cache file. Created if missing.
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self._cache = _CacheDB(cache_dir / "openai_cache.db")
        self.cost = CostTracker()

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

        Args:
            task_type: Key into MODEL_ROUTING (e.g. 'pico_extraction').
            messages: Standard OpenAI messages list.
            temperature: Sampling temperature. Default 0 for reproducibility.
            max_tokens: Output token cap.
            response_format: If set, passed directly as `response_format` for
                             structured outputs (JSON schema enforcement).
            model_override: Override the routing table with a specific model name.

        Returns:
            Parsed JSON response dict (same shape as openai API response).
        """
        model = model_override or MODEL_ROUTING.get(task_type, "gpt-4o")

        request_body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            request_body["response_format"] = response_format

        key = _cache_key("chat", model, request_body)

        # Cache lookup
        cached = self._cache.get(key)
        if cached is not None:
            self.cost.record(model, cached["prompt_tokens"], cached["completion_tokens"], cached=True)
            return json.loads(cached["response_body"])

        # Live API call with retries
        raw_response = self._call_chat_api(request_body)
        response_dict = raw_response.model_dump()

        # Extract token usage
        usage = response_dict.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Store in cache
        response_json = json.dumps(response_dict, default=str)
        self._cache.put(
            key=key,
            endpoint="chat",
            model=model,
            request_body=json.dumps(request_body, sort_keys=True),
            response_body=response_json,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        self.cost.record(model, prompt_tokens, completion_tokens, cached=False)
        return response_dict

    @retry(
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        reraise=True,
    )
    def _call_chat_api(self, request_body: dict[str, Any]) -> Any:
        """Make the actual API call. Decorated with tenacity retry logic."""
        return self._client.chat.completions.create(**request_body)

    # ── Convenience: extract content from chat response ────────────────────

    @staticmethod
    def extract_content(response: dict[str, Any]) -> str:
        """Get the assistant's text from a chat completion response."""
        return response["choices"][0]["message"]["content"]

    @staticmethod
    def extract_json(response: dict[str, Any]) -> dict[str, Any]:
        """Parse JSON from the assistant's text (for structured outputs)."""
        content = response["choices"][0]["message"]["content"]
        return json.loads(content)

    # ── Embeddings ────────────────────────────────────────────────────────

    def embed(
        self,
        texts: list[str],
        model: str = "text-embedding-3-large",
        *,
        batch_size: int = 100,
    ) -> list[list[float]]:
        """
        Embed a list of texts with caching and batching.

        Each unique text gets its own cache entry. Batches are sent in groups
        of `batch_size` to stay within API limits.

        Returns:
            List of embedding vectors (same order as input texts).
        """
        # OpenAI embeddings enforce a maximum input length; clamp oversized docs.
        prepared_texts = [_truncate_for_embedding(t, model=model, max_tokens=8192) for t in texts]
        all_embeddings: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []

        # Check cache for each text
        for i, text in enumerate(prepared_texts):
            key = _cache_key("embedding", model, {"input": text})
            cached = self._cache.get(key)
            if cached is not None:
                all_embeddings[i] = json.loads(cached["response_body"])
                self.cost.record_embedding(model, cached["prompt_tokens"], cached=True)
            else:
                uncached_indices.append(i)

        # Batch API calls for uncached texts
        for batch_start in range(0, len(uncached_indices), batch_size):
            batch_idx = uncached_indices[batch_start : batch_start + batch_size]
            batch_texts = [prepared_texts[i] for i in batch_idx]

            raw = self._call_embedding_api(model=model, texts=batch_texts)

            for j, idx in enumerate(batch_idx):
                embedding = raw.data[j].embedding
                token_count = _count_tokens(prepared_texts[idx], model)

                # Cache individual embedding
                key = _cache_key("embedding", model, {"input": prepared_texts[idx]})
                self._cache.put(
                    key=key,
                    endpoint="embedding",
                    model=model,
                    request_body=json.dumps({"input": prepared_texts[idx]}),
                    response_body=json.dumps(embedding),
                    prompt_tokens=token_count,
                    completion_tokens=0,
                )

                self.cost.record_embedding(model, token_count, cached=False)
                all_embeddings[idx] = embedding

        return all_embeddings  # type: ignore[return-value]

    @retry(
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        reraise=True,
    )
    def _call_embedding_api(self, model: str, texts: list[str]) -> Any:
        """Make embedding API call with retries."""
        return self._client.embeddings.create(model=model, input=texts)

    # ── Cost reporting ────────────────────────────────────────────────────

    def print_cost_summary(self) -> None:
        """Print human-readable cost summary to stdout."""
        s = self.cost.summary()
        print("\n╔══════════════════════════════════════╗")
        print("║       OpenAI API Cost Summary        ║")
        print("╠══════════════════════════════════════╣")
        print(f"║  Total calls:       {s['total_calls']:>14}  ║")
        print(f"║  Cache hits:        {s['cache_hits']:>14}  ║")
        print(f"║  Cache hit rate:    {s['cache_hit_rate']:>14}  ║")
        print(f"║  Prompt tokens:     {s['prompt_tokens']:>14}  ║")
        print(f"║  Completion tokens: {s['completion_tokens']:>14}  ║")
        print(f"║  Embedding tokens:  {s['embedding_tokens']:>14}  ║")
        print(f"║  Estimated cost:    {s['estimated_cost_usd']:>14}  ║")
        print("╚══════════════════════════════════════╝\n")

    def cache_stats(self) -> dict[str, int]:
        """Return cache size statistics."""
        return self._cache.stats()


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Smoke test: make one real chat call, verify cache hit on repeat, print cost.

    Requires OPENAI_API_KEY to be set. Run:
        python -m src.paper4.llm.openai_client
    """
    import tempfile

    print("=== CachedOpenAIClient Smoke Test ===\n")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[SKIP] OPENAI_API_KEY not set. Set it to run live smoke test.")
        print("[INFO] Testing cache key stability instead...\n")

        # Still test cache key determinism
        body1 = {"messages": [{"role": "user", "content": "hello"}], "model": "gpt-4o"}
        body2 = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}
        key1 = _cache_key("chat", "gpt-4o", body1)
        key2 = _cache_key("chat", "gpt-4o", body2)
        assert key1 == key2, f"Cache keys differ for equivalent bodies:\n  {key1}\n  {key2}"
        print("[PASS] Cache key stability: equivalent dicts produce same SHA-256")

        # Test cost tracker
        ct = CostTracker()
        ct.record("gpt-4o-mini", 100, 50, cached=False)
        ct.record("gpt-4o", 200, 100, cached=False)
        ct.record("gpt-4o", 0, 0, cached=True)
        s = ct.summary()
        assert s["total_calls"] == 3
        assert s["cache_hits"] == 1
        assert s["prompt_tokens"] == 300
        print(f"[PASS] Cost tracker: {s}")

        print("\n[PASS] All offline tests passed")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = CachedOpenAIClient(cache_dir=tmpdir)

            messages = [
                {"role": "system", "content": "Reply with exactly one word."},
                {"role": "user", "content": "What color is the sky on a clear day?"},
            ]

            # First call — should hit API
            print("[1] Making live API call (gpt-4o-mini)...")
            t0 = time.perf_counter()
            resp1 = client.chat("cbr_baseline", messages=messages)
            t1 = time.perf_counter()
            content1 = CachedOpenAIClient.extract_content(resp1)
            print(f"    Response: '{content1}'  ({t1-t0:.2f}s)")

            # Second call — should be cached
            print("[2] Repeating same call (should be cached)...")
            t2 = time.perf_counter()
            resp2 = client.chat("cbr_baseline", messages=messages)
            t3 = time.perf_counter()
            content2 = CachedOpenAIClient.extract_content(resp2)
            print(f"    Response: '{content2}'  ({t3-t2:.4f}s)")

            assert content1 == content2, "Cached response differs from original!"
            assert (t3 - t2) < 0.1, "Cache lookup too slow — likely not cached"

            client.print_cost_summary()

            stats = client.cache_stats()
            print(f"Cache stats: {stats}")
            assert stats["rows"] == 1, f"Expected 1 cached row, got {stats['rows']}"

            print("\n[PASS] Live smoke test passed — caching works!")
