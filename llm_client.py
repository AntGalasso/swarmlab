
"""llm_client.py -- PRE-EXISTING SCAFFOLDING (written before the Swarm Alignment Hackathon kickoff).

Async OpenRouter client + deterministic SQLite cache (common random numbers across policy arms).

COMPLIANCE GUARDRAIL
  * The ONLY live call this file makes by itself is run_connectivity_smoke_test(), which sends the
    harmless prompt "Ping." with the system prompt "Reply with the single word READY.".
  * No swarm task prompt, decoy string or canary tag is ever sent from here unless swarmlab.py is run
    with --allow-live-llm (and passes its own kickoff-date guard).

Cache key = SHA-256 of (model, messages, round(temperature, 4), crn_salt). Two policy arms that reach the
same (seed, step, agent) with an identical prompt therefore hit the disk instead of the API: identical
pre-intervention histories cost $0 and return in well under a millisecond.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import sqlite3
import sys
import time
from typing import Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

try:  # optional: .env support
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PRIMARY_MODEL = "qwen/qwen-2.5-7b-instruct"
FALLBACK_MODEL = "meta-llama/llama-3.1-8b-instruct"
H4_MODEL = "openai/gpt-4o-mini"

Messages = List[Dict[str, str]]
# transport(model, messages, temperature, max_tokens) -> (text, prompt_tokens, completion_tokens)
Transport = Callable[[str, Messages, float, int], Awaitable[Tuple[str, int, int]]]


# --------------------------------------------------------------------------------------- cache
class SQLiteLLMCache:
    """Disk cache in WAL mode. Tracks hits, live calls and token usage."""

    def __init__(self, db_path: str = "llm_cache.sqlite"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, model TEXT, text TEXT, pt INTEGER, ct INTEGER)"
        )
        self.conn.commit()
        self.hits = 0
        self.live_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @staticmethod
    def make_key(model: str, messages: Messages, temperature: float, crn_salt: str) -> str:
        payload = json.dumps([model, messages, round(float(temperature), 4), crn_salt], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _ensure_open(self) -> None:
        """Raise a clear error if the cache was closed (instead of a cryptic AttributeError)."""
        if self.conn is None:
            raise RuntimeError(
                "SQLiteLLMCache: cache is closed (close() was called); "
                "create a new SQLiteLLMCache instance to keep using the cache."
            )

    def get(self, key: str) -> Optional[str]:
        self._ensure_open()
        row = self.conn.execute("SELECT text FROM cache WHERE k=?", (key,)).fetchone()
        if row is not None:
            self.hits += 1
            return row[0]
        return None

    def put(self, key: str, model: str, text: str, pt: int, ct: int) -> None:
        self._ensure_open()
        self.conn.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?)", (key, model, text, pt, ct))
        self.conn.commit()
        self.live_calls += 1
        self.prompt_tokens += pt
        self.completion_tokens += ct

    def stats(self) -> Dict[str, int]:
        return {
            "cache_hits": self.hits,
            "live_calls": self.live_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    def close(self) -> None:
        """Close the SQLite connection and release database resources."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None


# --------------------------------------------------------------------------------- transports
def _api_key() -> Optional[str]:
    return os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")


def make_openrouter_transport() -> Transport:
    """Real transport (AsyncOpenAI pointed at OpenRouter). Imported lazily so offline tests need no key."""
    from openai import AsyncOpenAI

    key = _api_key()
    if not key:
        raise RuntimeError("Missing OPENROUTER_API_KEY (or OPENAI_API_KEY). Put it in .env, never in the code.")
    # Bounded request timeout: a stalled upstream provider must not wedge a smoke test or a sprint run.
    # SDK-level retries are disabled because LLMBackend owns the retry/backoff policy.
    client = AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=key, timeout=60.0, max_retries=0)

    async def transport(model: str, messages: Messages, temperature: float, max_tokens: int):
        r = await client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        choices = getattr(r, "choices", None)
        if not choices:
            extra = getattr(r, "model_extra", None) or {}
            err = extra.get("error") or getattr(r, "error", None)
            raise RuntimeError(f"OpenRouter returned no choices (upstream error: {str(err)[:200]})")
        text = (r.choices[0].message.content or "").strip()
        u = getattr(r, "usage", None)
        return text, int(getattr(u, "prompt_tokens", 0) or 0), int(getattr(u, "completion_tokens", 0) or 0)

    return transport


# ------------------------------------------------------------------------------------ backend
class LLMBackend:
    """Cached, rate-limited, retrying completion backend. Use .generate(batch) from synchronous code."""

    def __init__(self, model: str, cache: SQLiteLLMCache, transport: Optional[Transport] = None,
                 max_concurrency: int = 40, temperature: float = 0.7, max_tokens: int = 90, retries: int = 4):
        self.model, self.cache = model, cache
        self.transport = transport
        self.max_concurrency, self.temperature, self.max_tokens, self.retries = max_concurrency, temperature, max_tokens, retries
        self._loop = asyncio.new_event_loop()
        self._sem: Optional[asyncio.Semaphore] = None

    async def _one(self, system: str, user: str, crn_salt: str, temperature: Optional[float] = None) -> str:
        temp = self.temperature if temperature is None else temperature
        messages: Messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        key = self.cache.make_key(self.model, messages, temp, crn_salt)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        if self.transport is None:
            self.transport = make_openrouter_transport()
        assert self._sem is not None
        last: Optional[Exception] = None
        async with self._sem:
            for attempt in range(self.retries + 1):
                try:
                    text, pt, ct = await self.transport(self.model, messages, temp, self.max_tokens)
                    self.cache.put(key, self.model, text, pt, ct)
                    return text
                except Exception as e:  # noqa: BLE001
                    last = e
                    await asyncio.sleep(min(30.0, (2 ** attempt) + random.random()))
        raise RuntimeError(f"LLM call failed after {self.retries + 1} attempts: {last!r}")

    async def _batch(self, reqs: Sequence[Tuple[str, str, str]], temperature: Optional[float]) -> List[str]:
        self._sem = asyncio.Semaphore(self.max_concurrency)
        return list(await asyncio.gather(*[self._one(s, u, c, temperature) for s, u, c in reqs]))

    def generate(self, reqs: Sequence[Tuple[str, str, str]], temperature: Optional[float] = None) -> List[str]:
        """reqs: list of (system, user, crn_salt). Returns texts in the same order."""
        return self._loop.run_until_complete(self._batch(reqs, temperature))
    def close(self) -> None:
        """Close the event loop owned by this backend."""
        if not self._loop.is_closed():
            self._loop.close()


def make_llm_policy(model: str, cache: SQLiteLLMCache, max_concurrency: int = 40, temperature: float = 0.7,
                    max_tokens: int = 90, transport: Optional[Transport] = None) -> LLMBackend:
    return LLMBackend(model, cache, transport, max_concurrency, temperature, max_tokens)


# ------------------------------------------------------------------------------- smoke test
def run_connectivity_smoke_test(db_path: str = "llm_cache.sqlite", transport: Optional[Transport] = None) -> bool:
    """The ONLY live call permitted before kickoff: 'Reply with the single word READY.' / 'Ping.'"""
    if transport is None and not _api_key():
        print("No OPENROUTER_API_KEY / OPENAI_API_KEY found (.env). Cannot run the smoke test.")
        return False
    cache = SQLiteLLMCache(db_path)
    try:
        system, user = "Reply with the single word READY.", "Ping."
        rows, all_ok = [], True
        for label, models in (("primary", [PRIMARY_MODEL, FALLBACK_MODEL]), ("H4", [H4_MODEL])):
            used, first, lat1 = None, "", 0.0
            for slug in models:
                # Probe, not resilience test: fail fast (one attempt) and fall back to the next slug.
                backend = LLMBackend(slug, cache, transport, max_concurrency=1, temperature=0.0,
                                     max_tokens=8, retries=0)
                try:
                    t0 = time.perf_counter()
                    first = backend.generate([(system, user, "smoke-0")], temperature=0.0)[0]
                    lat1 = (time.perf_counter() - t0) * 1000
                    used = slug
                    break
                except Exception as e:  # slug unavailable -> try the fallback
                    print(f"[{label}] {slug} failed: {e!s:.120}")
            if used is None:
                rows.append((label, models[0], "-", float("nan"), float("nan"), "FAIL"))
                all_ok = False
                continue
            backend = LLMBackend(used, cache, transport, max_concurrency=1, temperature=0.0,
                                 max_tokens=8, retries=0)
            t0 = time.perf_counter()
            second = backend.generate([(system, user, "smoke-0")], temperature=0.0)[0]
            lat2 = (time.perf_counter() - t0) * 1000
            ok = (second == first) and (lat2 < 50.0)
            all_ok &= ok
            rows.append((label, used, first[:20].replace("\n", " "), lat1, lat2, "PASS" if ok else "FAIL"))
        print(f"\n{'Arm':<8}{'Model slug':<34}{'Output':<22}{'Live ms':>9}{'Cache ms':>10}  Exact match")
        for a, m, o, l1, l2, s in rows:
            print(f"{a:<8}{m:<34}{o:<22}{l1:>9.0f}{l2:>10.2f}  {s}")
        print("cache stats:", cache.stats())
        return all_ok
    finally:
        cache.close()


if __name__ == "__main__":
    sys.exit(0 if run_connectivity_smoke_test() else 1)
