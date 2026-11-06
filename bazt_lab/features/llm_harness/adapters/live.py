"""The live LLM backend: the llm_client scaffolding (OpenRouter base URL, SQLite request cache) with the provider pinned.

The only module of the feature that reaches the network, the openai SDK, llm_client, swarmlab (the kickoff guard) or the
API key. The key comes only from the environment (OPENROUTER_API_KEY; .env loaded with python-dotenv without overriding)
and is never logged. Nothing is sent before the kickoff time or without allow_live.

Requests are cached by llm_client.SQLiteLLMCache.make_key(model, messages, temperature, salt), so identical requests of
different arms (the round-0 prompts, a repeated text) are paid once; the cache also keeps token counts, which a hit
reports again. A reply still empty after every retry is returned empty (D114, unanswerable) and never cached as an answer.
"""
from __future__ import annotations

import asyncio
import os
import random
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import List, Mapping

import openai
from dotenv import load_dotenv

from bazt_lab.core import public as core
from bazt_lab.features.llm_harness.private.backend import Completion

CRN_SALT = "llm_harness"
MAX_TOKENS = 1024            # G-Safeguard's own limit
RETRIES = 4


def backend(settings: Mapping, *, allow_live: bool, cache_path: Path) -> LiveBackend:
    import swarmlab  # the kickoff time is defined there, in one place
    swarmlab._kickoff_guard(SimpleNamespace(mode="llm", allow_live_llm=allow_live))      # SystemExit before kickoff
    load_dotenv()
    key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set: put it in .env or the environment")
    providers = core.get(settings, "llm.providers")
    temperature = float(core.get(settings, "llm.temperature"))
    from openai import AsyncOpenAI

    import llm_client
    def client():
        return AsyncOpenAI(base_url=llm_client.OPENROUTER_BASE_URL, api_key=key, timeout=60.0, max_retries=0)
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    return LiveBackend(client, llm_client.SQLiteLLMCache(str(cache_path)), providers, temperature)


BadRequestError = openai.BadRequestError


def open_cache(path):
    """The SQLite request cache of llm_client (also for tests, which may not import llm_client themselves)."""
    import llm_client
    return llm_client.SQLiteLLMCache(str(path))


def bad_request_error(message: str = "bad request"):
    """An HTTP 400 as the openai SDK raises it (for tests of the retry rule)."""
    import httpx
    return openai.BadRequestError(message, response=httpx.Response(400, request=httpx.Request("POST", "https://x")),
                                  body=None)


class LiveBackend:
    retry_sleep = None             # tests set 0.0; None = exponential backoff with jitter

    def __init__(self, client_factory, cache, providers: Mapping, temperature: float) -> None:
        """`client_factory()` builds an AsyncOpenAI client. A client's connections belong to the event loop it was first
        used in, and every conversation runs in a new loop (asyncio.run), so one client is built per loop and closed in
        that loop when the conversation ends (see `_client`)."""
        self.client_factory, self.cache, self.providers, self.temperature = client_factory, cache, providers, temperature
        self._clients: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._closers: set = set()                   # the pending close tasks, held so none is collected before it runs
        # the cache keeps text and tokens only; what the response reported about its provider is kept beside it
        cache.conn.execute("CREATE TABLE IF NOT EXISTS provenance (k TEXT PRIMARY KEY, provider TEXT, quantization TEXT)")
        # D114: requests whose reply stayed empty after every retry. Never a cached answer: a later request is sent once
        # more, and an empty reply then ends it at once (no second round of retries)
        cache.conn.execute("CREATE TABLE IF NOT EXISTS empty_replies (k TEXT PRIMARY KEY)")
        cache.conn.commit()

    def _client(self, loop: asyncio.AbstractEventLoop):
        """The client of the running loop, built on first use. With it starts a task that only waits; asyncio.run cancels
        it when the conversation's coroutine returns, while the loop still runs, and the task then closes the client
        there. A client left open keeps its connections, and through them its closed loop, alive (about 3 MB per
        conversation); one closed after its loop is gone fails with 'Event loop is closed'."""
        if loop not in self._clients:
            client = self._clients[loop] = self.client_factory()

            async def close_at_end() -> None:
                try:
                    await loop.create_future()
                finally:
                    self._clients.pop(loop, None)
                    close = getattr(client, "close", None)             # AsyncOpenAI has one; a bare stand-in may not
                    if callable(close):
                        await close()
            closer = loop.create_task(close_at_end())
            self._closers.add(closer)
            closer.add_done_callback(self._closers.discard)
        return self._clients[loop]

    async def complete(self, messages: List[dict], model: str) -> Completion:
        if model not in self.providers:
            raise core.Unset(f"llm.providers.{model}")
        client = self._client(asyncio.get_running_loop())
        key = self.cache.make_key(model, messages, self.temperature, CRN_SALT + "|" + str(self.providers[model]))
        row = self.cache.conn.execute("SELECT text, pt, ct FROM cache WHERE k=?", (key,)).fetchone()
        if row is not None:
            self.cache.hits += 1
            seen = self.cache.conn.execute("SELECT provider, quantization FROM provenance WHERE k=?", (key,)).fetchone()
            return Completion(row[0], int(row[1]), int(row[2]), *(seen or (None, None)))
        known_empty = self.cache.conn.execute("SELECT 1 FROM empty_replies WHERE k=?", (key,)).fetchone() is not None
        last = None
        empty = None                         # the empty reply of the last attempt, if it was one
        for attempt in range(RETRIES + 1):
            try:
                r = await client.chat.completions.create(
                    model=model, messages=messages, temperature=self.temperature, max_tokens=MAX_TOKENS,
                    extra_body={"provider": {"order": [self.providers[model]], "allow_fallbacks": False}})
                if not r.choices:
                    raise RuntimeError("OpenRouter returned no choices")
                text = (r.choices[0].message.content or "").strip()
                usage = getattr(r, "usage", None)
                pt, ct = int(getattr(usage, "prompt_tokens", 0) or 0), int(getattr(usage, "completion_tokens", 0) or 0)
                extra = getattr(r, "model_extra", None) or {}
                provider = extra.get("provider") or getattr(r, "provider", None)
                quantization = extra.get("quantization")
                if not text:                                 # never cached as an answer; retried (D114)
                    last, empty = RuntimeError("the response has empty content"), Completion("", pt, ct, provider,
                                                                                              quantization)
                    if known_empty:                          # already empty after every retry once: no second round
                        return empty
                    await asyncio.sleep(self._pause(attempt))
                    continue
                self.cache.put(key, model, text, pt, ct)
                self.cache.conn.execute("INSERT OR REPLACE INTO provenance VALUES (?,?,?)", (key, provider, quantization))
                self.cache.conn.execute("DELETE FROM empty_replies WHERE k=?", (key,))
                self.cache.conn.commit()
                return Completion(text, pt, ct, provider, quantization)
            except openai.BadRequestError:
                raise                                       # HTTP 400: the request itself is wrong, retrying cannot help
            except Exception as e:
                last, empty = e, None
                await asyncio.sleep(self._pause(attempt))
        if empty is not None:        # D114: empty after every retry: an unanswerable reply, the run continues
            self.cache.conn.execute("INSERT OR REPLACE INTO empty_replies VALUES (?)", (key,))
            self.cache.conn.commit()
            return empty
        raise RuntimeError(f"LLM call failed after {RETRIES + 1} attempts: {last!r}")

    def _pause(self, attempt: int) -> float:
        return min(30.0, 2 ** attempt + random.random()) if self.retry_sleep is None else self.retry_sleep  # noqa: S311
