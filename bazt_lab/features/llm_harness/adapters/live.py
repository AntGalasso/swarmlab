"""The live LLM backend: the llm_client scaffolding (OpenRouter base URL, SQLite request cache) with the provider pinned.

The only module of the feature that reaches the network, the openai SDK, llm_client, swarmlab (the kickoff guard) or the
API key. The key comes only from the environment (OPENROUTER_API_KEY; .env loaded with python-dotenv without overriding)
and is never logged. Nothing is sent before the kickoff time or without allow_live.

Requests are cached by llm_client.SQLiteLLMCache.make_key(model, messages, temperature, salt), so identical requests of
different arms (the round-0 prompts, a repeated text) are paid once; the cache also keeps token counts, which a hit
reports again.
"""
from __future__ import annotations

import asyncio
import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import List, Mapping

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
    client = AsyncOpenAI(base_url=llm_client.OPENROUTER_BASE_URL, api_key=key, timeout=60.0, max_retries=0)
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    return LiveBackend(client, llm_client.SQLiteLLMCache(str(cache_path)), providers, temperature)


class LiveBackend:
    def __init__(self, client, cache, providers: Mapping, temperature: float) -> None:
        self.client, self.cache, self.providers, self.temperature = client, cache, providers, temperature
        # the cache keeps text and tokens only; what the response reported about its provider is kept beside it
        cache.conn.execute("CREATE TABLE IF NOT EXISTS provenance (k TEXT PRIMARY KEY, provider TEXT, quantization TEXT)")
        cache.conn.commit()

    async def complete(self, messages: List[dict], model: str) -> Completion:
        if model not in self.providers:
            raise core.Unset(f"llm.providers.{model}")
        key = self.cache.make_key(model, messages, self.temperature, CRN_SALT + "|" + str(self.providers[model]))
        row = self.cache.conn.execute("SELECT text, pt, ct FROM cache WHERE k=?", (key,)).fetchone()
        if row is not None:
            self.cache.hits += 1
            seen = self.cache.conn.execute("SELECT provider, quantization FROM provenance WHERE k=?", (key,)).fetchone()
            return Completion(row[0], int(row[1]), int(row[2]), *(seen or (None, None)))
        last = None
        for attempt in range(RETRIES + 1):
            try:
                r = await self.client.chat.completions.create(
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
                self.cache.put(key, model, text, pt, ct)
                self.cache.conn.execute("INSERT OR REPLACE INTO provenance VALUES (?,?,?)", (key, provider, quantization))
                self.cache.conn.commit()
                return Completion(text, pt, ct, provider, quantization)
            except Exception as e:
                last = e
                await asyncio.sleep(min(30.0, 2 ** attempt + random.random()))  # noqa: S311 - backoff jitter
        raise RuntimeError(f"LLM call failed after {RETRIES + 1} attempts: {last!r}")
