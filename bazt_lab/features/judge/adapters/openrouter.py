"""OpenRouter adapter: the openai SDK pointed at OpenRouter's OpenAI-compatible API, and the factory of the live judge.

The only module of the judge feature that reaches the network, the openai SDK, python-dotenv, the API key or swarmlab
(the default kickoff time lives there). Secrets come only from the environment (OPENROUTER_API_KEY; a .env file is
loaded with python-dotenv without overriding variables that are already set) and are never logged or put in an error.

Kickoff guard (README compliance rule): `live_judge` refuses, before anything is read or sent, unless the kickoff time
has passed and the caller passed allow_live=True (the launcher's --allow-live-llm). The kickoff time is
SWARMLAB_KICKOFF_UTC when set, else swarmlab.KICKOFF_UTC, the same rule as swarmlab._kickoff_guard.
"""
from __future__ import annotations

import datetime as dt
import math
import os
import time
from pathlib import Path
from typing import Callable, Mapping, Optional
from urllib.parse import urlsplit

import openai
from dotenv import load_dotenv

from bazt_lab.features.judge.private.config import JudgeConfig
from bazt_lab.features.judge.private.errors import JudgeCallFailed, JudgeConfigError, TransientError
from bazt_lab.features.judge.private.guard import check_live_allowed, parse_kickoff
from bazt_lab.features.judge.private.judge import LLMJudge
from bazt_lab.features.judge.private.retry import DEFAULT_RETRY, RetryPolicy

KEY_ENV = "OPENROUTER_API_KEY"
BASE_URL_ENV = "OPENROUTER_BASE_URL"                 # optional; the default is OpenRouter itself
KICKOFF_ENV = "SWARMLAB_KICKOFF_UTC"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
REQUEST_TIMEOUT_S = 60.0                             # every call has a deadline; a stalled provider must not wedge a run
LOOPBACK = ("localhost", "127.0.0.1", "::1")
SDK_FIELDS = ("model", "messages", "temperature", "max_tokens", "logprobs", "top_logprobs")   # the rest is extra_body


def kickoff_time(env: Mapping[str, str]) -> dt.datetime:
    text = env.get(KICKOFF_ENV)
    if text:                                           # an empty value counts as unset, as in swarmlab
        return parse_kickoff(text)
    import swarmlab  # the default start is defined in one place
    return swarmlab.KICKOFF_UTC


def _checked_base_url(url: str) -> str:
    """The bearer key is sent to this URL: https, or plain http only to this machine (a local gateway or a test)."""
    parts = urlsplit(url)
    if parts.hostname and (parts.scheme == "https" or (parts.scheme == "http" and parts.hostname in LOOPBACK)):
        return url
    raise JudgeConfigError(f"{BASE_URL_ENV} must be an https URL (http only on loopback): the API key is sent there")


def _retry_after(headers) -> Optional[float]:
    try:
        seconds = float(headers.get("retry-after"))     # delta-seconds; an HTTP date is not parsed (backoff applies)
    except (TypeError, ValueError):
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _status_error(e: openai.APIStatusError) -> Exception:
    """408, 429 and 5xx are worth retrying; so is a 402 that carries Retry-After (OpenRouter's in-flight budget).
    Every other status is final: 401/403 bad key, 402 out of credit, 400/404 the request or the pinned provider."""
    status = e.status_code
    retry_after = _retry_after(e.response.headers)
    body = e.body if isinstance(e.body, Mapping) else {}
    err = body.get("error")
    detail = str((err.get("message") if isinstance(err, Mapping) else None) or e.message)[:200]
    message = f"OpenRouter HTTP {status}: {detail}"
    if status == 404:
        message += (" (no endpoint of the pinned provider serves this request: check llm.judge.model and "
                    "llm.judge.provider, and that the provider returns logprobs)")
    if status in (408, 429) or status >= 500 or (status == 402 and retry_after is not None):
        return TransientError(message, retry_after)
    return JudgeCallFailed(message)


class OpenRouterTransport:
    """Transport of LLMJudge: request body in, the provider's raw JSON out. The openai SDK builds and sends the
    request; its own retries are off (max_retries=0) because the judge owns retry and backoff."""

    def __init__(self, api_key: str, base_url: str, *, timeout: float = REQUEST_TIMEOUT_S, http_client=None) -> None:
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0,
                                     http_client=http_client)

    def __call__(self, body: dict) -> dict:
        extra = {k: v for k, v in body.items() if k not in SDK_FIELDS}      # provider pin and reasoning switch
        try:
            raw = self._client.chat.completions.with_raw_response.create(
                **{k: body[k] for k in SDK_FIELDS}, extra_body=extra)
        except openai.APIStatusError as e:
            raise _status_error(e) from e
        except openai.APIConnectionError as e:                               # timeouts included
            raise TransientError(f"connection failed: {type(e).__name__}") from e
        try:
            return raw.http_response.json()          # the provider's JSON as sent: the parse rule never sees SDK models
        except ValueError as e:
            raise TransientError("the response body is not JSON") from e

    def close(self) -> None:
        self._client.close()


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def live_judge(settings: Mapping, *, allow_live: bool, cache_path: Optional[Path], max_workers: int = 16,
               retry: RetryPolicy = DEFAULT_RETRY, timeout: float = REQUEST_TIMEOUT_S,
               env: Optional[Mapping[str, str]] = None, now: Optional[Callable[[], dt.datetime]] = None,
               sleep: Callable[[float], None] = time.sleep, http_client=None) -> LLMJudge:
    """The PromptArmor judge over OpenRouter. `env`, `now`, `sleep` and `http_client` are seams for tests; in
    production they are the process environment (after .env), the wall clock, time.sleep and the SDK's own client.
    Raises KickoffGuardError (before kickoff / without allow_live), core.Unset or JudgeConfigError (settings, key,
    URL); in each case nothing has been sent."""
    if env is None:
        load_dotenv()
        env = os.environ
    check_live_allowed(allow_live=allow_live, kickoff=kickoff_time(env), now=(now or _utcnow)())
    JudgeConfig.from_settings(settings)              # refuse an unset provider (or any bad setting) before the key is read
    key = (env.get(KEY_ENV) or "").strip()
    if not key:
        raise JudgeConfigError(f"{KEY_ENV} is not set: put it in .env or the environment")
    transport = OpenRouterTransport(key, _checked_base_url(env.get(BASE_URL_ENV) or DEFAULT_BASE_URL),
                                    timeout=timeout, http_client=http_client)
    try:
        return LLMJudge(settings, transport, cache_path=cache_path, retry=retry, max_workers=max_workers, sleep=sleep)
    except Exception:
        transport.close()
        raise
