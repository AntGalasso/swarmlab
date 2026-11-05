"""Retry with exponential backoff and jitter for transient errors.

Operational constants, not experiment values: a retry changes latency, never a score (the cache and a
temperature-0 judge make the result a function of the request). The jitter is drawn from a core stream keyed by
the request, so a run is reproducible and no global RNG is touched.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Tuple, TypeVar

from bazt_lab.core import public as core
from bazt_lab.features.judge.private.errors import JudgeCallFailed, TransientError

T = TypeVar("T")
log = logging.getLogger("bazt_lab.judge")


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 6                # total tries, first one included
    base_delay: float = 1.0          # seconds before the first retry; doubles each time
    max_delay: float = 30.0          # cap on any single wait, a server's Retry-After included

    def __post_init__(self) -> None:
        if self.attempts < 1 or self.base_delay <= 0 or self.max_delay < self.base_delay:
            raise ValueError(f"invalid retry policy {self}")


DEFAULT_RETRY = RetryPolicy()


def call_with_retries(fn: Callable[[], T], policy: RetryPolicy, *, key: str,
                      sleep: Callable[[float], None] = time.sleep) -> Tuple[T, int]:
    """(fn(), number of retries used). Only TransientError is retried; anything else propagates at once."""
    last: TransientError | None = None
    for attempt in range(policy.attempts):
        try:
            return fn(), attempt
        except TransientError as e:
            last = e
            if attempt == policy.attempts - 1:
                break
            cap = min(policy.max_delay, policy.base_delay * 2.0 ** attempt)
            u = float(core.uniforms(core.stream("judge_backoff", key, attempt), (1,))[0])
            delay = cap * (0.5 + 0.5 * u)                       # equal jitter: in [cap/2, cap]
            if e.retry_after is not None:
                delay = max(delay, e.retry_after)
            delay = min(delay, policy.max_delay)
            log.warning("judge call failed (attempt %d/%d), retrying in %.1fs: %s", attempt + 1, policy.attempts, delay, e)
            sleep(delay)
    raise JudgeCallFailed(f"gave up after {policy.attempts} attempts, last error: {last}") from last
