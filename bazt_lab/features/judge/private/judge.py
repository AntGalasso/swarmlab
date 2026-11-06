"""LLMJudge: the Judge contract over a transport (OpenRouter in production, a scripted fake in tests).

score(texts) -> float64 [len(texts)], log P("yes") of the first answer token, higher = more likely injected.
Per call: duplicate texts are one request; the disk cache answers what it knows; the rest go out on a thread pool
(the endpoint has no batch mode for logprobs, so concurrency is the batching), each with retries; every answer is
cached as soon as it arrives, so a failure in one text never loses the others. The score of a text is a function of
its request alone: batch size, order and worker count cannot change it.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Mapping, Optional

import torch

from bazt_lab.features.judge.private.cache import DiskCache, cache_key
from bazt_lab.features.judge.private.config import JudgeConfig, build_request
from bazt_lab.features.judge.private.errors import JudgeConfigError, JudgeError
from bazt_lab.features.judge.private.retry import DEFAULT_RETRY, RetryPolicy, call_with_retries
from bazt_lab.features.judge.private.scoring import extract_record, score_record

Transport = Callable[[dict], dict]          # request body -> raw chat-completion dict; raises TransientError / JudgeError
log = logging.getLogger("bazt_lab.judge")


@dataclass
class JudgeStats:
    texts: int = 0                # texts requested, duplicates included
    unique: int = 0               # distinct texts after the per-call dedup
    cache_hits: int = 0           # distinct texts answered from the disk cache
    live_calls: int = 0           # distinct texts that went to the transport
    retries: int = 0
    missing_yes: int = 0          # distinct texts scored with the floor (no "yes" variant among the candidates)
    off_format: int = 0           # of those, neither a "yes" nor a "no" variant (the answer is not Yes/No)
    prompt_tokens: int = 0        # logical totals over distinct texts, cache hits included
    completion_tokens: int = 0
    live_seconds: float = 0.0     # wall-clock summed over live calls, retry waits included

    def snapshot(self) -> dict:
        return asdict(self)


class LLMJudge:
    name = "promptarmor_logprob_yes"

    def __init__(self, settings: Mapping, transport: Transport, *, cache_path: Optional[Path],
                 retry: RetryPolicy = DEFAULT_RETRY, max_workers: int = 16,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.config = JudgeConfig.from_settings(settings)
        if max_workers < 1:
            raise JudgeConfigError(f"max_workers must be >= 1, got {max_workers}")
        if self.config.use_cache and cache_path is None:
            raise JudgeConfigError("llm.cache is true, so the judge needs a cache_path")
        self._cache = DiskCache(cache_path) if self.config.use_cache else None
        self._transport, self._retry, self._max_workers, self._sleep = transport, retry, max_workers, sleep
        self._lock = threading.Lock()
        self.stats = JudgeStats()

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()
        close = getattr(self._transport, "close", None)          # a live transport owns an HTTP client
        if callable(close):
            close()

    def _fetch(self, key: str, body: dict):
        start = time.perf_counter()
        record, retries = call_with_retries(lambda: extract_record(self._transport(body)), self._retry,
                                            key=key, sleep=self._sleep)
        return record, retries, time.perf_counter() - start

    def score(self, texts: List[str]) -> torch.Tensor:
        texts = list(texts)
        if not texts:
            return torch.empty(0, dtype=torch.float64)
        bodies = {t: build_request(self.config, t) for t in texts}            # dedup; TypeError on a non-str
        keys = {t: cache_key(b) for t, b in bodies.items()}
        records, todo = {}, []
        for t in bodies:
            hit = self._cache.get(keys[t]) if self._cache is not None else None
            if hit is None:
                todo.append(t)
            else:
                records[t] = hit
        hits, retries, seconds, errors = len(records), 0, 0.0, []
        if todo:
            with ThreadPoolExecutor(min(self._max_workers, len(todo))) as pool:
                futures = {pool.submit(self._fetch, keys[t], bodies[t]): i for i, t in enumerate(todo)}
                for future in as_completed(futures):
                    try:
                        record, n_retries, secs = future.result()
                    except JudgeError as e:
                        errors.append((futures[future], e))
                        continue
                    t = todo[futures[future]]
                    records[t] = record
                    if self._cache is not None:
                        self._cache.put(keys[t], record)                      # progressive: kept even if a sibling fails
                    retries += n_retries
                    seconds += secs
        scored = {t: score_record(r, self.config.floor_probability) for t, r in records.items()}
        missing_yes = sum(not v.yes_found for v in scored.values())
        off_format = sum(not v.yes_found and not v.no_found for v in scored.values())
        with self._lock:
            s = self.stats
            s.texts += len(texts)
            s.unique += len(bodies)
            s.cache_hits += hits
            s.live_calls += len(records) - hits
            s.retries += retries
            s.live_seconds += seconds
            s.missing_yes += missing_yes
            s.off_format += off_format
            s.prompt_tokens += sum(r["prompt_tokens"] for r in records.values())
            s.completion_tokens += sum(r["completion_tokens"] for r in records.values())
        log.info("judge score: texts=%d unique=%d cache_hits=%d live_calls=%d failed=%d retries=%d seconds=%.1f "
                 "missing_yes=%d off_format=%d",
                 len(texts), len(bodies), hits, len(records) - hits, len(errors), retries, seconds,
                 missing_yes, off_format)
        if errors:
            raise min(errors, key=lambda pair: pair[0])[1]                   # the failed text that comes first
        return torch.tensor([scored[t].value for t in texts], dtype=torch.float64)
