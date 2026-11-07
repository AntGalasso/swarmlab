"""Bridge the real cached client to the conversation harness on one worker thread."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
import threading


class LiveBackendAdapter:
    """Run agents and async judge requests on the same client-owned event loop.

    The current provider transports do not expose first-token log probabilities;
    the harness must use its explicitly labelled structured-score path instead.
    """

    strict_judge_format = True

    def __init__(self, backend, own_cache=True, judge_backend=None, request_timeout=None):
        self.backend = backend
        self.model = backend.model
        self.judge_backend = judge_backend
        self.own_cache = own_cache
        self.calls = 0
        self._closed = False
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='swarmlab-live')
        # The worker thread must never block a block forever: the client already bounds each attempt, and this
        # is the outer bound on the whole retry ladder for one submitted request.
        self.request_timeout = float(request_timeout or getattr(backend, 'request_timeout', 30.0))
        self.retries = int(getattr(backend, 'retries', 2))
        self.call_timeout = self.request_timeout * (self.retries + 1) + 15.0

    def _request(self, reqs, temperature):
        with self._lock:
            self.calls += len(reqs)
        return self.backend.generate([(s, u, 'pr7-live-adapter-v1') for s, u in reqs],
                                     temperature=temperature)

    def generate_sync(self, reqs):
        if self._closed:
            raise RuntimeError('Live backend is closed.')
        future = self._executor.submit(self._request, reqs, 0.0)
        try:
            return future.result(timeout=self.call_timeout)
        except FutureTimeout as exc:
            raise RuntimeError(
                f'Live provider did not answer {len(reqs)} request(s) within {self.call_timeout:.0f}s '
                f'({self.retries + 1} attempts x {self.request_timeout:.0f}s); refusing to hang the block.'
            ) from exc

    async def complete(self, system, user, temperature=0.0):
        if self._closed:
            raise RuntimeError('Live backend is closed.')
        future = self._executor.submit(self._request, [(system, user)], temperature)
        try:
            return (await asyncio.wait_for(asyncio.wrap_future(future), timeout=self.call_timeout))[0]
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f'Live provider did not answer within {self.call_timeout:.0f}s; refusing to hang the block.'
            ) from exc

    async def first_token_yes_logprob(self, prompt):  # noqa: ARG002
        return None

    def _close_resources(self):
        close_transport = getattr(self.backend.transport, 'aclose', None)
        try:
            if close_transport is not None:
                self.backend._loop.run_until_complete(close_transport())
        finally:
            try:
                self.backend.close()
            finally:
                if self.own_cache:
                    self.backend.cache.close()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._executor.submit(self._close_resources).result()
        finally:
            self._executor.shutdown(wait=True)
            if self.judge_backend is not None:
                self.judge_backend.close()
