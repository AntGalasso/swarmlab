"""Failures during a run: told on stderr, once per operation, never silently and never fatally (D87: a tracking
problem must not stop an experiment; the results are on disk in the CheckpointStore)."""
from __future__ import annotations

import sys
from typing import Callable, Dict, Optional


class FailureLog:
    def __init__(self, redact: Callable[[str], str] = str) -> None:
        self._redact = redact
        self._counts: Dict[str, int] = {}

    @property
    def counts(self) -> Dict[str, int]:
        return dict(self._counts)

    def report(self, operation: str, error: BaseException) -> None:
        self._counts[operation] = self._counts.get(operation, 0) + 1
        if self._counts[operation] == 1:
            print(self._redact(f"tracking: {operation} failed and is skipped; the results stay on disk "
                               f"({type(error).__name__}: {error}). Further failures of {operation} are counted, "
                               "not printed."), file=sys.stderr)

    def summary(self) -> Optional[str]:
        if not self._counts:
            return None
        parts = ", ".join(f"{op}: {n}" for op, n in self._counts.items())
        return self._redact(f"tracking: {sum(self._counts.values())} operations failed during the run ({parts}); "
                            "the W&B run is incomplete, the results on disk are not.")
