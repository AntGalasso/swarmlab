"""The upload loop that runs inside the child process. It takes an uploader (ensure_repo, commit) so it is tested
without a network. A failure is reported once per kind on stderr and counted, and the loop goes on: the files stay on
disk and the next interval retries them (nothing is marked uploaded by a failed commit)."""
from __future__ import annotations

import sys
import threading
from typing import Callable, Dict, Iterable, List

from bazt_lab.features.sync.private import plan
from bazt_lab.features.sync.private.settings import SyncSettings


def _stderr(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


class Uploads:
    def __init__(self, settings: SyncSettings, paths: Iterable, uploader, report: Callable[[str], None] = _stderr,
                 redact: Callable[[str], str] = str) -> None:
        self.settings, self.paths, self.uploader = settings, list(paths), uploader
        self.report, self.redact = report, redact
        self.uploaded: Dict[str, tuple] = {}
        self.failures: Dict[str, int] = {}
        self.commits = 0
        self._repo_ready = False

    def _fail(self, kind: str, error: BaseException) -> None:
        self.failures[kind] = self.failures.get(kind, 0) + 1
        if self.failures[kind] == 1:
            self.report(self.redact(f"sync: {kind} failed and is skipped; the checkpoints stay on disk and the next "
                                    f"push retries ({type(error).__name__}: {error}). Further failures of {kind} are "
                                    "counted, not printed."))

    def ensure(self) -> bool:
        if not self._repo_ready:
            try:
                self.uploader.ensure_repo()
                self._repo_ready = True
            except Exception as error:               # an upload problem must not stop an experiment
                self._fail("repo", error)
        return self._repo_ready

    def pending(self) -> List[plan.Entry]:
        return plan.changed(plan.snapshot(self.paths, self.settings.include, self.settings.exclude), self.uploaded)

    def commit(self, entries: List[plan.Entry]) -> None:
        if not entries or not self.ensure():
            return
        try:
            self.uploader.commit(entries, f"sync: {len(entries)} changed file(s)")
        except Exception as error:
            self._fail("commit", error)
            return
        self.uploaded.update(plan.stamps(entries))
        self.commits += 1

    def push(self) -> None:
        self.commit(self.pending())


def run(uploads: Uploads, stop: threading.Event) -> None:
    """One commit per interval, and one more when `stop` is set (the final push)."""
    uploads.ensure()
    while True:
        stopped = stop.wait(uploads.settings.every_minutes * 60)
        uploads.push()
        if stopped:
            return
