"""Starting and stopping the upload child process. The child runs at a low OS priority so uploading never competes
with the experiments for CPU; this side only waits on it (a watcher thread blocked in `wait`, no polling)."""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Mapping, Optional, Sequence

from bazt_lab.features.sync.private.errors import SyncError
from bazt_lab.features.sync.private.settings import SyncSettings

ROOT = Path(__file__).resolve().parents[4]
UPLOADER: Optional[str] = None           # "module:callable" replacing the Hub uploader in the child (tests only)
WINDOWS_PRIORITY = {"below_normal": 0x00004000, "idle": 0x00000040}   # BELOW_NORMAL_ / IDLE_PRIORITY_CLASS
POSIX_NICE = {"below_normal": 10, "idle": 19}


def launch_options(os_name: str, level: str) -> dict:
    """The Popen options that give the child its low priority: creation flags on Windows, a nice value elsewhere."""
    if os_name == "nt":
        return {"creationflags": WINDOWS_PRIORITY[level]}
    return {"preexec_fn": lambda: os.nice(POSIX_NICE[level])}


def _tell(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


class SyncHandle:
    """What `start` returns. `stop()` ends the child after a final push; it never raises."""

    def __init__(self, proc: Optional[subprocess.Popen] = None, timeout: float = 0.0) -> None:
        self._proc, self._timeout = proc, timeout
        self.summary: Optional[dict] = None      # {"commits": n, "failures": {kind: n}} once stopped cleanly
        self.reported = 0                        # how many times the child's death was reported (at most once)
        self._stopping = False
        self._lock = threading.Lock()
        if proc is not None:
            threading.Thread(target=self._watch, daemon=True).start()

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc else None

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _report_death(self) -> None:
        with self._lock:
            if self.reported:
                return
            self.reported = 1
        _tell(f"sync: uploader process exited with code {self._proc.poll()}; checkpoints are no longer uploaded, "
              "the run continues and its results stay on disk.")

    def _watch(self) -> None:
        self._proc.wait()
        if not self._stopping:
            self._report_death()

    def stop(self) -> None:
        if self._proc is None or self._stopping:
            return
        self._stopping = True
        if self.alive:
            try:
                out, _ = self._proc.communicate("stop\n", timeout=self._timeout)
                self.summary = json.loads(out.strip().splitlines()[-1]) if out.strip() else None
            except subprocess.TimeoutExpired:
                _tell(f"sync: the final push did not finish within {self._timeout:g} s; the uploader keeps running "
                      "on its own and the checkpoints stay on disk.")
                return
            except (OSError, ValueError):
                pass
        if self.summary is None:
            self._report_death()


def start_child(settings: SyncSettings, paths: Sequence, environ: Mapping | None = None) -> SyncHandle:
    config = {"paths": [str(Path(p).resolve()) for p in paths], "settings": dataclasses.asdict(settings),
              "uploader": UPLOADER}
    env = dict(os.environ if environ is None else environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    try:
        proc = subprocess.Popen([sys.executable, "-m", "bazt_lab.features.sync.adapters.child", json.dumps(config)],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
                                cwd=ROOT, env=env, **launch_options(os.name, settings.priority))
    except OSError as error:
        raise SyncError(f"the uploader process could not be started: {error}") from error
    return SyncHandle(proc, settings.final_push_timeout_seconds)
