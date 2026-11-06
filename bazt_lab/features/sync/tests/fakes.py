"""Stand-ins for the Hub uploader: no network. `Recorder` is used in-process; `make_uploader` is loaded by name in a
real child process (config["uploader"]) and appends one JSON line per event to the file named by BAZT_SYNC_FAKE_LOG.
BAZT_SYNC_FAKE_FAIL=1 makes every commit fail with the token in the error text."""
from __future__ import annotations

import json
import os

from bazt_lab.features.sync.adapters import env


class Recorder:
    """Records ensure_repo and commit calls; `fail` makes the next commits raise it."""

    def __init__(self):
        self.repos = 0
        self.commits = []          # one list of repo paths per commit
        self.contents = []         # {repo path: bytes} per commit
        self.fail = None

    def ensure_repo(self):
        self.repos += 1

    def commit(self, files, message):
        if self.fail is not None:
            raise self.fail
        self.commits.append(sorted(f.repo_path for f in files))
        self.contents.append({f.repo_path: f.local.read_bytes()[:f.size] for f in files})
        return list(files)


class FileRecorder:
    def __init__(self, config):
        self.path = os.environ["BAZT_SYNC_FAKE_LOG"]

    def _write(self, event):
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")

    def ensure_repo(self):
        self._write({"event": "repo", "pid": os.getpid()})

    def commit(self, files, message):
        if os.environ.get("BAZT_SYNC_FAKE_FAIL"):
            raise RuntimeError(f"401 rejected token {os.environ.get(env.VARIABLE)}")
        self._write({"event": "commit", "files": sorted(f.repo_path for f in files), "pid": os.getpid()})
        return list(files)


def make_uploader(config):
    return FileRecorder(config)
