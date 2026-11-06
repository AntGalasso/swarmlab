"""The Hugging Face Hub uploader (huggingface_hub 2.4.0): one create_commit per batch of changed files. The only
module of this feature that imports huggingface_hub, and only the child process loads it.

Each file of a batch is copied (as one consistent version, see shared_file.py) to a private staging directory and
closed at once; the commit uploads the staged copies. No experiment file is open while the network is used, so the
experiment's own deletes and renames never meet a sync handle for longer than one local file copy. The caller
(private/loop.py) hands over a bounded batch, so a push never holds thousands of handles or gigabytes of staging."""
from __future__ import annotations

import dataclasses
import os
import tempfile
from pathlib import Path
from typing import List

from huggingface_hub import CommitOperationAdd, HfApi

from bazt_lab.features.sync.adapters import env
from bazt_lab.features.sync.adapters.shared_file import copy_version


class HubUploader:
    def __init__(self, config: dict) -> None:
        self.repo_id, self.repo_type, self.private = (config["settings"][k] for k in ("repo_id", "repo_type", "private"))
        self.api = HfApi(token=os.environ[env.VARIABLE])

    def ensure_repo(self) -> None:
        self.api.create_repo(self.repo_id, repo_type=self.repo_type, private=self.private, exist_ok=True)

    def commit(self, files, message: str) -> List:
        """Commits the files that can still be read; returns them, each with the stamp of the version uploaded. A file
        that vanished or cannot be read since listing is skipped (it is not returned, so the next push retries it)."""
        with tempfile.TemporaryDirectory(prefix="bazt-sync-", ignore_cleanup_errors=True) as stage:
            operations, sent = [], []
            for number, entry in enumerate(files):
                staged = Path(stage) / str(number)
                try:
                    mtime_ns, size = copy_version(entry.local, staged)
                except OSError:
                    continue
                operations.append(CommitOperationAdd(path_in_repo=entry.repo_path, path_or_fileobj=str(staged)))
                sent.append(dataclasses.replace(entry, mtime_ns=mtime_ns, size=size))
            if operations:
                self.api.create_commit(repo_id=self.repo_id, repo_type=self.repo_type, operations=operations,
                                       commit_message=message)
            return sent


def make_uploader(config: dict) -> HubUploader:
    return HubUploader(config)
