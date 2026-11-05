"""The Hugging Face Hub uploader (huggingface_hub 2.4.0): one create_commit per push with every changed file. The
only module of this feature that imports huggingface_hub, and only the child process loads it."""
from __future__ import annotations

import os

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub._commit_scheduler import PartialFileIO  # what CommitScheduler uses for files still growing

from bazt_lab.features.sync.adapters import env


class HubUploader:
    def __init__(self, config: dict) -> None:
        self.repo_id, self.repo_type, self.private = (config["settings"][k] for k in ("repo_id", "repo_type", "private"))
        self.api = HfApi(token=os.environ[env.VARIABLE])

    def ensure_repo(self) -> None:
        self.api.create_repo(self.repo_id, repo_type=self.repo_type, private=self.private, exist_ok=True)

    def commit(self, files, message: str) -> None:
        # Capped to the size seen at listing, so a file appended to during the upload is not torn.
        operations = [CommitOperationAdd(path_in_repo=f.repo_path, path_or_fileobj=PartialFileIO(f.local, f.size))
                      for f in files]
        self.api.create_commit(repo_id=self.repo_id, repo_type=self.repo_type, operations=operations,
                               commit_message=message)


def make_uploader(config: dict) -> HubUploader:
    return HubUploader(config)
