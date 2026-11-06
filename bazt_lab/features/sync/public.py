"""Public interface of the sync feature: push checkpoint folders to a Hugging Face repository from a separate,
low-priority process, so a crash never loses them and uploading never takes the experiments' CPU.

    handle = start(settings, [paths])      # refuses (SyncError / core.Unset) before anything runs; no-op handle if off
    ...                                    # run; handle.alive says whether the uploader still runs
    handle.stop()                          # final push, then the uploader ends; never raises

See CONTRACT.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from bazt_lab.features.sync.adapters import env, process
from bazt_lab.features.sync.adapters.process import SyncHandle
from bazt_lab.features.sync.private import settings as sync_settings
from bazt_lab.features.sync.private.errors import SyncError

__all__ = ["SyncError", "SyncHandle", "start"]


def start(settings: Mapping, paths: Sequence) -> SyncHandle:
    wanted = sync_settings.read(settings)
    if wanted is None:
        return SyncHandle()
    if not paths:
        raise SyncError("sync is enabled but no path was given to upload")
    names = [Path(p).name for p in paths]
    if len(set(names)) != len(names) or not all(names):
        raise SyncError(f"sync paths must have distinct names, each is mirrored under a folder of its own name: {names}")
    env.require_token()
    return process.start_child(wanted, paths)
