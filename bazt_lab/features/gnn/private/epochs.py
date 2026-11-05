"""Per-epoch training state on disk: one atomic file per epoch, the newest few kept besides the best epoch.

`<gnn.artifacts_dir>/epochs/<training unit id>/epoch_NNNN.pt` holds everything a crashed training needs to continue:
epoch, model, optimizer, scheduler, history, best validation accuracy and the global RNG state (training itself draws
only from keyed streams, so the RNG state is recorded for completeness and needed by nothing).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional

import torch

from bazt_lab.core import public as core


def epoch_dir(settings: Mapping, unit_id: str) -> Path:
    return Path(core.get(settings, "gnn.artifacts_dir")) / "epochs" / unit_id


def _name(epoch: int) -> str:
    return f"epoch_{epoch:04d}.pt"


def save_epoch(directory: Path, state: Mapping, keep: int, best_epoch: int) -> None:
    """Write `state` (its "epoch" names the file) atomically, then drop every file but the newest `keep`, the
    best epoch and this one."""
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / _name(state["epoch"])
    tmp = final.with_name(final.name + ".tmp")
    try:
        torch.save({**state, "rng": torch.get_rng_state()}, tmp)
        os.replace(tmp, final)
    finally:
        tmp.unlink(missing_ok=True)
    files = sorted(directory.glob("epoch_*.pt"))
    protected = {_name(best_epoch), final.name, *(f.name for f in files[-keep:])}
    for f in files:
        if f.name not in protected:
            f.unlink()


def _load(path: Path) -> Optional[dict]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:                             # a damaged file is skipped, the older one is used
        return None


def load_latest(directory: Path) -> Optional[dict]:
    """The newest epoch file that loads completely, or None."""
    for f in sorted(Path(directory).glob("epoch_*.pt"), reverse=True):
        state = _load(f)
        if state is not None:
            return state
    return None
