"""A trained GNN on disk: `{"state_dict", "meta"}` through torch.save, replaced atomically, read with weights_only."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Tuple

import torch

from bazt_lab.core import public as core
from bazt_lab.features.gnn.adapters.pyg_model import MyGAT, build_model


def save_checkpoint(path, model: torch.nn.Module, meta: Mapping) -> Path:
    """Write the model's state_dict and `meta` (plain values; `meta["arch"]` rebuilds the model) to `path`."""
    if "arch" not in meta:
        raise ValueError("checkpoint metadata needs 'arch', the arguments that rebuild the model")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "meta": dict(meta)}, tmp)
        core.atomic_replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def load_checkpoint(path) -> Tuple[MyGAT, dict]:
    """(model in eval mode, metadata) of a checkpoint written by `save_checkpoint`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no GNN checkpoint at {path}")
    blob = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(blob["meta"]["arch"])
    model.load_state_dict(blob["state_dict"])
    return model.eval(), blob["meta"]
