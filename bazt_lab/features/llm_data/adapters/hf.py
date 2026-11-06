"""Hugging Face Hub adapter: the only module of this feature that imports `datasets`.

Secrets and offline switches come from the environment (HF_TOKEN, HF_HUB_OFFLINE, HF_DATASETS_OFFLINE); a repo-root
`.env` fills what the process leaves unset. `datasets` and `huggingface_hub` read these variables once at import, so
the `.env` is loaded before the first import. Nothing here prints or logs them.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[4] / ".env"


def _load_env() -> None:
    load_dotenv(ENV_PATH)          # never overrides a variable the process already has; a missing file is fine


@functools.cache
def _datasets():
    _load_env()
    import datasets

    return datasets


@functools.cache
def _split(dataset: str, config: Optional[str], split: str, revision: str):
    """The pinned split with `_row` (its position before shuffling), resolved once per process: every
    `load_dataset` call asks the Hub again, and a run draws once per seed and task. A failed load is not cached."""
    ds = _datasets().load_dataset(dataset, config, split=split, revision=revision)
    return ds.add_column("_row", list(range(len(ds))))          # shuffle depends on len(ds) only


def draw(dataset: str, config: Optional[str], split: str, revision: str, seed: int, k: int) -> List[dict]:
    """The spec's draw: `shuffle(seed=seed).select(range(k))` of the split pinned at `revision`.

    Every row is a plain dict with one extra key, `_row`: its position in the split before shuffling (the id).
    A cached split is reused when HF_HUB_OFFLINE / HF_DATASETS_OFFLINE are set, provided `revision` is the same.
    """
    ds = _split(dataset, config, split, revision)
    if k > len(ds):
        raise ValueError(f"k={k} exceeds the {len(ds)} rows of {dataset} {config} {split}")
    return list(ds.shuffle(seed=seed, keep_in_memory=True).select(range(k)))   # in memory: no cache file written
