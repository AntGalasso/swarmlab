"""Seeded sampling of task items (spec 4.1: a seed draws its questions) and the sources record for manifests."""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from bazt_lab.core import public as core
from bazt_lab.features.llm_data.adapters import hf
from bazt_lab.features.llm_data.private import vendored
from bazt_lab.features.llm_data.private.items import HUB, TASKS, Source, TaskItem, hub_item

_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_PARTS = ("train", "test")
TRAIN_FRACTION = 0.8        # G-Safeguard's own split of the vendored files (gen_injecagent_data, gen_poisonrag_data)


def _setting(settings: Mapping, key: str, pattern: re.Pattern, what: str) -> str:
    value = core.get(settings, key)
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"{key} must be {what}, got {value!r}")
    return value


def _revision(settings: Mapping, task: str) -> str:
    return _setting(settings, f"llm_data.revisions.{task}", _COMMIT,
                    "a full 40-character commit sha (a branch name or short sha does not pin)")


def _digest(settings: Mapping, task: str) -> str:
    return _setting(settings, f"llm_data.vendored_sha256.{task}", _DIGEST, "a lowercase 64-character sha256")


def _count(name: str, value, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")


def sample(task: str, seed: int, k: int, settings: Mapping, *, part: Optional[str] = None,
           root: Optional[Path] = None) -> List[TaskItem]:
    """The `k` items that `seed` draws for `task`, in draw order.

    Hub tasks: the spec's `shuffle(seed=seed).select(range(k))`, run by `datasets` (its own numpy generator).
    Vendored tasks: the positions ordered by one float64 uniform per record from `core.stream("llm_data", "sample",
    task, part or "all", seed)`. Either way the draw depends on (task, seed, pinned data) only, never on the arm,
    so every arm sees the same questions, and a larger k extends the draw. `part` ("train" or "test", vendored
    tasks only) restricts the pool to G-Safeguard's 80/20 positional split; ids keep their position in the whole
    file. `root` is the repository root that holds the vendored files (tests only).
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; known: {', '.join(TASKS)}")
    _count("seed", seed, 0)
    _count("k", k, 1)
    if part is not None and (task in HUB or part not in _PARTS):
        raise ValueError(f"part must be one of {_PARTS} and applies to the vendored tasks only, got {part!r} "
                         f"for {task}")
    if task in HUB:
        revision = _revision(settings, task)
        dataset, config, split = HUB[task]
        return [hub_item(task, row, revision) for row in hf.draw(dataset, config, split, revision, seed, k)]
    records, digest = vendored.read(task, _digest(settings, task), root)
    cut = int(len(records) * TRAIN_FRACTION)
    pool = records if part is None else records[:cut] if part == "train" else records[cut:]
    if k > len(pool):
        raise ValueError(f"k={k} exceeds the {len(pool)} {task} records of part {part or 'all'}")
    src = vendored.source(task, part, digest)
    uniforms = core.uniforms(core.stream("llm_data", "sample", task, part or "all", seed), (len(pool),))
    return [vendored.item(task, *pool[i], src) for i in uniforms.argsort(stable=True)[:k].tolist()]


def sources(settings: Mapping, root: Optional[Path] = None) -> Dict[str, dict]:
    """Per task: dataset, config, split, revision and sha256, for the manifest. Reads the vendored files."""
    out = {}
    for task in TASKS:
        if task in HUB:
            out[task] = dataclasses.asdict(Source(*HUB[task], _revision(settings, task), None))
        else:
            _, digest = vendored.read(task, _digest(settings, task), root)
            out[task] = dataclasses.asdict(vendored.source(task, None, digest))
    return out
