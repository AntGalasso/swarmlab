"""Test doubles: a settings tree with small test grids (fixture values, not experiment choices) and a fake Platform
whose final ASR is a known function of the unit's attack_params."""
from __future__ import annotations

import copy
from typing import Callable, Dict, List, Mapping

from bazt_lab.core import public as core

CONDITIONS = ["gray_box", "white_box", "embedding_evasion", "attack_rate"]


def make_settings(over: Mapping | None = None) -> dict:
    s: dict = {}
    base = {
        "runtime.n_insiders": 3,
        "mock.every_sink": True,
        "mock.batch_size": 64,
        "mock.main_seeds": [1, 30],
        "mock.heldout_seeds": [31, 60],
        "mock.tune_seeds": [9001, 9030],
        "mock.adaptive.conditions": list(CONDITIONS),
        "mock.adaptive.search_seeds": [8001, 8003],
        "mock.adaptive.insider_sets": "contiguous",
        "mock.adaptive.start_fractions": [0.0, 0.2, 0.4, 0.6],
        "mock.adaptive.rates": [1.0, 0.5, 0.3, 0.2, 0.1],
        "mock.adaptive.detectabilities": [1.0, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1],
        "mock.adaptive.evasion_cost": 1.0,
        "mock.adaptive.max_candidates": 5000,
        "mock.adaptive.knowledge.gray_box": "cell",
        "mock.adaptive.knowledge.white_box": "episode",
        "mock.adaptive.knowledge.embedding_evasion": "cell",
        "mock.adaptive.knowledge.attack_rate": "cell",
    }
    for k, v in {**base, **(over or {})}.items():
        core.put(s, k, copy.deepcopy(v))
    return s


def drop(settings: dict, dotted: str) -> dict:
    """A copy of `settings` without the key `dotted`."""
    out = copy.deepcopy(settings)
    node = out
    *path, leaf = dotted.split(".")
    for part in path:
        node = node[part]
    del node[leaf]
    return out


def unit(**over) -> dict:
    """One evaluation unit of the mock_adaptive block."""
    u = dict(block="mock_adaptive", platform="mock", arm="X", audit=0.1, intervention=0.1, topology="small_world",
             attack="infection", agents=6, seed=1, sink=0, condition="gray_box")
    u.update(over)
    return u


def units_of(cell: Mapping, seeds, sinks) -> List[dict]:
    return [dict(cell, seed=s, sink=k) for s in seeds for k in sinks]


class FakePlatform:
    """The Platform contract: run_batch skips units the store marks done, appends one row per unit to the store
    (under unit["id"]), marks it done and returns the rows in unit order. The row carries the unit fields and
    `asr_final = asr_fn(arm_name, unit)`. `calls` records every call's size, units and thresholds."""
    name = "mock"

    def __init__(self, asr_fn: Callable[[str, dict], float], fail_after_calls: int | None = None):
        self.asr_fn = asr_fn
        self.calls: List[dict] = []
        self.fail_after_calls = fail_after_calls

    def run_batch(self, units, arm, settings, store, tracker) -> List[dict]:
        if self.fail_after_calls is not None and len(self.calls) >= self.fail_after_calls:
            raise RuntimeError("simulated crash")
        todo = [u for u in units if not store.is_done(u["id"])]
        self.calls.append(dict(arm=arm.name, n=len(todo), units=todo, thresholds=todo[0]["thresholds"] if todo else None))
        rows: List[Dict] = []
        for u in todo:
            row = {k: v for k, v in u.items() if k != "thresholds"}
            row["asr_final"] = float(self.asr_fn(arm.name, u))
            store.append_rows(u["id"], [row])
            store.mark_done(u["id"])
            rows.append(row)
        return rows

    @property
    def submitted(self) -> int:
        return sum(c["n"] for c in self.calls)


def arm_spec(name: str = "X"):
    return core.ArmSpec(name=name, factory=lambda: None)


def arm_of(name: str):
    return arm_spec(name)
