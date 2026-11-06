"""Settings an arm or a platform will need during a run, resolved before the first unit so that a block never fails
hours in because a key is unset (core.Unset names it)."""
from __future__ import annotations

from typing import Iterable, Mapping

from bazt_lab.core import public as core

MATCHED_SUFFIX = "_matched"
UPSTREAM = "BAZT_upstream"          # the architecture track's package: it has its own parameters


def family(arm: str) -> str:
    """BAZT_no_budget and the other ablations are BAZT; BAZT_upstream is not."""
    return "BAZT" if arm.startswith("BAZT_") and arm != UPSTREAM else arm


def sanitizes(arm: str) -> bool:
    """BAZT, its ablations and BAZT_upstream can sanitize; the other arms never do."""
    return arm == UPSTREAM or family(arm) == "BAZT"


GNN_ARMS = ("GNN", "GNN_recalibrated")
GNN_KEYS = ("gnn.cut", "gnn.carry_flags", "gnn.fallback_family")
CALIBRATION_MODES = ("closed_loop", "open_loop")
ORACLE = "oracle_alloc"
ORACLE_SOURCE = "random_alloc"       # D96: oracle_alloc acts on random_alloc's calibrated audit_score


def required_keys(s: Mapping, platform: str, arms: Iterable[str], block: str) -> list:
    """Dotted keys the arms (and their capped variants) and the platform read during the run."""
    families = {family(a) for a in arms}
    keys = []
    if families & {"C", "BAZT"}:
        keys += [f"conventional.c.judge_score_bins.{platform}", "conventional.c.presence_window_rounds"]
    if block != "gnn_train" and families & set(core.get(s, "llm.judge_arms")):
        keys += (["llm.judge.provider", "llm.judge.top_logprobs", "llm.judge.floor_probability"] if platform == "llm"
                 else ["mock.auditor.binormal_method"])
    if platform == "llm" and families & {"BAZT", UPSTREAM}:
        keys.append("runtime.sanitize_success")      # BAZT prices sanitize with it (the mock lists it among its own)
    if UPSTREAM in set(arms):
        keys.append("bazt_upstream.verdict_probability")
    if families & set(GNN_ARMS) and block != "gnn_train":
        keys += list(GNN_KEYS)
    return keys


def _range(s: Mapping, key: str) -> tuple:
    value = core.get(s, key)
    ok = isinstance(value, list) and len(value) == 2 and all(isinstance(x, int) and not isinstance(x, bool)
                                                              for x in value) and value[0] <= value[1]
    if not ok:
        raise ValueError(f"{key} must be an inclusive range [first, last] of integers, got {value!r}")
    return value[0], value[1]


def heldout_overlap(s: Mapping, seeds: set) -> bool:
    lo, hi = _range(s, "mock.heldout_seeds")
    return any(lo <= x <= hi for x in seeds)


def check_seeds(s: Mapping, platform: str, block: str) -> None:
    """The seed sets of a platform are pairwise disjoint (and, on the LLM side, as many as seeds_per_cell)."""
    if platform == "mock":
        named = {k: _range(s, k) for k in ("mock.main_seeds", "mock.heldout_seeds", "mock.tune_seeds")}
        if core.is_set(s, "mock.adaptive.search_seeds"):
            named["mock.adaptive.search_seeds"] = _range(s, "mock.adaptive.search_seeds")
        items = list(named.items())
        for i, (a, (alo, ahi)) in enumerate(items):
            for b, (blo, bhi) in items[i + 1:]:
                if alo <= bhi and blo <= ahi:
                    raise ValueError(f"{a} {[alo, ahi]} overlap {b} {[blo, bhi]}: seed splits must be disjoint")
    elif block != "gnn_train":
        n = core.get(s, "llm.seeds_per_cell")
        sets = {}
        for key in ("runtime.llm_seeds", "runtime.llm_calibration_seeds"):
            value = core.get(s, key)
            if not isinstance(value, list):
                raise ValueError(f"{key} must be a list like [1, 2, 3], got {value!r}")
            if len(value) != n or len(set(value)) != len(value):
                raise ValueError(f"{key} must hold llm.seeds_per_cell={n} distinct seeds, got {value!r}")
            sets[key] = set(value)
        if sets["runtime.llm_seeds"] & sets["runtime.llm_calibration_seeds"]:
            raise ValueError("runtime.llm_seeds overlap runtime.llm_calibration_seeds: calibration seeds must differ "
                             "from evaluation seeds")


def resolve(s: Mapping, keys: Iterable[str]) -> None:
    for key in keys:
        core.get(s, key)
