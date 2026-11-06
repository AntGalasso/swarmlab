"""The attacker's strategy space: a finite set of `attack_params` per (family, N, condition).

Every grid comes from settings (mock.adaptive.*); nothing here holds an experiment value. Candidate 0 is always `{}`,
the family's default attack, so a best response is never worse than the non-adaptive attack. Ties go to the
earlier candidate, so the order below is part of the definition.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

from bazt_lab.core import public as core

CONDITIONS = ("gray_box", "white_box", "embedding_evasion", "attack_rate")
INSIDER_FAMILIES = ("insider_pi", "insider_ta", "insider_ma")
SINGLE_ENTRY_FAMILIES = ("infection", "aitm")
FAMILIES = INSIDER_FAMILIES + SINGLE_ENTRY_FAMILIES
KNOWLEDGE = ("cell", "seed", "episode")        # granularity at which the strategy may vary (see CONTRACT.md)
INSIDER_SETS = ("contiguous", "all")


class Refusal(core.Unset, ValueError):
    """A setting or plan the search refuses. A core.Unset, so the runner preflight prints `Refusing <block>: ...` with
    the message; also a ValueError for callers that catch that."""

    def __str__(self) -> str:
        return str(self.args[0])


@dataclass(frozen=True)
class Space:
    n_insiders: int
    insider_sets: str
    start_fractions: Tuple[float, ...]          # ascending
    rates: Tuple[float, ...]                    # descending, contains 1.0
    detectabilities: Tuple[float, ...]          # descending, contains 1.0
    evasion_cost: float
    max_candidates: int
    search_seeds: Optional[Tuple[int, int]]     # None when no configured condition searches at cell level
    every_sink: bool
    main_seeds: Tuple[int, int]
    knowledge: Mapping[str, str]                # condition -> KNOWLEDGE value, for the configured conditions


def _grid(settings, key, lo_open: bool, hi_open: bool, must_contain_one: bool = False) -> List[float]:
    values = [float(v) for v in core.get(settings, key)]
    full = f"{key}: "
    if not values:
        raise Refusal(full + "must not be empty")
    if len(set(values)) != len(values):
        raise Refusal(full + f"has duplicate values {values}")
    for v in values:
        if math.isnan(v):
            raise Refusal(full + "has NaN")
        if (v <= 0.0 if lo_open else v < 0.0) or (v >= 1.0 if hi_open else v > 1.0):
            raise Refusal(full + f"value {v} outside {'(' if lo_open else '['}0, 1{')' if hi_open else ']'}")
    if must_contain_one and 1.0 not in values:
        raise Refusal(full + "must contain 1.0 (the unmodified attack), so every condition's space contains gray_box's")
    return values


def _seed_range(settings, key) -> Tuple[int, int]:
    lo, hi = (int(v) for v in core.get(settings, key))
    if lo > hi:
        raise Refusal(f"{key}: lower bound {lo} above upper bound {hi}")
    return lo, hi


def load_space(settings: Mapping) -> Space:
    """Read and validate every key the search needs. Raises core.Unset naming a missing key, ValueError otherwise."""
    conditions = list(core.get(settings, "mock.adaptive.conditions"))
    for c in conditions:
        if c not in CONDITIONS:
            raise Refusal(f"mock.adaptive.conditions: unknown condition {c!r}; known: {list(CONDITIONS)}")
    insider_sets = core.get(settings, "mock.adaptive.insider_sets")
    if insider_sets not in INSIDER_SETS:
        raise Refusal(f"mock.adaptive.insider_sets must be one of {list(INSIDER_SETS)}, got {insider_sets!r}")
    cost = float(core.get(settings, "mock.adaptive.evasion_cost"))
    if not 0.0 <= cost <= 1.0:
        raise Refusal(f"mock.adaptive.evasion_cost must be in [0, 1], got {cost}")
    cap = int(core.get(settings, "mock.adaptive.max_candidates"))
    if cap < 1:
        raise Refusal(f"mock.adaptive.max_candidates must be at least 1, got {cap}")
    knowledge: Dict[str, str] = {}
    for c in conditions:
        k = core.get(settings, f"mock.adaptive.knowledge.{c}")
        if k not in KNOWLEDGE:
            raise Refusal(f"mock.adaptive.knowledge.{c} must be one of {list(KNOWLEDGE)}, got {k!r}")
        if c == "gray_box" and k != "cell":
            raise Refusal("mock.adaptive.knowledge.gray_box must be 'cell': gray-box does not know the allocation")
        if c == "white_box" and k == "cell":
            raise Refusal("mock.adaptive.knowledge.white_box must be 'seed' or 'episode': white-box knows the allocation")
        knowledge[c] = k
    search = None
    if "cell" in knowledge.values():                 # only a cell-level strategy is optimised on search seeds
        search = _seed_range(settings, "mock.adaptive.search_seeds")
        main = _seed_range(settings, "mock.main_seeds")
        held = _seed_range(settings, "mock.heldout_seeds")
        tune = _seed_range(settings, "mock.tune_seeds")
        for name, (lo, hi) in (("mock.main_seeds", main), ("mock.heldout_seeds", held), ("mock.tune_seeds", tune)):
            if search[0] <= hi and lo <= search[1]:
                raise Refusal(f"mock.adaptive.search_seeds {list(search)} overlap {name} {[lo, hi]}: the attacker "
                              f"must be optimised on seeds that are not evaluated or calibrated on")
    n_insiders = core.get(settings, "runtime.n_insiders")
    if isinstance(n_insiders, bool) or not isinstance(n_insiders, int) or n_insiders < 1:
        raise Refusal(f"runtime.n_insiders must be an integer >= 1, got {n_insiders!r}")
    return Space(
        n_insiders=n_insiders, insider_sets=insider_sets,
        start_fractions=tuple(sorted(_grid(settings, "mock.adaptive.start_fractions", False, True))),
        rates=tuple(sorted(_grid(settings, "mock.adaptive.rates", True, False, True), reverse=True)),
        detectabilities=tuple(sorted(_grid(settings, "mock.adaptive.detectabilities", True, False, True), reverse=True)),
        evasion_cost=cost, max_candidates=cap, search_seeds=search, every_sink=bool(core.get(settings, "mock.every_sink")),
        main_seeds=_seed_range(settings, "mock.main_seeds"), knowledge=knowledge)


def _n_entry_sets(family: str, N: int, space: Space) -> int:
    if family in SINGLE_ENTRY_FAMILIES:
        return N
    if family in INSIDER_FAMILIES:
        n = space.n_insiders
        if n >= N:
            raise Refusal(f"runtime.n_insiders={n} leaves no victim among the {N} agents")
        return N - n + 1 if space.insider_sets == "contiguous" else math.comb(N, n)
    raise Refusal(f"unknown attack family {family!r}; known: {list(FAMILIES)}")


def entry_sets(family: str, N: int, space: Space) -> List[List[int]]:
    """Degree ranks (0 = highest out-degree, ties by agent index) that play the attacker role: the entry agent
    (infection), the victim (aitm) or the n_insiders insiders."""
    _n_entry_sets(family, N, space)
    if family in SINGLE_ENTRY_FAMILIES:
        return [[r] for r in range(N)]
    n = space.n_insiders
    if space.insider_sets == "contiguous":
        return [list(range(r, r + n)) for r in range(N - n + 1)]
    return [list(c) for c in itertools.combinations(range(N), n)]


def candidate_count(family: str, N: int, condition: str, space: Space) -> int:
    if condition not in CONDITIONS:
        raise Refusal(f"unknown condition {condition!r}; known: {list(CONDITIONS)}")
    rates = len(space.rates) if condition == "attack_rate" else 1
    dets = len(space.detectabilities) if condition == "embedding_evasion" else 1
    return 1 + _n_entry_sets(family, N, space) * len(space.start_fractions) * rates * dets


def checked_count(family: str, N: int, condition: str, space: Space) -> int:
    """candidate_count, refused above mock.adaptive.max_candidates (before anything is enumerated)."""
    n = candidate_count(family, N, condition, space)
    if n > space.max_candidates:
        raise Refusal(f"{family} N={N} {condition}: {n} candidates exceed mock.adaptive.max_candidates="
                         f"{space.max_candidates}; shrink the grids or raise the cap")
    return n


def _params(entry: List[int], start: float, rate: float, det: float, cost: float) -> dict:
    p: dict = {"entry_ranks": entry}
    if start != 0.0:
        p["start_frac"] = start
    if rate != 1.0:
        p["rate"] = rate
    if det != 1.0:
        p["detectability"] = det
        potency = round(1.0 - cost * (1.0 - det), 12)
        if potency != 1.0:
            p["potency"] = potency
    return p


def candidates(family: str, N: int, condition: str, space: Space) -> List[dict]:
    """The strategy space, in tie-break order: `{}` first, then entry set (ascending), start (ascending),
    rate (descending), detectability (descending). gray_box and white_box vary entry and start; embedding_evasion
    adds detectability (with its potency cost); attack_rate adds rate."""
    checked_count(family, N, condition, space)
    rates = space.rates if condition == "attack_rate" else (1.0,)
    dets = space.detectabilities if condition == "embedding_evasion" else (1.0,)
    out: List[dict] = [{}]
    for entry, start, rate, det in itertools.product(entry_sets(family, N, space), space.start_fractions, rates, dets):
        out.append(_params(entry, start, rate, det, space.evasion_cost))
    return out
