"""llm_adaptive_check (D110): the LLM attacker replays the mock's gray-box best response as its schedule.

For each unit of the block the runner finds one cell of the mock_adaptive grid (the same arm, family, N and condition;
the budget cell, the topology and the clock are owner keys, below), reads that cell's best response from the table
the mock search saved, and attaches `attack_schedule = {clock, start_frac, start_round, rate}` to the unit. Only the
attack start and the attack rate transfer: detectability does not (LLM text is judged as written, D110), and the
degree ranks of the entry agents are the mock's placement, not a schedule.

Owner keys (unset in profiles.toml, refused by name while unset):
  llm.adaptive_check.mock_cell            [audit, intervention] of mock.adaptive.cells whose best response is replayed
                                          (the LLM check runs uncapped)
  llm.adaptive_check.mock_topology.<t>    the mock topology (one of mock.topologies) whose best response LLM topology t
                                          replays
  llm.adaptive_check.start_clock          "seed_run": start_frac of the seed run's rounds (the LLM episode, D54), or
                                          "conversation": start_frac of each conversation's rounds
  llm.adaptive_check.mock_arm.<arm>       read only for an LLM arm the mock_adaptive grid does not have: the mock arm
                                          whose best response it replays

The start round is the mock's own rule (`adaptive.attack_rounds`, round 1 + floor(start_frac x T)) on the clock's T, in
the trace's rounds: index 0 = round 0. The entries used are pinned in the block directory (`best_responses.json`), so a
launch after the table changed is refused instead of mixing rows of two schedules.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from bazt_lab.core import public as core
from bazt_lab.features.runner.private import units as u

SOURCE = "mock_adaptive"                 # the block whose best responses the check replays
PIN = "best_responses.json"
CLOCKS = ("seed_run", "conversation")
KEY = "llm.adaptive_check"


def _text(cell: Mapping) -> str:
    return json.dumps(dict(cell), sort_keys=True)


def _mock_cell(s: Mapping, grid: u.MockGrid) -> tuple:
    value = core.get(s, f"{KEY}.mock_cell")
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{KEY}.mock_cell must be [audit, intervention] as in mock.adaptive.cells, got {value!r}")
    cell = (u._cell(value[0]), u._cell(value[1]))
    if cell not in {c for _, c in grid.arms_cells}:
        raise ValueError(f"{KEY}.mock_cell {value!r} is not a cell of mock.adaptive.cells "
                         f"{core.get(s, 'mock.adaptive.cells')}: the mock search has no best response there")
    return cell


def mock_unit(s: Mapping, unit: Mapping, grid: u.MockGrid) -> dict:
    """The mock_adaptive unit (without an episode) whose cell the LLM unit replays."""
    audit, intervention = _mock_cell(s, grid)
    arms = {a for a, _ in grid.arms_cells}
    arm = unit["arm"] if unit["arm"] in arms else core.get(s, f"{KEY}.mock_arm.{unit['arm']}")
    if arm not in arms:
        raise ValueError(f"{KEY}.mock_arm.{unit['arm']} = {arm!r} is not an arm of mock.adaptive.arms {sorted(arms)}")
    topology = core.get(s, f"{KEY}.mock_topology.{unit['topology']}")
    if topology not in grid.topologies:
        raise ValueError(f"{KEY}.mock_topology.{unit['topology']} = {topology!r} is not one of mock.topologies "
                         f"{list(grid.topologies)}")
    if unit["attack"] not in grid.families:
        raise ValueError(f"family {unit['attack']!r} of {KEY}.families is not searched by mock_adaptive "
                         f"(mock.families {[f for f in grid.families if f != 'none']})")
    if unit["agents"] not in grid.agent_counts:
        raise ValueError(f"llm.agents = {unit['agents']} is not searched by mock_adaptive (mock.frontier_agents "
                         f"{list(grid.agent_counts)})")
    if unit.get("condition") not in [grid._extra(v).get("condition") for v in grid.variants]:
        raise ValueError(f"condition {unit.get('condition')!r} of {KEY}.condition is not searched by mock_adaptive "
                         f"(mock.adaptive.conditions {core.get(s, 'mock.adaptive.conditions')})")
    variant = next(v for v in grid.variants if grid._extra(v).get("condition") == unit.get("condition"))
    return grid._unit(arm, audit, intervention, topology, unit["attack"], unit["agents"], None, None,
                      grid._extra(variant))


class Replay:
    """The schedules of the block's units: `annotate(unit)` adds attack_schedule; `pin()` is what the block pins."""

    def __init__(self, clock: str, entries: Mapping[tuple, tuple], attack_rounds) -> None:
        self.clock, self.entries, self.attack_rounds = clock, dict(entries), attack_rounds

    @staticmethod
    def _key(unit: Mapping) -> tuple:
        return unit["arm"], unit["topology"], unit["attack"], unit["agents"], unit.get("condition")

    def annotate(self, unit: Mapping) -> dict:
        if unit["attack"] == "none":
            return dict(unit)
        _, params = self.entries[self._key(unit)]
        T = int(unit["rounds"]) * (int(unit["conversations"]) if self.clock == "seed_run" else 1)
        on = self.attack_rounds(T, params["start_frac"], 1.0)
        start = int(on.nonzero()[0, 0]) if bool(on.any()) else T
        return {**unit, "attack_schedule": {"clock": self.clock, "start_frac": params["start_frac"],
                                            "start_round": start, "rate": params["rate"]}}

    def pin(self) -> dict:
        """The mock cells replayed with the start and rate read from each: what the block's rows depend on."""
        used = {cell: dict(params) for cell, params in self.entries.values()}
        return {"source": SOURCE, "clock": self.clock, "entries": dict(sorted(used.items()))}


def load(s: Mapping, grid, adaptive: Any, root: Path, source_grid) -> Replay:
    """The replay of the block's units, refused (core.Unset or ValueError) before anything is written."""
    clock = core.get(s, f"{KEY}.start_clock")
    if clock not in CLOCKS:
        raise ValueError(f"{KEY}.start_clock must be one of {list(CLOCKS)}, got {clock!r}")
    cells: Dict[tuple, dict] = {}
    for unit in grid:
        if unit["attack"] != "none":
            cells.setdefault(Replay._key(unit), mock_unit(s, unit, source_grid))
    directory = Path(root) / s["profile"] / SOURCE
    try:
        table = adaptive.load_best_responses(directory, s)
    except FileNotFoundError as e:
        raise ValueError(f"it replays the gray-box best responses of {SOURCE} (D110), and {directory} holds none "
                         f"({e}): run --block {SOURCE} first") from None
    entries, missing = {}, []
    for key, cell in cells.items():
        try:
            params = table.choice(cell).attack_params
        except LookupError:
            missing.append(cell)
            continue
        entries[key] = (_text(_without_episode(cell)), {"start_frac": float(params.get("start_frac", 0.0)),
                                                        "rate": float(params.get("rate", 1.0))})
    if missing:
        shown = "; ".join(_describe(c) for c in missing[:3])
        raise ValueError(f"the best responses of {SOURCE} in {directory} lack {len(missing)} cell(s) the check replays, "
                         f"e.g. {shown}: run --block {SOURCE} (a crashed search resumes)")
    return Replay(clock, entries, adaptive.attack_rounds)


def _without_episode(cell: Mapping) -> dict:
    return {k: v for k, v in cell.items() if k not in ("block", "seed", "sink")}


def _describe(cell: Mapping) -> str:
    return ", ".join(f"{k}={cell[k]}" for k in ("arm", "audit", "intervention", "topology", "attack", "agents",
                                                 "condition"))


def check_pin(path: Path, replay: Replay) -> None:
    """A block that pinned other entries than the table now gives is refused (ValueError)."""
    if not path.exists():
        return
    saved = json.loads(path.read_text(encoding="utf-8"))
    now = replay.pin()
    if saved != now:
        changed = sorted(c for c in set(saved.get("entries", {})) | set(now["entries"])
                         if saved.get("entries", {}).get(c) != now["entries"].get(c))
        shown = "; ".join(f"{c}: {saved.get('entries', {}).get(c)} -> {now['entries'].get(c)}" for c in changed[:2])
        clock = f"clock {saved.get('clock')!r} -> {now['clock']!r}; " if saved.get("clock") != now["clock"] else ""
        raise ValueError(f"{path} pins other best responses than {SOURCE} now holds ({clock}{shown}): rows of two "
                         f"schedules would mix; use another --results-root or --profile")
