"""Unit expansion: every block of the spec as a list of unit dicts.

An LLM unit is one seed run (its conversations in sequence on one graph). A mock unit is one episode.

The mock frontier has millions of episodes, so a mock block is a `MockGrid`: it knows its size in closed form and
yields its units lazily. Its axes nest as arm cell, topology, family, agents, seed, sink, so the units of one batch
group (`group_key`) are consecutive and a launch needs no more than one group in memory.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Mapping, Sequence

from bazt_lab.core import public as core
from bazt_lab.features.runner.private import needs

# What the runner adds to a unit before handing it to a platform. None takes part in the unit's identity (the replayed
# schedule of llm_adaptive_check and the BAZT rounds of the matched-timing controls are pinned in their block directory
# instead, private/replay.py and private/matched.py).
ANNOTATIONS = ("id", "thresholds", "twin_id", "attack_schedule", "timing_schedule", "twin_timing_schedule")
# The fields the units of one Platform.run_batch call share (absent fields count as None).
GROUP_FIELDS = ("arm", "audit", "intervention", "topology", "attack", "agents", "cost_condition", "preset",
                "condition", "attack_params")


def unit_id(unit: Mapping) -> str:
    base = {k: v for k, v in unit.items() if k not in ANNOTATIONS}
    digest = hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()[:16]
    return f"{unit['arm']}-{digest}"


def group_key(unit: Mapping) -> tuple:
    return tuple(json.dumps(v, sort_keys=True) if isinstance(v, dict) else v
                 for v in (unit.get(f) for f in GROUP_FIELDS))


def cell_key(unit: Mapping) -> tuple:
    """The calibration cell of a unit: (arm, topology, N, backbone, task). The mock has no backbone and no task."""
    return unit["arm"], unit.get("topology"), unit.get("agents"), unit.get("backbone"), unit.get("task")


# ==================================================================================================== containers
def _once(values) -> list:
    """The values that are not None, in order, each once; equality, not hashing (a value may be a list)."""
    out: list = []
    for v in values:
        if v is not None and v not in out:
            out.append(v)
    return out


class UnitList(list):
    """The units of a block that is small enough to hold: every LLM block and the mock calibration."""

    def by_agents(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for u in self:
            counts[u["agents"]] = counts.get(u["agents"], 0) + 1
        return counts

    def arms(self) -> set:
        return {u["arm"] for u in self}

    def cells(self) -> set:
        return {cell_key(u) for u in self}

    def probes(self) -> Iterator[dict]:
        """The first unit of each (arm, topology, attack, conditions, capped or not)."""
        seen = set()
        for u in self:
            key = (*group_key(u)[:1], _flag(u.get("audit")), _flag(u.get("intervention")), *group_key(u)[3:5],
                   *group_key(u)[6:])
            if key not in seen:
                seen.add(key)
                yield u

    def seed_set(self) -> set:
        return {u["seed"] for u in self if "seed" in u}

    def conditions(self) -> list:
        """The adaptive conditions the units carry, in unit order, each once (a condition may be any settings value)."""
        return _once(u.get("condition") for u in self)

    def cost_groups(self) -> Dict[tuple, int]:
        out: Dict[tuple, int] = {}
        for u in self:
            kind = "uncapped" if u.get("audit") == "uncapped" and u.get("intervention") == "uncapped" else "capped"
            key = (u["arm"], kind, u["agents"])
            out[key] = out.get(key, 0) + 1
        return out


def _flag(value) -> bool:
    return value == "uncapped"


@dataclass(frozen=True)
class MockGrid:
    """The mock episodes of one block, never held in memory. `arms_cells` is a list of (arm, (audit, intervention)).

    A block that repeats its grid under several conditions lists them in `variants`: a value stored in `variant_field`,
    or a dict of unit fields (a robustness preset with its attack_params).
    """
    block: str
    arms_cells: Sequence
    topologies: Sequence
    families: Sequence
    agent_counts: Sequence
    seeds: Sequence
    every_sink: bool
    fixed: Mapping = field(default_factory=dict)       # fields every unit of the block carries
    variant_field: str | None = None
    variants: Sequence = (None,)

    def _extra(self, variant) -> dict:
        if isinstance(variant, Mapping):
            return {**self.fixed, **variant}
        return {**self.fixed, **({self.variant_field: variant} if self.variant_field else {})}

    def _unit(self, arm, audit, interv, topo, fam, n, seed, sink, extra) -> dict:
        return dict(block=self.block, platform="mock", arm=arm, audit=audit, intervention=interv, topology=topo,
                    attack=fam, agents=n, seed=seed, sink=sink, **extra)

    def _ordered_families(self) -> list:
        """Attack-free first (D109): an attacked episode's clean reference is its attack-free twin's own run."""
        return sorted(self.families, key=lambda f: f != "none")

    def __iter__(self) -> Iterator[dict]:
        for variant in self.variants:
            extra = self._extra(variant)
            for fam, ((arm, (audit, interv)), topo, n, seed) in itertools.product(
                    self._ordered_families(),
                    itertools.product(self.arms_cells, self.topologies, self.agent_counts, self.seeds)):
                for sink in (range(n) if self.every_sink else (None,)):
                    yield self._unit(arm, audit, interv, topo, fam, n, seed, sink, extra)

    def probes(self) -> Iterator[dict]:
        """One small representative episode per (arm, capped or not per budget, topology, family, variant): the
        smallest N, the first seed, the first sink. Counted from the axes, never from the whole grid."""
        if not len(self):
            return
        seen = set()
        for variant in self.variants:
            extra = self._extra(variant)
            for (arm, (audit, interv)), topo, fam in itertools.product(self.arms_cells, self.topologies,
                                                                      self.families):
                key = (repr(variant), arm, _flag(audit), _flag(interv), topo, fam)
                if key not in seen:
                    seen.add(key)
                    n = min(self.agent_counts)
                    yield self._unit(arm, audit, interv, topo, fam, n, self.seeds[0],
                                     0 if self.every_sink else None, extra)

    def by_agents(self) -> Dict[int, int]:
        per_sink = (len(self.arms_cells) * len(self.topologies) * len(self.families) * len(self.seeds)
                    * len(self.variants))
        counts: Dict[int, int] = {}
        for n in self.agent_counts:
            counts[n] = counts.get(n, 0) + per_sink * (n if self.every_sink else 1)
        return counts

    def __len__(self) -> int:
        return sum(self.by_agents().values())

    def arms(self) -> set:
        return {arm for arm, _ in self.arms_cells} if len(self) else set()

    def cells(self) -> set:
        if not len(self):
            return set()
        return {(arm, topo, n, None, None) for arm, _ in self.arms_cells for topo in self.topologies
                for n in self.agent_counts}

    def seed_set(self) -> set:
        return set(self.seeds) if len(self) else set()

    def conditions(self) -> list:
        """The adaptive conditions the units carry, in grid order, each once; counted from the variants."""
        return _once(self._extra(v).get("condition") for v in self.variants) if len(self) else []

    def cost_groups(self) -> Dict[tuple, int]:
        """{(arm, "capped" | "uncapped", N): episodes}, counted from the axes."""
        per = len(self.topologies) * len(self.families) * len(self.seeds) * len(self.variants)
        out: Dict[tuple, int] = {}
        for arm, (audit, interv) in self.arms_cells:
            kind = "uncapped" if audit == "uncapped" and interv == "uncapped" else "capped"
            for n in self.agent_counts:
                key = (arm, kind, n)
                out[key] = out.get(key, 0) + per * (n if self.every_sink else 1)
        return out


# ======================================================================================================== helpers
def _cell(value) -> float | str:
    """A budget level: a percent from profiles.toml becomes a fraction of deliveries; "uncapped" stays."""
    return value if value == "uncapped" else float(value) / 100.0


def capped_cells(levels) -> List[tuple]:
    """Every (audit, intervention) pair of the crossed levels except the fully uncapped one."""
    axis = [_cell(v) for v in levels] + ["uncapped"]
    return [(a, i) for a, i in itertools.product(axis, axis) if not (a == "uncapped" and i == "uncapped")]


def _seed_range(bounds) -> List[int]:
    lo, hi = bounds
    return list(range(int(lo), int(hi) + 1))


# ============================================================================================================ LLM
def _llm_seeds(s: Mapping, key: str, strict: bool) -> List:
    try:
        return list(core.get(s, f"runtime.{key}"))
    except core.Unset:
        if strict:
            raise
        return [f"<{key}[{i}]>" for i in range(core.get(s, "llm.seeds_per_cell"))]


def _backbones(s: Mapping, strict: bool) -> List[str]:
    try:
        return list(core.get(s, "llm.backbones"))
    except core.Unset:
        if strict:
            raise
        return [f"<backbone {i + 1}>" for i in range(core.get(s, "llm.n_backbones"))]


def _per_seed(s: Mapping, total_key: str) -> int:
    total, seeds = core.get(s, total_key), core.get(s, "llm.seeds_per_cell")
    if total % seeds:
        sys.exit(f"{total_key}={total} is not divisible by llm.seeds_per_cell={seeds}.")
    return total // seeds


def _llm_attack_cells(s: Mapping) -> List[tuple]:
    attacks = [tuple(c) for c in core.get(s, "llm.attack_cells")]
    return attacks + [("none", task) for task in core.get(s, "llm.attack_free_tasks")]


def sanitize_modes(s: Mapping, arm: str) -> List[str]:
    """D101: an arm that can sanitize runs once per mode of the profile; every other arm once, with the first mode."""
    modes = list(core.get(s, "llm.sanitize_modes"))
    if not modes:
        raise ValueError("llm.sanitize_modes must list at least one mode")
    return modes if needs.sanitizes(arm) else modes[:1]


def _llm_units(s: Mapping, block: str, arms, cells, strict: bool) -> UnitList:
    conv = _per_seed(s, "llm.conversations_per_cell")
    units = UnitList()
    for arm, (audit, interv), backbone, topology, (attack, task), seed in itertools.product(
            arms, cells, _backbones(s, strict), core.get(s, "llm.topologies"), _llm_attack_cells(s),
            _llm_seeds(s, "llm_seeds", strict)):
        for mode in sanitize_modes(s, arm):
            units.append(dict(block=block, platform="llm", arm=arm, audit=audit, intervention=interv,
                              backbone=backbone, topology=topology, attack=attack, task=task, seed=seed,
                              conversations=conv, agents=core.get(s, "llm.agents"), rounds=core.get(s, "llm.rounds"),
                              sanitize_mode=mode))
    return units


def units_llm_headline(s, strict):
    return _llm_units(s, "llm_headline", core.get(s, "llm.headline_arms"), [("uncapped", "uncapped")], strict)


def units_llm_capped(s, strict):
    cells = [(_cell(a), _cell(i)) for a, i in core.get(s, "llm.capped_cells")]
    return _llm_units(s, "llm_capped", core.get(s, "llm.capped_arms"), cells, strict)


def units_llm_calibration(s, strict):
    conv = _per_seed(s, "llm.calibration_conversations_per_cell")
    return UnitList(
        dict(block="llm_calibration", platform="llm", arm=arm, audit="uncapped", intervention="uncapped",
             backbone=b, topology=t, attack="none", task=task, seed=seed, conversations=conv,
             agents=core.get(s, "llm.agents"), rounds=core.get(s, "llm.rounds"), sanitize_mode=sanitize_modes(s, arm)[0])
        for arm, b, t, task, seed in itertools.product(
            core.get(s, "llm.calibrated_arms"), _backbones(s, strict), core.get(s, "llm.topologies"),
            core.get(s, "llm.attack_free_tasks"), _llm_seeds(s, "llm_calibration_seeds", strict)))


def units_gnn_train(s, strict):
    try:
        backbone = core.get(s, "gnn_train.backbone")
    except core.Unset:
        if strict:
            raise
        backbone = "<gnn backbone>"
    return UnitList(
        dict(block="gnn_train", platform="llm", arm="GNN", attack=attack, task=task, edge_probability=p,
             attackers=k, backbone=backbone, conversations=core.get(s, "gnn_train.conversations_per_setting"),
             agents=core.get(s, "gnn_train.agents"), rounds=core.get(s, "gnn_train.rounds"),
             sanitize_mode=sanitize_modes(s, "GNN")[0])
        for (attack, task), p, k in itertools.product(
            [tuple(f) for f in core.get(s, "gnn_train.families")], core.get(s, "gnn_train.edge_probabilities"),
            core.get(s, "gnn_train.attackers")))


def units_llm_adaptive_check(s, strict):
    arms, families, condition, tasks = (core.get(s, f"llm.adaptive_check.{k}")
                                        for k in ("arms", "families", "condition", "tasks"))
    return UnitList(
        dict(block="llm_adaptive_check", platform="llm", arm=arm, audit="uncapped", intervention="uncapped",
             backbone=b, topology=t, attack=fam, task=task, condition=condition, seed=seed,
             conversations=_per_seed(s, "llm.conversations_per_cell"), agents=core.get(s, "llm.agents"),
             rounds=core.get(s, "llm.rounds"), sanitize_mode=mode)
        for arm, fam, task, b, t, seed in itertools.product(
            arms, families, tasks, _backbones(s, strict), core.get(s, "llm.topologies"), _llm_seeds(s, "llm_seeds", strict))
        for mode in sanitize_modes(s, arm))


# =========================================================================================================== mock
def _frontier_arms_cells(s: Mapping) -> List[tuple]:
    capped = capped_cells(core.get(s, "budgets.levels_percent"))
    pairs = [(arm, ("uncapped", "uncapped")) for arm in core.get(s, "mock.uncapped_arms")]
    pairs += [(arm, c) for arm in core.get(s, "mock.capped_arms") + core.get(s, "mock.allocation_arms")
              for c in capped]
    return pairs


def units_mock_frontier(s, strict, seeds_key="mock.main_seeds", block="mock_frontier"):
    return MockGrid(block, _frontier_arms_cells(s), core.get(s, "mock.topologies"),
                    core.get(s, "mock.families") + ["none"], core.get(s, "mock.frontier_agents"),
                    _seed_range(core.get(s, seeds_key)), core.get(s, "mock.every_sink"))


def units_mock_heldout(s, strict):
    return units_mock_frontier(s, strict, seeds_key="mock.heldout_seeds", block="mock_heldout")


def units_mock_calibration(s, strict):
    return UnitList(
        dict(block="mock_calibration", platform="mock", arm=arm, audit="uncapped", intervention="uncapped",
             topology=t, attack="none", agents=n, seed=seed, sink=None)
        for arm, t, n, seed in itertools.product(
            core.get(s, "mock.calibrated_arms"), core.get(s, "mock.topologies"),
            core.get(s, "mock.epidemiology_agents"), _seed_range(core.get(s, "mock.tune_seeds"))))


def preset_variants(s: Mapping) -> list:
    """The auditor presets of the robustness block (spec section 5). The adaptive preset is a range of detection
    rates: one variant per end point of auditor_presets.adaptive_tpr, carried as attack_params['auditor_tpr'], with
    the false-positive rates of the main preset."""
    variants = []
    for preset in core.get(s, "mock.presets_robustness.presets"):
        if preset == "adaptive":
            tpr = core.get(s, "auditor_presets.adaptive_tpr")
            if (not isinstance(tpr, list) or len(tpr) != 2 or not all(isinstance(x, (int, float)) for x in tpr)
                    or not 0 <= tpr[0] <= tpr[1] <= 1):
                raise ValueError(f"auditor_presets.adaptive_tpr must be [low, high] with 0 <= low <= high <= 1, "
                                 f"got {tpr!r}")
            variants += [{"preset": "adaptive", "attack_params": {"auditor_tpr": x}} for x in dict.fromkeys(tpr)]
        elif core.is_set(s, f"auditor_presets.{preset}"):
            variants.append({"preset": preset})
        else:
            raise ValueError(f"mock.presets_robustness.presets: unknown preset {preset!r}; auditor_presets defines "
                             f"{sorted(k for k in core.get(s, 'auditor_presets') if k not in ('main', 'adaptive_tpr'))}")
    return variants


def scoped(name: str, topologies_key="mock.topologies", agents_key="mock.frontier_agents", variant_key=None,
           variant_field=None, fixed=None, reference_key=None, variants=None):
    """A mock block whose arms and cells are set in [mock.<name>]; it raises Unset until both are set.
    `reference_key` names a setting holding an arm the block always includes (the analysis compares against it)."""
    def build(s, strict):
        arms = list(core.get(s, f"mock.{name}.arms"))
        if reference_key and core.get(s, reference_key) not in arms:
            arms.insert(0, core.get(s, reference_key))
        arms_cells = [(a, (_cell(au), _cell(iv))) for a in arms for au, iv in core.get(s, f"mock.{name}.cells")]
        if variants:
            found = variants(s)
        else:
            found = core.get(s, variant_key) if variant_key else (None,)
        return MockGrid(f"mock_{name}", arms_cells, core.get(s, topologies_key), core.get(s, "mock.families") + ["none"],
                        core.get(s, agents_key), _seed_range(core.get(s, "mock.main_seeds")),
                        core.get(s, "mock.every_sink"), fixed or {}, variant_field if variant_key else None, found)
    return build
