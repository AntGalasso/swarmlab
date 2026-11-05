"""Synthetic result rows with planted effects, written in the launcher's layout:
results/experiments/<profile>/<block>/units/<unit file>.jsonl, one `{"unit": {...}, **metrics}` per line.

Not a test module (pytest does not collect it). Numpy draws are fine here: this is test data, not an experiment.
"""
from __future__ import annotations

import itertools
import json
import math
import zlib
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np

UNIT_FIELDS = ("block", "platform", "arm", "audit", "intervention", "topology", "attack", "agents", "seed", "sink",
               "task", "backbone", "conversation", "condition", "preset", "conversations", "rounds")
UNCAPPED = ("uncapped", "uncapped")


def write_rows(root: Path, rows: Iterable[dict], per_file: int = 40) -> Path:
    """Write flat rows (unit fields + metrics) under root/<block>/units/. Returns root."""
    by_block: Dict[str, List[dict]] = {}
    for row in rows:
        by_block.setdefault(row["block"], []).append(row)
    for block, items in by_block.items():
        units = Path(root) / block / "units"
        units.mkdir(parents=True, exist_ok=True)
        for i in range(0, len(items), per_file):
            lines = []
            for row in items[i:i + per_file]:
                unit = {k: row[k] for k in UNIT_FIELDS if k in row}
                lines.append(json.dumps({"unit": unit, **{k: v for k, v in row.items() if k not in UNIT_FIELDS}}))
            (units / f"{items[i]['arm']}-{i // per_file:04d}.jsonl").write_text("\n".join(lines) + "\n")
    return Path(root)


def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def episodes(
        arms: Dict[str, Callable[[tuple, str, int], float]],
        *, block: str = "mock_frontier", platform: str = "mock", cells: Sequence[tuple] = (UNCAPPED,),
        cells_by_arm: Dict[str, Sequence[tuple]] | None = None, topologies: Sequence[str] = ("chain",),
        agents: Sequence[int] = (12,), seeds: Sequence[int] = range(1, 31), sinks: Sequence[int] = range(3),
        conversations: Sequence[int] | None = None, families: Sequence[str] = ("insider_pi",), tasks=(None,),
        backbones=(None,), utility: Dict[str, Callable[[tuple, str, int], float]] | None = None,
        seed_sd: float = 0.05, noise_sd: float = 0.02, condition=None, preset=None, rng_seed: int = 0,
        shared_seed: int = 0,
        spend: Callable[[str, tuple], tuple] | None = None,
        family_effect: Callable[[str, str], float] | None = None) -> List[dict]:
    """Rows for every (arm, cell, topology, N, family, seed, sink|conversation), plus attack-free rows.

    arms[arm](cell, topology, n_agents) is the planted mean ASR; utility[arm] the planted mean clean accuracy.
    A shared episode effect (seed_sd) is common to every arm and cell, so paired differences are tight while the
    marginal values are not: the common-random-numbers situation of the real runs. `seed_sd` is also added per seed.
    family_effect(arm, family) shifts the planted ASR of one attack family. The episode effects depend on
    (shared_seed, episode) only, so two calls with the same shared_seed (two blocks) share them; noise_sd noise does not.
    """
    rng = np.random.default_rng(rng_seed)
    unit_axis = conversations if conversations is not None else sinks
    unit_name = "conversation" if conversations is not None else "sink"
    cache: Dict[tuple, float] = {}

    def shared(*key, sd):                        # one draw per key, common to every arm, cell and call with this shared_seed
        if key not in cache:
            cache[key] = float(np.random.default_rng([shared_seed, zlib.crc32(repr(key).encode())]).normal(0, sd))
        return cache[key]

    rows: List[dict] = []
    for arm, asr in arms.items():
        for cell in (cells_by_arm or {}).get(arm, cells):
            for topo, n, attack, task, bb, seed, u in itertools.product(
                    topologies, agents, list(families) + ["none"], tasks, backbones, seeds, unit_axis):
                eff = (shared("seed", seed, topo, n, sd=seed_sd)
                       + shared("episode", topo, n, attack, task, bb, seed, u, sd=seed_sd / 2))
                noise = float(rng.normal(0, noise_sd))
                row = dict(block=block, platform=platform, arm=arm, audit=cell[0], intervention=cell[1],
                           topology=topo, attack=attack, agents=n, seed=seed, task=task, backbone=bb,
                           condition=condition, preset=preset, **{unit_name: u})
                if attack == "none":
                    acc = (utility or {}).get(arm, lambda *a: 0.9)(cell, topo, n)
                    row.update(asr_final=math.nan, accuracy_attack=math.nan,
                               accuracy_clean=clip01(acc + 0.2 * eff + noise), protected_sink_harm=math.nan,
                               attack_free=True)
                else:
                    mean = asr(cell, topo, n) + (family_effect(arm, attack) if family_effect else 0.0)
                    row.update(asr_final=clip01(mean + eff + noise), accuracy_attack=clip01(1 - mean - eff - noise),
                               accuracy_clean=math.nan, attack_free=False,
                               protected_sink_harm=float(rng.random() < clip01(mean + eff)))
                sa, si = spend(arm, cell) if spend else (0.0, 0.0)
                row.update(spend_audit_total=sa, spend_intervention_total=si, spend_total=sa + si)
                rows.append(row)
    return rows


def settings(**over) -> dict:
    """A complete settings tree for the analysis; `over` maps dotted keys (with __ for dots) to values."""
    from bazt_lab.core import public as core
    blocks = {
        "budget": {"mock": ["mock_frontier"], "llm": ["llm_capped"]},
        "efficiency": {"mock": ["mock_frontier"], "llm": ["llm_capped"]},
        "topology": {"mock": ["mock_epidemiology"], "llm": ["llm_headline"]},
        "adaptive": {"mock": ["mock_adaptive"], "llm": ["llm_adaptive_check"]},
        "adaptive_reference": {"mock": ["mock_frontier"], "llm": ["llm_headline"]},
        "ladder": {"mock": ["mock_frontier"], "llm": ["llm_headline"]},
        "ablations": {"mock": ["mock_ablations"], "llm": []},
        "rank": {"mock": ["mock_frontier"], "llm": ["llm_headline"]},
        "containment": {"mock": ["mock_frontier"], "llm": ["llm_capped"]},
    }
    tree = {
        "profile": "test",
        "runtime": {"ni_margin_accuracy": 0.01, "rank_agreement_threshold": 0.8, "recovery_margin": 0.01},
        "analysis": {
            "alpha": 0.05, "bootstrap_resamples": 1000, "ci_level": 0.95,
            "baseline": "C", "challengers": ["BAZT"], "ladder": ["A", "B", "C"],
            "ablations": ["BAZT_no_budget", "BAZT_no_topology", "BAZT_no_graduated"], "ablation_reference": "BAZT",
            "containment_reference": "A", "containment_metrics": ["agents_reached_auc", "asr_final"],
            "efficiency_tolerance": 0.0, "adaptive_contrast": "advantage_shrinks_less_than_c_security",
            "rank_agreement_statistic": "spearman",
            "rank_agreement_arms": [["A", "A"], ["B", "B"], ["C", "C"], ["BAZT", "BAZT"]],
            "utility_endpoint": "accuracy_clean",
            "extra_metrics": {"protected_sink_harm": ["binary", "attacked"]},
            "pair_unit": {"mock": "seed", "llm": "episode"},
            "split_by": {"mock": [], "llm": []},
            "topology_classes": {
                "mock": {"heterogeneous": ["barabasi_albert", "star_hub", "hub_star"],
                         "homogeneous": ["small_world", "watts_strogatz"]},
                "llm": {"heterogeneous": ["barabasi_albert", "star_hub", "hub_star"],
                        "homogeneous": ["small_world", "watts_strogatz"]}},
            "blocks": blocks,
        },
    }
    for dotted, value in over.items():
        core.put(tree, dotted.replace("__", "."), value)
    return tree
