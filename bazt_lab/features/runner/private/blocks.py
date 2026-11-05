"""The registry of blocks (EXPERIMENT_PLAN.md): which units each runs, on which platform, and how."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping

from bazt_lab.features.runner.private import units as u


@dataclass(frozen=True)
class Block:
    name: str
    platform: str                       # "mock" | "llm"
    kind: str                           # "calibrate": Platform.calibrate per cell; "train": GNN training units,
                                        # no thresholds; "evaluate": Platform.run_batch with thresholds attached
    description: str
    units: Callable[[Mapping, bool], object]    # (settings, strict) -> a UnitList or MockGrid; strict forbids placeholders
    requires_freeze: bool = False
    adaptive: bool = False              # the best-response search runs first and annotates the units


BLOCKS: Dict[str, Block] = {b.name: b for b in [
    Block("llm_calibration", "llm", "calibrate", "attack-free calibration of B, C, BAZT (alpha deny rate)",
          u.units_llm_calibration),
    Block("gnn_train", "llm", "train", "G-Safeguard GNN training, upstream setup", u.units_gnn_train),
    Block("llm_headline", "llm", "evaluate", "uncapped A, B, C, BAZT, GNN", u.units_llm_headline),
    Block("llm_capped", "llm", "evaluate", "capped C and BAZT at the profile's budget cells", u.units_llm_capped),
    Block("llm_adaptive_check", "llm", "evaluate", "held-out LLM adaptive check", u.units_llm_adaptive_check),
    Block("mock_calibration", "mock", "calibrate", "attack-free calibration on the tune split",
          u.units_mock_calibration),
    Block("mock_frontier", "mock", "evaluate", "full 7x7 budget frontier, every sink", u.units_mock_frontier),
    Block("mock_epidemiology", "mock", "evaluate", "H-topology at 12 and 100 agents",
          u.scoped("epidemiology", topologies_key="mock.epidemiology.topologies",
                   agents_key="mock.epidemiology_agents")),
    Block("mock_ablations", "mock", "evaluate", "BAZT minus one component", u.scoped("ablations", reference_key="analysis.ablation_reference")),
    Block("mock_adaptive", "mock", "evaluate", "adaptive attacker conditions",
          u.scoped("adaptive", variant_key="mock.adaptive.conditions", variant_field="condition"), adaptive=True),
    Block("mock_presets_robustness", "mock", "evaluate", "robustness auditor presets",
          u.scoped("presets_robustness", variants=u.preset_variants)),
    Block("mock_costs_robustness", "mock", "evaluate", "CAGE-4 cost weights",
          u.scoped("costs_robustness", fixed={"cost_condition": "robustness"})),
    Block("mock_legacy", "mock", "evaluate", "existing swarmlab arms with matched-timing controls",
          u.scoped("legacy")),
    Block("mock_heldout", "mock", "evaluate", "frontier on held-out seeds 31-60, after the freeze",
          u.units_mock_heldout, requires_freeze=True),
]}

# The block whose thresholds.json the evaluation blocks of a platform read.
CALIBRATION_BLOCK = {"mock": "mock_calibration", "llm": "llm_calibration"}
