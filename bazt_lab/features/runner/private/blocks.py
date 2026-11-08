"""The registry of blocks (docs/EXPERIMENT_PLAN.md): which units each runs, on which platform, and how."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping

from bazt_lab.core import public as core
from bazt_lab.features.runner.private import units as u
from bazt_lab.features.runner.private.wiring import FeatureMissing, Wiring


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
    open_decision: str = ""             # what is still to decide before the block's platform can run its units, said
                                        # with the reason it cannot (`unrunnable`)
    replays: str = ""                   # the block whose results this block's units replay (private/replay.py): run-all
                                        # runs it first, and a launch is refused until its results cover the units
    timing_source: str = ""             # D116: the block whose BAZT rows give the matched-timing controls their rounds
                                        # (private/matched.py): run-all runs it first; only such a block runs *_matched


BLOCKS: Dict[str, Block] = {b.name: b for b in [
    Block("llm_calibration", "llm", "calibrate", "attack-free calibration of B, C, BAZT (alpha deny rate)",
          u.units_llm_calibration),
    Block("gnn_train", "llm", "train", "G-Safeguard GNN training, upstream setup", u.units_gnn_train),
    Block("llm_headline", "llm", "evaluate", "uncapped A, B, C, BAZT, GNN", u.units_llm_headline),
    Block("llm_capped", "llm", "evaluate", "capped C and BAZT at the profile's budget cells", u.units_llm_capped),
    Block("mock_calibration", "mock", "calibrate", "attack-free calibration on the tune split",
          u.units_mock_calibration),
    Block("mock_frontier", "mock", "evaluate", "full 7x7 budget frontier, every sink", u.units_mock_frontier),
    Block("mock_epidemiology", "mock", "evaluate", "H-topology at 12 and 100 agents",
          u.scoped("epidemiology", topologies_key="mock.epidemiology.topologies",
                   agents_key="mock.epidemiology_agents")),
    Block("mock_ablations", "mock", "evaluate", "BAZT minus one component", u.scoped("ablations", reference_key="analysis.ablation_reference")),
    Block("mock_adaptive", "mock", "evaluate", "adaptive attacker conditions",
          u.scoped("adaptive", variant_key="mock.adaptive.conditions", variant_field="condition"), adaptive=True),
    # after mock_adaptive: the check replays its gray-box best responses (D110)
    Block("llm_adaptive_check", "llm", "evaluate", "held-out LLM adaptive check, the mock's gray-box best response replayed",
          u.units_llm_adaptive_check, replays="mock_adaptive",
          open_decision="D110: the held-out LLM adaptive check replays the mock's gray-box best response only"),
    Block("mock_presets_robustness", "mock", "evaluate", "robustness auditor presets",
          u.scoped("presets_robustness", variants=u.preset_variants)),
    Block("mock_costs_robustness", "mock", "evaluate", "CAGE-4 cost weights",
          u.scoped("costs_robustness", fixed={"cost_condition": "robustness"})),
    # after mock_frontier: the matched-timing controls act at the rounds BAZT intervened in the same unit (D116)
    Block("mock_legacy", "mock", "evaluate", "existing swarmlab arms with matched-timing controls",
          u.scoped("legacy"), timing_source="mock_frontier"),
    Block("mock_heldout", "mock", "evaluate", "frontier on held-out seeds 31-60, after the freeze",
          u.units_mock_heldout, requires_freeze=True),
]}

# The block whose thresholds.json the evaluation blocks of a platform read.
CALIBRATION_BLOCK = {"mock": "mock_calibration", "llm": "llm_calibration"}


def unrunnable(block: Block, s: Mapping, wiring: Wiring) -> str | None:
    """Why the block's platform cannot run its units at all (an adaptive condition it does not run), or None. What a
    platform runs is its `Platform.adaptive_conditions`, asked through the wiring, and only when a unit carries a
    condition (the wiring builds the platform on that first ask: it imports its feature, nothing else). A grid the
    settings cannot build yet, or a platform whose feature is not built, is None too: the launch refusals name what is
    missing."""
    try:
        carried = block.units(s, False).conditions()
    except (core.Unset, ValueError, SystemExit):
        return None
    if not carried:
        return None
    try:
        runs = tuple(wiring.platform(block.platform).adaptive_conditions)
    except FeatureMissing:
        return None
    refused = [c for c in carried if c not in runs]
    if not refused:
        return None
    platform = block.platform.upper()
    what = (f"the {platform} platform has no adaptive condition yet" if not runs else
            f"the {platform} platform runs the adaptive conditions {', '.join(map(repr, runs))} only")
    reason = f"{what} (its units carry condition {', '.join(map(repr, refused))})"
    return f"{reason}; {block.open_decision}" if block.open_decision else reason
