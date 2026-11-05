"""bazt/arms.py -- Arm registry for the BAZT runner.

Registers every BAZT arm (ablation flags on a shared `BaztConfig`), `static_zta`
(reusing `env_ladder.HardConstraintSpec` movement/channel policy), and
cost-matched wrappers for legacy arms (`no_defense`, `score_only`,
`centrality_only`, `risk_aware`, `risk_aware_tiered`).  With `--equalize-budget`,
pooled `B` is mapped to `K = B // cost_isolate` for hard-only baselines so
budget comparisons are 100% cost-equivalent.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

from bazt.config import BaztConfig
import env_ladder as el


BAZT_ARMS = ("bazt", "bazt_no_prov", "bazt_no_topo", "bazt_no_pacing",
             "bazt_no_util", "bazt_binary", "bazt_unlimited")
STATIC_ZTA = "static_zta"
COST_MATCHED_LEGACY = ("no_defense", "score_only", "centrality_only", "risk_aware", "risk_aware_tiered")
ALL_ARMS = list(BAZT_ARMS) + [STATIC_ZTA] + list(COST_MATCHED_LEGACY)


def bazt_config(policy: str, equalize_budget: bool = False, B: int = 12) -> BaztConfig:
    """`BaztConfig` for a BAZT arm; ablations switch one mechanism off."""
    base = BaztConfig(B=B)
    if policy == "bazt":
        return base
    if policy == "bazt_no_prov":
        return replace(base, use_provenance=False)
    if policy == "bazt_no_topo":
        return replace(base, use_topology=False)
    if policy == "bazt_no_pacing":
        return replace(base, use_pacing=False)
    if policy == "bazt_no_util":
        return replace(base, use_utility=False)
    if policy == "bazt_binary":
        return replace(base, binary_actions=True)
    if policy == "bazt_unlimited":
        return replace(base, unlimited=True)
    raise ValueError(f"unknown BAZT arm: {policy}")


def static_zta_spec() -> el.HardConstraintSpec:
    """Static zero-trust architecture: directed-DAG movement + schema-only channel."""
    return el.HardConstraintSpec(movement_direction_mode="directed_dag",
                                 action_channel_mode="structured_kv_only")


def equalized_K(bazt_cfg: BaztConfig) -> int:
    """Hard-only baseline budget `K = B // cost_isolate` for cost-equivalent comparison."""
    return max(1, int(bazt_cfg.B) // max(1, bazt_cfg.cost_isolate))


def is_bazt_arm(policy: str) -> bool:
    return policy in BAZT_ARMS
