"""bazt/config.py -- Frozen `BaztConfig` for the BAZT defence.

All fields default to the values validated on the tune split (9001-9015).
Never tune on main (1-30) or heldout (31-60).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass(frozen=True)
class BaztConfig:
    # ------------------------------------------------------------------ budget
    budget_mode: str = "pooled"          # "pooled" | "tiered"
    B: int = 12                          # pooled capacity (actions)
    K_hard: int = 3                      # tiered hard cap for ISOLATE / BLOCK_EDGE
    B_soft: int = 6                      # tiered soft cap for INSPECT / SANITIZE / RESTRICT
    # ------------------------------------------------------------------ costs
    cost_inspect: int = 1
    cost_sanitize: int = 1
    cost_restrict: int = 2
    cost_block: int = 2
    cost_isolate: int = 3
    # ------------------------------------------------------------------ monitor / inspection / TTL
    q_inspect: float = 0.8               # P(inspect returns a conclusive verdict)
    inspect_ttl: int = 2                 # steps an inspection verdict overrides fused p'
    ttl_restrict: int = 3                # steps a RESTRICT node stays restricted
    tau_release: float = 0.3             # restricted node reassesses and releases when p' < tau_release
    quarantine_ttl: Optional[int] = None  # optional hard TTL for quarantined (isolated) nodes
    p_strip: float = 0.9                 # mock probability trusted_sanitize strips (used by taint attenuation)
    # ------------------------------------------------------------------ provenance & fusion
    w_prov: float = 1.0                  # weight of lineage taint in the log-odds
    delta_hop: float = 0.7               # per-hop taint decay (< 1 -> monotone decay across hops)
    own_decay: float = 0.5               # per-step decay of a node's own taint
    # ------------------------------------------------------------------ optimizer & pacing
    lam: float = 1.0                     # utility cost weight (0 -> ignore utility)
    rho_san: float = 0.15                # SANITIZE utility cost coefficient
    rho_res: float = 0.35                # RESTRICT utility cost coefficient
    theta0: float = 0.15                 # base acceptance density threshold
    kappa: float = 1.5                   # pacing: theta = theta0 * exp(kappa * z())
    n_privileged: int = 3                # tool-holding / privileged nodes
    # ------------------------------------------------------------------ filtering cost toggle
    strip_degrades_weight: bool = True   # stripped messages lose consensus weight
    stripped_edge_weight: float = 0.35   # weight of a stripped (schema-only) message vs intact 1.0 / 0.6
    # ------------------------------------------------------------------ external monitor override
    probe_score_hook: Optional[Callable] = None  # Track-4 open-weight activation probe s^{probe}_{i,t}
    # ------------------------------------------------------------------ ablation booleans
    unlimited: bool = False              # infinite pooled budget
    use_pacing: bool = True              # theta rises as budget is spent
    use_topology: bool = True            # horizon-bounded reachability in H
    use_provenance: bool = True          # lineage taint in ell (w_prov -> 0 disables)
    use_utility: bool = True             # lam_eff = lam if True else 0
    binary_actions: bool = False         # only ISOLATE candidates


def cost_of(kind, cfg: "BaztConfig") -> int:
    k = kind.value if hasattr(kind, "value") else str(kind)
    return {
        "INSPECT": cfg.cost_inspect,
        "SANITIZE": cfg.cost_sanitize,
        "RESTRICT": cfg.cost_restrict,
        "BLOCK_EDGE": cfg.cost_block,
        "ISOLATE": cfg.cost_isolate,
    }[k]


def message_weight(intact: bool, noisy: bool, cfg: BaztConfig) -> float:
    """Consensus weight of a delivered message.

    Intact message: calibrated precision weight (1.0 high-precision / 0.6 noisy sensor).
    Stripped (schema-only) message: loses the rationale, drops to `stripped_edge_weight`,
    creating a genuine, measurable MAE degradation when benign nodes are over-sanitized.
    """
    if not intact:
        return cfg.stripped_edge_weight
    return 0.6 if noisy else 1.0
