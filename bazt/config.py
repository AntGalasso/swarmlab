"""Frozen configuration of the BAZT gateway (one object = one fully specified experiment arm)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

BUDGET_MODES = ("pooled", "tiered", "dual_pct")


@dataclass(frozen=True)
class BaztConfig:
    # --- budget ---------------------------------------------------------------------------------
    budget_mode: str = "pooled"            # "pooled": one pool of B cost units | "tiered": hard + soft pools
    B: int = 12                            # pooled capacity, in cost units
    K_hard: int = 3                        # tiered: hard pool = K_hard isolate-equivalents (K_hard*cost_isolate)
    B_soft: int = 6                        # tiered: soft pool = B_soft sanitize-equivalents (B_soft*cost_sanitize)
    cost_inspect: int = 1
    cost_sanitize: int = 1
    cost_restrict: int = 2
    cost_block: int = 2
    cost_isolate: int = 3
    # dual_pct: both caps are FRACTIONS of the total episode edge deliveries ((T-1) * E_dir).
    # None = uncapped. audit pays INSPECT; intervention pays every delivery-changing action.
    audit_pct: Optional[float] = None
    intervention_pct: Optional[float] = None
    # --- inspection / state lifetimes -----------------------------------------------------------
    q_inspect: float = 0.8                 # probability a paid inspection is conclusive
    inspect_ttl: int = 2                   # a verdict stays valid for this many steps (including its own)
    ttl_restrict: int = 3                  # a restriction lasts this many steps
    tau_release: float = 0.3               # restricted node is released early once p' drops below this
    quarantine_ttl: Optional[int] = None   # None = isolation is permanent (legacy semantics)
    # --- provenance & evidence fusion -----------------------------------------------------------
    w_prov: float = 1.0                    # weight of lineage taint in the log-odds fusion
    delta_hop: float = 0.7                 # taint attenuation per hop
    own_decay: float = 0.8                 # per-step decay of a node's own taint when nothing refreshes it
    n_ingress: int = 2                     # designated external-facing nodes (patient-zero mode only)
    # --- optimizer & pacing ---------------------------------------------------------------------
    lam: float = 1.0                       # utility-cost weight (same meaning as swarmlab.Config.lam)
    rho_san: float = 0.15                  # utility cost of one sanitize, as a fraction of centrality C_i
    rho_res: float = 0.35                  # utility cost of one restriction, as a fraction of C_i
    theta0: float = 0.15                   # base density threshold
    kappa: float = 1.5                     # pacing steepness: theta = theta0 * exp(kappa * spent/capacity)
    n_privileged: int = 3                  # tool-holding nodes
    priv_bonus: float = 0.5                # extra benefit of restricting a privileged node (denies its tools)
    # --- non-trivial filtering cost (see bazt/channel.py) -----------------------------------------
    strip_degrades_weight: bool = True
    stripped_edge_weight: float = 0.35     # consensus weight scale of a schema-only (rationale-free) message
    precision_hp: float = 1.0              # weight scale of an intact message from a high-precision sensor
    precision_lp: float = 0.6              # ... from a noisy sensor
    noisy_frac: float = 0.4                # fraction of nodes with a noisy sensor
    # --- optional external monitor (open-weight activation probe, Track 4) ----------------------
    probe_score_hook: Optional[Callable] = None   # f(node, t, StepView) -> Optional[float], rolling-score scale
    # --- ablations ------------------------------------------------------------------------------
    unlimited: bool = False
    use_pacing: bool = True
    use_topology: bool = True
    use_provenance: bool = True
    use_utility: bool = True
    binary_actions: bool = False

    def __post_init__(self):
        if self.budget_mode not in BUDGET_MODES:
            raise ValueError(f"budget_mode must be one of {BUDGET_MODES}, got {self.budget_mode!r}")
        for name in ("cost_inspect", "cost_sanitize", "cost_restrict", "cost_block", "cost_isolate"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("B", "K_hard", "B_soft"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if not 0.0 <= self.q_inspect <= 1.0:
            raise ValueError("q_inspect must be in [0, 1]")
        if self.inspect_ttl < 1 or self.ttl_restrict < 1:
            raise ValueError("inspect_ttl and ttl_restrict must be >= 1")
        if self.quarantine_ttl is not None and self.quarantine_ttl < 1:
            raise ValueError("quarantine_ttl must be None or >= 1")
        if not 0.0 < self.stripped_edge_weight <= 1.0:
            raise ValueError("stripped_edge_weight must be in (0, 1]")
        for name in ("audit_pct", "intervention_pct"):
            v = getattr(self, name)
            if v is not None and v < 0.0:
                raise ValueError(f"{name} must be None (uncapped) or >= 0")

    @property
    def total_capacity(self) -> float:
        """Total budget in cost units (inf when unlimited).

        dual_pct caps are fractions of an episode-dependent denominator, so they cannot be expressed here;
        the ledger resolves them from the effective graph. `inf` keeps cost-per-arm comparisons honest.
        """
        if self.unlimited:
            return float("inf")
        if self.budget_mode == "pooled":
            return float(self.B)
        if self.budget_mode == "dual_pct":
            return float("inf")
        return float(self.K_hard * self.cost_isolate + self.B_soft * self.cost_sanitize)
