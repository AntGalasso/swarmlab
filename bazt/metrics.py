"""Section-15 metrics: gateway cost, harm exposure, utility side effects (equations in docs/METRICS_SPEC.md §15).

Ground truth (`infected_series`) is used ONLY here, after the last decision of the episode.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

from bazt.actions import ActionKind
from bazt.audit import AuditLog
from bazt.ledger import BudgetLedger

BAZT_METRICS = ["enforcement_spend", "spend_inspect", "spend_sanitize", "spend_restrict", "spend_block",
                "spend_isolate", "harm_exposure_auc", "message_throughput", "degraded_node_step_frac",
                "false_action_rate", "recovery_time", "cost_per_prevented_compromise", "budget_violations",
                "inspections_conclusive"]


@dataclass
class EpisodeLog:
    """Raw facts the gateway records while it runs (no ground truth in here)."""
    n: int
    T: int
    e_dir: int                                           # directed send pairs of the effective graph
    delivered: Dict[int, List[Tuple[int, int, bool]]] = field(default_factory=dict)   # t -> (src, dst, stripped)
    degraded: Dict[int, Set[int]] = field(default_factory=dict)                       # t -> restricted|sanitized|isolated
    actions: List[Tuple[int, ActionKind, int]] = field(default_factory=list)          # non-inspect actions
    inspections_conclusive: int = 0


def episode_metrics(log: EpisodeLog, infected_series: Sequence[Sequence[bool]], ledger: BudgetLedger,
                    audit: AuditLog) -> dict:
    T, N = log.T, log.n
    capacity = max((T - 1) * log.e_dir, 0)               # step-T messages are never consumed
    harm = delivered = 0
    for t in range(1, T):
        for src, _dst, stripped in log.delivered.get(t, ()):
            delivered += 1
            if infected_series[t - 1][src] and not stripped:
                harm += 1
    ever = [False] * N
    ever_at: List[List[bool]] = []
    for row in infected_series:
        ever = [e or bool(x) for e, x in zip(ever, row)]
        ever_at.append(ever)
    false_actions = sum(1 for t, _k, node in log.actions if not ever_at[t - 1][node])
    recovery = float("nan")
    if log.actions:
        t_first = min(t for t, _k, _n in log.actions)
        recovery = float(T - t_first + 1)                # censored: never recovered inside the horizon
        for t in range(t_first, T + 1):
            if not any(infected_series[t - 1]):
                recovery = float(t - t_first)
                break
    sk = ledger.spent_by_kind
    return {
        "enforcement_spend": ledger.spent,
        "spend_inspect": sk[ActionKind.INSPECT], "spend_sanitize": sk[ActionKind.SANITIZE],
        "spend_restrict": sk[ActionKind.RESTRICT], "spend_block": sk[ActionKind.BLOCK_EDGE],
        "spend_isolate": sk[ActionKind.ISOLATE],
        "harm_exposure_auc": harm / capacity if capacity else 0.0,
        "message_throughput": delivered / capacity if capacity else 0.0,
        "degraded_node_step_frac": sum(len(s) for s in log.degraded.values()) / (N * T) if N * T else 0.0,
        "false_action_rate": false_actions / len(log.actions) if log.actions else 0.0,
        "recovery_time": recovery,
        "cost_per_prevented_compromise": float("nan"),   # needs the paired no_defense run: see add_paired_metrics
        "budget_violations": audit.overspent_rows(),
        "inspections_conclusive": log.inspections_conclusive,
    }


def legacy_spend_columns(row: dict, cost_isolate: int, cost_sanitize: int) -> None:
    """Price a legacy arm in the same cost units, so every arm has an `enforcement_spend`."""
    iso = row.get("quarantines_used", 0) * cost_isolate
    san = row.get("sanitizations_used", 0) * cost_sanitize
    row.update(enforcement_spend=float(iso + san), spend_isolate=float(iso), spend_sanitize=float(san),
               spend_inspect=0.0, spend_restrict=0.0, spend_block=0.0, budget_violations=0)
    for key in BAZT_METRICS:                      # gateway-only metrics are not applicable to a legacy arm
        row.setdefault(key, float("nan"))


def add_paired_metrics(rows: List[dict], keys: Tuple[str, ...] = ("topology", "regime", "seed")) -> None:
    """cost_per_prevented_compromise = spend / (nodes spared vs no_defense on the SAME world, by CRN).

    NaN when the arm prevented nothing (an undefined ratio is not reported as 0 or inf). Mirrors how
    `swarmlab.add_deltas` pairs rows against the no_defense baseline.
    """
    base = {tuple(r[k] for k in keys): r for r in rows if r["policy"] == "no_defense"}
    for r in rows:
        b = base.get(tuple(r[k] for k in keys))
        r["cost_per_prevented_compromise"] = float("nan")
        if b is None:
            continue
        prevented = (b["final_ever_infected_rate"] - r["final_ever_infected_rate"]) * r["N"]
        if prevented > 1e-9 and not math.isnan(r.get("enforcement_spend", float("nan"))):
            r["cost_per_prevented_compromise"] = r["enforcement_spend"] / prevented
