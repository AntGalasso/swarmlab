"""bazt/metrics.py -- §15 BAZT enforcement metrics.

Pure functions over the audit log, ledger, and episode series. Ground-truth
evaluation metrics (e.g. `harm_exposure_auc`, `false_action_rate`,
`cost_per_prevented_compromise`) are computed here in `extra_metrics` at
`t == T`, never inside `decide()`.
"""

from __future__ import annotations

from typing import Dict, List

import swarmlab as sl


def enforcement_spend(rows: List[dict]) -> float:
    return float(sum(r["cost"] for r in rows))


def spend_by_kind(rows: List[dict], kind: str) -> float:
    return float(sum(r["cost"] for r in rows if r["action"] == kind))


def harm_exposure_auc(infected_series: List[int], N: int) -> float:
    """AUC-style ever-infected exposure = mean_t(c_t) / N (same as swarmlab §1)."""
    if not infected_series or N == 0:
        return 0.0
    return float(sum(infected_series) / len(infected_series)) / N


def message_throughput(rows: List[dict], delivered: int, N: int, T: int) -> float:
    denom = max(1, N * T)
    return delivered / denom


def degraded_node_step_frac(rows: List[dict], N: int, T: int) -> float:
    """Fraction of node-steps on which a node was restricted or sanitized (degraded channel)."""
    degraded = set()
    for r in rows:
        if r["action"] in ("RESTRICT", "SANITIZE"):
            degraded.add((r["t"], r["target"]))
    return len(degraded) / max(1, N * T)


def false_action_rate(rows: List[dict], healthy: set) -> float:
    """Enforcement actions spent on nodes that never carried the payload (healthy)."""
    total = len(rows)
    if total == 0:
        return 0.0
    wasted = sum(1 for r in rows if r["target"] in healthy)
    return wasted / total


def recovery_time(rows: List[dict], T: int) -> int:
    """First step at which no further enforcement action was needed (T+1 if never)."""
    if not rows:
        return T + 1
    return int(max(r["t"] for r in rows)) if rows else T + 1


def cost_per_prevented_compromise(spend: float, prevented: int) -> float:
    if prevented <= 0:
        return float("inf") if spend > 0 else 0.0
    return spend / prevented


def budget_violations(rows: List[dict]) -> int:
    return sum(1 for r in rows if r["reason"] == "BUDGET_VIOLATION")


def compute_all(
    rows: List[dict],
    infected_series: List[int],
    N: int,
    T: int,
    delivered: int,
    healthy: set,
    spend: float,
    prevented: int,
) -> dict:
    return {
        "enforcement_spend": enforcement_spend(rows),
        "spend_inspect": spend_by_kind(rows, "INSPECT"),
        "spend_sanitize": spend_by_kind(rows, "SANITIZE"),
        "spend_restrict": spend_by_kind(rows, "RESTRICT"),
        "spend_block": spend_by_kind(rows, "BLOCK_EDGE"),
        "spend_isolate": spend_by_kind(rows, "ISOLATE"),
        "harm_exposure_auc": harm_exposure_auc(infected_series, N),
        "message_throughput": message_throughput(rows, delivered, N, T),
        "degraded_node_step_frac": degraded_node_step_frac(rows, N, T),
        "false_action_rate": false_action_rate(rows, healthy),
        "recovery_time": float(recovery_time(rows, T)),
        "cost_per_prevented_compromise": cost_per_prevented_compromise(spend, prevented),
        "budget_violations": float(budget_violations(rows)),
    }
