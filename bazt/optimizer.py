"""bazt/optimizer.py -- Deterministic two-phase `BudgetedOptimizer.plan(...)`.

Zero RNG inside the optimizer. Ties broken strictly by
`(-density, node_id, peer_id_or_-1, action_rank)`.

Phase 1: evaluate `INSPECT` candidates with `net > 0` and `density >= theta()`,
execute affordable inspections via the oracle callback, update `p'` for
inspected nodes.

Phase 2: recompute remaining action candidates (`SANITIZE`, `RESTRICT`,
`BLOCK_EDGE`, `ISOLATE`), keep the best candidate with `net > 0` per target,
sort by tie-break key, accept greedily while `ledger.can_afford(kind)` and
`density >= theta()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from bazt.actions import ActionKind, Candidate
from bazt.ledger import BudgetLedger


@dataclass
class Plan:
    isolate: List[int] = field(default_factory=list)
    sanitize: List[int] = field(default_factory=list)
    restrict: List[int] = field(default_factory=list)
    block_edges: List[Tuple[int, int]] = field(default_factory=list)
    inspect: List[int] = field(default_factory=list)
    spend_by_kind: Dict[str, int] = field(default_factory=dict)


class BudgetedOptimizer:
    def __init__(self) -> None:
        pass

    def plan(
        self,
        candidates: List[Candidate],
        ledger: BudgetLedger,
        inspect_oracle: Callable[[int], Optional[float]],
        p_prime: List[float],
    ) -> Plan:
        theta = ledger.theta()
        plan = Plan()

        # ------------------------------------------------------------------ Phase 1: inspections
        inspects = [
            c for c in candidates
            if c.kind == ActionKind.INSPECT and c.net > 0 and c.density >= theta
        ]
        inspects.sort(key=lambda c: c.tie_key())
        for c in inspects:
            if not ledger.can_afford(ActionKind.INSPECT):
                continue
            verdict = inspect_oracle(c.target)
            plan.inspect.append(c.target)
            plan.spend_by_kind[ActionKind.INSPECT.value] = (
                plan.spend_by_kind.get(ActionKind.INSPECT.value, 0) + 1
            )
            if verdict is not None:
                p_prime[c.target] = float(verdict)

        # ------------------------------------------------------------------ Phase 2: remaining actions
        remaining = [
            c for c in candidates
            if c.kind != ActionKind.INSPECT and c.net > 0 and c.density >= ledger.theta()
        ]
        # keep best candidate per target node
        best: Dict[int, Candidate] = {}
        for c in remaining:
            cur = best.get(c.target)
            if cur is None or c.tie_key() < cur.tie_key():
                best[c.target] = c
        selected = sorted(best.values(), key=lambda c: c.tie_key())

        for c in selected:
            if not ledger.can_afford(c.kind):
                continue
            if c.density < ledger.theta():
                continue
            ledger.charge(c.kind)
            plan.spend_by_kind[c.kind.value] = plan.spend_by_kind.get(c.kind.value, 0) + 1
            if c.kind == ActionKind.ISOLATE:
                plan.isolate.append(c.target)
            elif c.kind == ActionKind.SANITIZE:
                plan.sanitize.append(c.target)
            elif c.kind == ActionKind.RESTRICT:
                plan.restrict.append(c.target)
            elif c.kind == ActionKind.BLOCK_EDGE:
                plan.block_edges.append((c.target, c.peer))
        return plan
