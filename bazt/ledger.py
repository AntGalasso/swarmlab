"""Budget accounting. Overspending is an error, never a silent clip.

Modes
  pooled    one pool of ``B`` cost units (legacy).
  tiered    hard pool (ISOLATE / BLOCK_EDGE) + soft pool (the rest), legacy.
  dual_pct  ``(audit_pct, intervention_pct)`` as FRACTIONS of the total episode edge deliveries,
            ``total_episode_deliveries = (T - 1) * E_dir``. ``None`` = uncapped. The audit pool pays
            INSPECT; the intervention pool pays every delivery-changing action (SANITIZE, RESTRICT,
            BLOCK_EDGE, ISOLATE). This is the ledger shared by ``arm_B``, ``arm_C`` and ``bazt`` so all
            three are compared under IDENTICAL caps.

Exhaustion
  ``BudgetViolation`` is still raised by ``charge`` when an action cannot be afforded. On top of that the
  ledger records a ``budget_exhausted(arm, pool, step)`` event once per pool, which the gateways turn into
  the mandated fallback: from that step on the defender obeys only the ``arm_A`` identity/allowlist rules.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from bazt.actions import ActionKind, HARD_KINDS, cost_of
from bazt.config import BaztConfig


class BudgetViolation(RuntimeError):
    """Raised when an action is charged that the budget cannot afford."""


@dataclass(frozen=True)
class Charge:
    kind: ActionKind
    cost: int
    pool: str
    spent: float                 # pool spend AFTER this charge
    remaining: float             # pool capacity - spent (inf when unlimited)


_DUAL_POOLS = ("audit", "intervention")


class BudgetLedger:
    """pooled: one pool of B cost units. tiered: hard pool (ISOLATE/BLOCK_EDGE) + soft pool (the rest).
    dual_pct: audit pool (INSPECT) + intervention pool (every delivery-changing action).

    All pools are denominated in cost units so spend is comparable across budget modes.
    """

    def __init__(self, cfg: BaztConfig, total_deliveries: int = 0):
        self.cfg = cfg
        self.total_deliveries = max(0, int(total_deliveries))
        inf = float("inf")
        if cfg.budget_mode == "pooled":
            self._cap: Dict[str, float] = {"pool": inf if cfg.unlimited else float(cfg.B)}
        elif cfg.budget_mode == "tiered":
            self._cap = {"hard": inf if cfg.unlimited else float(cfg.K_hard * cfg.cost_isolate),
                         "soft": inf if cfg.unlimited else float(cfg.B_soft * cfg.cost_sanitize)}
        else:                                                   # dual_pct
            tot = float(max(1, self.total_deliveries))
            self._cap = {
                "audit": inf if (cfg.unlimited or cfg.audit_pct is None) else float(cfg.audit_pct) * tot,
                "intervention": inf if (cfg.unlimited or cfg.intervention_pct is None)
                else float(cfg.intervention_pct) * tot,
            }
        self._spent: Dict[str, float] = {k: 0.0 for k in self._cap}
        self.spent_by_kind: Dict[ActionKind, float] = {k: 0.0 for k in ActionKind}
        self.exhausted_events: List[dict] = []                  # [{"pool", "step", "cap"}]

    # ---------------------------------------------------------------------------- queries
    def pool_of(self, kind: ActionKind) -> str:
        if self.cfg.budget_mode == "pooled":
            return "pool"
        if self.cfg.budget_mode == "tiered":
            return "hard" if kind in HARD_KINDS else "soft"
        return "audit" if kind == ActionKind.INSPECT else "intervention"

    def capacity(self, pool: str) -> float:
        return self._cap[pool]

    @property
    def pools(self) -> tuple:
        return tuple(self._cap)

    @property
    def spent(self) -> float:
        return sum(self._spent.values())

    @property
    def capacity_total(self) -> float:
        return sum(self._cap.values())

    def remaining(self, pool: str) -> float:
        return self._cap[pool] - self._spent[pool]

    def can_afford(self, kind: ActionKind) -> bool:
        pool = self.pool_of(kind)
        return self._spent[pool] + cost_of(self.cfg, kind) <= self._cap[pool]

    def exhausted_pools(self) -> List[str]:
        """Pools with no budget left at all (a strict test, so a partially drained pool is not 'exhausted')."""
        return [p for p in self._cap if not math.isinf(self._cap[p]) and self.remaining(p) <= 0.0]

    def mark_exhausted(self, step: int) -> List[dict]:
        """Record one ``budget_exhausted`` event per newly drained pool. Idempotent per pool."""
        seen = {e["pool"] for e in self.exhausted_events}
        new = []
        for p in self.exhausted_pools():
            if p in seen:
                continue
            ev = {"pool": p, "step": int(step), "cap": float(self._cap[p])}
            self.exhausted_events.append(ev)
            new.append(ev)
        return new

    @property
    def any_exhausted(self) -> bool:
        return bool(self.exhausted_events)

    def z(self) -> float:
        """Budget pressure in [0, 1]: fraction of the total capacity already spent."""
        cap = self.capacity_total
        if self.cfg.unlimited or cap == 0 or math.isinf(cap):
            return 0.0
        return self.spent / cap

    def theta(self) -> float:
        """Density threshold: flat theta0, or pacing theta0 * exp(kappa * z) as the budget drains."""
        if not self.cfg.use_pacing:
            return self.cfg.theta0
        return self.cfg.theta0 * math.exp(self.cfg.kappa * self.z())

    # ---------------------------------------------------------------------------- mutation
    def charge(self, kind: ActionKind) -> Charge:
        pool, cost = self.pool_of(kind), cost_of(self.cfg, kind)
        if self._spent[pool] + cost > self._cap[pool]:
            raise BudgetViolation(f"{kind.name} costs {cost} but pool {pool!r} has only "
                                  f"{self.remaining(pool):g} of {self._cap[pool]:g} left")
        self._spent[pool] += cost
        self.spent_by_kind[kind] += cost
        return Charge(kind, cost, pool, self._spent[pool], self.remaining(pool))

    def try_charge(self, kind: ActionKind) -> Optional[Charge]:
        """Charge if affordable, else return None (never raises). Used by the dual_pct gateways."""
        if not self.can_afford(kind):
            return None
        return self.charge(kind)

    # ---------------------------------------------------------------------------- reporting
    def spend_by_pool(self) -> Dict[str, float]:
        return dict(self._spent)

    def audit_spend(self) -> float:
        return self._spent.get("audit", self.spent_by_kind[ActionKind.INSPECT])

    def intervention_spend(self) -> float:
        if "intervention" in self._spent:
            return self._spent["intervention"]
        return sum(v for k, v in self.spent_by_kind.items() if k != ActionKind.INSPECT)
