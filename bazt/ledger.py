"""Budget accounting. Overspending is an error, never a silent clip."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

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


class BudgetLedger:
    """pooled: one pool of B cost units. tiered: hard pool (ISOLATE/BLOCK_EDGE) + soft pool (the rest).

    All pools are denominated in cost units so spend is comparable across budget modes.
    """

    def __init__(self, cfg: BaztConfig):
        self.cfg = cfg
        inf = float("inf")
        if cfg.budget_mode == "pooled":
            self._cap: Dict[str, float] = {"pool": inf if cfg.unlimited else float(cfg.B)}
        else:
            self._cap = {"hard": inf if cfg.unlimited else float(cfg.K_hard * cfg.cost_isolate),
                         "soft": inf if cfg.unlimited else float(cfg.B_soft * cfg.cost_sanitize)}
        self._spent: Dict[str, float] = {k: 0.0 for k in self._cap}
        self.spent_by_kind: Dict[ActionKind, float] = {k: 0.0 for k in ActionKind}

    # ---------------------------------------------------------------------------- queries
    def pool_of(self, kind: ActionKind) -> str:
        if self.cfg.budget_mode == "pooled":
            return "pool"
        return "hard" if kind in HARD_KINDS else "soft"

    def capacity(self, pool: str) -> float:
        return self._cap[pool]

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
