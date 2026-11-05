"""bazt/ledger.py -- `BudgetLedger`: enforcement budget, never silently clips.

Supports `"pooled"` (`capacity = B`, or `inf` if `unlimited=True`) and
`"tiered"` (`K_hard` for `ISOLATE`/`BLOCK_EDGE`, `B_soft` for
`INSPECT`/`SANITIZE`/`RESTRICT`).

`charge(kind)` raises `BudgetViolation` if unaffordable -- never silently clips.

    z()   = spent / capacity  (0.0 if unlimited or capacity == 0)
    theta() = theta0 * exp(kappa * z()) if use_pacing else theta0
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bazt.config import BaztConfig, cost_of


class BudgetViolation(Exception):
    """Raised when an action would exceed the enforcement budget."""


_HARD_KINDS = {"ISOLATE", "BLOCK_EDGE"}
_SOFT_KINDS = {"INSPECT", "SANITIZE", "RESTRICT"}


@dataclass
class BudgetLedger:
    cfg: BaztConfig
    # pooled
    capacity: float = 0.0
    spent: float = 0.0
    # tiered
    hard_spent: float = 0.0
    soft_spent: float = 0.0
    per_kind: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        cfg = self.cfg
        if cfg.budget_mode == "tiered":
            self.capacity = float(cfg.K_hard) + float(cfg.B_soft)
        else:
            self.capacity = float("inf") if cfg.unlimited else float(cfg.B)
        self.per_kind = {k: 0 for k in (_HARD_KINDS | _SOFT_KINDS)}

    # ------------------------------------------------------------------ queries
    def can_afford(self, kind: str) -> bool:
        cost = cost_of(kind, self.cfg)
        if self.cfg.budget_mode == "tiered":
            cap = float(self.cfg.K_hard) if kind in _HARD_KINDS else float(self.cfg.B_soft)
            spent = self.hard_spent if kind in _HARD_KINDS else self.soft_spent
            return spent + cost <= cap + 1e-9
        return self.spent + cost <= self.capacity + 1e-9

    def z(self) -> float:
        if self.cfg.unlimited or self.capacity == 0:
            return 0.0
        return self.spent / self.capacity if self.cfg.budget_mode != "tiered" else (
            (self.hard_spent + self.soft_spent) / self.capacity
        )

    def theta(self) -> float:
        if not self.cfg.use_pacing:
            return self.cfg.theta0
        return self.cfg.theta0 * __import__("math").exp(self.cfg.kappa * self.z())

    # ------------------------------------------------------------------ mutation
    def charge(self, kind: str) -> None:
        if not self.can_afford(kind):
            raise BudgetViolation(
                f"cannot afford {kind} (cost={cost_of(kind, self.cfg)}, "
                f"spent={self.spent}, capacity={self.capacity})"
            )
        cost = cost_of(kind, self.cfg)
        k = kind.value if hasattr(kind, "value") else str(kind)
        self.per_kind[k] += cost
        if self.cfg.budget_mode == "tiered":
            if kind in _HARD_KINDS:
                self.hard_spent += cost
            else:
                self.soft_spent += cost
        else:
            self.spent += cost
