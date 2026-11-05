"""bazt/audit.py -- `AuditLog`: one row per enforcement action with full §15 context.

Fields: `t, action, target, cost, pool, spent, remaining, p_prime, s_mon, taint,
H, net, density, theta, reason`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class AuditEntry:
    t: int
    action: str
    target: int
    cost: int
    pool: float
    spent: float
    remaining: float
    p_prime: float
    s_mon: float
    taint: float
    H: float
    net: float
    density: float
    theta: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "t": self.t, "action": self.action, "target": self.target,
            "cost": self.cost, "pool": self.pool, "spent": self.spent,
            "remaining": self.remaining, "p_prime": self.p_prime, "s_mon": self.s_mon,
            "taint": self.taint, "H": self.H, "net": self.net, "density": self.density,
            "theta": self.theta, "reason": self.reason,
        }


class AuditLog:
    def __init__(self) -> None:
        self.entries: List[AuditEntry] = []

    def record(self, t, action, target, cost, pool, spent, remaining,
               p_prime, s_mon, taint, H, net, density, theta, reason) -> None:
        self.entries.append(AuditEntry(
            t=t, action=action, target=target, cost=cost, pool=pool, spent=spent,
            remaining=remaining, p_prime=p_prime, s_mon=s_mon, taint=taint, H=H,
            net=net, density=density, theta=theta, reason=reason,
        ))

    def rows(self) -> List[dict]:
        return [e.to_dict() for e in self.entries]
