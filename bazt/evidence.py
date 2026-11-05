"""Log-odds evidence fusion and inspection verdicts."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import swarmlab as sl
from bazt.config import BaztConfig

P_POSITIVE = 0.98      # verdict: the node really is emitting the attack
P_NEGATIVE = 0.02      # verdict: the node is clean


def fused_probability(sl_cfg: sl.Config, cfg: BaztConfig, roll: float, tau: float) -> float:
    """sigma(a*roll - b + w_prov*tau).

    With provenance off (or w_prov = 0) the expression is exactly the monitor's own
    `sigma(a*roll - b)`, bit for bit, so the ablation degenerates to the plain monitor.
    """
    z = sl_cfg.a * roll - sl_cfg.b
    if cfg.use_provenance and cfg.w_prov:
        z += cfg.w_prov * tau
    return sl.sigmoid(z)


class InspectionState:
    """Verdicts of paid, conclusive inspections. An inconclusive review leaves no trace here."""

    def __init__(self, ttl: int):
        self.ttl = ttl
        self._v: Dict[int, Tuple[float, int]] = {}     # node -> (p', last valid step)

    def record(self, node: int, t: int, infected: bool) -> float:
        p = P_POSITIVE if infected else P_NEGATIVE
        self._v[node] = (p, t + self.ttl - 1)
        return p

    def override(self, node: int, t: int) -> Optional[float]:
        v = self._v.get(node)
        return v[0] if v is not None and t <= v[1] else None

    def clean_active(self, node: int, t: int) -> bool:
        return self.override(node, t) == P_NEGATIVE

    def active_nodes(self, t: int) -> frozenset:
        return frozenset(n for n, (_, last) in self._v.items() if t <= last)
