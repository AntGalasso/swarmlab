"""Forward contagion reach on the ACTIVE graph with OPEN directed edges (reuses swarmlab.reach_H)."""
from __future__ import annotations

from typing import Dict, Sequence, Set, Tuple

import swarmlab as sl
from bazt.config import BaztConfig


class ContagionModel:
    """Memoised H_i and H_edge(i, j) for one decision step.

    Horizon rule: with h_t = T - t == 0 nothing can still spread, so every reach is 0 and the optimizer
    buys nothing at the last step -- also when topology awareness is ablated.
    """

    def __init__(self, sl_cfg: sl.Config, cfg: BaztConfig, out_adj: Sequence[Sequence[int]], active: Set[int],
                 p_prime: Sequence[float], h_t: int):
        self.sl_cfg, self.cfg, self.out_adj, self.active, self.p, self.h_t = sl_cfg, cfg, out_adj, active, p_prime, h_t
        self._H: Dict[int, float] = {}
        self._He: Dict[Tuple[int, int], float] = {}

    def H(self, i: int) -> float:
        if self.h_t <= 0:
            return 0.0
        if not self.cfg.use_topology:
            return 1.0
        if i not in self._H:
            self._H[i] = sl.reach_H(self.out_adj, i, self.h_t, self.sl_cfg, self.active, self.p)
        return self._H[i]

    def H_edge(self, i: int, j: int) -> float:
        """(1 - p'_j) * (1 + H_j without i): value of cutting the open directed edge i -> j."""
        if self.h_t <= 0:
            return 0.0
        key = (i, j)
        if key not in self._He:
            beyond = 0.0
            if self.cfg.use_topology:
                self.active.discard(i)               # reach from j must not route back through i
                try:
                    beyond = sl.reach_H(self.out_adj, j, self.h_t - 1, self.sl_cfg, self.active, self.p)
                finally:
                    self.active.add(i)
            self._He[key] = (1.0 - self.p[j]) * (1.0 + beyond)
        return self._He[key]
