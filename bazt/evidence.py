"""bazt/evidence.py -- Log-odds fusion + InspectionState.

`ell_{i,t} = a * roll_{i,t} - b + (cfg.w_prov if cfg.use_provenance else 0.0) * tau_{i,t}`

Matching `swarmlab`'s monitor calibration (`a`, `b`) so that when
`w_prov = 0` (or `use_provenance=False`), `sigma(ell_{i,t}) == p_monitor[i]`
bit-for-bit.

`InspectionState`: a conclusive inspect verdict overrides fused `p'` for
`inspect_ttl` steps (`0.98` if truly emitting, `0.02` if clean); an inconclusive
`q_inspect` draw retains the fused `sigma(ell)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import swarmlab as sl
from bazt.config import BaztConfig


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-max(-60.0, min(60.0, z))))


def fused_ell(roll: float, tau: float, w_prov: float, a: float = 2.5, b: float = 2.0) -> float:
    return a * roll - b + w_prov * tau


def fused_p(roll: float, tau: float, cfg: BaztConfig, a: float = 2.5, b: float = 2.0) -> float:
    w_prov = (cfg.w_prov if cfg.use_provenance else 0.0)
    return float(sigmoid(fused_ell(roll, tau, w_prov, a, b)))


@dataclass
class InspectionState:
    cfg: BaztConfig
    N: int
    topology: str
    seed: int
    T: int
    # pre-drawn (T, N) inconclusive-review draws; never touches swarmlab's CRN streams
    u_inspect: np.ndarray = field(init=False, repr=False)
    verdict: List[Optional[float]] = field(default_factory=list)
    valid_until: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.u_inspect = sl.stream_rng("bazt-inspect", self.topology, self.seed).random(
            (self.T, self.N)
        )
        self.verdict = [None] * self.N
        self.valid_until = [0] * self.N

    def inspect(self, i: int, t: int, is_emitting: bool) -> Optional[float]:
        """Spend one inspection budget on node i at step t.

        Returns the conclusive verdict `p'` (0.98 emitting / 0.02 clean), or None
        when the `q_inspect` draw fails (inconclusive -> retain fused `p'`).
        """
        if not (0 <= i < self.N and 1 <= t <= self.T):
            return None
        if float(self.u_inspect[t - 1, i]) < self.cfg.q_inspect:
            vp = 0.98 if is_emitting else 0.02
            self.verdict[i] = vp
            self.valid_until[i] = t + self.cfg.inspect_ttl - 1
            return vp
        return None

    def active(self, i: int, t: int) -> Optional[float]:
        """Active inspection verdict overriding fused `p'`, or None."""
        if self.verdict[i] is not None and self.valid_until[i] >= t:
            return float(self.verdict[i])
        return None
