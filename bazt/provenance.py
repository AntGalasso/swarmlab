"""Lineage tracking: message ids, parent links and taint propagation.

Taint is the gateway's belief that a node's state descends from an untrusted source through
unmediated hops. It is NOT ground truth: it is built only from who delivered what to whom.
"""
from __future__ import annotations

from typing import Callable, Dict, FrozenSet, List, Sequence, Set, Tuple

from bazt.config import BaztConfig

MsgId = Tuple[int, int]       # (step, source node)


class LineageTracker:
    def __init__(self, n: int, ingress: FrozenSet[int], recv: Sequence[Sequence[int]], cfg: BaztConfig,
                 p_strip: float):
        self.n, self.ingress, self.recv, self.cfg, self.p_strip = n, ingress, recv, cfg, p_strip
        self.enabled = bool(cfg.use_provenance and cfg.w_prov)
        self.tau: List[float] = [0.0] * n
        self.parents: Dict[MsgId, Tuple[MsgId, ...]] = {}
        self._prev_out: Dict[int, bool] = {}           # sender -> its step t-1 message was stripped
        self._step = 0

    def record_outgoing(self, outgoing: Dict[int, bool]) -> None:
        """Called after delivery of step t: who emitted a message, and was it successfully stripped."""
        self._prev_out = dict(outgoing)

    def _tau_eff(self, j: int) -> float:
        """Taint a neighbour passes on: only (1 - p_strip) of it survives a successful strip."""
        return (1.0 - self.p_strip) * self.tau[j] if self._prev_out[j] else self.tau[j]

    def advance(self, t: int, is_open: Callable[[int, int], bool], active: Set[int]) -> List[float]:
        """Compute tau_{.,t} from the deliveries of step t-1 (or the ingress prior at t = 1)."""
        if not self.enabled:
            return self.tau
        if t == 1:
            self.tau = [1.0 if i in self.ingress else 0.0 for i in range(self.n)]
            return self.tau
        new = [self.cfg.own_decay * x for x in self.tau]
        for i in active:
            parents = []
            for j in self.recv[i]:
                if j in self._prev_out and is_open(j, i):
                    parents.append((t - 1, j))
                    inc = self.cfg.delta_hop * self._tau_eff(j)
                    if inc > new[i]:
                        new[i] = inc
            if parents:
                self.parents[(t, i)] = tuple(parents)
        self.tau = new
        return new

    def clear(self, node: int) -> None:
        """A clean inspection verdict resets the node's lineage."""
        self.tau[node] = 0.0
