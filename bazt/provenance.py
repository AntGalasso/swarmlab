"""bazt/provenance.py -- LineageTracker: message IDs, parent links, taint dynamics.

Taint reflects *unmediated untrusted flow*:

  * At `t = 1`, ingress nodes that ingest an external untrusted source receive
    `tau_{i,1} = 1.0`.
  * For `t > 1`, incoming taint along an open delivered edge `j -> i` is
    `delta_hop * tau_eff(j, t-1)`, where
    `tau_eff(j, t-1) = (1 - p_strip) * tau_{j,t-1}` if `j`'s outgoing message at
    `t-1` was successfully stripped (sanitize / restricted), else `tau_{j,t-1}`.
  * `tau_{i,t} = max(own_decay * tau_{i,t-1}, max_{j -> i} incoming_taint)`.
  * If `inspect(i)` returned a `clean` verdict active at `t`, `tau_{i,t}` is
    cleared to `0.0`.
  * Taint decays monotonically across hops (`delta_hop = 0.7`) and drops to `0`
    when `use_provenance=False` (equivalently `w_prov = 0`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set, Tuple

import swarmlab as sl
from bazt.config import BaztConfig


@dataclass
class LineageTracker:
    cfg: BaztConfig
    N: int
    ingress_nodes: Set[int] = field(default_factory=set)
    tau: List[float] = field(default_factory=list)
    # (t, node) -> whether that node's outgoing message at step t was stripped
    stripped_at: dict = field(default_factory=dict)
    # node -> step until which an inspect-clean verdict forces tau to 0
    clean_until: dict = field(default_factory=dict)
    t: int = 0

    def __post_init__(self) -> None:
        if not self.tau:
            self.tau = [0.0] * self.N

    # ------------------------------------------------------------------ seeding
    def seed_ingress(self, t: int, is_seed_flags: List[bool]) -> None:
        """At t=1, ingress nodes that ingest an external untrusted source get tau=1.0."""
        if t != 1 or not self.cfg.use_provenance:
            return
        for i in self.ingress_nodes:
            if 0 <= i < self.N and is_seed_flags[i]:
                self.tau[i] = 1.0

    # ------------------------------------------------------------------ step advance
    def begin_step(self, t: int) -> None:
        """Decay each node's own taint at the start of step t (t>1)."""
        self.t = t
        if not self.cfg.use_provenance:
            self.tau = [0.0] * self.N
            return
        if t > 1:
            d = self.cfg.own_decay
            self.tau = [d * v for v in self.tau]

    # ------------------------------------------------------------------ delivery observation
    def mark_stripped(self, t: int, node: int, stripped: bool) -> None:
        self.stripped_at[(t, node)] = bool(stripped)

    def on_delivery(self, t: int, edges: List[Tuple[int, int, bool]]) -> None:
        """Propagate taint across the delivered edges of step t.

        `edges` = (src, dst, sanitized) for each delivered message src->dst.
        Resulting tau is the taint used by decisions at step t+1.
        """
        self.t = t
        if not self.cfg.use_provenance:
            self.tau = [0.0] * self.N
            return
        new_tau = [self.cfg.own_decay * v for v in self.tau]
        for src, dst, sanitized in edges:
            if not (0 <= dst < self.N and 0 <= src < self.N):
                continue
            base = self.tau[src]
            if sanitized:
                base = (1.0 - self.cfg.p_strip) * base  # successful strip removes most taint
            incoming = self.cfg.delta_hop * base
            if incoming > new_tau[dst]:
                new_tau[dst] = incoming
        # inspect-clear: clean verdict active at t forces tau to 0
        for i in range(self.N):
            if self.clean_until.get(i, 0) >= t:
                new_tau[i] = 0.0
        self.tau = new_tau

    def inspect_clean(self, node: int, until_step: int) -> None:
        self.clean_until[node] = max(self.clean_until.get(node, 0), until_step)

    def value(self, node: int) -> float:
        if not self.cfg.use_provenance:
            return 0.0
        return float(self.tau[node])
