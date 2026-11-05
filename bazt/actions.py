"""bazt/actions.py -- Pure candidate scoring for BAZT enforcement actions.

`ActionKind`: INSPECT, SANITIZE, RESTRICT, BLOCK_EDGE, ISOLATE.

All scores are deterministic given `(p', H_i, H_edge, C_i, deg_i, cfg)`; no RNG
inside the action layer.

    b_isolate        = p' * H_i
    b_restrict       = p_strip * p' * H_i * (min(ttl_restrict, h_t) / max(h_t, 1))
                     (+ bonus reduction on privileged exposure if privileged)
    b_sanitize       = p_strip * p' * H_i * (1.0 / max(h_t, 1))
    b_block(i -> j)  = p' * H_edge(i, j)
    b_inspect        = q_inspect * min(p' * H_i, (1.0 - p') * lam_eff * C_i)   # EVPI >= 0

    u_a: 0.0 (INSPECT), rho_san * C_i (SANITIZE), rho_res * C_i (RESTRICT),
         C_i / max(deg_i, 1) (BLOCK_EDGE), C_i (ISOLATE)

    net   = b_a - lam_eff * (1.0 - p') * u_a    (lam_eff = lam if use_utility else 0)
    density = net / cost_a
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import swarmlab as sl
from bazt.config import BaztConfig, cost_of


class ActionKind(Enum):
    INSPECT = "INSPECT"
    SANITIZE = "SANITIZE"
    RESTRICT = "RESTRICT"
    BLOCK_EDGE = "BLOCK_EDGE"
    ISOLATE = "ISOLATE"


@dataclass
class Candidate:
    kind: ActionKind
    target: int        # node id
    peer: int = -1     # peer node id for BLOCK_EDGE, else -1
    benefit: float = 0.0
    utility: float = 0.0
    cost: int = 0
    net: float = 0.0
    density: float = 0.0
    action_rank: int = 0  # tie-break rank within kind (lower = preferred)

    def tie_key(self):
        # (-density, node_id, peer_id_or_-1, action_rank)
        return (-self.density, self.target, self.peer if self.peer >= 0 else -1, self.action_rank)


def centrality(i: int, t: int, deg: List[int], maxdeg: int, T: int) -> float:
    return sl.centrality(i, t, deg, maxdeg, T)


def score_candidates(
    p_prime: List[float],
    H: List[float],
    H_edge: dict,            # (i, j) -> float
    deg: List[int],
    maxdeg: int,
    T: int,
    t: int,
    h_t: int,
    active: List[int],
    adj: List[List[int]],
    cfg: BaztConfig,
    privileged: set,
) -> List[Candidate]:
    """Build every affordable-relevant candidate action with benefit/utility/net/density."""
    lam_eff = cfg.lam if cfg.use_utility else 0.0
    cands: List[Candidate] = []

    def centrality_of(i: int) -> float:
        return centrality(i, t, deg, maxdeg, T)

    # --- ISOLATE (node-level)
    if not cfg.binary_actions or True:  # isolate always available (binary restricts to ONLY isolate)
        for i in active:
            Ci = centrality_of(i)
            b = p_prime[i] * H[i]
            u = Ci
            net = b - lam_eff * (1.0 - p_prime[i]) * u
            cost = cost_of("ISOLATE", cfg)
            cands.append(Candidate(ActionKind.ISOLATE, i, -1, b, u, cost, net, net / cost if cost else 0.0, 0))

    if cfg.binary_actions:
        # only ISOLATE candidates
        for c in cands:
            if c.kind == ActionKind.ISOLATE:
                c.density = c.net / c.cost if c.cost else 0.0
        return _rank(cands)

    # --- SANITIZE
    for i in active:
        Ci = centrality_of(i)
        b = cfg.p_strip * p_prime[i] * H[i] * (1.0 / max(h_t, 1))
        u = cfg.rho_san * Ci
        net = b - lam_eff * (1.0 - p_prime[i]) * u
        cost = cost_of("SANITIZE", cfg)
        cands.append(Candidate(ActionKind.SANITIZE, i, -1, b, u, cost, net, net / cost if cost else 0.0, 1))

    # --- RESTRICT
    for i in active:
        Ci = centrality_of(i)
        ttl_factor = min(cfg.ttl_restrict, h_t) / max(h_t, 1)
        b = cfg.p_strip * p_prime[i] * H[i] * ttl_factor
        if i in privileged:
            b *= 1.0 + cfg.rho_res  # bonus reduction on privileged exposure
        u = cfg.rho_res * Ci
        net = b - lam_eff * (1.0 - p_prime[i]) * u
        cost = cost_of("RESTRICT", cfg)
        cands.append(Candidate(ActionKind.RESTRICT, i, -1, b, u, cost, net, net / cost if cost else 0.0, 2))

    # --- BLOCK_EDGE (directed open edges i -> j)
    for i in active:
        for j in adj[i]:
            if j not in active:
                continue
            he = H_edge.get((i, j), 0.0)
            b = p_prime[i] * he
            Ci = centrality_of(i)
            u = Ci / max(deg[i], 1)
            net = b - lam_eff * (1.0 - p_prime[i]) * u
            cost = cost_of("BLOCK_EDGE", cfg)
            cands.append(Candidate(ActionKind.BLOCK_EDGE, i, j, b, u, cost, net, net / cost if cost else 0.0, 3))

    # --- INSPECT
    for i in active:
        Ci = centrality_of(i)
        b = cfg.q_inspect * min(p_prime[i] * H[i], (1.0 - p_prime[i]) * lam_eff * Ci)
        if b < 0:
            b = 0.0
        u = 0.0
        net = b  # u=0 -> net = b
        cost = cost_of("INSPECT", cfg)
        cands.append(Candidate(ActionKind.INSPECT, i, -1, b, u, cost, net, net / cost if cost else 0.0, 4))

    return _rank(cands)


def _rank(cands: List[Candidate]) -> List[Candidate]:
    for rank, c in enumerate(sorted(cands, key=lambda c: c.tie_key())):
        c.action_rank = rank
    return sorted(cands, key=lambda c: c.tie_key())
