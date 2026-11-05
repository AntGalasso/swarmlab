"""Pure candidate scoring: benefit, utility cost, net value and density of every enforcement action.

No state, no RNG, no budget here -- the optimizer decides what to buy, this module only prices it.
For ISOLATE the net value equals `swarmlab.risk_score` (p*H - lam*(1-p)*C), so the BAZT arm with
every extra mechanism ablated degenerates to the legacy `risk_aware` score.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple

from bazt.config import BaztConfig


class ActionKind(IntEnum):
    """Order = tie-break rank (cheapest / least intrusive first)."""
    INSPECT = 0
    SANITIZE = 1
    RESTRICT = 2
    BLOCK_EDGE = 3
    ISOLATE = 4


HARD_KINDS = frozenset({ActionKind.BLOCK_EDGE, ActionKind.ISOLATE})
SOFT_KINDS = frozenset({ActionKind.INSPECT, ActionKind.SANITIZE, ActionKind.RESTRICT})
_COST_FIELD = {ActionKind.INSPECT: "cost_inspect", ActionKind.SANITIZE: "cost_sanitize",
               ActionKind.RESTRICT: "cost_restrict", ActionKind.BLOCK_EDGE: "cost_block",
               ActionKind.ISOLATE: "cost_isolate"}


def cost_of(cfg: BaztConfig, kind: ActionKind) -> int:
    return int(getattr(cfg, _COST_FIELD[kind]))


@dataclass(frozen=True)
class Candidate:
    kind: ActionKind
    node: int
    peer: int            # receiver of a BLOCK_EDGE, -1 for node-level actions
    benefit: float
    util_cost: float     # u_a (before the lam_eff * (1 - p') factor)
    net: float
    cost: int
    density: float
    p_prime: float
    H: float

    @property
    def sort_key(self) -> Tuple[float, int, int, int]:
        """Deterministic total order: best density first, then node id, peer id, action rank."""
        return (-self.density, self.node, self.peer, int(self.kind))


def lam_eff(cfg: BaztConfig) -> float:
    return cfg.lam if cfg.use_utility else 0.0


def _build(kind: ActionKind, node: int, peer: int, benefit: float, util: float, p: float, H: float,
           cfg: BaztConfig) -> Candidate:
    net = benefit - lam_eff(cfg) * (1.0 - p) * util
    cost = cost_of(cfg, kind)
    return Candidate(kind, node, peer, benefit, util, net, cost, net / cost, p, H)


def _horizon_fraction(span: int, h_t: int) -> float:
    """Share of the remaining horizon covered by an action that lasts `span` steps."""
    return min(span, h_t) / max(h_t, 1)


def score_isolate(node: int, p: float, H: float, C: float, cfg: BaztConfig) -> Candidate:
    return _build(ActionKind.ISOLATE, node, -1, p * H, C, p, H, cfg)


def score_restrict(node: int, p: float, H: float, C: float, h_t: int, p_strip: float, privileged: bool,
                   cfg: BaztConfig) -> Candidate:
    frac = _horizon_fraction(cfg.ttl_restrict, h_t)
    benefit = p_strip * p * H * frac
    if privileged:                       # least privilege: a restricted tool-holder cannot act on its tools
        benefit += cfg.priv_bonus * p * frac
    return _build(ActionKind.RESTRICT, node, -1, benefit, cfg.rho_res * C, p, H, cfg)


def score_sanitize(node: int, p: float, H: float, C: float, h_t: int, p_strip: float,
                   cfg: BaztConfig) -> Candidate:
    benefit = p_strip * p * H * (1.0 / max(h_t, 1))      # one step of stripping
    return _build(ActionKind.SANITIZE, node, -1, benefit, cfg.rho_san * C, p, H, cfg)


def score_block(node: int, peer: int, p: float, H_edge: float, C: float, deg_i: int,
                cfg: BaztConfig) -> Candidate:
    return _build(ActionKind.BLOCK_EDGE, node, peer, p * H_edge, C / max(deg_i, 1), p, H_edge, cfg)


def score_inspect(node: int, p: float, H: float, C: float, cfg: BaztConfig) -> Candidate:
    """Expected value of perfect information, bounded by the cost of the error it can avoid.

    q * min(p*H, (1-p)*lam*C): confirming a true positive saves p*H of contagion, clearing a false
    positive saves (1-p)*lam*C of utility; the smaller of the two bounds what a verdict is worth.
    """
    benefit = cfg.q_inspect * min(p * H, (1.0 - p) * lam_eff(cfg) * C)
    return _build(ActionKind.INSPECT, node, -1, benefit, 0.0, p, H, cfg)
