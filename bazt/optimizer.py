"""Deterministic two-phase budgeted optimizer. No RNG: identical inputs always give the identical plan."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import swarmlab as sl
from bazt.actions import (ActionKind, Candidate, cost_of, score_block, score_inspect, score_isolate,
                          score_restrict, score_sanitize)
from bazt.config import BaztConfig
from bazt.contagion import ContagionModel
from bazt.ledger import BudgetLedger, Charge


@dataclass
class Situation:
    """What the optimizer may know at step t. `p_prime` is updated in place by conclusive inspections."""
    t: int
    T: int
    active: Tuple[int, ...]
    p_prime: List[float]
    deg: Sequence[int]
    maxdeg: int
    privileged: frozenset
    out_adj: Sequence[Sequence[int]]
    restricted: frozenset                  # already restricted: no point sanitizing / restricting again
    verdict_active: frozenset              # nodes with a live inspection verdict: no point re-inspecting
    inspect: Callable[[int], Optional[float]]   # pays the oracle; new p' if conclusive, None if not


@dataclass(frozen=True)
class Decision:
    cand: Candidate
    charge: Charge
    theta: float
    reason: str


@dataclass
class Plan:
    isolate: List[int] = field(default_factory=list)
    restrict: List[int] = field(default_factory=list)
    sanitize: List[int] = field(default_factory=list)
    block: List[Tuple[int, int]] = field(default_factory=list)
    inspected: Dict[int, Optional[float]] = field(default_factory=dict)
    decisions: List[Decision] = field(default_factory=list)


class BudgetedOptimizer:
    def __init__(self, cfg: BaztConfig, sl_cfg: sl.Config):
        self.cfg, self.sl_cfg = cfg, sl_cfg

    # ------------------------------------------------------------------------------------ phase 1
    def _inspect_phase(self, sit: Situation, ledger: BudgetLedger, plan: Plan) -> Optional[ContagionModel]:
        """Buy inspections. Returns the reach model if no verdict changed any p' (so phase 2 can reuse it)."""
        aset = set(sit.active)
        model = ContagionModel(self.sl_cfg, self.cfg, sit.out_adj, aset, sit.p_prime, sit.T - sit.t)
        cands = []
        for i in sit.active:
            if i in sit.verdict_active:
                continue
            c = score_inspect(i, sit.p_prime[i], model.H(i), sl.centrality(i, sit.t, sit.deg, sit.maxdeg, sit.T),
                              self.cfg)
            if c.net > 0:
                cands.append(c)
        changed = False
        for c in sorted(cands, key=lambda c: c.sort_key):
            theta = ledger.theta()
            if c.density < theta:
                break                                   # sorted by density, theta never decreases
            if not ledger.can_afford(c.kind):
                continue
            charge = ledger.charge(c.kind)
            new_p = sit.inspect(c.node)
            plan.inspected[c.node] = new_p
            if new_p is not None:
                sit.p_prime[c.node] = new_p
                changed = True
            plan.decisions.append(Decision(c, charge, theta, "inconclusive" if new_p is None else
                                           ("confirmed" if new_p > 0.5 else "cleared")))
        return None if changed else model

    # ------------------------------------------------------------------------------------ phase 2
    def _node_candidates(self, i: int, sit: Situation, model: ContagionModel) -> List[Candidate]:
        p, H = sit.p_prime[i], model.H(i)
        C = sl.centrality(i, sit.t, sit.deg, sit.maxdeg, sit.T)
        out = [score_isolate(i, p, H, C, self.cfg)]
        if not self.cfg.binary_actions and i not in sit.restricted:
            h_t = sit.T - sit.t
            out.append(score_sanitize(i, p, H, C, h_t, self.sl_cfg.p_strip, self.cfg))
            out.append(score_restrict(i, p, H, C, h_t, self.sl_cfg.p_strip, i in sit.privileged, self.cfg))
        return out

    def _edge_candidates(self, i: int, sit: Situation, model: ContagionModel, aset: set,
                         theta_min: float) -> List[Candidate]:
        """BLOCK_EDGE candidates of sender i.

        Exact pruning (never changes the plan): H_edge <= (1 - p_j)(1 + H_j), and an action is only accepted
        when density >= theta, with theta non-decreasing within a step. So an edge whose best possible net
        value is below theta_min * cost can never be accepted and its reach need not be computed.
        """
        p = sit.p_prime[i]
        C = sl.centrality(i, sit.t, sit.deg, sit.maxdeg, sit.T)
        floor = (self.cfg.lam if self.cfg.use_utility else 0.0) * (1.0 - p) * C / max(sit.deg[i], 1)
        need = floor + theta_min * cost_of(self.cfg, ActionKind.BLOCK_EDGE)
        out = []
        for j in sit.out_adj[i]:
            if j not in aset:
                continue
            if p * (1.0 - sit.p_prime[j]) * (1.0 + model.H(j)) < need:
                continue
            c = score_block(i, j, p, model.H_edge(i, j), C, sit.deg[i], self.cfg)
            if c.net > 0:
                out.append(c)
        return out

    def plan(self, sit: Situation, ledger: BudgetLedger) -> Plan:
        plan = Plan()
        model = None if self.cfg.binary_actions else self._inspect_phase(sit, ledger, plan)
        aset = set(sit.active)
        if model is None:                                   # a verdict changed p': reach must be recomputed
            model = ContagionModel(self.sl_cfg, self.cfg, sit.out_adj, aset, sit.p_prime, sit.T - sit.t)
        theta_min = ledger.theta()
        best: Dict[Tuple[int, int], Candidate] = {}        # best positive-net candidate per target
        for i in sit.active:
            pool = self._node_candidates(i, sit, model)
            if not self.cfg.binary_actions:
                pool += self._edge_candidates(i, sit, model, aset, theta_min)
            for c in sorted(pool, key=lambda c: c.sort_key):
                if c.net > 0:
                    key = (c.node, c.peer)
                    if key not in best or c.sort_key < best[key].sort_key:
                        best[key] = c
        isolated = set()
        for c in sorted(best.values(), key=lambda c: c.sort_key):
            theta = ledger.theta()
            if c.density < theta:
                break
            if c.node in isolated or not ledger.can_afford(c.kind):
                continue
            charge = ledger.charge(c.kind)
            plan.decisions.append(Decision(c, charge, theta, c.kind.name.lower()))
            if c.kind == ActionKind.ISOLATE:
                plan.isolate.append(c.node)
                isolated.add(c.node)
            elif c.kind == ActionKind.RESTRICT:
                plan.restrict.append(c.node)
            elif c.kind == ActionKind.SANITIZE:
                plan.sanitize.append(c.node)
            else:
                plan.block.append((c.node, c.peer))
        return plan
