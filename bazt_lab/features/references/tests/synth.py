"""Synthetic episodes for the references tests: contexts, graphs and a scripted fake platform.

The fake platform steps a defender round by round exactly as the core contract describes: it builds the
Observation, applies the returned Actions, charges them (per affected delivery; agent-level and persistent
actions: affected deliveries x remaining rounds, counted from the current round) and fails the test on a budget
violation, which the real platform would answer by rejecting the episode's actions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch

from bazt_lab.core import public as core
from bazt_lab.core.public import ActionKind, Actions, EpisodeContext, Observation

INF = float("inf")
COSTS = {k.value: 1.0 for k in ActionKind}


def adj_from_edges(n: int, edges, symmetric: bool = False) -> torch.Tensor:
    a = torch.zeros(n, n, dtype=torch.bool)
    for i, j in edges:
        a[i, j] = True
        if symmetric:
            a[j, i] = True
    return a


def star(n: int) -> torch.Tensor:
    return adj_from_edges(n, [(0, i) for i in range(1, n)], symmetric=True)


def chain(n: int, symmetric: bool = False) -> torch.Tensor:
    return adj_from_edges(n, [(i, i + 1) for i in range(n - 1)], symmetric=symmetric)


def make_ctx(adj: torch.Tensor, T: int, *, B: Optional[int] = None, audit_cap=INF, intervention_cap=INF,
             costs=None, thresholds=None, settings=None, arm: str = "x", unit_ids=None, sink: int = 0) -> EpisodeContext:
    """`adj` is [N, N] (repeated over B) or [B, N, N]. Caps are scalars or [B] tensors."""
    if adj.dim() == 2:
        adj = adj.unsqueeze(0).expand(B or 1, -1, -1).clone()
    B, N = adj.shape[0], adj.shape[1]
    ids = tuple(unit_ids) if unit_ids is not None else tuple(f"u{b}" for b in range(B))
    cap = lambda x: x.double() if torch.is_tensor(x) else torch.full((B,), float(x), dtype=torch.float64)  # noqa: E731
    return EpisodeContext(
        B=B, N=N, T=T, adj=adj, sink=torch.full((B,), sink, dtype=torch.int64),
        audit_cap=cap(audit_cap), intervention_cap=cap(intervention_cap),
        costs=dict(COSTS if costs is None else costs), thresholds=dict(thresholds or {}),
        settings=settings or {}, arm=arm, unit_ids=ids,
        arm_stream=lambda ns, b: core.stream(ns, arm, ids[b]),
    )


@dataclass
class Round:
    t: int
    obs: Observation
    actions: Actions
    charged: torch.Tensor          # float64 [B] intervention units charged this round
    audited: torch.Tensor          # bool [B, N] senders audited this round
    requested: torch.Tensor        # bool [B, N] senders the arm asked to audit this round
    audit_scores: torch.Tensor     # float64 [B, N]


@dataclass
class FakePlatform:
    ctx: EpisodeContext
    monitor: Optional[Callable[[int], torch.Tensor]] = None      # t -> float64 [B, N]
    infected: Optional[Callable[[int], torch.Tensor]] = None     # t -> bool [B, N]
    audit_hi: float = 3.0                                        # auditor score of an infected sender
    audit_lo: float = -3.0
    audit_override: Optional[torch.Tensor] = None                # float64 [B, N]: score of an infected sender
    expose_oracle: bool = True
    log: List[Round] = field(default_factory=list)

    def __post_init__(self):
        c = self.ctx
        self.active = torch.ones(c.B, c.N, dtype=torch.bool)
        self.blocked = torch.zeros(c.B, c.N, c.N, dtype=torch.bool)
        self.rem_audit = c.audit_cap.clone()
        self.rem_int = c.intervention_cap.clone()

    def _inf(self, t: int) -> torch.Tensor:
        c = self.ctx
        return self.infected(t) if self.infected else torch.zeros(c.B, c.N, dtype=torch.bool)

    def step(self, defender, t: int) -> Round:
        c = self.ctx
        attempts = c.adj & self.active[:, :, None] & self.active[:, None, :] & ~self.blocked
        inf_now = self._inf(t)
        audited = torch.zeros(c.B, c.N, dtype=torch.bool)
        requested = torch.zeros(c.B, c.N, dtype=torch.bool)
        scores_out = torch.full((c.B, c.N), float("nan"), dtype=torch.float64)

        def audit(mask: torch.Tensor) -> torch.Tensor:
            out = torch.full((c.B, c.N), float("nan"), dtype=torch.float64)
            requested.logical_or_(mask)
            need = mask.sum(-1).double() * c.costs["inspect"]
            ok = (need <= self.rem_audit)[:, None] & mask
            hi = self.audit_override if self.audit_override is not None else torch.full((c.B, c.N), self.audit_hi)
            out = torch.where(ok, torch.where(inf_now, hi.double(), torch.full((c.B, c.N), self.audit_lo).double()), out)
            self.rem_audit = self.rem_audit - ok.sum(-1).double() * c.costs["inspect"]
            audited.logical_or_(ok)
            scores_out.copy_(torch.where(ok, out, scores_out))
            return out

        obs = Observation(
            t=t, T=c.T, active=self.active.clone(), attempts=attempts,
            monitor=self.monitor(t) if self.monitor else None,
            remaining_audit=self.rem_audit.clone(), remaining_intervention=self.rem_int.clone(), audit=audit,
            oracle=inf_now.clone if self.expose_oracle else None)
        acts = defender.decide(obs)
        charged = self._charge(acts, attempts, t)
        self.rem_int = self.rem_int - charged
        r = Round(t, obs, acts, charged, audited, requested, scores_out)
        self.log.append(r)
        return r

    def _charge(self, a: Actions, attempts: torch.Tensor, t: int) -> torch.Tensor:
        c = self.ctx
        left = c.T - t + 1
        total = torch.zeros(c.B, dtype=torch.float64)
        for kind in ("deny", "sanitize", "delay"):
            m = getattr(a, kind)
            if m is not None:
                total += c.costs[kind] * (m & attempts).sum((-1, -2)).double()
        if a.edge_quarantine is not None:
            new = a.edge_quarantine & ~self.blocked & c.adj
            total += c.costs["edge_quarantine"] * new.sum((-1, -2)).double() * left
            self.blocked |= new
        if a.isolation is not None:
            new = a.isolation & self.active
            total += c.costs["isolation"] * (new.double() * c.adj.sum(-1).double()).sum(-1) * left
            self.active = self.active & ~a.isolation
        assert bool((total <= self.rem_int).all()), f"budget violation at round {t}: {total} > {self.rem_int}"
        return total

    def run(self, defender, rounds: Optional[int] = None) -> List[Round]:
        defender.reset(self.ctx)
        for t in range(1, (rounds or self.ctx.T) + 1):
            self.step(defender, t)
        return self.log
