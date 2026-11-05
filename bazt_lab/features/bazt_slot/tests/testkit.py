"""Small synthetic episodes for the tests of the BAZT slot.

Every number in `make_settings` is a TEST value chosen to make a scenario small and checkable. None of them is an
experiment decision (those are the owner's, see CONTRACT.md "Settings keys read").

`Fake` plays the platform for one batch. It owns the two budget ledgers, answers `obs.audit` with scripted judge
scores (NaN where the budget cannot pay), applies the persistent actions and charges them with the rules written in
`core.Actions`: one unit of the action's cost per affected delivery, agent-level actions per delivery they remove
(out-degree x remaining rounds). It raises on an overspend instead of clipping, so a test fails when a defender
buys what it cannot pay for.
"""
from __future__ import annotations

import math

import torch

from bazt_lab.core import public as core

INF = math.inf
NAN = math.nan
COSTS_MAIN = {kind.value: 1.0 for kind in core.ActionKind}
COSTS_ROBUST = {**COSTS_MAIN, "sanitize": 2.0, "privilege_degradation": 2.0, "isolation": 3.0, "reinstatement": 5.0}

TEST_VALUES = {
    "bazt.katz_decay": 0.5, "bazt.eigen_iterations": 300, "bazt.sink_weight": 1.0, "bazt.utility_weight": 1.0,
    "bazt.utility_loss.sanitize": 0.15, "bazt.utility_loss.delay": 0.1, "bazt.utility_loss.edge_quarantine": 0.3,
    "bazt.utility_loss.rate_limit": 0.3, "bazt.utility_loss.privilege_degradation": 0.35,
    "bazt.utility_loss.isolation": 1.0,
    "bazt.flag_logodds": 4.0, "bazt.clear_logodds": 3.0, "bazt.memory_decay": 0.8, "bazt.prior": 0.1,
    "bazt.inspect_conclusive": 0.8,
    "runtime.sanitize_success": 0.9, "runtime.rate_limit_per_identity": 1,
}


def make_settings(**over) -> dict:
    s: dict = {}
    for key, value in {**TEST_VALUES, **{k.replace("__", "."): v for k, v in over.items()}}.items():
        core.put(s, key, value)
    return s


def full_adj(B: int, N: int) -> torch.Tensor:
    return (~torch.eye(N, dtype=torch.bool)).expand(B, N, N).clone()


def adj_of(B: int, edges, N: int) -> torch.Tensor:
    a = torch.zeros(B, N, N, dtype=torch.bool)
    for i, j in edges:
        a[:, i, j] = True
    return a


def make_ctx(adj, arm="BAZT", *, thresholds=None, audit_cap=INF, intervention_cap=INF, settings=None, costs=None,
             T=6, sink=0):
    B, N, _ = adj.shape
    ids = tuple(f"u{b}" for b in range(B))
    return core.EpisodeContext(
        B=B, N=N, T=T, adj=adj, sink=torch.full((B,), sink, dtype=torch.int64),
        audit_cap=torch.full((B,), float(audit_cap), dtype=torch.float64),
        intervention_cap=torch.full((B,), float(intervention_cap), dtype=torch.float64),
        costs=dict(COSTS_MAIN if costs is None else costs), thresholds=dict(thresholds or {}),
        settings=make_settings() if settings is None else settings, arm=arm, unit_ids=ids,
        arm_stream=lambda ns, b: core.stream(ns, arm, ids[b]), platform="mock")


class Overspend(AssertionError):
    pass


class Fake:
    def __init__(self, ctx, *, quarantine_charged_once=False):
        self.ctx = ctx
        B, N = ctx.B, ctx.N
        self.remaining_audit = ctx.audit_cap.clone()
        self.remaining_intervention = ctx.intervention_cap.clone()
        self.isolated = torch.zeros(B, N, dtype=torch.bool)
        self.scores = torch.zeros(B, N, dtype=torch.float64)       # what the auditor says about each sender
        self.audit_calls = []
        self.spent_audit = torch.zeros(B, dtype=torch.float64)
        self.spent_intervention = torch.zeros(B, dtype=torch.float64)
        self.quarantine_charged_once = quarantine_charged_once
        self.history = []

    def audit(self, mask):
        self.audit_calls.append(mask.clone())
        out = torch.full(mask.shape, NAN, dtype=torch.float64)
        unit = self.ctx.costs["inspect"]
        for b in range(mask.shape[0]):
            for i in range(mask.shape[1]):
                if mask[b, i] and self.remaining_audit[b] >= unit:
                    out[b, i] = self.scores[b, i]
                    self.remaining_audit[b] -= unit
                    self.spent_audit[b] += unit
        return out

    def obs(self, t, *, monitor=None, authentic=None, attempts=None, T=None):
        """`T` is the episode length the defender is told (core contract: obs.T counts the whole episode)."""
        ctx = self.ctx
        self.t, self.T = t, ctx.T if T is None else T
        self.attempts = ctx.adj.clone() if attempts is None else attempts
        return core.Observation(
            t=t, T=self.T, active=~self.isolated, attempts=self.attempts, monitor=monitor,
            remaining_audit=self.remaining_audit.clone(), remaining_intervention=self.remaining_intervention.clone(),
            audit=self.audit, authentic=authentic)

    def charge_of(self, actions):
        """float64 [B]: intervention units the platform charges for these actions at the current round."""
        ctx, t = self.ctx, self.t
        c = ctx.costs
        rounds_left = self.T - t + 1
        live = self.attempts & ctx.adj & ~self.isolated[:, :, None] & ~self.isolated[:, None, :]
        outdeg = live.sum(-1).to(torch.float64)
        total = torch.zeros(ctx.B, dtype=torch.float64)
        for kind, per in (("sanitize", 1.0), ("delay", 1.0)):
            m = getattr(actions, kind)
            if m is not None:
                total += m.flatten(1).sum(-1) * c[kind] * per
        if actions.edge_quarantine is not None:
            n = actions.edge_quarantine.flatten(1).sum(-1).to(torch.float64)
            total += n * c["edge_quarantine"] * (1.0 if self.quarantine_charged_once else rounds_left)
        if actions.isolation is not None:
            total += (actions.isolation.double() * outdeg).sum(-1) * c["isolation"] * rounds_left
        if actions.privilege_degradation is not None:
            total += (actions.privilege_degradation.double() * outdeg).sum(-1) * c["privilege_degradation"] * rounds_left
        if actions.rate_limit is not None:
            limit = core.get(self.ctx.settings, "runtime.rate_limit_per_identity")
            total += (actions.rate_limit.double() * (outdeg - limit).clamp_min(0)).sum(-1) * c["rate_limit"] * rounds_left
        if actions.reinstatement is not None:               # per delivery the agent sends again, over the rounds left
            iso_after = self.isolated & ~actions.reinstatement
            out_after = (ctx.adj & ~iso_after[:, None, :]).sum(-1).to(torch.float64)
            total += (actions.reinstatement.double() * out_after).sum(-1) * c["reinstatement"] * rounds_left
        return total

    def settle(self, actions):
        """Charge the round's actions to the intervention ledger and apply isolation / reinstatement."""
        charge = self.charge_of(actions)
        if (charge > self.remaining_intervention + 1e-9).any():
            raise Overspend(f"round {self.t}: charge {charge.tolist()} > remaining {self.remaining_intervention.tolist()}")
        self.remaining_intervention -= charge
        self.spent_intervention += charge
        if actions.reinstatement is not None:
            self.isolated &= ~actions.reinstatement
        if actions.isolation is not None:
            self.isolated |= actions.isolation
        self.history.append(actions)
        return charge


# ------------------------------------------------------------------------------------------------ drivers
class FakePipeline:
    """The A+B+C pipeline as BAZT sees it: A's free denies and a judge statistic that is the bought audit score."""

    def reset(self, ctx):
        self.ctx = ctx

    def step(self, obs, scores):
        from bazt_lab.features.bazt_slot.private.pipeline import Result
        gate = self.ctx.adj if obs.authentic is None else self.ctx.adj & obs.authentic
        return Result(obs.attempts & ~gate, {"judge": scores[:, :, None].expand_as(obs.attempts).clone()})

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


def monitor_of(B, N, hot=(), level=0.97, base=0.03):
    """float64 [B, N]: the cheap monitor reads `level` for the agents in `hot` and `base` for the others."""
    m = torch.full((B, N), base, dtype=torch.float64)
    for i in hot:
        m[:, i] = level
    return m


def run(defender, fake, rounds, *, monitor=None, scores=None):
    """Play `rounds` rounds against the fake platform, charging every round; returns the list of Actions."""
    if scores is not None:
        fake.scores = scores.clone()
    out = []
    for t in range(1, rounds + 1):
        m = monitor(t) if callable(monitor) else monitor
        actions = defender.decide(fake.obs(t, monitor=m))
        fake.settle(actions)
        out.append(actions)
    return out
