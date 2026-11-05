"""Small synthetic episodes for the tests of the conventional arms.

`Fake` plays the platform for one batch: it owns the budget ledger, answers `obs.audit` with given judge scores
(NaN where the budget cannot pay, in sender order), and charges one intervention unit per deny that is not one of
A's free rules.
"""
from __future__ import annotations

import math

import torch

from bazt_lab.core import public as core

INF = math.inf
NAN = math.nan
BINS = [[-5.0, 0.0], [-1.0, 4.0], [INF, 12.0]]      # judge score <= -5: 0, (-5, -1]: 4, above -1: 12
COSTS = {kind.value: 1.0 for kind in core.ActionKind}


def make_settings(rate_limit=2, bins=BINS, window=3) -> dict:
    s: dict = {}
    if rate_limit is not None:
        core.put(s, "runtime.rate_limit_per_identity", rate_limit)
    if bins is not None:
        core.put(s, "conventional.c.judge_score_bins", {"mock": bins, "llm": bins})
    if window is not None:
        core.put(s, "conventional.c.presence_window_rounds", window)
    return s


def full_adj(B: int, N: int) -> torch.Tensor:
    return (~torch.eye(N, dtype=torch.bool)).expand(B, N, N).clone()


def make_ctx(adj, arm, *, platform="mock", thresholds=None, audit_cap=INF, intervention_cap=INF, settings=None, costs=None, T=4,
             ids=None):
    B, N, _ = adj.shape
    ids = tuple(f"u{b}" for b in range(B)) if ids is None else tuple(ids)
    return core.EpisodeContext(
        B=B, N=N, T=T, adj=adj, sink=torch.zeros(B, dtype=torch.int64),
        audit_cap=torch.full((B,), float(audit_cap), dtype=torch.float64),
        intervention_cap=torch.full((B,), float(intervention_cap), dtype=torch.float64),
        costs=dict(COSTS if costs is None else costs), thresholds=dict(thresholds or {}),
        settings=make_settings() if settings is None else settings, arm=arm, platform=platform, unit_ids=ids,
        arm_stream=lambda ns, b: core.stream(ns, arm, ids[b]))


class Fake:
    def __init__(self, ctx):
        self.ctx = ctx
        self.remaining_audit = ctx.audit_cap.clone()
        self.remaining_intervention = ctx.intervention_cap.clone()
        self.audit_calls = []
        self.scores = torch.zeros(ctx.B, ctx.N, dtype=torch.float64)

    def audit(self, mask):
        self.audit_calls.append(mask.clone())
        out = torch.full(mask.shape, NAN, dtype=torch.float64)
        for b in range(mask.shape[0]):
            for i in range(mask.shape[1]):
                if mask[b, i] and self.remaining_audit[b] >= 1.0:
                    out[b, i] = self.scores[b, i]
                    self.remaining_audit[b] -= 1.0
        return out

    def obs(self, t, attempts=None, *, scores=None, authentic=None, monitor=None, active=None):
        ctx = self.ctx
        if scores is not None:
            self.scores = torch.as_tensor(scores, dtype=torch.float64).expand(ctx.B, ctx.N).clone()
        self.attempts = ctx.adj.clone() if attempts is None else attempts
        self.authentic = authentic
        active = torch.ones(ctx.B, ctx.N, dtype=torch.bool) if active is None else active
        return core.Observation(
            t=t, T=ctx.T, active=active, attempts=self.attempts,
            monitor=monitor, remaining_audit=self.remaining_audit, remaining_intervention=self.remaining_intervention,
            audit=self.audit, authentic=authentic)

    def settle(self, actions):
        """Charge one unit per deny that is not one of A's rules (outside adj or not authentic)."""
        free = ~self.ctx.adj if self.authentic is None else (~self.ctx.adj | ~self.authentic)
        if actions.deny is not None:
            self.remaining_intervention -= (actions.deny & ~free).flatten(1).sum(-1).to(torch.float64)


def adj_of(B, edges, N):
    """bool [B, N, N] with the given directed edges (i, j) in every episode."""
    a = torch.zeros(B, N, N, dtype=torch.bool)
    for i, j in edges:
        a[:, i, j] = True
    return a
