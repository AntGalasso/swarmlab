"""Allocation baselines (spec 2.7): random_alloc, oracle_alloc and bra_greedy.

All three run the same audit-then-block pipeline, so they differ only in WHOM they audit:
  1. pace the audit budget evenly over the rounds (cumulative allowance floor(cap * t / T));
  2. rank the senders that have something to send and audit the top ones (one audit unit per sender text);
  3. block every audited sender whose audit score exceeds the calibrated threshold "audit_score", highest score
     first, while the intervention budget lasts. A flag is remembered (D98): the sender's outgoing edges are
     quarantined for the rest of the episode, priced once as the platform prices it (cost x out-degree x remaining
     rounds), and a cut sender has nothing left to audit.
oracle_alloc differs in step 2 only (D96): it audits the senders it knows to be infected; the auditor's verdict still
decides, through the same threshold.
Budget accounting goes through Actions only: the arm asks for audits it can pay for and proposes quarantines it can pay for.
"""
from __future__ import annotations

import torch

from bazt_lab.core import public as core
from bazt_lab.features.references.private import common


class _Allocation(common.Arm):
    thresholded = True
    _state = ("last_audit", "n_audited", "cut")
    statistic = "audit_score"

    def reset(self, ctx: core.EpisodeContext) -> None:
        self.ctx = ctx
        self.c_inspect = common.cost_of(ctx, core.ActionKind.INSPECT)
        self.c_cut = common.cost_of(ctx, core.ActionKind.EDGE_QUARANTINE)
        if self.c_inspect <= 0:
            raise ValueError("costs.*.inspect must be > 0: the audit budget is paced in audit units")
        self.last_audit = torch.zeros(ctx.B, ctx.N, dtype=torch.int64)     # round of the agent's last audit, 0 = never
        self.n_audited = torch.zeros(ctx.B, dtype=torch.float64)           # audit units requested so far
        self.cut = torch.zeros(ctx.B, ctx.N, dtype=torch.bool)             # senders whose out-edges are quarantined
        self.setup(ctx)

    def setup(self, ctx: core.EpisodeContext) -> None:
        pass

    def priority(self, obs: core.Observation, send: torch.Tensor):
        """-> (score [B, N] float64, eligible [B, N] bool, tie-break key [B, N] float64 or None)."""
        raise NotImplementedError

    def flagged(self, done: torch.Tensor, audit: torch.Tensor) -> torch.Tensor:
        """bool [B, N]: audited senders whose deliveries are blocked this round."""
        return done & ~self.cut & (audit > common.threshold_of(self.ctx, self.statistic))

    def _allowance(self, obs: core.Observation, eligible: torch.Tensor) -> torch.Tensor:
        """int64 [B]: audits this round = min(paced allowance, affordable, eligible). The cumulative allowance
        floor(cap * t / T) is the paper's constant per-round audit ratio for a total budget; unspent allowance carries."""
        cap_n = self.ctx.audit_cap / self.c_inspect
        room = torch.floor(cap_n * obs.t / self.ctx.T) - self.n_audited
        afford = torch.floor(obs.remaining_audit / self.c_inspect)
        k = torch.minimum(torch.minimum(room, afford), eligible.sum(-1).double())
        return k.clamp(min=0).to(torch.int64)

    def decide(self, obs: core.Observation) -> core.Actions:
        B, N = obs.active.shape
        send = obs.attempts.any(-1) & obs.active
        score, eligible, tie = self.priority(obs, send)
        k = self._allowance(obs, eligible)
        order = common.rank_desc(score, eligible, tie)
        chosen = eligible & (order.argsort(dim=-1) < k[:, None])
        if chosen.any():
            audit = obs.audit(chosen)
        else:
            audit = torch.full((B, N), float("nan"), dtype=torch.float64)
        done = chosen & ~audit.isnan()                                     # audits the platform actually performed
        self.last_audit = torch.where(done, torch.full_like(self.last_audit, obs.t), self.last_audit)
        self.n_audited = self.n_audited + done.sum(-1).double()
        flagged = self.flagged(done, audit)
        cost = self.c_cut * self.ctx.adj.sum(-1).double() * (self.ctx.T - obs.t + 1)
        blocked = common.fit_in_order(common.rank_desc(audit, flagged), cost, flagged, obs.remaining_intervention)
        self.cut = self.cut | blocked
        return core.Actions(edge_quarantine=self.ctx.adj & blocked[:, :, None], statistics={self.statistic: audit})


class _RandomOrder(_Allocation):
    """Pre-draws one uniform per (episode, round, agent) from this arm's own stream."""
    def setup(self, ctx):
        self.u = common.predraw(ctx, f"{self.name}.pick", (ctx.T, ctx.N))


class RandomAlloc(_RandomOrder):
    """Lower bound: audits are spent on senders chosen uniformly at random (top-k of iid uniforms)."""
    name = "random_alloc"

    def priority(self, obs, send):
        return self.u[:, obs.t - 1], send, None


class OracleAlloc(_RandomOrder):
    """Ceiling for allocation (D96): audits go only to senders that are infected now (obs.oracle), in random order, and
    each verdict still goes through the auditor's ROC: the sender is cut when its audit score exceeds the calibrated
    "audit_score" threshold, as for the other allocation arms. With ground truth only on WHERE to look, this bounds
    allocation, not detection. The threshold is the one of the same cell's random_alloc (an attack-free oracle run
    audits nobody, so it cannot calibrate itself): the runner puts it in ctx.thresholds; without it the arm audits
    and never acts."""
    name = "oracle_alloc"
    needs_truth = True

    def priority(self, obs, send):
        if obs.oracle is None:
            raise ValueError("oracle_alloc needs obs.oracle: register it with ArmSpec.needs_truth=True")
        return self.u[:, obs.t - 1], send & obs.oracle(), None


class BraGreedy(_Allocation):
    """BRA-Audit greedy placement (Wang, Lin, Lou, Li, arXiv 2608.14668, Alg. 1, Eq. 7): audit the senders with the
    largest weight w(i) = AuditGap(i) * TopoImpact(i), AuditGap = rounds since the agent's last audit (never = round
    index), TopoImpact = 1 + number of descendants of its message in the round-unrolled dependency graph.
    Ties go to the arm stream. Deviations from the paper are listed in CONTRACT.md."""
    name = "bra_greedy"

    def setup(self, ctx):
        self.curve = common.downstream_curve(ctx.adj, ctx.T)               # [B, N, T+1], static graph
        self.tie = common.predraw(ctx, f"{self.name}.tie", (ctx.T, ctx.N))

    def priority(self, obs, send):
        gap = (obs.t - self.last_audit).double()
        impact = 1.0 + self.curve[:, :, self.ctx.T - obs.t]
        return gap * impact, send, self.tie[:, obs.t - 1]
