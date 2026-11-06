"""The existing swarmlab decision rules (swarmlab._decide, risk_score, reach_H) ported to torch, vectorised over B.

Each rule keeps swarmlab's selection logic; what the spec changes is the money and the thresholds:
  * the budget is the cell's intervention cap, not a count K: a hard quarantine becomes ISOLATION, priced as the
    platform prices it (cost x out-degree x remaining rounds), and the walk down the ranking takes every candidate
    that still fits (skip-and-continue), where swarmlab took the first K;
  * thresholds are calibrated at alpha (spec 2.8, replacing the J-objective tuning): the monitor probability p is the
    statistic "monitor" (tau_p), the risk score R is the statistic "risk" (tau_R / tau_quar). A missing threshold
    means +inf (calibration run: collect the statistic, never act). Comparisons are strict, as in the core contract;
  * the structure constants (gamma, dmax, lam, confirm_steps, ...) are settings keys under mock.legacy.*.
No rule calls the auditor: swarmlab's arms use the free monitor only.

`own_timing=False` removes the rule's own timing gate (the fixed windows, random_capped's random steps) so that a
MatchedTiming wrapper decides when the arm may act (spec 2.7, D32).
"""
from __future__ import annotations

import torch

from bazt_lab.core import public as core
from bazt_lab.features.references.private import common

NAN = float("nan")


class _Legacy(common.Arm):
    keys: tuple = ()                 # settings keys under mock.legacy.* this rule reads
    needs_monitor = True
    thresholded = True

    def __init__(self, own_timing: bool = True) -> None:
        self.own_timing = own_timing

    def reset(self, ctx: core.EpisodeContext) -> None:
        self.ctx = ctx
        self.c_isolate = common.cost_of(ctx, core.ActionKind.ISOLATION)
        self.par = {k: core.get(ctx.settings, f"mock.legacy.{k}") for k in self.keys}
        self.setup(ctx)

    def setup(self, ctx: core.EpisodeContext) -> None:
        pass

    def decide(self, obs: core.Observation) -> core.Actions:
        if self.needs_monitor and obs.monitor is None:
            raise ValueError(f"{self.name} reads the mock's cheap monitor (obs.monitor); it is mock-only")
        return self.act(obs)

    def act(self, obs: core.Observation) -> core.Actions:
        raise NotImplementedError

    # --- shared pieces -------------------------------------------------------------------------------------
    def isolation_cost(self, obs: core.Observation) -> torch.Tensor:
        """float64 [B, N]: isolating an agent removes its out-degree deliveries for every remaining round."""
        return self.c_isolate * self.ctx.adj.sum(-1).double() * (self.ctx.T - obs.t + 1)

    def isolate(self, obs: core.Observation, order: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """bool [B, N]: candidates (walked in `order`) that the remaining intervention budget pays for."""
        return common.fit_in_order(order, self.isolation_cost(obs), cand, obs.remaining_intervention)

    def reach(self, obs: core.Observation, weight: torch.Tensor) -> torch.Tensor:
        """swarmlab.reach_H for every agent: horizon T - t, gamma path decay, dmax hops, over active agents."""
        return common.reach_h(self.ctx.adj, obs.active, self.ctx.T - obs.t, self.par["gamma"], self.par["dmax"], weight)

    def risk(self, obs: core.Observation, p: torch.Tensor):
        """swarmlab.risk_score: (R, C) with R = p * H - lam * (1 - p) * C, both float64 [B, N]."""
        H = self.reach(obs, 1.0 - p)
        deg = self.ctx.adj.sum(-1).to(self.ctx.dtype)
        C = 0.5 * deg / deg.amax(-1, keepdim=True).clamp(min=1.0) + 0.5 * (1.0 - obs.t / self.ctx.T)
        return p * H - self.par["lam"] * (1.0 - p) * C, C


class FixedWindow(_Legacy):
    """fixed_early / fixed_mid / fixed_late: a snapshot audit inside one third of the horizon (swarmlab.window_of):
    isolate the active agents whose monitor probability is above the threshold, highest first."""

    def __init__(self, which: str, own_timing: bool = True) -> None:
        super().__init__(own_timing)
        if which not in ("early", "mid", "late"):
            raise ValueError(f"unknown window {which!r}")
        self.which = which
        self.name = f"fixed_{which}"

    def act(self, obs):
        T = self.ctx.T
        th = T // 3
        lo, hi = {"early": (1, th), "mid": (th + 1, 2 * th), "late": (2 * th + 1, T)}[self.which]
        p = obs.monitor
        cand = obs.active & (p > common.threshold_of(self.ctx, "monitor"))
        if self.own_timing and not lo <= obs.t <= hi:
            cand = torch.zeros_like(cand)
        iso = self.isolate(obs, common.rank_desc(p, cand), cand)
        return core.Actions(isolation=iso, statistics={"monitor": p})


class ScoreOnly(_Legacy):
    """score_only: monitor-only alarm that needs `confirm_steps` consecutive crossings (a one-step spike is not
    enough), at any step."""
    name = "score_only"
    keys = ("confirm_steps",)
    _state = ("alarm_run",)

    def setup(self, ctx):
        self.alarm_run = torch.zeros(ctx.B, ctx.N, dtype=torch.int64, device=ctx.device)

    def act(self, obs):
        p = obs.monitor
        above = p > common.threshold_of(self.ctx, "monitor")
        run = torch.where(above, self.alarm_run + 1, torch.zeros_like(self.alarm_run))
        self.alarm_run = torch.where(obs.active, run, self.alarm_run)       # swarmlab updates active agents only
        cand = obs.active & above & (self.alarm_run >= self.par["confirm_steps"])
        iso = self.isolate(obs, common.rank_desc(p, cand), cand)
        return core.Actions(isolation=iso, statistics={"monitor": p})


class CentralityOnly(_Legacy):
    """centrality_only: rank by raw structural reach, no evidence, no threshold; acts from the first round on and
    spends the budget at once."""
    name = "centrality_only"
    keys = ("gamma", "dmax")
    needs_monitor = False
    thresholded = False

    def act(self, obs):
        reach = self.reach(obs, torch.ones(self.ctx.B, self.ctx.N, dtype=self.ctx.dtype, device=self.ctx.device))
        cand = obs.active & (reach > 0)
        # sorted(((reach, i) ...), reverse=True): equal reach goes to the higher index
        iso = self.isolate(obs, common.rank_desc(reach, cand, ties_high=True), cand)
        return core.Actions(isolation=iso)


class RandomCapped(_Legacy):
    """random_capped: a random active agent at random steps. swarmlab draws K random steps without replacement
    (rand_steps) and one agent per step; here K is the setting mock.legacy.random_steps, the agent is the active one
    with the smallest pre-drawn key (uniform over the active agents), and when it does not fit the budget the next
    random one that does is taken. One agent per acting step."""
    name = "random_capped"
    keys = ("random_steps",)
    needs_monitor = False
    thresholded = False

    def setup(self, ctx):
        steps = int(self.par["random_steps"])
        when = common.predraw(ctx, f"{self.name}.when", (ctx.T,)).argsort(dim=-1)[:, :min(steps, ctx.T)]
        self.when = torch.zeros(ctx.B, ctx.T, dtype=torch.bool, device=ctx.device).scatter_(-1, when, True)
        self.who = common.predraw(ctx, f"{self.name}.who", (ctx.T, ctx.N))

    def act(self, obs):
        gate = self.when[:, obs.t - 1] if self.own_timing else torch.ones(self.ctx.B, dtype=torch.bool, device=self.ctx.device)
        cand = obs.active & gate[:, None]
        order = common.rank_desc(self.who[:, obs.t - 1], cand)
        fits = self.isolate(obs, order, cand)
        first = torch.where(fits, order.argsort(dim=-1), self.ctx.N).argmin(-1)   # best-ranked one that fits
        pick = torch.nn.functional.one_hot(first, self.ctx.N).bool() & fits.any(-1, keepdim=True)
        return core.Actions(isolation=pick)


class RiskAware(_Legacy):
    """risk_aware: R = p * H - lam * (1 - p) * C, hard quarantine of the agents with R above the threshold,
    riskiest first (equal R: higher index first, as sorted(reverse=True))."""
    name = "risk_aware"
    keys = ("lam", "gamma", "dmax")

    def act(self, obs):
        p = obs.monitor
        R, _ = self.risk(obs, p)
        cand = obs.active & (R > common.threshold_of(self.ctx, "risk"))
        iso = self.isolate(obs, common.rank_desc(R, cand, ties_high=True), cand)
        return core.Actions(isolation=iso, statistics={"risk": R})


class RiskAwareTiered(_Legacy):
    """risk_aware_tiered: tier 2 isolates R above the "risk" threshold (tau_quar); tier 1 sanitizes this round's
    outgoing deliveries of (a) an agent with an actual alarm (p above the "monitor" threshold) whose R lies in
    [tau_san, tau_quar], (b) a central agent (C >= tau_central) whose evidence is unconfirmed (alarm, p < tau_confirm),
    (c) a confirmed-risk agent the budget cannot isolate (graceful degradation). Both tiers draw on the single
    intervention budget: hard actions first, then sanitizations riskiest first (equal R: lower index first)."""
    name = "risk_aware_tiered"
    keys = ("lam", "gamma", "dmax", "tau_san", "tau_central", "tau_confirm")

    def reset(self, ctx):
        super().reset(ctx)
        self.c_sanitize = common.cost_of(ctx, core.ActionKind.SANITIZE)

    def act(self, obs):
        p = obs.monitor
        R, C = self.risk(obs, p)
        quar = obs.active & (R > common.threshold_of(self.ctx, "risk"))
        alarm = obs.active & (p > common.threshold_of(self.ctx, "monitor"))
        ambiguous = alarm & ~quar & (R >= self.par["tau_san"])
        shield = alarm & ~quar & (C >= self.par["tau_central"]) & (p < self.par["tau_confirm"])
        iso = self.isolate(obs, common.rank_desc(R, quar, ties_high=True), quar)
        soft = (ambiguous | shield | (quar & ~iso))
        left = obs.remaining_intervention - (iso.double() * self.isolation_cost(obs)).sum(-1)
        cost = self.c_sanitize * obs.attempts.sum(-1).double()
        san = common.fit_in_order(common.rank_desc(R, soft), cost, soft, left)
        return core.Actions(isolation=iso, sanitize=obs.attempts & san[:, :, None],
                            statistics={"risk": R, "monitor": p})
