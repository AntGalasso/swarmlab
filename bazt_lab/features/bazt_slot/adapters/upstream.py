"""BAZT_upstream: the architecture track's `bazt` package driven from our Observation (decision D86).

The pure parts of `bazt` do the thinking: `BudgetedOptimizer.plan` (candidate pricing from `bazt.actions`, the two-phase
greedy, pacing from `bazt.ledger`), `bazt.evidence.InspectionState` for verdicts. This module only translates:

  Observation (monitor p, authentic, budgets)  ->  bazt.optimizer.Situation       one episode at a time
  bazt Plan                                    ->  core.Actions
      INSPECT    -> obs.audit on one sender (verdict: judge score over the calibrated threshold of the statistic `upstream_judge`)
      SANITIZE   -> sanitize, every delivery the node attempts this round
      RESTRICT   -> privilege_degradation (persistent; upstream restricts for ttl_restrict steps, see CONTRACT.md)
      BLOCK_EDGE -> edge_quarantine
      ISOLATE    -> isolation

Episodes are planned in a Python loop over b (the upstream optimizer is scalar). Nothing here draws a random number.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch

import swarmlab as sl
from bazt.actions import ActionKind as UK
from bazt.config import BaztConfig
from bazt.evidence import InspectionState
from bazt.ledger import BudgetLedger
from bazt.optimizer import BudgetedOptimizer, Situation
from bazt_lab.core import public as core
from bazt_lab.features.bazt_slot.private import jitter

K = core.ActionKind
INF = math.inf
STAT = "upstream_judge"
_UPSTREAM_COST = {"cost_inspect": K.INSPECT, "cost_sanitize": K.SANITIZE, "cost_restrict": K.PRIVILEGE_DEGRADATION,
                  "cost_block": K.EDGE_QUARANTINE, "cost_isolate": K.ISOLATION}


def _units(x: float) -> int:
    return max(1, round(x))


class BaztUpstream:
    name = "BAZT_upstream"

    def reset(self, ctx: core.EpisodeContext) -> None:
        self.ctx = ctx
        s = ctx.settings
        self.calibrating = len(ctx.thresholds) == 0
        self.verdict_threshold = float(ctx.thresholds.get(STAT, INF))     # a missing threshold flags nothing
        self.prior = float(core.get(s, "bazt.prior"))
        strip = float(core.get(s, "runtime.sanitize_success"))
        self.sl_cfg = sl.Config(N=ctx.N, T=ctx.T, p_strip=strip)
        costs = {field: _units(float(ctx.costs[kind.value])) for field, kind in _UPSTREAM_COST.items()}
        self.cfgs = []
        for b in range(ctx.B):
            cap = float(ctx.intervention_cap[b])
            self.cfgs.append(BaztConfig(B=0 if math.isinf(cap) else int(cap), unlimited=math.isinf(cap), **costs))
        self.ledgers = [BudgetLedger(c) for c in self.cfgs]
        self.optimizers = [BudgetedOptimizer(c, self.sl_cfg) for c in self.cfgs]
        self.verdicts = [InspectionState(c.inspect_ttl) for c in self.cfgs]
        self.step = 0
        self.quarantined = torch.zeros(ctx.B, ctx.N, ctx.N, dtype=torch.bool)
        self.restricted = torch.zeros(ctx.B, ctx.N, dtype=torch.bool)
        self.exhausted = torch.zeros(ctx.B, dtype=torch.bool)

    # ------------------------------------------------------------------------------------------------ state
    def state_dict(self) -> dict:
        spent = torch.tensor([[led._spent[k] for k in sorted(led._spent)] for led in self.ledgers], dtype=torch.float64)
        kinds = torch.tensor([[led.spent_by_kind[k] for k in UK] for led in self.ledgers], dtype=torch.float64)
        verdicts = [[(n, p, last) for n, (p, last) in v._v.items()] for v in self.verdicts]
        return {"step": self.step, "quarantined": self.quarantined.clone(), "restricted": self.restricted.clone(),
                "exhausted": self.exhausted.clone(), "ledger_spent": spent, "ledger_kinds": kinds, "verdicts": verdicts}

    def load_state_dict(self, state: Mapping) -> None:
        self.step = int(state["step"])
        for key in ("quarantined", "restricted", "exhausted"):
            setattr(self, key, state[key].clone())
        for b, led in enumerate(self.ledgers):
            for k, v in zip(sorted(led._spent), state["ledger_spent"][b].tolist(), strict=True):
                led._spent[k] = v
            for k, v in zip(UK, state["ledger_kinds"][b].tolist(), strict=True):
                led.spent_by_kind[k] = v
            self.verdicts[b]._v = {int(n): (float(p), int(last)) for n, p, last in state["verdicts"][b]}

    # ------------------------------------------------------------------------------------------------ a round
    def decide(self, obs: core.Observation) -> core.Actions:
        ctx = self.ctx
        self.step += 1
        B, N = ctx.B, ctx.N
        gate = ctx.adj if obs.authentic is None else ctx.adj & obs.authentic
        actions = core.Actions(deny=obs.attempts & ~gate)
        attempted = obs.attempts & gate & ~self.quarantined
        self.scores = torch.full((B, N), math.nan, dtype=torch.float64)
        self.noise = (jitter.unit(ctx, STAT, self.step, per_sender=True)[:, :, 0] - 0.5) * jitter.SCALE
        if self.calibrating:                          # collect an unbiased sample of the judge score, act on nothing
            self.scores = obs.audit(attempted.any(-1))
            actions.statistics[STAT] = self._reported(attempted)
            return actions
        masks = {name: torch.zeros(B, N, dtype=torch.bool) for name in ("isolation", "privilege_degradation")}
        edges = {name: torch.zeros(B, N, N, dtype=torch.bool) for name in ("sanitize", "edge_quarantine")}
        for b in range(B):
            self._plan_episode(b, obs, attempted, masks, edges)
        for name, mask in {**masks, **edges}.items():
            if mask.any():
                setattr(actions, name, mask)
        self.restricted |= masks["privilege_degradation"]
        self.quarantined |= edges["edge_quarantine"]
        actions.events += self._exhausted(obs)
        if not self.scores.isnan().all():
            actions.statistics[STAT] = self._reported(attempted)
        return actions

    def _plan_episode(self, b: int, obs: core.Observation, attempted, masks, edges) -> None:
        ctx, t = self.ctx, obs.t
        N, left = ctx.N, obs.T - obs.t + 1
        active = obs.active[b]
        live = ctx.adj[b] & ~self.quarantined[b] & active[:, None] & active[None, :]
        verdicts = self.verdicts[b]
        monitor = None if obs.monitor is None else obs.monitor[b].tolist()
        p = [self.prior if monitor is None else monitor[i] for i in range(N)]
        for i in range(N):
            override = verdicts.override(i, t)
            if override is not None:
                p[i] = override
        out_adj = [live[i].nonzero().flatten().tolist() for i in range(N)]
        deg = [len(o) for o in out_adj]

        def inspect(i: int):
            request = torch.zeros(ctx.B, N, dtype=torch.bool)
            request[b, i] = True
            score = float(obs.audit(request)[b, i])
            if math.isnan(score):
                return None
            self.scores[b, i] = score
            return verdicts.record(i, t, score + float(self.noise[b, i]) > self.verdict_threshold)

        sit = Situation(t=t, T=obs.T, active=tuple(active.nonzero().flatten().tolist()), p_prime=p, deg=deg,
                        maxdeg=max(*deg, 1), privileged=frozenset(), out_adj=out_adj,
                        restricted=frozenset(self.restricted[b].nonzero().flatten().tolist()),
                        verdict_active=verdicts.active_nodes(t), inspect=inspect)
        plan = self.optimizers[b].plan(sit, self.ledgers[b])
        remaining = float(obs.remaining_intervention[b])
        c = ctx.costs
        charged = (ctx.adj[b] & active[:, None] & active[None, :]).sum(1).tolist()
        for d in plan.decisions:
            kind, i, j = d.cand.kind, d.cand.node, d.cand.peer
            if kind == UK.INSPECT:
                continue
            if kind == UK.SANITIZE:
                price = float(attempted[b, i].sum()) * c["sanitize"]
            elif kind == UK.BLOCK_EDGE:
                price = left * c["edge_quarantine"]
            elif kind == UK.RESTRICT:
                price = charged[i] * left * c["privilege_degradation"]
            else:
                price = charged[i] * left * c["isolation"]
            if price > remaining:                                       # the platform would reject the whole round
                continue
            remaining -= price
            if kind == UK.SANITIZE:
                edges["sanitize"][b, i] = attempted[b, i]
            elif kind == UK.BLOCK_EDGE:
                edges["edge_quarantine"][b, i, j] = True
            elif kind == UK.RESTRICT:
                masks["privilege_degradation"][b, i] = True
            else:
                masks["isolation"][b, i] = True

    def _reported(self, attempted: torch.Tensor) -> torch.Tensor:
        """upstream_judge [B, N, N]: the audit score of the sender on each of its deliveries, plus the keyed tie-break jitter
        (one per sender and round, guarded by the smallest gap; the verdict above uses the same draw at the fixed scale)."""
        stat = torch.where(attempted, self.scores[:, :, None], math.nan)
        return jitter.jitter(stat, self.ctx, STAT, self.step, per_sender=True)

    def _exhausted(self, obs: core.Observation) -> list:
        """budget_exhausted once per episode when the intervention budget cannot pay the cheapest action (one unit)."""
        capped = torch.isfinite(obs.remaining_intervention)
        low = torch.tensor([not math.isinf(led.capacity_total) and led.remaining("pool") < 1 for led in self.ledgers])
        now = capped & low & ~self.exhausted
        self.exhausted |= now
        return [{"event": "budget_exhausted", "arm": self.ctx.arm, "b": b, "unit_id": self.ctx.unit_ids[b],
                 "budget": "intervention", "t": obs.t} for b in now.nonzero().flatten().tolist()]


def reference_plan(monitor, cap: float, N: int, T: int, strip: float):
    """The plan `bazt` itself makes at round 1 of a complete graph with one unit per action and no usable audit: what the
    adapter must reproduce (used by the tests, which may not import bazt)."""
    cfg = BaztConfig(unlimited=math.isinf(cap), B=0 if math.isinf(cap) else int(cap), cost_inspect=1, cost_sanitize=1,
                     cost_restrict=1, cost_block=1, cost_isolate=1)
    out_adj = [[j for j in range(N) if j != i] for i in range(N)]
    sit = Situation(t=1, T=T, active=tuple(range(N)), p_prime=list(monitor), deg=[N - 1] * N, maxdeg=N - 1,
                    privileged=frozenset(), out_adj=out_adj, restricted=frozenset(), verdict_active=frozenset(),
                    inspect=lambda i: None)
    return BudgetedOptimizer(cfg, sl.Config(N=N, T=T, p_strip=strip)).plan(sit, BudgetLedger(cfg))


__all__ = ["BaztUpstream", "reference_plan"]
