"""BAZT (spec 2.5): the A+B+C pipeline plus three components, and the three ablations of spec 2.6.

  enforcement budget   candidates are ranked by value density and bought while the cumulative price of what they waste
                       on a benign target stays within the remaining budget (Hua et al., arXiv 2507.15886, section
                       3.2): the budget lasts the horizon. The hard cap on real cost is separate (`select`).
  topology-aware       what a compromised agent or edge puts at stake over the remaining rounds: truncated Katz reach
  contagion            and the NetShield / NetMelt eigen-drop (`contagion`, `pricing`).
  graduated selection  the eight actions are priced alike, expected loss reduction minus weighted utility loss, and
                       bought by density within both remaining budgets (`pricing`, `select`).

A round: (1) free evidence and the memory of earlier verdicts give each agent a posterior; (2) audits are bought by
value of information; (3) the pipeline sees the bought scores and reports its statistics, a flagged or cleared sender
updates the memory; (4) the posterior is recomputed, the actions are priced and the best are bought. While the
thresholds are missing (a calibration run) BAZT audits every sender, acts on none and only reports statistics.
Nothing here draws a random number.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Dict, Mapping

import torch

from bazt_lab.core import public as core
from bazt_lab.features.bazt_slot.private import jitter, risk, select
from bazt_lab.features.bazt_slot.private import params as P
from bazt_lab.features.bazt_slot.private import pricing as pr
from bazt_lab.features.bazt_slot.private.pipeline import Pipeline

Tensor = torch.Tensor
K = core.ActionKind
STAT_RISK = "bazt_risk"
STAT_JUDGE = "judge"                  # the audit score B thresholds; also the verdict on an isolated agent (D119)
SENDER_LEVEL = ("judge",)             # statistics that are one value per sender: they get one jitter per sender
INF = math.inf
_TINY = 1e-12

# Ascending intrusiveness: argmax takes the first of equal densities, so the lighter action wins a tie.
EDGE_KINDS = (K.DELAY, K.SANITIZE, K.EDGE_QUARANTINE)
AGENT_KINDS = (K.RATE_LIMIT, K.PRIVILEGE_DEGRADATION, K.ISOLATION, K.REINSTATEMENT)
FIELD = {k: k.value for k in EDGE_KINDS + AGENT_KINDS}


@dataclass(frozen=True)
class Variant:
    budget: bool = True        # price benign traffic and pace the budget; False plans as if both budgets were infinite
    topology: bool = True      # contagion over the graph; False values every agent as one unit at stake
    graduated: bool = True     # eight actions; False allows or isolates only


VARIANTS = {
    "BAZT": Variant(),
    "BAZT_no_budget": Variant(budget=False),
    "BAZT_no_topology": Variant(topology=False),
    "BAZT_no_graduated": Variant(graduated=False),
}


@dataclass
class Best:
    """The best valid kind per target (an agent or an edge), as index into the list of kinds it was chosen from."""
    kind: Tensor
    density: Tensor
    cost: Tensor
    recurrence: Tensor
    valid: Tensor


def kinds_of(v: Variant) -> frozenset:
    return frozenset(EDGE_KINDS + AGENT_KINDS) if v.graduated else frozenset({K.ISOLATION})


class Bazt:
    def __init__(self, name: str = "BAZT", pipeline=None) -> None:
        if name not in VARIANTS:
            raise ValueError(f"unknown BAZT variant {name!r}; choose from {sorted(VARIANTS)}")
        self.name, self.variant, self._pipeline = name, VARIANTS[name], pipeline
        self.ctx: core.EpisodeContext | None = None

    # ------------------------------------------------------------------------------------------ Defender
    def reset(self, ctx: core.EpisodeContext) -> None:
        self.ctx, self.p = ctx, P.read(ctx.settings)
        for kind in (K.INSPECT, *(EDGE_KINDS + AGENT_KINDS)):
            pr.unit_cost(ctx.costs, kind)                                 # refuse now, naming the missing key
        if not self.variant.budget and not (torch.isinf(ctx.audit_cap).all() and torch.isinf(ctx.intervention_cap).all()):
            raise ValueError(f"{self.name} plans with infinite budgets and is meant for the uncapped cell; "
                             f"it was given a capped budget (audit {ctx.audit_cap.tolist()}, "
                             f"intervention {ctx.intervention_cap.tolist()})")
        self.pipeline = self._pipeline if self._pipeline is not None else Pipeline()
        self.pipeline.reset(ctx)
        B, N = ctx.B, ctx.N
        self.dev, self.dt = ctx.device, ctx.dtype
        self.step = 0
        self.keys = core.episode_keys(ctx.unit_ids)
        self.memory = torch.zeros(B, N, dtype=self.dt, device=self.dev)              # decaying log-odds of paid verdicts
        # free evidence of each agent's last active round (D119): an isolated agent sends nothing, so it keeps this instead of the prior
        self.held = torch.full((B, N), risk.prior_logodds(self.p.prior), dtype=self.dt, device=self.dev)
        self.quarantined = torch.zeros(B, N, N, dtype=torch.bool, device=self.dev)
        self.degraded = torch.zeros(B, N, dtype=torch.bool, device=self.dev)
        self.limited = torch.zeros(B, N, dtype=torch.bool, device=self.dev)
        self.exhausted = torch.zeros(B, 2, dtype=torch.bool, device=self.dev)              # [audit, intervention]
        self.pair: tuple | None = None                                    # eigenvectors, warm start of the next round
        # float64 [B] Katz decay of each episode, katz_decay / rho(ctx.adj) (D115); a function of ctx, so not state
        self.decay = pr.katz_decay(ctx.adj, self.p) if self.variant.topology else None

    def state_dict(self) -> dict:
        state = {"step": self.step, "memory": self.memory.clone(), "held": self.held.clone(), "quarantined": self.quarantined.clone(),
                 "degraded": self.degraded.clone(), "limited": self.limited.clone(),
                 "exhausted": self.exhausted.clone(), "pipeline": self.pipeline.state_dict()}
        if self.pair is not None:
            state["eigen_u"], state["eigen_v"] = self.pair[0].clone(), self.pair[1].clone()
        return state

    def load_state_dict(self, state: Mapping) -> None:
        self.step = int(state["step"])
        for key in ("memory", "held", "quarantined", "degraded", "limited", "exhausted"):
            setattr(self, key, state[key].clone())
        self.pair = (state["eigen_u"].clone(), state["eigen_v"].clone()) if "eigen_u" in state else None
        self.pipeline.load_state_dict(state["pipeline"])

    # -------------------------------------------------------------------------------------------- a round
    def decide(self, obs: core.Observation) -> core.Actions:
        ctx, p, v = self.ctx, self.p, self.variant
        self.step += 1
        B, N = ctx.B, ctx.N
        active = obs.active
        gate = ctx.adj if obs.authentic is None else ctx.adj & obs.authentic
        structural = ctx.adj & active[:, :, None] & active[:, None, :] & ~self.quarantined
        attempted = obs.attempts & gate & structural                       # the deliveries this round decides on
        senders = attempted.any(-1)
        calibrating = len(ctx.thresholds) == 0
        threshold = float(ctx.thresholds.get(STAT_RISK, INF))
        left = float(obs.T - obs.t + 1)
        inf_budget = torch.full((B,), INF, dtype=torch.float64, device=self.dev)
        audit_budget = obs.remaining_audit if v.budget else inf_budget
        action_budget = obs.remaining_intervention if v.budget else inf_budget

        free = risk.free_logodds(obs.monitor, active, p.prior, self.dt, self.held)
        self.held = free.to(self.dt).clone()           # raw evidence, before the jitter: what an isolated agent keeps
        if obs.monitor is not None:                    # the statistic is the free evidence plus its keyed tie-break jitter
            free = jitter.jitter(free[:, :, None].expand(B, N, N), ctx, STAT_RISK, self.step, per_sender=True, keys=self.keys)[:, :, 0]

        # (2) audits, bought by value of information
        def world(r: Tensor) -> pr.World:
            return pr.World(obs.t, obs.T, ctx.adj, structural, self.quarantined, active, ~active, self.degraded,
                            self.limited, r, ctx.sink, self.decay)

        r_pre = risk.sigmoid(free + p.memory_decay * self.memory)
        w_pre = world(r_pre)
        rc_pre = pr.reach(w_pre, pr.transmission(w_pre, p), p, v.topology, structure=False)
        voi = pr.inspect_value(w_pre, p, rc_pre)
        c_audit = pr.unit_cost(ctx.costs, K.INSPECT)
        audit_cost = torch.full((B, N), c_audit, dtype=torch.float64, device=self.dev)
        if calibrating:                                                    # collect an unbiased sample of every statistic
            audit_mask = select.choose(torch.zeros(B, N, dtype=self.dt, device=self.dev), audit_cost, senders, audit_budget)
        else:
            pace = ({"p_right": r_pre, "recurrence": torch.full_like(r_pre, left), "price_limit": audit_budget}
                    if v.budget else {})
            audit_mask = select.choose(voi / max(c_audit, _TINY), audit_cost, senders & (voi > 0), audit_budget, **pace)
        nan = torch.full((B, N), math.nan, dtype=self.dt, device=self.dev)
        scores = obs.audit(audit_mask).masked_fill(~audit_mask, math.nan) if audit_mask.any() else nan
        spent_audit = audit_mask.sum(-1).to(torch.float64) * c_audit

        # (2b) D119: an isolated agent sends nothing, so only an audit paid from what the senders left can clear it. A
        # platform that cannot audit an agent (no text) answers NaN and charges nothing: it then stays isolated.
        judge_at = float(ctx.thresholds.get(STAT_JUDGE, INF))
        iso_mask = torch.zeros(B, N, dtype=torch.bool, device=self.dev)
        if not calibrating and K.REINSTATEMENT in kinds_of(v) and math.isfinite(judge_at) and not active.all():
            value = p.q_inspect * (1.0 - r_pre) * p.utility_weight * p.rho["isolation"] * pr.centrality(w_pre)
            left_audit = audit_budget - spent_audit
            pace = ({"p_right": 1.0 - r_pre, "recurrence": torch.full_like(r_pre, left), "price_limit": left_audit}
                    if v.budget else {})
            iso_mask = select.choose(value / max(c_audit, _TINY), audit_cost, ~active & (value > 0), left_audit, **pace)
        # D123: through the platform's own channel for isolated agents (the plain audit never grants one); a platform
        # without it keeps the plain audit
        ask = obs.audit_isolated if obs.audit_isolated is not None else obs.audit
        iso_scores = ask(iso_mask).masked_fill(~iso_mask, math.nan) if iso_mask.any() else nan
        iso_audited = ~iso_scores.isnan()
        scores = torch.where(iso_audited, iso_scores, scores)
        spent_audit = spent_audit + iso_audited.sum(-1).to(torch.float64) * c_audit

        # (3) the pipeline sees what was bought; a flagged or cleared sender updates the memory
        result = self.pipeline.step(obs, scores)
        flagged = torch.zeros(B, N, dtype=torch.bool, device=self.dev)
        statistics = {name: jitter.jitter(stat, ctx, name, self.step, per_sender=name in SENDER_LEVEL, keys=self.keys) for name, stat in result.statistics.items()}
        for name, stat in statistics.items():
            hit = stat > float(ctx.thresholds.get(name, INF))              # NaN never exceeds a threshold
            flagged |= (hit & attempted).any(-1) if hit.dim() == 3 else hit
        flagged |= iso_audited & (scores > judge_at)                       # B's judge rule on the score of an isolated agent
        self.memory = risk.remember(self.memory, flagged, ~scores.isnan(), p.memory_decay, p.flag_logodds,
                                    p.clear_logodds)
        cleared_iso = risk.cleared(iso_audited, flagged)

        # (4) the posterior, the priced actions, and what the budgets buy
        r = risk.sigmoid(free + self.memory)
        w = world(r)
        A = pr.transmission(w, p)
        rc = pr.reach(w, A, p, v.topology, self.pair)
        self.pair = rc.pair
        elig = active & risk.eligible(free, self.memory, threshold, p.clear_logodds)
        kinds = kinds_of(v)
        priced = pr.price(w, p, ctx.costs, elig, kinds, v.topology, rc)
        if K.REINSTATEMENT in priced:                  # D119: an isolated agent is lifted only in a round an audit cleared it
            back = priced[K.REINSTATEMENT]
            priced[K.REINSTATEMENT] = dataclasses.replace(back, valid=back.valid & (active | cleared_iso))

        agent_kinds = [k for k in AGENT_KINDS if k in kinds]
        edge_kinds = [k for k in EDGE_KINDS if k in kinds]
        agent = self._best(priced, agent_kinds, B, N, None, self.dev, self.dt)
        edge = self._best(priced, edge_kinds, B, N, attempted, self.dev, self.dt)
        # P(the purchase is warranted): the target is compromised; for a reinstatement, that it is benign.
        agent_right = r
        if K.REINSTATEMENT in agent_kinds:
            agent_right = torch.where(agent.kind == agent_kinds.index(K.REINSTATEMENT), 1.0 - r, r)
        right = torch.cat([agent_right, r[:, :, None].expand(B, N, N).reshape(B, -1)], 1)

        def both(f):
            return torch.cat([f(agent).reshape(B, -1), f(edge).reshape(B, -1)], 1)

        density, cost, valid = both(lambda x: x.density), both(lambda x: x.cost), both(lambda x: x.valid)
        pace = ({"p_right": right, "recurrence": both(lambda x: x.recurrence), "price_limit": action_budget}
                if v.budget else {})
        chosen = select.choose(density, cost, valid, action_budget, **pace)
        pick_agent, pick_edge = chosen[:, :N], chosen[:, N:].reshape(B, N, N)
        if K.ISOLATION in agent_kinds:                 # an isolated agent neither sends nor receives: no edge action on it
            iso = pick_agent & (agent.kind == agent_kinds.index(K.ISOLATION))
            pick_edge = pick_edge & ~(iso[:, :, None] | iso[:, None, :])
            chosen = torch.cat([pick_agent, pick_edge.reshape(B, -1)], 1)

        actions = core.Actions(deny=result.deny_a)
        for i, kind in enumerate(agent_kinds):
            mask = pick_agent & (agent.kind == i)
            if mask.any():
                setattr(actions, FIELD[kind], mask)
        for i, kind in enumerate(edge_kinds):
            mask = pick_edge & (edge.kind == i)
            if mask.any():
                setattr(actions, FIELD[kind], mask)
        self._remember_actions(actions, active)
        actions.threshold_restricted = self._threshold_restricted(actions, active)

        spent_action = (chosen * cost).sum(-1)
        actions.events += self._exhausted(obs, spent_audit, spent_action)
        actions.statistics = statistics
        if obs.monitor is not None:
            actions.statistics[STAT_RISK] = torch.where(attempted, free[:, :, None], math.nan)
        return actions

    # ------------------------------------------------------------------------------------------ helpers
    @staticmethod
    def _best(priced: Dict[core.ActionKind, pr.Priced], kinds, B: int, N: int, mask: Tensor | None, device=None,
              dtype=torch.float64) -> Best:
        """Per target the kind with the highest value density among the valid ones (ties: the lighter action). A running
        maximum over the kinds (strictly greater replaces, so the first kind wins a tie, as argmax does): no [K, B, N, N]
        stack and no reduction over its leading dimension."""
        shape = (B, N) if mask is None else (B, N, N)
        if not kinds:
            z = torch.zeros(shape, dtype=dtype, device=device)
            return Best(torch.zeros(shape, dtype=torch.int64, device=device), z, z, z,
                        torch.zeros(shape, dtype=torch.bool, device=device))
        first = priced[kinds[0]]
        any_valid = first.valid if mask is None else first.valid & mask
        best = torch.where(any_valid, first.net / first.cost.clamp_min(_TINY), torch.full_like(first.net, -INF))
        idx = torch.zeros(shape, dtype=torch.int64, device=first.net.device)
        cost, recurrence = first.cost, first.recurrence
        for i, k in enumerate(kinds[1:], 1):
            c = priced[k]
            valid = c.valid if mask is None else c.valid & mask
            dens = torch.where(valid, c.net / c.cost.clamp_min(_TINY), torch.full_like(c.net, -INF))
            better = dens > best
            best = torch.where(better, dens, best)
            idx = torch.where(better, i, idx)
            cost, recurrence = torch.where(better, c.cost, cost), torch.where(better, c.recurrence, recurrence)
            any_valid = any_valid | valid
        return Best(idx, best, cost, recurrence, any_valid)

    def _remember_actions(self, actions: core.Actions, active: Tensor) -> None:
        """Mirror the engine's persistent state, in the engine's order: reinstatement lifts every restriction of the agent
        (quarantines on its edges in either direction), then isolation, then the new quarantines, degradations and rate
        limits, none of which applies to an agent isolated in this same Actions or to an edge of one."""
        B, N = active.shape
        no = torch.zeros(B, N, dtype=torch.bool, device=self.dev)
        back = no if actions.reinstatement is None else actions.reinstatement
        iso = ~active & ~back
        if actions.isolation is not None:
            iso = iso | actions.isolation
        self.degraded &= ~back
        self.limited &= ~back
        self.quarantined &= ~(back[:, :, None] | back[:, None, :])
        if actions.edge_quarantine is not None:
            self.quarantined |= actions.edge_quarantine & self.ctx.adj & ~(iso[:, :, None] | iso[:, None, :])
        if actions.privilege_degradation is not None:
            self.degraded |= actions.privilege_degradation & ~iso
        if actions.rate_limit is not None:
            self.limited |= actions.rate_limit & ~iso

    def _threshold_restricted(self, actions: core.Actions, active: Tensor) -> Tensor:
        """bool [B, N, N]: the deliveries restricted this round because a calibrated statistic put their sender over its
        threshold (D108). Every priced restriction of BAZT is gated by eligibility, which is the `bazt_risk` threshold or a
        positive verdict of the pipeline's `judge` / `rl_minus_ts`: this round's sanitize and delay, the edge quarantines in
        force, every delivery of a degraded sender and of an isolated one. Not in it: A's denies, the persistent rate limit,
        nothing that is not an action of the arm. Call after `_remember_actions` (the state is already updated)."""
        B, N = active.shape
        iso = ~active
        back = actions.reinstatement
        if back is not None:
            iso = iso & ~back
        if actions.isolation is not None:
            iso = iso | actions.isolation
        mask = self.quarantined | self.degraded[:, :, None] | iso[:, :, None] | iso[:, None, :]
        for field in (actions.sanitize, actions.delay):
            if field is not None:
                mask = mask | field
        return mask & self.ctx.adj

    def _exhausted(self, obs: core.Observation, spent_audit: Tensor, spent_action: Tensor) -> list:
        """budget_exhausted, once per budget and episode, when the budget cannot pay the cheapest thing it buys."""
        costs = self.ctx.costs
        unit_audit = pr.unit_cost(costs, K.INSPECT)
        unit_action = min(pr.unit_cost(costs, k) for k in EDGE_KINDS + AGENT_KINDS if pr.unit_cost(costs, k) > 0)
        left = torch.stack([obs.remaining_audit - spent_audit, obs.remaining_intervention - spent_action], 1)
        capped = torch.isfinite(torch.stack([obs.remaining_audit, obs.remaining_intervention], 1))
        now = capped & (left < torch.tensor([unit_audit, unit_action], dtype=torch.float64, device=self.dev)) & ~self.exhausted
        self.exhausted |= now
        return [{"event": "budget_exhausted", "arm": self.ctx.arm, "b": b, "unit_id": self.ctx.unit_ids[b],
                 "budget": ("audit", "intervention")[k], "t": obs.t} for b, k in now.nonzero().tolist()]
