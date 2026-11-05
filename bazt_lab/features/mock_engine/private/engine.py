"""The mock round loop, vectorised over a batch of episodes (B). The only Python loop is over rounds.

One round t (1-based; swarmlab.run_episode with the spec's changes):
  1. consensus: every active agent moves toward the values delivered to it last round (Metropolis weights of the base
     graph); a delivery that was denied, sanitized, delayed or cut leaves the receiver's weight on itself
  2. each active agent writes: it carries the payload if it is a seed in an attack round, keeps it (p_stay) from its OWN
     previous output, or adopts it after k exposures with probability 1 - (1 - beta)^k. Its memory is the raw output,
     so a denied, sanitized or delayed delivery never changes it (D10)
  3. cheap monitor: score of the written text + noise, rolling mean over m rounds, p = sigmoid(a * rolling - b)
  4. the defender sees an Observation and returns Actions; they are applied and charged (enforce.py)
  5. deliveries are made: unmodified, sanitized (payload stripped with probability sanitize_success), delayed or lost

Everything that is random comes pre-drawn from world.py, so the loop consumes no random numbers.
"""
from __future__ import annotations

import math
from typing import Callable, Mapping, Optional, Sequence, Tuple

import torch

from bazt_lab.core import public as core
from bazt_lab.features.mock_engine.private import enforce, goal
from bazt_lab.features.mock_engine.private.model import Model
from bazt_lab.features.mock_engine.private.world import World, audit_priority, metropolis, rate_priority

Tensor = torch.Tensor
INSPECT = enforce.INDEX[core.ActionKind.INSPECT.value]


class NullDefender:
    """No defence at all: used for the clean reference run and for fidelity checks."""
    name = "no_defense"

    def reset(self, ctx) -> None:
        pass

    def decide(self, obs) -> core.Actions:
        return core.Actions()

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state) -> None:
        pass


def fraction(unit: Mapping, field: str) -> float:
    """A budget of a unit: 'uncapped' (inf) or a fraction in (0, 1] of all edge deliveries of the episode."""
    v = unit[field]
    if v == "uncapped":
        return math.inf
    if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v <= 1:
        return float(v)
    raise ValueError(f"unit[{field!r}] must be 'uncapped' or a fraction in (0, 1], got {v!r}")


def caps(units: Sequence[Mapping], adj: Tensor, T: int, field: str) -> Tensor:
    """float64 [B]: a budget is a fraction of all edge deliveries of the episode (directed edges x rounds), or inf."""
    deliveries = adj.sum((1, 2)).to(torch.float64) * T
    share = torch.tensor([fraction(u, field) for u in units], dtype=torch.float64)
    return torch.where(torch.isinf(share), share, share * deliveries)


def timing_schedule(units: Sequence[Mapping], T: int) -> Optional[Tensor]:
    """bool [B, T] when every unit carries `timing_schedule` (T bools, 0-based rounds), else None."""
    if not all("timing_schedule" in u for u in units):
        return None
    rows = [list(u["timing_schedule"]) for u in units]
    if any(len(r) != T for r in rows):
        raise ValueError(f"unit['timing_schedule'] must hold T={T} booleans, got lengths {sorted({len(r) for r in rows})}")
    return torch.tensor(rows, dtype=torch.bool)


def make_context(units: Sequence[Mapping], world: World, arm: str, settings: Mapping,
                 thresholds: Mapping[str, float]) -> core.EpisodeContext:
    ids = tuple(u["id"] for u in units)
    return core.EpisodeContext(
        B=world.B, N=world.N, T=world.T, adj=world.adj.clone(), sink=world.sink.clone(),
        audit_cap=caps(units, world.adj, world.T, "audit"), intervention_cap=caps(units, world.adj, world.T, "intervention"),
        costs=enforce.costs_of(settings, units[0].get("cost_condition") or "main"), thresholds=dict(thresholds),
        settings=settings, arm=arm, unit_ids=ids, platform="mock", timing_schedule=timing_schedule(units, world.T), arm_stream=lambda ns, b: core.stream(ns, arm, ids[b]))


def _zeros(B: int, N: int, dtype=torch.bool) -> Tensor:
    return torch.zeros(B, N, N, dtype=dtype)


def simulate(world: World, model: Model, defender, ctx: core.EpisodeContext, needs_truth: bool = False,
             collect: Optional[Callable[[Mapping, Tensor, Tensor], None]] = None,
             units: Optional[Sequence[Mapping]] = None) -> Tuple[core.EpisodeTrace, Tensor]:
    """Run the batch. -> (EpisodeTrace, estimates float64 [B, T, N]). `collect(statistics, live, active)` receives the
    decision statistics a defender reports each round (threshold calibration). `units` become trace.unit_ids."""
    defender = defender if defender is not None else NullDefender()
    defender.reset(ctx)
    B, N, T = world.B, world.N, world.T
    adj, costs = world.adj, ctx.costs
    c_inspect = float(costs["inspect"])
    positive = [v for k, v in costs.items() if k != "inspect" and v > 0]
    floor_intervention = min(positive) if positive else 0.0
    Wm = metropolis(adj)
    shift3, rewrite3 = world.shift[:, None, None], world.rewrite[:, None, None]
    victim = torch.zeros(B, N, dtype=torch.bool)
    hit = (world.victim >= 0).nonzero().flatten()
    victim[hit, world.victim[hit]] = True
    keep = 1.0 - model.beta

    est = world.x.clone()
    active = torch.ones(B, N, dtype=torch.bool)
    state = enforce.Persistent.zeros(B, N)
    rem_audit, rem_interv = ctx.audit_cap.clone(), ctx.intervention_cap.clone()
    nothing = (_zeros(B, N), _zeros(B, N), torch.zeros(B, N, dtype=torch.float64), _zeros(B, N))
    fresh, late, pending = nothing, nothing, nothing        # arrivals next round / delayed ones due next round / due after
    carry = torch.zeros(B, N, dtype=torch.bool)             # the agent's own last output carried the payload
    hist = torch.zeros(B, N, model.m, dtype=torch.float64)
    p_mon = torch.full((B, N), 1.0 / (1.0 + math.exp(model.b)), dtype=torch.float64)

    infected_t, active_t, correct_t, flagged_t = (torch.zeros(B, T, N, dtype=torch.bool) for _ in range(4))
    attempts_t = adj[:, None].expand(B, T, N, N).contiguous()
    delivered_t, restricted_t, payload_t, tampered_t = (torch.zeros(B, T, N, N, dtype=torch.bool) for _ in range(4))
    est_t = torch.zeros(B, T, N, dtype=torch.float64)
    spend_audit, spend_interv = torch.zeros(B, T, dtype=torch.float64), torch.zeros(B, T, dtype=torch.float64)
    violations = torch.zeros(B, dtype=torch.int64)
    counts = torch.zeros(B, len(enforce.KINDS), dtype=torch.int64)
    exhausted = torch.zeros(B, dtype=torch.int64)
    events: list = []

    for t in range(1, T + 1):
        ti = t - 1
        # 1. consensus on last round's deliveries (a delayed one is used only where no fresh one arrived)
        k = torch.zeros(B, N, dtype=torch.int64)
        if t > 1:
            (a1, p1, e1, s1), (a2, p2, e2, s2) = fresh, late
            val = torch.where(a1, e1[:, :, None] + shift3 * s1, e2[:, :, None] + shift3 * s2)      # [b, sender, receiver]
            gain = (Wm * (a1 | a2) * (val - est[:, None, :])).sum(1)
            est = torch.where(active, est + gain, est)
            k = (a1 & p1).sum(1) + (a2 & p2).sum(1)
        # 2. what each active agent writes
        on = world.attack_on[:, ti]
        adopt = (k > 0) & (world.u_adopt[:, ti] < 1.0 - keep ** k.to(torch.float64))
        infected = ((world.seed & on[:, None]) | (carry & (world.u_stay[:, ti] < model.p_stay)) | adopt) & active
        header = world.decoy & world.burst[:, ti] & (world.u_decoy[:, ti] < model.p_decoy) & active
        carry = torch.where(active, infected, carry)
        # 3. cheap monitor
        mu = torch.where(infected, world.mu_mal[:, None].expand(B, N), header.to(torch.float64) * model.mu_dec)
        hist = torch.where(active[..., None], torch.cat([hist[..., 1:], (mu + model.sd * world.mon_z[:, ti])[..., None]], -1), hist)
        p_mon = torch.where(active, torch.sigmoid(model.a * hist.sum(-1) / model.m - model.b), p_mon)
        # 4. the defender
        live = adj & active[:, :, None] & active[:, None, :]
        rewritten = live & victim[:, None, :] & (world.u_aitm[:, ti][:, :, None] < rewrite3) & on[:, None, None]
        authentic = ~rewritten
        score = world.aud_z[:, ti] + torch.where(infected, world.d_pay[:, None].expand(B, N),
                                                 header.to(torch.float64) * world.d_dec[:, None])
        has_text = active & live.any(2)
        paid_round, inspected, calls = torch.zeros(B, dtype=torch.float64), torch.zeros(B, dtype=torch.int64), torch.zeros(B, dtype=torch.int64)

        def audit(request: Tensor, score=score, has_text=has_text, t=t, calls=calls, paid_round=paid_round,
                  inspected=inspected) -> Tensor:
            if not isinstance(request, torch.Tensor) or request.dtype != torch.bool or tuple(request.shape) != (B, N):
                raise ValueError(f"audit takes a bool tensor of shape {(B, N)}")
            want = request & has_text
            calls.add_(want.any(1))                                 # per episode: only its own non-empty requests count
            granted = want
            if c_inspect > 0:
                afford = torch.floor(rem_audit / c_inspect)
                granted = want.clone()
                rows = (want.sum(1) > afford).nonzero().flatten()      # over-request: a keyed random subset, not agent 0 first
                if len(rows):
                    prio = audit_priority(world, rows, t, calls[rows]).masked_fill(~want[rows], math.inf)
                    granted[rows] = want[rows] & (prio.argsort(1).argsort(1) < afford[rows][:, None])
            spent = granted.sum(1)
            rem_audit.sub_(spent * c_inspect)
            paid_round.add_(spent * c_inspect)
            inspected.add_(spent)
            return torch.where(granted, score, torch.full_like(score, math.nan))

        open_now = live & ~state.quar                         # what a defender can still act on: a cut edge is not offered
        obs = core.Observation(
            t=t, T=T, active=active.clone(), attempts=open_now.clone(), monitor=p_mon.clone(),
            remaining_audit=rem_audit.clone(), remaining_intervention=rem_interv.clone(), audit=audit,
            authentic=authentic.clone(), oracle=(lambda inf=infected: inf.clone()) if needs_truth else None)
        actions = defender.decide(obs)
        if not isinstance(actions, core.Actions):
            raise TypeError(f"{getattr(defender, 'name', type(defender).__name__)}.decide must return Actions, "
                            f"got {type(actions).__name__}")
        events.extend(actions.events)
        out = enforce.apply_actions(actions, state, live, authentic, adj, rem_interv, t, T, costs, model.rate_limit,
                                    lambda rows, t=t: rate_priority(world, rows, t))
        state, rem_interv = out.state, rem_interv - out.cost
        if collect is not None and actions.statistics:
            collect(actions.statistics, open_now, active)
        # 5. deliveries; a sanitized copy loses its payload with probability sanitize_success
        stripped = out.sanitized & (world.u_strip[:, ti][:, :, None] < model.sanitize_success)
        biased = infected[:, :, None] | rewritten                   # the reported number is shifted (sanitizing keeps the number)
        carries = biased & ~stripped                                # the payload text survives unless stripped
        due = out.delayed if t < T else torch.zeros_like(out.delayed)                 # a delay past the horizon is lost
        late, pending = pending, (due, carries & due, est, biased & due)
        fresh = (out.arrive, carries & out.arrive, est, biased & out.arrive)
        # trace
        infected_t[:, ti], correct_t[:, ti] = infected, (est - world.mean[:, None]).abs() <= model.eps_correct
        active_t[:, ti], flagged_t[:, ti] = active & ~state.iso, state.rate | state.degr
        delivered_t[:, ti], restricted_t[:, ti], tampered_t[:, ti] = out.arrive & ~out.sanitized, out.restricted, rewritten
        payload_t[:, ti], est_t[:, ti] = (out.arrive | due) & carries, est
        spend_audit[:, ti], spend_interv[:, ti] = paid_round, out.cost
        violations += out.rejected
        counts += out.counts
        counts[:, INSPECT] += inspected
        spent_out = (rem_audit < c_inspect) | (rem_interv < floor_intervention)
        exhausted = torch.where((exhausted == 0) & spent_out, torch.full_like(exhausted, t), exhausted)
        active = ~state.iso

    trace = core.EpisodeTrace(
        unit_ids=tuple(units) if units is not None else ctx.unit_ids, infected=infected_t, attackers=world.attackers.clone(),
        active=active_t, attempts=attempts_t, delivered=delivered_t, restricted=restricted_t, payload_delivered=payload_t,
        correct=correct_t, goal_reached=goal.reached(world.family, infected_t, est, world.sink, world.victim, world.attackers,
                                                     world.mean, world.shift, model.eps_correct),
        sink=world.sink.clone(), spend_audit=spend_audit, spend_intervention=spend_interv, exhausted_step=exhausted,
        violations=violations, action_counts=counts, attack_free=world.family == "none",
        meta={"engine": "mock_engine", "events": events, "flagged": flagged_t, "tampered": tampered_t})
    return trace, est_t
