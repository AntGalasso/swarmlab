"""The mock round loop, vectorised over a batch of episodes (B). The only Python loop is over rounds.

One round t (1-based; swarmlab.run_episode with the spec's changes):
  1. consensus: every active agent moves toward the values delivered to it last round (Metropolis weights of the base
     graph); a delivery that was denied, sanitized, delayed or cut leaves the receiver's weight on itself
  2. each active agent writes: it carries the payload if it is a seed in an attack round, keeps it (p_stay) from its OWN
     previous output, or adopts it after k exposures with probability 1 - (1 - beta)^k. Its memory is the raw output,
     so a denied, sanitized or delayed delivery never changes it (D10). An isolated agent writes and sends nothing and
     keeps its last own output (`carry`); D111: it stays infected while isolated (containment, not cure), so the trace's
     `infected` and `obs.oracle` are that carried state, and a reinstated agent comes back with it
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
from bazt_lab.features.mock_engine.private.bits import band, bandnot, bnot, bor
from bazt_lab.features.mock_engine.private.model import Model
from bazt_lab.features.mock_engine.private.world import World, audit_priority, metropolis, rate_priority

Tensor = torch.Tensor
INSPECT = enforce.INDEX[core.ActionKind.INSPECT.value]
APPLIED = [i for k, i in enforce.INDEX.items() if k != core.ActionKind.INSPECT.value]   # the kinds that intervene (D116)


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
    """bool [B, T] when every unit carries `timing_schedule`, else None. A schedule is T booleans, entry r = round r+1
    (0-based, the `intervened_by_round` column of BAZT's row of the same unit, D116); a list of round numbers or of
    numbers is refused, never read as booleans."""
    if not all("timing_schedule" in u for u in units):
        return None
    rows = [u["timing_schedule"] for u in units]
    for u, r in zip(units, rows, strict=True):
        if not isinstance(r, (list, tuple)) or not all(type(x) is bool for x in r):
            raise ValueError(f"unit {u.get('id')!r}: unit['timing_schedule'] must be a list of T={T} booleans (entry r = "
                             f"round r+1), got {r!r}")
    if any(len(r) != T for r in rows):
        raise ValueError(f"unit['timing_schedule'] must hold T={T} booleans, got lengths {sorted({len(r) for r in rows})}")
    return torch.tensor([list(r) for r in rows], dtype=torch.bool)


def make_context(units: Sequence[Mapping], world: World, arm: str, settings: Mapping,
                 thresholds: Mapping[str, float]) -> core.EpisodeContext:
    ids = tuple(u["id"] for u in units)
    device, dtype = core.device(settings), core.dtype(settings)
    return core.EpisodeContext(
        B=world.B, N=world.N, T=world.T, adj=world.adj.to(device).clone(), sink=world.sink.to(device).clone(),
        audit_cap=caps(units, world.adj, world.T, "audit").to(device),
        intervention_cap=caps(units, world.adj, world.T, "intervention").to(device),
        costs=enforce.costs_of(settings, units[0].get("cost_condition") or "main"), thresholds=dict(thresholds),
        settings=settings, arm=arm, unit_ids=ids, platform="mock", timing_schedule=timing_schedule(units, world.T),
        arm_stream=lambda ns, b: core.stream(ns, arm, ids[b]), device=device, dtype=dtype)


def _zeros(B: int, N: int, device, dtype=torch.bool) -> Tensor:
    return torch.zeros(B, N, N, dtype=dtype, device=device)


# The two numeric kernels of a round are pure functions of tensors, so core.compiled can wrap them (torch.compile on a GPU;
# eager when compute.compile is off or the compiled result disagrees). The defender's call sits between them.
def _consensus(est, a1, e1, s1, p1, a2, e2, s2, p2, Wm, shift3, active):
    """Metropolis consensus on last round's deliveries (a delayed one is used only where no fresh one arrived) and the
    number k of payload-carrying deliveries each agent received. Arrivals are (arrive, payload, estimate, biased)."""
    val = torch.where(a1, e1[:, :, None] + shift3 * s1, e2[:, :, None] + shift3 * s2)      # [b, sender, receiver]
    gain = (Wm * bor(a1, a2) * (val - est[:, None, :])).sum(1)
    k = band(a1, p1).sum(1) + band(a2, p2).sum(1)
    return torch.where(active, est + gain, est), k


def _write(k, carry, active, on, seed, u_adopt, u_stay, decoy, burst, u_decoy, hist, p_mon, mu_mal, mon_z,
           keep, p_stay, p_decoy, mu_dec, sd, a, b, m):
    """What each active agent writes (carries the payload, decoy header) and the cheap monitor on it. `infected` is this
    round's text, so it is False for an isolated agent (no text, nothing sent or audited); `carry` is the payload state
    every agent holds after the round: this round's text when active, else its last own output (D111)."""
    adopt = (k > 0) & (u_adopt < 1.0 - keep ** k.to(u_adopt.dtype))
    infected = band(bor(band(seed, on[:, None]), band(carry, u_stay < p_stay), adopt), active)
    header = band(decoy, burst, u_decoy < p_decoy, active)
    carry = torch.where(active, infected, carry)
    mu = torch.where(infected, mu_mal[:, None].expand_as(mon_z), header.to(mon_z.dtype) * mu_dec)
    hist = torch.where(active[..., None], torch.cat([hist[..., 1:], (mu + sd * mon_z)[..., None]], -1), hist)
    p_mon = torch.where(active, torch.sigmoid(a * hist.sum(-1) / m - b), p_mon)
    return infected, header, carry, hist, p_mon


def simulate(world: World, model: Model, defender, ctx: core.EpisodeContext, needs_truth: bool = False,
             collect: Optional[Callable[[Mapping, Tensor, Tensor], None]] = None,
             units: Optional[Sequence[Mapping]] = None, extras: Optional[dict] = None) -> Tuple[core.EpisodeTrace, Tensor]:
    """Run the batch. -> (EpisodeTrace, estimates float64 [B, T, N]), both on the CPU. Tensors are created on `ctx.device`;
    real-valued state (estimates, monitor, scores) uses `ctx.dtype`; accounting (budgets, costs, spend) is float64 or int64
    whatever the dtype. `collect(statistics, live, active)` receives the decision statistics a defender reports each round
    (threshold calibration). `units` become trace.unit_ids. `extras`, if given, receives the Actions.threshold_restricted
    record (bool [B, T, N, N]) when the defender reported it."""
    dev, dt = ctx.device, ctx.dtype
    world = world.on(dev, dt)
    defender = defender if defender is not None else NullDefender()
    defender.reset(ctx)
    consensus = core.compiled(_consensus, ctx.settings, "mock_engine.consensus")
    write = core.compiled(_write, ctx.settings, "mock_engine.write")
    B, N, T = world.B, world.N, world.T
    adj, costs = world.adj, ctx.costs
    c_inspect = float(costs["inspect"])
    positive = [v for k, v in costs.items() if k != "inspect" and v > 0]
    floor_intervention = min(positive) if positive else 0.0
    Wm = metropolis(adj).to(dt)
    shift3, rewrite3 = world.shift[:, None, None], world.rewrite[:, None, None]
    aitm = world.family == "aitm"
    victim = torch.zeros(B, N, dtype=torch.bool, device=dev)
    hit = (world.victim >= 0).nonzero().flatten()
    victim[hit, world.victim[hit]] = True
    keep = 1.0 - model.beta

    est = world.x.clone()
    active = torch.ones(B, N, dtype=torch.bool, device=dev)
    state = enforce.Persistent.zeros(B, N, dev)
    rem_audit, rem_interv = ctx.audit_cap.clone(), ctx.intervention_cap.clone()
    nothing = (_zeros(B, N, dev), _zeros(B, N, dev), torch.zeros(B, N, dtype=dt, device=dev), _zeros(B, N, dev))
    fresh, late, pending = nothing, nothing, nothing        # arrivals next round / delayed ones due next round / due after
    carry = torch.zeros(B, N, dtype=torch.bool, device=dev)     # the agent's own last output carried the payload
    hist = torch.zeros(B, N, model.m, dtype=dt, device=dev)
    p_mon = torch.full((B, N), 1.0 / (1.0 + math.exp(model.b)), dtype=dt, device=dev)

    infected_t, active_t, correct_t, flagged_t = (torch.zeros(B, T, N, dtype=torch.bool, device=dev) for _ in range(4))
    attempts_t = adj[:, None].expand(B, T, N, N).contiguous()
    delivered_t, restricted_t, payload_t, tampered_t = (torch.zeros(B, T, N, N, dtype=torch.bool, device=dev) for _ in range(4))
    est_t = torch.zeros(B, T, N, dtype=dt, device=dev)
    spend_audit, spend_interv = torch.zeros(B, T, dtype=torch.float64, device=dev), torch.zeros(B, T, dtype=torch.float64, device=dev)
    violations = torch.zeros(B, dtype=torch.int64, device=dev)
    counts = torch.zeros(B, len(enforce.KINDS), dtype=torch.int64, device=dev)
    exhausted = torch.zeros(B, dtype=torch.int64, device=dev)
    intervened_t = torch.zeros(B, T, dtype=torch.bool, device=dev)    # D116: a round with an applied action other than inspect
    events: list = []
    thr_t, thr_given = torch.zeros(B, T, N, N, dtype=torch.bool, device=dev), False     # restricted because a statistic crossed its threshold
    no_rewrite = torch.zeros(B, N, N, dtype=torch.bool, device=dev)
    audit_unbounded = bool(torch.isinf(rem_audit).all())            # an uncapped audit budget never needs a subset

    for t in range(1, T + 1):
        ti = t - 1
        # 1. consensus on last round's deliveries
        k = torch.zeros(B, N, dtype=torch.int64, device=dev)
        if t > 1:
            (a1, p1, e1, s1), (a2, p2, e2, s2) = fresh, late
            est, k = consensus(est, a1, e1, s1, p1, a2, e2, s2, p2, Wm, shift3, active)
        # 2. what each active agent writes, 3. the cheap monitor
        on = world.attack_on[:, ti]
        infected, header, carry, hist, p_mon = write(
            k, carry, active, on, world.seed, world.u_adopt[:, ti], world.u_stay[:, ti], world.decoy, world.burst[:, ti],
            world.u_decoy[:, ti], hist, p_mon, world.mu_mal, world.mon_z[:, ti], keep, model.p_stay, model.p_decoy,
            model.mu_dec, model.sd, model.a, model.b, model.m)
        # 4. the defender
        live = band(adj, active[:, :, None], active[:, None, :])
        rewritten = band(live, victim[:, None, :], world.u_aitm[:, ti][:, :, None] < rewrite3, on[:, None, None]) if aitm else no_rewrite
        authentic = bnot(rewritten) if aitm else bnot(no_rewrite)
        score = world.aud_z[:, ti] + torch.where(infected, world.d_pay[:, None].expand(B, N),
                                                 header.to(dt) * world.d_dec[:, None])
        has_text = band(active, live.any(2))
        paid_round, inspected = torch.zeros(B, dtype=torch.float64, device=dev), torch.zeros(B, dtype=torch.int64, device=dev)
        calls = torch.zeros(B, dtype=torch.int64, device=dev)
        calls_iso = torch.zeros(B, dtype=torch.int64, device=dev)

        def grant(want: Tensor, calls: Tensor, stream: str, t=t, paid_round=paid_round, inspected=inspected) -> Tensor:
            """The requested senders the audit budget pays for: all of them, or a keyed random subset when the episode asks
            for more than it can afford (hard cap, exact). Debits the budget and the round's spend."""
            granted = want
            if c_inspect > 0 and not audit_unbounded:
                afford = torch.floor(rem_audit / c_inspect)
                over = want.sum(1) > afford                        # over-request: a keyed random subset, not agent 0 first
                if bool(over.any()):                               # the one host sync of the audit path
                    prio = audit_priority(world, torch.arange(B, device=dev), t, calls, stream).masked_fill(bnot(want), math.inf)
                    subset = band(want, (prio.argsort(1).argsort(1) < afford[:, None]))
                    granted = torch.where(over[:, None], subset, want)
            spent = granted.sum(1)
            rem_audit.sub_(spent * c_inspect)
            paid_round.add_(spent * c_inspect)
            inspected.add_(spent)
            return granted

        def check(request: Tensor, what: str) -> None:
            if not isinstance(request, torch.Tensor) or request.dtype != torch.bool or tuple(request.shape) != (B, N):
                raise ValueError(f"{what} takes a bool tensor of shape {(B, N)}")

        def audit(request: Tensor, score=score, has_text=has_text, calls=calls) -> Tensor:
            check(request, "audit")
            want = band(request.to(dev), has_text)
            calls.add_(want.any(1))                                 # per episode: only its own non-empty requests count
            granted = grant(want, calls, "audit")
            return torch.where(granted, score, torch.full_like(score, math.nan))

        def audit_isolated(request: Tensor, carried=carry, calls=calls_iso, active=active, ti=ti) -> Tensor:
            """D123: the auditor scores an isolated agent from the payload state it carries (D111), with the same auditor
            model and noise as `audit`; an agent that is not isolated is not granted. Own call count and priority stream,
            so `audit` numbers its calls exactly as before."""
            check(request, "audit_isolated")
            want = band(request.to(dev), bnot(active))
            calls.add_(want.any(1))
            granted = grant(want, calls, "audit_isolated")
            seen = world.aud_z[:, ti] + torch.where(carried, world.d_pay[:, None].expand(B, N), torch.zeros_like(world.aud_z[:, ti]))
            return torch.where(granted, seen, torch.full_like(seen, math.nan))

        open_now = bandnot(live, state.quar)                  # what a defender can still act on: a cut edge is not offered
        obs = core.Observation(
            t=t, T=T, active=active.clone(), attempts=open_now.clone(), monitor=p_mon.clone(),
            remaining_audit=rem_audit.clone(), remaining_intervention=rem_interv.clone(), audit=audit,
            authentic=authentic.clone(), oracle=(lambda inf=carry: inf.clone()) if needs_truth else None,   # D111 state
            audit_isolated=audit_isolated)
        actions = defender.decide(obs)
        if not isinstance(actions, core.Actions):
            raise TypeError(f"{getattr(defender, 'name', type(defender).__name__)}.decide must return Actions, "
                            f"got {type(actions).__name__}")
        events.extend(actions.events)
        thr = actions.threshold_restricted
        out = enforce.apply_actions(actions, state, live, authentic, adj, rem_interv, t, T, costs, model.rate_limit,
                                    lambda rows, t=t: rate_priority(world, rows, t))
        state, rem_interv = out.state, rem_interv - out.cost
        if thr is not None:
            if not isinstance(thr, torch.Tensor) or thr.dtype != torch.bool or tuple(thr.shape) != (B, N, N):
                raise ValueError(f"Actions.threshold_restricted must be a bool tensor of shape {(B, N, N)}")
            thr_t[:, ti], thr_given = thr.to(dev), True        # as reported; the caller counts it with attempts & ~delivered
        if collect is not None and actions.statistics:
            collect(actions.statistics, open_now, active)
        # 5. deliveries; a sanitized copy loses its payload with probability sanitize_success
        stripped = band(out.sanitized, world.u_strip[:, ti][:, :, None] < model.sanitize_success)
        biased = bor(infected[:, :, None], rewritten) if aitm else infected[:, :, None].expand(B, N, N)
        carries = bandnot(biased, stripped)                         # the payload text survives unless stripped
        due = out.delayed if t < T - 1 else torch.zeros_like(out.delayed)             # consumed at round t + 2: lost past the horizon
        late, pending = pending, (due, band(carries, due), est, band(biased, due))
        fresh = (out.arrive, band(carries, out.arrive), est, band(biased, out.arrive))
        # trace
        # D111: an isolated agent that carried the payload stays infected (containment, not cure), as on the LLM platform
        infected_t[:, ti], correct_t[:, ti] = carry, (est - world.mean[:, None]).abs() <= model.eps_correct
        active_t[:, ti], flagged_t[:, ti] = active, bor(state.rate, state.degr, state.iso)   # DURING the round / after its decision
        delivered_t[:, ti], restricted_t[:, ti], tampered_t[:, ti] = bandnot(out.arrive, out.sanitized), out.restricted, rewritten
        payload_t[:, ti], est_t[:, ti] = band(bor(out.arrive, due), carries), est
        spend_audit[:, ti], spend_interv[:, ti] = paid_round, out.cost
        violations += out.rejected
        counts += out.counts
        intervened_t[:, ti] = out.counts[:, APPLIED].sum(1) > 0          # after the budget rule: a rejected round has none
        counts[:, INSPECT] += inspected
        spent_out = (rem_audit < c_inspect) | (rem_interv < floor_intervention)
        exhausted = torch.where((exhausted == 0) & spent_out, torch.full_like(exhausted, t), exhausted)
        active = bnot(state.iso)

    cpu = torch.device("cpu")
    est_final = est_t.to(cpu, torch.float64)
    infected_c = infected_t.to(cpu)
    trace = core.EpisodeTrace(
        unit_ids=tuple(units) if units is not None else ctx.unit_ids, infected=infected_c, attackers=world.attackers.to(cpu).clone(),
        active=active_t.to(cpu), attempts=attempts_t.to(cpu), delivered=delivered_t.to(cpu), restricted=restricted_t.to(cpu),
        payload_delivered=payload_t.to(cpu), correct=correct_t.to(cpu),
        goal_reached=goal.reached(world.family, infected_c, est_final[:, -1].contiguous(), world.sink.to(cpu), world.victim.to(cpu),
                                  world.attackers.to(cpu), world.mean.to(cpu, torch.float64), world.shift.to(cpu, torch.float64),
                                  model.eps_correct),
        sink=world.sink.to(cpu).clone(), spend_audit=spend_audit.to(cpu), spend_intervention=spend_interv.to(cpu),
        exhausted_step=exhausted.to(cpu), violations=violations.to(cpu), action_counts=counts.to(cpu),
        attack_free=world.family == "none",
        meta={"engine": "mock_engine", "dtype": str(dt).removeprefix("torch."), "events": events, "flagged": flagged_t.to(cpu),
              "tampered": tampered_t.to(cpu), "intervened": intervened_t.to(cpu)})
    if extras is not None and thr_given:
        extras["threshold_restricted"] = thr_t.to(cpu)    # [B, T, N, N]: for the closed loop only, never part of the trace or its rows
    return trace, est_final
