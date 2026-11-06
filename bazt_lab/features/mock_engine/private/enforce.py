"""Applying a defender's Actions to one round of deliveries (spec section 5, core contract `Actions`).

Order: reinstatement, isolation, edge quarantine, privilege degradation, rate limit, deny, delay, sanitize. A delivery
is charged by the first action that touches it: an action on a delivery an earlier action already removed, or that
cannot happen (an endpoint is isolated), costs nothing and counts for nothing.

Pricing (cost = unit cost of the action's kind x affected deliveries):
  deny, delay, sanitize     one delivery this round; denying what the identity tier already drops (a delivery that
                            failed authentication) is free: those are A's rules
  edge quarantine           the edge's deliveries over the remaining rounds t..T
  isolation                 live out-degree x remaining rounds (the deliveries the agent no longer sends)
  privilege degradation     live out-degree x remaining rounds (every outgoing delivery is sanitized)
  rate limit                (live out-degree - limit)+ x remaining rounds (the deliveries cut every round)
  reinstatement             out-degree after the reinstatement x remaining rounds, only if the agent had a persistent
                            restriction (otherwise a free no-op)
A persistent action on an agent or edge that already carries it is a free no-op. An episode whose actions of the round
cost more than its remaining intervention budget has ALL of them rejected: it keeps its earlier persistent state,
pays nothing and a violation is counted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Callable, Dict, Mapping

import torch

from bazt_lab.core import public as core
from bazt_lab.features.mock_engine.private.bits import band, bandnot, bor

Tensor = torch.Tensor
KINDS = [k.value for k in core.ActionKind]
INDEX = {k: i for i, k in enumerate(KINDS)}
COST_CONDITIONS = ("main", "robustness")
EDGE_FIELDS = ("deny", "sanitize", "delay", "edge_quarantine")
AGENT_FIELDS = ("rate_limit", "privilege_degradation", "isolation", "reinstatement")


def costs_of(settings: Mapping, condition: str) -> Dict[str, float]:
    """Cost per affected delivery of every ActionKind: [costs.<condition>] of the settings."""
    if condition not in COST_CONDITIONS:
        raise ValueError(f"unknown cost condition {condition!r}; the settings define {COST_CONDITIONS}")
    return {k: float(core.get(settings, f"costs.{condition}.{k}")) for k in KINDS}


@dataclass
class Persistent:
    """Persistent defence state of a batch: it lasts until a reinstatement."""
    iso: Tensor        # bool [B, N] isolated: neither sends nor receives
    quar: Tensor       # bool [B, N, N] quarantined edges
    rate: Tensor       # bool [B, N] rate-limited senders
    degr: Tensor       # bool [B, N] degraded senders: every outgoing delivery is sanitized

    @classmethod
    def zeros(cls, B: int, N: int, device=None) -> Persistent:
        agent = lambda: torch.zeros(B, N, dtype=torch.bool, device=device)              # noqa: E731
        return cls(agent(), torch.zeros(B, N, N, dtype=torch.bool, device=device), agent(), agent())

    def clone(self) -> Persistent:
        return Persistent(*(getattr(self, f.name).clone() for f in fields(self)))


@dataclass
class Round:
    state: Persistent       # persistent state after the actions
    arrive: Tensor          # bool [B, N, N] arrives next round (unmodified or sanitized)
    delayed: Tensor         # bool [B, N, N] arrives one round later
    sanitized: Tensor       # bool [B, N, N] modified by a sanitize or by the sender's degradation
    restricted: Tensor      # bool [B, N, N] denied, sanitized, delayed or quarantined (not isolation losses)
    cost: Tensor            # float64 [B] intervention units charged
    rejected: Tensor        # bool [B] the episode's actions were over budget and all rejected
    counts: Tensor          # int64 [B, len(ActionKind)] actions applied this round


def _checked(actions: core.Actions, B: int, N: int) -> Dict[str, Tensor]:
    out = {}
    for name, shape in [(f, (B, N, N)) for f in EDGE_FIELDS] + [(f, (B, N)) for f in AGENT_FIELDS]:
        v = getattr(actions, name)
        if v is None:
            continue
        if not isinstance(v, torch.Tensor) or v.dtype != torch.bool or tuple(v.shape) != shape:
            got = (v.dtype, tuple(v.shape)) if isinstance(v, torch.Tensor) else type(v).__name__
            raise ValueError(f"Actions.{name} must be a bool tensor of shape {shape}, got {got}")
        out[name] = v
    return out


def _total(x: Tensor, dims) -> Tensor:
    return x.sum(dims).to(torch.float64)


def _apply(acts: Mapping[str, Tensor], st: Persistent, live: Tensor, auth: Tensor, adj: Tensor, t: int, T: int,
           c: Mapping[str, float], limit: int, rate_u: Callable[[Tensor], Tensor]) -> Round:
    B, N, _ = live.shape
    rem = float(T - t + 1)
    dev = live.device
    no_agent, no_edge = torch.zeros(B, N, dtype=torch.bool, device=dev), torch.zeros(B, N, N, dtype=torch.bool, device=dev)
    given = lambda name: acts.get(name, no_edge if name in EDGE_FIELDS else no_agent)      # noqa: E731
    counts = torch.zeros(B, len(KINDS), dtype=torch.int64, device=dev)

    # reinstatement lifts isolation, degradation, rate limit and the quarantines of the agent's edges
    had = st.iso | st.rate | st.degr | st.quar.any(2) | st.quar.any(1)
    back = given("reinstatement") & had if "reinstatement" in acts else no_agent
    iso1, rate1, degr1 = st.iso & ~back, st.rate & ~back, st.degr & ~back
    quar1 = st.quar & ~(back[:, :, None] | back[:, None, :])
    out_after = (adj & ~iso1[:, None, :]).sum(2)
    # an isolated agent sends nothing in the round it is reinstated, so it restores one round fewer
    cost = c["reinstatement"] * _total(back * out_after * (rem - st.iso.to(torch.float64)), 1)
    counts[:, INDEX["reinstatement"]] = back.sum(1)

    iso_new = given("isolation") & ~iso1
    if "isolation" in acts:
        # only deliveries no earlier persistent action removed: quarantined edges are gone, a rate-limited sender keeps `limit`
        still = (live & ~iso1[:, None, :] & ~quar1).sum(2)
        still = torch.where(rate1, still.clamp(max=limit), still)
        cost = cost + c["isolation"] * _total(iso_new * still, 1) * rem
    counts[:, INDEX["isolation"]] = iso_new.sum(1)
    iso2 = iso1 | iso_new
    iso_block = bor(iso2[:, :, None], iso2[:, None, :])

    q_new = given("edge_quarantine") & adj & ~quar1 & ~iso_block if "edge_quarantine" in acts else no_edge
    cost = cost + c["edge_quarantine"] * _total(q_new, (1, 2)) * rem
    counts[:, INDEX["edge_quarantine"]] = q_new.sum((1, 2))
    quar2 = quar1 | q_new

    open_ = bandnot(bandnot(live, iso_block), quar2)         # deliveries that survive the persistent cuts so far
    n_open = open_.sum(2)
    surviving = torch.where(rate1, n_open.clamp(max=limit), n_open)    # a rate-limited sender keeps `limit` deliveries
    d_new = given("privilege_degradation") & ~degr1 & ~iso2
    cost = cost + c["privilege_degradation"] * _total(d_new * surviving, 1) * rem
    counts[:, INDEX["privilege_degradation"]] = d_new.sum(1)
    degr2 = degr1 | d_new

    l_new = given("rate_limit") & ~rate1 & ~iso2
    cost = cost + c["rate_limit"] * _total(l_new * (n_open - limit).clamp_min(0), 1) * rem
    counts[:, INDEX["rate_limit"]] = l_new.sum(1)
    rate2 = rate1 | l_new
    rate_drop = torch.zeros_like(live)
    need = band(rate2, n_open > limit)
    rows = need.any(1).nonzero().flatten()
    if len(rows):
        cand = open_[rows] & need[rows][:, :, None]
        rank = rate_u(rows).masked_fill(~cand, math.inf).argsort(-1).argsort(-1)       # lowest draws are kept
        rate_drop[rows] = cand & (rank >= limit)

    after_persistent = bandnot(open_, rate_drop)
    deny = band(given("deny"), after_persistent)
    cost = cost + c["deny"] * _total(band(deny, auth), (1, 2))
    counts[:, INDEX["deny"]] = deny.sum((1, 2))
    after_deny = bandnot(after_persistent, deny)
    delay = band(given("delay"), after_deny)
    cost = cost + c["delay"] * _total(delay, (1, 2))
    counts[:, INDEX["delay"]] = delay.sum((1, 2))
    explicit = band(given("sanitize"), after_deny)
    fresh = bandnot(bandnot(explicit, delay), degr2[:, :, None].expand(B, N, N))        # a delivery already delayed or already sanitized by a degradation is charged once
    cost = cost + c["sanitize"] * _total(fresh, (1, 2))
    counts[:, INDEX["sanitize"]] = fresh.sum((1, 2))
    sanitized = bor(explicit, band(degr2[:, :, None], after_deny))

    restricted = bandnot(live, iso_block)
    restricted = band(restricted, bor(quar2, rate_drop, deny, delay, sanitized))
    return Round(Persistent(iso2, quar2, rate2, degr2), bandnot(after_deny, delay), delay, sanitized, restricted, cost,
                 torch.zeros(B, dtype=torch.bool, device=dev), counts)


def apply_actions(actions: core.Actions, state: Persistent, live: Tensor, authentic: Tensor, adj: Tensor,
                  remaining: Tensor, t: int, T: int, costs: Mapping[str, float], rate_limit: int,
                  rate_u: Callable[[Tensor], Tensor]) -> Round:
    """One round. `live` bool [B, N, N]: deliveries that can happen (base edge, both endpoints active at the start of
    the round); `authentic` bool [B, N, N]; `remaining` float64 [B] intervention units left; `rate_u(rows)` float64
    [len(rows), N, N] draws that rank a rate-limited sender's receivers (the lowest are kept)."""
    B, N, _ = live.shape
    acts = _checked(actions, B, N)
    out = _apply(acts, state, live, authentic, adj, t, T, costs, rate_limit, rate_u)
    over = out.cost > remaining
    if over.any():
        kept = _apply({}, state, live, authentic, adj, t, T, costs, rate_limit, rate_u)
        pick = lambda a, b: torch.where(over.view(-1, *([1] * (a.dim() - 1))), b, a)       # noqa: E731
        merged = {f.name: pick(getattr(out, f.name), getattr(kept, f.name)) for f in fields(Round)
                  if f.name not in ("state", "rejected")}
        state = Persistent(*(pick(getattr(out.state, f.name), getattr(kept.state, f.name)) for f in fields(Persistent)))
        out = Round(state=state, rejected=over, **merged)
    return out
