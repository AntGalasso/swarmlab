"""Applying a defender's Actions to one round of one conversation (spec section 5, core contract `Actions`).

Order: reinstatement, isolation, edge quarantine, privilege degradation, rate limit, deny, delay, sanitize. A delivery
is charged by the first action that touches it. Pricing, cost = unit cost of the action's kind x affected deliveries:
  deny, delay, sanitize     one delivery this round; denying a delivery that failed authentication is free (A's rule)
  edge quarantine           the edge's deliveries over the remaining rounds t..T
  isolation, degradation    live out-degree x remaining rounds
  rate limit                (live out-degree - limit)+ x remaining rounds
  reinstatement             out-degree after it x remaining rounds, only if the agent had a persistent restriction
A persistent action on something that already carries it is a free no-op. If the round's actions cost more than the
remaining intervention budget, ALL of them are rejected: the earlier persistent state stays, nothing is paid and a
violation is counted. One episode (B = 1): masks here are [N, N] and [N]; Actions tensors are [1, N, N] and [1, N].
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping

import torch

from bazt_lab.core import public as core

Tensor = torch.Tensor
KINDS = [k.value for k in core.ActionKind]
INDEX = {k: i for i, k in enumerate(KINDS)}
EDGE_FIELDS = ("deny", "sanitize", "delay", "edge_quarantine")
AGENT_FIELDS = ("rate_limit", "privilege_degradation", "isolation", "reinstatement")


@dataclass
class State:
    """Persistent defence state of a seed run, until a reinstatement."""
    iso: Tensor        # bool [N] isolated: neither sends nor receives
    quar: Tensor       # bool [N, N] quarantined edges
    rate: Tensor       # bool [N] rate-limited senders
    degr: Tensor       # bool [N] degraded senders: every outgoing delivery is sanitized

    @classmethod
    def zeros(cls, n: int) -> State:
        agent = lambda: torch.zeros(n, dtype=torch.bool)               # noqa: E731
        return cls(agent(), torch.zeros(n, n, dtype=torch.bool), agent(), agent())

    def clone(self) -> State:
        return State(self.iso.clone(), self.quar.clone(), self.rate.clone(), self.degr.clone())

    def as_dict(self) -> dict:
        return {"iso": self.iso, "quar": self.quar, "rate": self.rate, "degr": self.degr}

    @classmethod
    def from_dict(cls, d: Mapping) -> State:
        return cls(d["iso"].clone(), d["quar"].clone(), d["rate"].clone(), d["degr"].clone())


@dataclass
class Round:
    state: State
    arrive: Tensor          # bool [N, N] arrives next update (unmodified or sanitized)
    delayed: Tensor         # bool [N, N] arrives one round later
    sanitized: Tensor       # bool [N, N] modified by a sanitize or by the sender's degradation
    restricted: Tensor      # bool [N, N] denied, sanitized, delayed, quarantined or rate-dropped (not isolation)
    cost: float             # intervention units charged
    rejected: bool          # the actions were over budget and all rejected
    counts: Tensor          # int64 [len(ActionKind)] actions applied


def costs_of(settings: Mapping, condition: str) -> dict:
    """Cost per affected delivery of every ActionKind: [costs.<condition>] of the settings."""
    if condition not in ("main", "robustness"):
        raise ValueError(f"unknown cost condition {condition!r}; the settings define main and robustness")
    return {k: float(core.get(settings, f"costs.{condition}.{k}")) for k in KINDS}


def _checked(actions: core.Actions, n: int) -> dict:
    out = {}
    for name, shape in [(f, (1, n, n)) for f in EDGE_FIELDS] + [(f, (1, n)) for f in AGENT_FIELDS]:
        v = getattr(actions, name)
        if v is None:
            continue
        if not isinstance(v, torch.Tensor) or v.dtype != torch.bool or tuple(v.shape) != shape:
            got = (v.dtype, tuple(v.shape)) if isinstance(v, torch.Tensor) else type(v).__name__
            raise ValueError(f"Actions.{name} must be a bool tensor of shape {shape}, got {got}")
        out[name] = v[0]
    return out


def _pass(acts: Mapping, st: State, live: Tensor, auth: Tensor, adj: Tensor, t: int, T: int, c: Mapping,
          limit: int, rate_u: Callable[[], Tensor]) -> Round:
    n = live.shape[0]
    rem = float(T - t + 1)
    no_edge, no_agent = torch.zeros(n, n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)
    given = lambda name: acts.get(name, no_edge if name in EDGE_FIELDS else no_agent)      # noqa: E731
    total = lambda x: float(x.sum())                                                        # noqa: E731
    counts = torch.zeros(len(KINDS), dtype=torch.int64)

    had = st.iso | st.rate | st.degr | st.quar.any(1) | st.quar.any(0)
    back = given("reinstatement") & had
    iso1, rate1, degr1 = st.iso & ~back, st.rate & ~back, st.degr & ~back
    quar1 = st.quar & ~(back[:, None] | back[None, :])
    out_after = (adj.bool() & ~iso1[None, :]).sum(1)
    # an isolated agent sends nothing in the round it is reinstated, so it restores one round fewer
    cost = c["reinstatement"] * total(back * out_after * (rem - st.iso.to(torch.float64)))
    counts[INDEX["reinstatement"]] = back.sum()

    iso_new = given("isolation") & ~iso1
    # only deliveries no earlier persistent action removed: quarantined edges are gone, a rate-limited sender keeps `limit`
    still = (live & ~iso1[None, :] & ~quar1).sum(1)
    still = torch.where(rate1, still.clamp(max=limit), still)
    cost += c["isolation"] * total(iso_new * still) * rem
    counts[INDEX["isolation"]] = iso_new.sum()
    iso2 = iso1 | iso_new
    iso_block = iso2[:, None] | iso2[None, :]

    q_new = given("edge_quarantine") & adj.bool() & ~quar1 & ~iso_block
    cost += c["edge_quarantine"] * total(q_new) * rem
    counts[INDEX["edge_quarantine"]] = q_new.sum()
    quar2 = quar1 | q_new

    open_ = live & ~iso_block & ~quar2
    n_open = open_.sum(1)
    surviving = torch.where(rate1, n_open.clamp(max=limit), n_open)       # a rate-limited sender keeps `limit` deliveries
    d_new = given("privilege_degradation") & ~degr1 & ~iso2
    cost += c["privilege_degradation"] * total(d_new * surviving) * rem
    counts[INDEX["privilege_degradation"]] = d_new.sum()
    degr2 = degr1 | d_new

    l_new = given("rate_limit") & ~rate1 & ~iso2
    cost += c["rate_limit"] * total(l_new * (n_open - limit).clamp_min(0)) * rem
    counts[INDEX["rate_limit"]] = l_new.sum()
    rate2 = rate1 | l_new
    need = rate2 & (n_open > limit)
    rate_drop = torch.zeros_like(live)
    if need.any():
        cand = open_ & need[:, None]
        rank = rate_u().masked_fill(~cand, math.inf).argsort(-1).argsort(-1)       # the lowest draws are kept
        rate_drop = cand & (rank >= limit)

    after_persistent = open_ & ~rate_drop
    deny = given("deny") & after_persistent
    cost += c["deny"] * total(deny & auth)
    counts[INDEX["deny"]] = deny.sum()
    after_deny = after_persistent & ~deny
    delay = given("delay") & after_deny
    cost += c["delay"] * total(delay)
    counts[INDEX["delay"]] = delay.sum()
    explicit = given("sanitize") & after_deny
    fresh = explicit & ~delay & ~degr2[:, None]       # a delivery already delayed or already sanitized by a degradation is charged once
    cost += c["sanitize"] * total(fresh)
    counts[INDEX["sanitize"]] = fresh.sum()
    sanitized = explicit | (degr2[:, None] & after_deny)

    restricted = live & ~iso_block & (quar2 | rate_drop | deny | delay | sanitized)
    return Round(State(iso2, quar2, rate2, degr2), after_deny & ~delay, delay, sanitized, restricted, cost, False,
                 counts)


def apply(actions: core.Actions, state: State, live: Tensor, authentic: Tensor, adj: Tensor, remaining: float,
          t: int, T: int, costs: Mapping, rate_limit: int, rate_u: Callable[[], Tensor]) -> Round:
    """One delivery round. `live` bool [N, N]: deliveries that can happen (a base edge, both endpoints active);
    `remaining` intervention units left; `rate_u()` float64 [N, N] draws ranking a rate-limited sender's receivers."""
    acts = _checked(actions, live.shape[0])
    out = _pass(acts, state, live, authentic, adj, t, T, costs, rate_limit, rate_u)
    if out.cost > remaining:
        out = _pass({}, state, live, authentic, adj, t, T, costs, rate_limit, rate_u)
        out.rejected = True
    return out
