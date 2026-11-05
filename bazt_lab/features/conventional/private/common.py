"""Pieces shared by arms A, B and C."""
from __future__ import annotations

import math

import torch

from bazt_lab.core import public as core

Tensor = torch.Tensor

STAT_JUDGE = "judge"            # B: judge score of the sender's text, per delivery
STAT_C = "rl_minus_ts"          # C: risk level minus trust score, per delivery
_LOWEST = torch.finfo(torch.float64).min


def a_gate(adj: Tensor, attempts: Tensor, authentic: Tensor | None) -> Tensor:
    """Deliveries A lets through: on the allowlist (the graph's edges) and authenticated (mTLS or signature)."""
    ok = attempts & adj
    return ok if authentic is None else ok & authentic


def unit_cost(ctx: core.EpisodeContext, kind: core.ActionKind) -> float:
    try:
        return float(ctx.costs[kind.value])
    except KeyError:
        raise core.Unset(f"costs.<condition>.{kind.value}") from None


class BudgetedArm:
    """B and C: A's rules, then a budgeted policy; once either budget of an episode is exhausted the episode gets
    A's rules only (spec 2.4). Subclasses supply `_allocate` and `_policy`.

    Exhausted means the budget cannot pay one more unit of what the arm spends it on (an audit unit, a deny). With
    unit costs and whole-number caps this is exactly "reached zero"; a fractional remainder that can buy nothing
    counts as exhausted too. The event is emitted in the round whose charges exhausted the budget, once per episode.
    """
    name = ""
    state_keys: tuple = ()          # names of the tensor attributes a subclass carries across rounds

    def __init__(self) -> None:
        self.ctx: core.EpisodeContext | None = None

    # ---- Defender protocol
    def reset(self, ctx: core.EpisodeContext) -> None:
        self.ctx = ctx
        self.unit_audit = unit_cost(ctx, core.ActionKind.INSPECT)
        self.unit_deny = unit_cost(ctx, core.ActionKind.DENY)
        self.step = 0
        self.exhausted = torch.zeros(ctx.B, dtype=torch.bool)
        self._allocate(ctx)

    def state_dict(self) -> dict:
        state = {"step": self.step, "exhausted": self.exhausted.clone()}
        state.update({k: getattr(self, k).clone() for k in self.state_keys})
        return state

    def load_state_dict(self, state) -> None:
        self.step = int(state["step"])
        self.exhausted = state["exhausted"].clone()
        for k in self.state_keys:
            setattr(self, k, state[k].clone())

    def decide(self, obs: core.Observation) -> core.Actions:
        ctx = self.ctx
        self.step += 1
        ok = a_gate(ctx.adj, obs.attempts, obs.authentic)
        actions = core.Actions(deny=obs.attempts & ~ok)                  # A's rules: free
        ra, ri = obs.remaining_audit.clone(), obs.remaining_intervention.clone()
        actions.events += self._latch(ra, ri, obs.t)
        live = ~self.exhausted
        evaluated = ok & live[:, None, None]
        senders = evaluated.any(-1)
        audited = obs.audit(senders) if senders.any() else torch.full(senders.shape, math.nan, dtype=torch.float64)
        audited = audited.masked_fill(~senders, math.nan)
        want, priority, statistics = self._policy(obs, evaluated, audited)
        final = self._fund(want, priority, ri)
        actions.deny = actions.deny | final
        actions.statistics = statistics
        paid = (~audited.isnan()).sum(-1).to(torch.float64)
        actions.events += self._latch(ra - paid * self.unit_audit,
                                      ri - final.flatten(1).sum(-1).to(torch.float64) * self.unit_deny, obs.t)
        return actions

    # ---- subclass hooks
    def _allocate(self, ctx: core.EpisodeContext) -> None:
        raise NotImplementedError

    def _policy(self, obs: core.Observation, evaluated: Tensor, audited: Tensor):
        """-> (deliveries to deny, suspicion used to rank them when the budget cannot pay all, statistics). Any
        state the policy learns from this round is updated here; it does not depend on how many denies are funded."""
        raise NotImplementedError

    # ---- shared mechanics
    def _latch(self, remaining_audit: Tensor, remaining_intervention: Tensor, t: int) -> list:
        starved_audit = remaining_audit < self.unit_audit
        now = (starved_audit | (remaining_intervention < self.unit_deny)) & ~self.exhausted
        self.exhausted = self.exhausted | now
        return [{"event": "budget_exhausted", "arm": self.name, "b": b, "unit_id": self.ctx.unit_ids[b],
                 "budget": "audit" if starved_audit[b] else "intervention", "t": t}
                for b in now.nonzero().flatten().tolist()]

    def _draw(self, namespace: str, rows: Tensor) -> Tensor:
        """float64 uniforms [len(rows), N, N] from the arm stream of (namespace, step, episode): they depend on the
        episode and the round only, never on the batch."""
        N = self.ctx.N
        return torch.stack([core.uniforms(self.ctx.arm_stream(f"{namespace}|{self.step}", int(b)), (N, N))
                            for b in rows.tolist()])

    def _fund(self, want: Tensor, priority: Tensor, remaining: Tensor) -> Tensor:
        """Keep the denies an episode can pay for: the most suspicious first, ties broken at random (keyed)."""
        if self.unit_deny <= 0:
            return want
        afford = torch.floor(remaining / self.unit_deny)                # inf while uncapped
        over = want.flatten(1).sum(-1) > afford
        if not over.any():
            return want
        rows = over.nonzero().flatten()
        B, N, _ = want.shape
        w = want[rows].flatten(1)
        key = priority[rows].flatten(1).nan_to_num(nan=_LOWEST).masked_fill(~w, -math.inf)
        shuffle = self._draw("fund", rows).flatten(1).argsort(-1)
        best = key.gather(-1, shuffle).argsort(dim=-1, descending=True, stable=True)   # stable= needs dim= by keyword
        order = shuffle.gather(-1, best)
        rank = torch.empty_like(order).scatter_(-1, order, torch.arange(N * N).expand_as(order).contiguous())
        keep = (rank < afford[rows].long()[:, None]) & w
        out = want.clone()
        out[rows] = keep.view(len(rows), N, N)
        return out
