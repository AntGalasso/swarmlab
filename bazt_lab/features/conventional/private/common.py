"""Pieces shared by arms A, B and C."""
from __future__ import annotations

import math

import torch

from bazt_lab.core import public as core

Tensor = torch.Tensor

STAT_JUDGE = "judge"            # B: judge score of the sender's text, per delivery
STAT_C = "rl_minus_ts"          # C: risk level minus trust score, per delivery


def mask_and(*masks: Tensor) -> Tensor:
    """Logical and of bool masks through uint8 (the bool kernels are several times slower on CPU); same result."""
    out = core.as_uint8(masks[0])
    for m in masks[1:]:
        out = out & core.as_uint8(m)
    return core.as_bool(out)


def a_gate(adj: Tensor, attempts: Tensor, authentic: Tensor | None) -> Tensor:
    """Deliveries A lets through: on the allowlist (the graph's edges) and authenticated (mTLS or signature)."""
    return mask_and(attempts, adj) if authentic is None else mask_and(attempts, adj, authentic)


def a_drops(attempts: Tensor, ok: Tensor) -> Tensor:
    """Attempted deliveries A drops. `ok` is a subset of `attempts`, so xor is attempts & ~ok."""
    return core.as_bool(core.as_uint8(attempts) ^ core.as_uint8(ok))


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
        self.keys = core.episode_keys(ctx.unit_ids).to(ctx.device)      # one key per episode: draws never depend on B
        self.unit_audit = unit_cost(ctx, core.ActionKind.INSPECT)
        self.unit_deny = unit_cost(ctx, core.ActionKind.DENY)
        self.step = 0
        self.exhausted = torch.zeros(ctx.B, dtype=torch.bool, device=ctx.device)
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
        actions = core.Actions(deny=a_drops(obs.attempts, ok))           # A's rules: free
        ra, ri = obs.remaining_audit.clone(), obs.remaining_intervention.clone()
        actions.events += self._latch(ra, ri, obs.t)
        live = ~self.exhausted
        evaluated = mask_and(ok, live[:, None, None].expand_as(ok))
        senders = evaluated.any(-1)
        if senders.any():
            audited = obs.audit(senders).masked_fill(~senders, math.nan)
        else:
            audited = torch.full(senders.shape, math.nan, dtype=ctx.dtype, device=ctx.device)
        want, priority, statistics, crossed = self._policy(obs, evaluated, audited)
        final = self._fund(want, priority, ri)
        actions.deny = core.as_bool(core.as_uint8(actions.deny) | core.as_uint8(final))
        actions.statistics = statistics
        actions.threshold_restricted = final & crossed                  # D108: what alpha counts
        paid = (~audited.isnan()).sum(-1, dtype=torch.float64)      # accounting stays float64
        actions.events += self._latch(ra - paid * self.unit_audit,
                                      ri - final.flatten(1).sum(-1, dtype=torch.float64) * self.unit_deny, obs.t)
        return actions

    # ---- subclass hooks
    def _allocate(self, ctx: core.EpisodeContext) -> None:
        raise NotImplementedError

    def _policy(self, obs: core.Observation, evaluated: Tensor, audited: Tensor):
        """-> (deliveries to deny, suspicion used to rank them when the budget cannot pay all, statistics, deliveries
        whose statistic crossed its threshold: the subset of the denies that alpha counts, D108). Any
        state the policy learns from this round is updated here; it does not depend on how many denies are funded."""
        raise NotImplementedError

    # ---- shared mechanics
    def _latch(self, remaining_audit: Tensor, remaining_intervention: Tensor, t: int) -> list:
        starved_audit = remaining_audit < self.unit_audit
        now = (starved_audit | (remaining_intervention < self.unit_deny)) & ~self.exhausted
        self.exhausted = self.exhausted | now
        if not now.any():
            return []
        return [{"event": "budget_exhausted", "arm": self.name, "b": b, "unit_id": self.ctx.unit_ids[b],
                 "budget": "audit" if starved_audit[b] else "intervention", "t": t}
                for b in now.nonzero().flatten().tolist()]

    def _draw(self, namespace: str, rows: Tensor | None = None, dtype: torch.dtype = torch.float64) -> Tensor:
        """Uniforms [len(rows) or B, N, N] from the keyed stream (arm, namespace) at counter = this round: one batched
        call, and a draw depends on the episode id, the arm, the namespace, the round and the delivery index only,
        never on the batch, its order or the device."""
        keys = self.keys if rows is None else self.keys[rows]
        N = self.ctx.N
        return core.keyed_uniform(keys, f"{self.name}|{namespace}", self.step, (N, N), self.ctx.device, dtype)

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
        lowest = torch.finfo(priority.dtype).min
        key = priority[rows].flatten(1).nan_to_num(nan=lowest).masked_fill(~w, -math.inf)
        shuffle = self._draw("fund", rows).flatten(1).argsort(-1)       # float64 scores: the order never depends on the dtype
        best = key.gather(-1, shuffle).argsort(dim=-1, descending=True, stable=True)   # stable= needs dim= by keyword
        order = shuffle.gather(-1, best)
        rank = torch.empty_like(order).scatter_(-1, order, torch.arange(N * N, device=order.device).expand_as(order).contiguous())
        keep = (rank < afford[rows].long()[:, None]) & w
        out = want.clone()
        out[rows] = keep.view(len(rows), N, N)
        return out
