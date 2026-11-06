"""Helpers shared by the reference arms. Shapes: B episodes, N agents; masks bool, scores float64, indices int64.

Edge convention (core contract): adj[b, i, j] is the directed edge i -> j.
"""
from __future__ import annotations

from typing import Mapping, Optional

import torch

from bazt_lab.core import public as core

INF = float("inf")


def cost_of(ctx: core.EpisodeContext, kind: core.ActionKind) -> float:
    """Cost per affected delivery of an action, from ctx.costs; a missing entry refuses with its key."""
    try:
        return float(ctx.costs[kind.value])
    except KeyError:
        raise core.Unset(f"costs.<main|robustness>.{kind.value}") from None


def threshold_of(ctx: core.EpisodeContext, name: str) -> float:
    """Calibrated threshold of a statistic; missing means +inf (never act), as in the core contract."""
    return float(ctx.thresholds.get(name, INF))


def active_edges(adj: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """bool [B, N, N]: base edges between two active agents."""
    return adj & active[:, :, None] & active[:, None, :]


def rank_desc(score: torch.Tensor, valid: torch.Tensor, tie: Optional[torch.Tensor] = None,
              ties_high: bool = False) -> torch.Tensor:
    """int64 [B, N]: agent indices by score, highest first; invalid (and NaN) agents last.

    Ties go to the smaller `tie` key when given (an arm-stream draw), else to the lower index, or to the higher
    index with ties_high (what Python's `sorted(..., reverse=True)` on (score, index) tuples does)."""
    valid = valid & ~torch.isnan(score)
    key = torch.where(valid, score, torch.full_like(score, -INF))
    if tie is not None:
        first = tie.argsort(dim=-1, stable=True)
    elif ties_high:
        first = torch.arange(score.shape[-1] - 1, -1, -1, dtype=torch.int64, device=score.device).expand_as(score)
    else:
        return key.argsort(dim=-1, descending=True, stable=True)
    by_score = key.gather(-1, first).argsort(dim=-1, descending=True, stable=True)
    return first.gather(-1, by_score)


def fit_in_order(order: torch.Tensor, cost: torch.Tensor, valid: torch.Tensor, budget: torch.Tensor) -> torch.Tensor:
    """bool [B, N]: walk `order` and take each valid agent whose cost still fits the remaining `budget` [B]; an agent
    that does not fit is skipped and the walk continues (skip-and-continue knapsack).

    Until the first skip the walk is a prefix sum, so one cumsum answers a batch in which nothing is skipped (an
    uncapped budget, or one that pays for every candidate): no per-rank loop and no host sync beyond the single
    `any()` that decides whether a skip happened anywhere. Only a batch with a skip pays the N-step walk, with no
    `.item()` inside it (costs are exact in float64, so the prefix test is exact)."""
    B, N = order.shape
    cost_o = cost.gather(-1, order)
    valid_o = valid.gather(-1, order)
    prefix = torch.where(valid_o, cost_o, torch.zeros_like(cost_o)).cumsum(-1)
    fits = valid_o & (prefix <= budget[:, None])
    chosen = torch.zeros(B, N, dtype=torch.bool, device=order.device)
    if bool((valid_o & ~fits).any()):                      # somewhere an agent does not fit: walk the order exactly
        fits = torch.zeros_like(valid_o)
        left = budget.clone()
        for r in range(N):
            c = cost_o[:, r]
            ok = valid_o[:, r] & (c <= left)
            left = left - torch.where(ok, c, torch.zeros_like(c))
            fits[:, r] = ok
    return chosen.scatter_(-1, order, fits)


def downstream_curve(adj: torch.Tensor, K: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """`dtype` [B, N, K+1]: out[b, i, R] = number of descendants of agent i's message when R rounds remain.

    The execution is unrolled over rounds into a dependency DAG: node (j, s) depends on (j, s-1) (the agent's own
    memory) and on (i, s-1) for every edge i -> j. The descendants of (i, r) are the nodes (j, r+k), k = 1..R, with
    j within k hops of i (self included), so out[., i, R] = sum_{k=1..R} |ball_k(i)| and out[., i, 0] = 0.
    Boolean ball growth by dense bmm: O(K B N^3), run once per episode batch."""
    B, N, _ = adj.shape
    eye = torch.eye(N, dtype=torch.bool, device=adj.device).expand(B, N, N)
    step = (adj | eye).to(torch.float32)
    ball = eye.to(torch.float32)
    sizes = [torch.zeros(B, N, dtype=dtype, device=adj.device)]
    for _ in range(K):
        ball = (torch.bmm(ball, step) > 0).to(torch.float32)
        sizes.append(ball.sum(-1).to(dtype))
    return torch.stack(sizes, -1).cumsum(-1)


def reach_h(adj: torch.Tensor, active: torch.Tensor, h: int, gamma: float, dmax: int,
            weight: torch.Tensor) -> torch.Tensor:
    """`weight.dtype` [B, N]: swarmlab.reach_H for every source agent at once.

    Horizon-bounded reachability from i over ACTIVE agents: each agent first reached at hop d <= min(h, dmax)
    contributes weight[agent] * gamma**(d-1) (weight is 1, or 1 - p for the monitor-discounted form)."""
    B, N, _ = adj.shape
    total = torch.zeros(B, N, dtype=weight.dtype, device=adj.device)
    depth = min(h, dmax)
    if depth <= 0:
        return total
    step = active_edges(adj, active).to(torch.float32)
    seen = torch.eye(N, dtype=torch.bool, device=adj.device).expand(B, N, N).clone()
    frontier = seen.to(torch.float32)
    for d in range(1, depth + 1):
        new = (torch.bmm(frontier, step) > 0) & ~seen
        total += (new.to(weight.dtype) * weight[:, None, :]).sum(-1) * gamma ** (d - 1)
        seen |= new
        frontier = new.to(torch.float32)
    return total


class Arm:
    """Base of the reference defenders: name, context, and the tensor / plain state listed in `_state`.

    Everything random is pre-drawn in reset() from ctx.arm_stream, so a restored arm only needs `_state`."""
    name: str = ""
    needs_truth: bool = False
    thresholded: bool = False       # acts only when a calibrated statistic exceeds its threshold (no threshold: never)
    _state: tuple = ()
    ctx: core.EpisodeContext

    def state_dict(self) -> dict:
        return {k: (v.clone() if torch.is_tensor(v) else v) for k in self._state for v in [getattr(self, k)]}

    def load_state_dict(self, state: Mapping) -> None:
        for k in self._state:
            v = state[k]
            setattr(self, k, v.clone() if torch.is_tensor(v) else v)


def predraw(ctx: core.EpisodeContext, namespace: str, shape) -> torch.Tensor:
    """float64 uniforms [B, *shape] on ctx.device for all rounds at once, in ONE keyed batched call. A draw depends only
    on (episode id, arm, namespace, element index): never on B, on the position of the episode or on the device. They
    stay float64 whatever compute.dtype is: they only order candidates, and the order must not depend on the dtype."""
    keys = core.episode_keys(ctx.unit_ids).to(ctx.device)
    return core.keyed_permutation_scores(keys, f"{ctx.arm}|{namespace}", 0, tuple(shape), ctx.device)
