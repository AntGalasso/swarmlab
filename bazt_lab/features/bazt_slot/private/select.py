"""Budgeted greedy selection over a batch of episodes: stable argsort and cumsum, no Python loop over episodes.

Shapes: B episodes, M candidates each. `order` is int64 [B, M], candidate indices from best to worst; scores and
costs are float64, masks bool.

The enforcement-budget component (Hua et al., arXiv 2507.15886, section 3.1-3.2) is the cut in `choose`. Their budget
is priced on benign traffic (Eq. 2: what the rule spends on aligned outputs), and the Neyman-Pearson rule (Eq. 4) buys
every output whose likelihood ratio clears a threshold, the threshold being set to exhaust that budget. Here a
purchase's benign price is P(its target is benign) x cost x the rounds it recurs (`benign_price`); the purchases are
ordered by likelihood ratio per unit of benign price (`np_order`: the posterior odds over the price, which is the
likelihood ratio itself when prices are equal) and the cut sits where the cumulative benign price reaches the allowance.
A tight allowance therefore keeps only the near-certain purchases and the budget lasts the horizon instead of going at
the first round. What passes the cut is then bought by value density (the selection component) under the hard cap
(`fill`): real costs never exceed what the platform will pay.
"""
from __future__ import annotations

import torch

Tensor = torch.Tensor
_TINY = 1e-300


def rank(score: Tensor, valid: Tensor) -> Tensor:
    """Candidate indices by score, highest first; ties by index (stable); invalid and NaN candidates last."""
    valid = valid & ~score.isnan()
    key = torch.where(valid, score, torch.full_like(score, -torch.inf))
    return key.argsort(dim=-1, descending=True, stable=True)


def prefix(order: Tensor, weight: Tensor, valid: Tensor, limit: Tensor) -> Tensor:
    """bool [B, M] in candidate order: the valid candidates taken while their cumulative weight stays <= limit [B].

    Stops at the first valid candidate that does not fit (weights are >= 0, so the cumulative sum is monotone)."""
    taken = torch.where(valid, weight, torch.zeros_like(weight)).gather(-1, order).cumsum(-1) <= limit[:, None]
    taken = taken & valid.gather(-1, order)
    return torch.zeros_like(valid).scatter_(-1, order, taken)


def fill(order: Tensor, cost: Tensor, valid: Tensor, budget: Tensor) -> Tensor:
    """bool [B, M]: walk the ranking and take every valid candidate whose cost still fits, skipping those that do not.

    Equal to the sequential skip-and-continue greedy: a prefix pass takes candidates until the first misfit, then the
    candidates that can no longer fit are dropped and the next pass continues behind it. Each pass takes at least one
    candidate, so it ends."""
    chosen = torch.zeros_like(valid)
    left = budget.clone()
    live = valid.clone()
    while True:
        live = live & (cost <= left[:, None])
        take = prefix(order, cost, live, left)
        if not take.any():
            return chosen
        chosen |= take
        left = left - (cost * take).sum(-1)
        live = live & ~take


def benign_price(p_right: Tensor, cost: Tensor, recurrence: Tensor) -> Tensor:
    """Expected spend on a benign target: P(the purchase is wrong) x cost x the number of rounds it recurs.
    `p_right` is the probability the purchase is warranted (the target is compromised; for a reinstatement, benign)."""
    return (1.0 - p_right) * cost * recurrence


def np_order(p_right: Tensor, price: Tensor, valid: Tensor) -> Tensor:
    """Candidate indices by likelihood ratio per unit of benign price, highest first (ties by index; invalid last).
    A free, certain purchase (price 0) comes first."""
    key = torch.where(price > 0, p_right / price.clamp_min(_TINY), torch.full_like(price, torch.inf))
    return rank(key, valid)


def choose(density: Tensor, cost: Tensor, valid: Tensor, budget: Tensor, p_right: Tensor | None = None,
           recurrence: Tensor | None = None, price_limit: Tensor | None = None) -> Tensor:
    """bool [B, M]: the candidates to buy. `budget` [B] is the hard cap on real cost. With `p_right`, `recurrence`
    [B, M] and `price_limit` [B] the Neyman-Pearson cut applies first: only candidates the hard cap can pay count
    towards the allowance, and the allowed ones are the likelihood-ratio prefix whose cumulative benign price stays
    within `price_limit`."""
    valid = valid & (cost <= budget[:, None])
    if p_right is not None:
        price = benign_price(p_right, cost, recurrence)
        valid = valid & prefix(np_order(p_right, price, valid), price, valid, price_limit)
    return fill(rank(density, valid), cost, valid, budget)
