"""Budgeted greedy selection: stable argsort + cumsum, skip-and-continue under a hard cap, and the benign-priced
(Neyman-Pearson) cut that exhausts the budget over the horizon (Hua et al., arXiv 2507.15886, section 3.2)."""
import math

import pytest
import torch

from bazt_lab.features.bazt_slot.private import select as sel

F64 = torch.float64


def t(x, dtype=F64):
    return torch.as_tensor(x, dtype=dtype)


def sequential_greedy(density, cost, valid, budget):
    """Reference: walk the items by density (ties by index), take every item that still fits, skip the rest."""
    chosen, left = [False] * len(density), budget
    for i in sorted(range(len(density)), key=lambda i: (-density[i], i)):
        if valid[i] and cost[i] <= left:
            chosen[i], left = True, left - cost[i]
    return chosen


def test_rank_orders_by_score_descending_with_ties_by_index_and_invalid_last():
    score = t([[1.0, 3.0, 3.0, 2.0, 9.0]])
    valid = t([[1, 1, 1, 1, 0]], torch.bool)
    assert sel.rank(score, valid).tolist() == [[1, 2, 3, 0, 4]]


def test_rank_treats_nan_as_invalid():
    score = t([[math.nan, 1.0, 2.0]])
    assert sel.rank(score, torch.ones(1, 3, dtype=torch.bool)).tolist() == [[2, 1, 0]]


def test_prefix_stops_at_the_first_item_that_does_not_fit():
    """A threshold rule: once the cumulative weight crosses the limit nothing behind it is taken, even if cheaper."""
    order = t([[0, 1, 2, 3]], torch.int64)
    weight = t([[2.0, 3.0, 1.0, 1.0]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    assert sel.prefix(order, weight, valid, t([4.0])).tolist() == [[True, False, False, False]]
    assert sel.prefix(order, weight, valid, t([6.0])).tolist() == [[True, True, True, False]]


def test_prefix_ignores_invalid_items_and_infinite_limit_takes_all_valid():
    order = t([[0, 1, 2]], torch.int64)
    weight = t([[5.0, 5.0, 5.0]])
    valid = t([[1, 0, 1]], torch.bool)
    assert sel.prefix(order, weight, valid, t([math.inf])).tolist() == [[True, False, True]]


def test_fill_skips_an_unaffordable_item_and_continues_down_the_ranking():
    order = t([[0, 1, 2, 3]], torch.int64)
    cost = t([[1.0, 9.0, 1.0, 1.0]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    assert sel.fill(order, cost, valid, t([3.0])).tolist() == [[True, False, True, True]]


def test_fill_matches_the_sequential_greedy_on_random_instances():
    gen = torch.Generator().manual_seed(11)
    for _ in range(60):
        M = int(torch.randint(1, 30, (1,), generator=gen))
        density = torch.rand(M, generator=gen, dtype=F64)
        cost = torch.randint(1, 8, (M,), generator=gen).to(F64)
        valid = torch.rand(M, generator=gen, dtype=F64) < 0.8
        budget = float(torch.randint(0, 40, (1,), generator=gen))
        got = sel.fill(sel.rank(density[None], valid[None]), cost[None], valid[None], t([budget]))
        assert got[0].tolist() == sequential_greedy(density.tolist(), cost.tolist(), valid.tolist(), budget)


def test_fill_never_exceeds_the_budget_and_uses_each_episode_budget_alone():
    gen = torch.Generator().manual_seed(5)
    B, M = 40, 25
    cost = torch.randint(1, 6, (B, M), generator=gen).to(F64)
    density = torch.rand(B, M, generator=gen, dtype=F64)
    valid = torch.rand(B, M, generator=gen, dtype=F64) < 0.9
    budget = torch.randint(0, 30, (B,), generator=gen).to(F64)
    chosen = sel.fill(sel.rank(density, valid), cost, valid, budget)
    assert ((cost * chosen).sum(-1) <= budget).all() and not (chosen & ~valid).any()
    for b in range(B):
        one = sel.fill(sel.rank(density[b:b + 1], valid[b:b + 1]), cost[b:b + 1], valid[b:b + 1], budget[b:b + 1])
        assert torch.equal(chosen[b], one[0])


def test_fill_with_zero_budget_takes_nothing_and_with_infinite_budget_takes_every_valid_item():
    cost = t([[1.0, 2.0, 3.0]])
    valid = t([[1, 1, 0]], torch.bool)
    order = sel.rank(t([[3.0, 2.0, 1.0]]), valid)
    assert not sel.fill(order, cost, valid, t([0.0])).any()
    assert sel.fill(order, cost, valid, t([math.inf])).tolist() == [[True, True, False]]


def test_choose_spends_the_budget_over_the_horizon_by_pricing_benign_traffic():
    """Each purchase is priced by what it wastes if its target is benign. The cut is where the cumulative benign price
    reaches the budget, so a tight budget keeps only the near-certain purchases."""
    density = t([[9.0, 8.0, 7.0, 6.0]])
    p_right = t([[0.95, 0.90, 0.40, 0.30]])
    cost = t([[1.0, 1.0, 1.0, 1.0]])
    recurrence = t([[4.0, 4.0, 4.0, 4.0]])
    assert sel.benign_price(p_right, cost, recurrence).tolist() == [pytest.approx([0.2, 0.4, 2.4, 2.8])]
    valid = torch.ones(1, 4, dtype=torch.bool)
    tight = sel.choose(density, cost, valid, budget=t([10.0]), p_right=p_right, recurrence=recurrence, price_limit=t([1.0]))
    assert tight.tolist() == [[True, True, False, False]]
    loose = sel.choose(density, cost, valid, budget=t([10.0]), p_right=p_right, recurrence=recurrence, price_limit=t([5.5]))
    assert loose.tolist() == [[True, True, True, False]]


def test_the_benign_priced_cut_is_a_threshold_on_the_likelihood_ratio_not_on_the_value_density():
    """Neyman-Pearson (Hua et al.): admit every purchase whose likelihood ratio clears a threshold chosen to exhaust
    the budget. Here the best density belongs to the least certain target; it must not displace the certain ones."""
    density = t([[9.0, 8.0, 7.0]])
    p_right = t([[0.5, 0.95, 0.9]])                       # prices 0.5, 0.05, 0.1 at cost 1, recurrence 1
    ones = t([[1.0, 1.0, 1.0]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    got = sel.choose(density, ones, valid, budget=t([9.0]), p_right=p_right, recurrence=ones, price_limit=t([0.2]))
    assert got.tolist() == [[False, True, True]]


def test_the_cut_exhausts_its_allowance_and_admits_nothing_below_the_threshold():
    p_right = t([[0.9, 0.8, 0.7, 0.6]])                   # prices 0.1, 0.2, 0.3, 0.4: cumulative 0.1, 0.3, 0.6, 1.0
    ones = t([[1.0, 1.0, 1.0, 1.0]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    density = t([[1.0, 2.0, 3.0, 4.0]])
    for limit, expected in ((0.3, 2), (0.59, 2), (0.6, 3), (1.0, 4), (0.05, 0)):
        got = sel.choose(density, ones, valid, budget=t([9.0]), p_right=p_right, recurrence=ones, price_limit=t([limit]))
        assert got.tolist()[0] == [i < expected for i in range(4)], limit


def test_a_purchase_the_hard_cap_can_never_pay_does_not_use_up_the_allowance():
    density = t([[9.0, 8.0]])
    cost = t([[30.0, 6.0]])
    p_right = t([[0.6, 0.6]])                             # benign prices 12 and 2.4 against an allowance of 12
    valid = torch.ones(1, 2, dtype=torch.bool)
    got = sel.choose(density, cost, valid, budget=t([12.0]), p_right=p_right, recurrence=t([[1.0, 1.0]]),
                     price_limit=t([12.0]))
    assert got.tolist() == [[False, True]]


def test_the_cut_prices_each_episode_against_its_own_allowance():
    density = t([[3.0, 2.0], [3.0, 2.0]])
    p_right = t([[0.9, 0.9], [0.9, 0.9]])
    ones = torch.ones(2, 2, dtype=F64)
    valid = torch.ones(2, 2, dtype=torch.bool)
    got = sel.choose(density, ones, valid, budget=t([9.0, 9.0]), p_right=p_right, recurrence=ones,
                     price_limit=t([0.1, 1.0]))
    assert got.tolist() == [[True, False], [True, True]]


def test_choose_without_a_price_is_the_plain_hard_cap_greedy():
    density = t([[3.0, 2.0, 1.0]])
    cost = t([[2.0, 2.0, 1.0]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    assert sel.choose(density, cost, valid, budget=t([3.0])).tolist() == [[True, False, True]]


def test_benign_price_of_a_certain_target_is_zero_and_of_a_certainly_benign_one_is_full_cost():
    price = sel.benign_price(t([1.0, 0.0]), t([3.0, 3.0]), t([2.0, 2.0]))
    assert price.tolist() == [0.0, 6.0]
