"""Shared helpers: pricing, ranking, budgeted greedy selection, downstream counts, horizon-bounded reach."""
import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.references.private import common
from bazt_lab.features.references.tests.synth import adj_from_edges, chain, make_ctx, star


def test_cost_of_names_the_missing_key():
    ctx = make_ctx(chain(3), 3, costs={"inspect": 1.0})
    assert common.cost_of(ctx, core.ActionKind.INSPECT) == 1.0
    with pytest.raises(core.Unset) as e:
        common.cost_of(ctx, core.ActionKind.DENY)
    assert "deny" in e.value.args[0]


def test_rank_desc_orders_valid_first_and_breaks_ties_by_index_or_tie_key():
    score = torch.tensor([[1.0, 3.0, 3.0, 2.0, 9.0]], dtype=torch.float64)
    valid = torch.tensor([[True, True, True, True, False]])
    assert common.rank_desc(score, valid).tolist() == [[1, 2, 3, 0, 4]]                    # ties: lower index
    assert common.rank_desc(score, valid, ties_high=True).tolist() == [[2, 1, 3, 0, 4]]      # ties: higher index
    tie = torch.tensor([[0.0, 0.9, 0.1, 0.0, 0.0]], dtype=torch.float64)
    assert common.rank_desc(score, valid, tie=tie).tolist() == [[2, 1, 3, 0, 4]]             # ties: smaller tie key
    nan = torch.tensor([[float("nan"), 1.0]], dtype=torch.float64)
    assert common.rank_desc(nan, torch.ones(1, 2, dtype=torch.bool)).tolist() == [[1, 0]]    # NaN counts as invalid


def test_fit_in_order_skips_what_does_not_fit_and_continues():
    order = torch.tensor([[2, 0, 1, 3], [0, 1, 2, 3]])
    cost = torch.tensor([[1.0, 2.0, 9.0, 1.0], [5.0, 1.0, 1.0, 1.0]], dtype=torch.float64)
    valid = torch.tensor([[True, True, True, True], [True, True, True, False]])
    budget = torch.tensor([4.0, 3.0], dtype=torch.float64)
    got = common.fit_in_order(order, cost, valid, budget)
    # episode 0, order 2,0,1,3 with costs 9,1,2,1 and budget 4: agent 2 is skipped, then 0, 1 and 3 fit (1+2+1)
    assert got[0].tolist() == [True, True, False, True]
    # episode 1: agent 0 costs 5 > 3 and is skipped; agents 1 and 2 fit (1+1); agent 3 is invalid
    assert got[1].tolist() == [False, True, True, False]
    inf = common.fit_in_order(order, cost, valid, torch.full((2,), float("inf"), dtype=torch.float64))
    assert inf.tolist() == valid.tolist()


def test_downstream_curve_counts_nodes_of_the_unrolled_dependency_graph():
    # chain 0->1->2->3 with memory edges (i, r) -> (i, r+1): descendants of (i, r) after k rounds are the nodes
    # within k hops, counted once per round. Hand computation for i=0: k=1 {0,1}=2, k=2 {0,1,2}=3, k=3 4 -> 2,5,9.
    curve = common.downstream_curve(chain(4).unsqueeze(0), 3)
    assert curve.dtype == torch.float64 and curve.shape == (1, 4, 4)
    assert curve[0, 0].tolist() == [0.0, 2.0, 5.0, 9.0]
    assert curve[0, 1].tolist() == [0.0, 2.0, 5.0, 8.0]
    assert curve[0, 3].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_downstream_curve_separates_hub_from_leaf_on_a_symmetric_star():
    curve = common.downstream_curve(star(5).unsqueeze(0), 3)
    assert curve[0, 0, 3] > curve[0, 1, 3]          # the hub reaches everything first
    assert torch.equal(curve[0, 1], curve[0, 4])    # leaves are symmetric


def test_reach_is_horizon_bounded_decayed_and_discounted():
    a = star(5).unsqueeze(0)                          # hub 0, leaves 1..4
    act = torch.ones(1, 5, dtype=torch.bool)
    w = torch.ones(1, 5, dtype=torch.float64)
    H = common.reach_h(a, act, h=3, gamma=0.7, dmax=3, weight=w)
    assert H[0, 0].item() == pytest.approx(4.0)                           # four leaves at depth 1
    assert H[0, 1].item() == pytest.approx(1.0 + 3 * 0.7)                # hub at depth 1, three leaves at depth 2
    assert common.reach_h(a, act, h=1, gamma=0.7, dmax=3, weight=w)[0, 1].item() == pytest.approx(1.0)
    assert common.reach_h(a, act, h=0, gamma=0.7, dmax=3, weight=w).abs().sum().item() == 0.0
    inactive = torch.tensor([[True, True, False, True, True]])
    H2 = common.reach_h(a, inactive, h=3, gamma=0.7, dmax=3, weight=w)
    assert H2[0, 0].item() == pytest.approx(3.0)                          # an isolated leaf is not reachable
    disc = torch.tensor([[0.0, 0.5, 0.0, 0.0, 0.0]], dtype=torch.float64)
    assert common.reach_h(a, act, h=1, gamma=0.7, dmax=3, weight=1.0 - disc)[0, 0].item() == pytest.approx(3.5)


def test_reach_follows_edge_direction():
    a = adj_from_edges(3, [(0, 1), (1, 2)]).unsqueeze(0)
    act = torch.ones(1, 3, dtype=torch.bool)
    H = common.reach_h(a, act, h=3, gamma=0.5, dmax=3, weight=torch.ones(1, 3, dtype=torch.float64))
    assert H[0].tolist() == [1.0 + 0.5, 1.0, 0.0]
