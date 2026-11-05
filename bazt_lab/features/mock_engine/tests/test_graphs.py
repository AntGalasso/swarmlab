"""Graphs, effective diameter and the horizon rule (spec 4.2). Random graphs use keyed streams, never the global RNG."""
import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.mock_engine.private import graphs


def draw(topology, N, seeds, **p):
    shape = graphs.draw_shape(topology, N, p.get("ws_k"), p.get("ba_m"))
    u = None if shape is None else core.batch_uniforms([("g", topology, N, s) for s in seeds], shape)
    return graphs.build_graphs(topology, N, u, len(seeds), **p)


def degrees(adj):
    return adj.sum(2)


@pytest.mark.parametrize("topology", ["star_hub", "star_leaf"])
def test_star_is_a_hub_with_leaves(topology):
    adj = draw(topology, 6, [1, 2])
    assert adj.dtype == torch.bool and adj.shape == (2, 6, 6)
    assert adj[:, 0, 1:].all() and adj[:, 1:, 0].all() and not adj[:, 1:, 1:].any()
    assert not adj[:, 0, 0].any()


def test_chain_links_neighbours_only():
    adj = draw("chain", 5, [1])[0]
    expected = torch.tensor([[abs(i - j) == 1 for j in range(5)] for i in range(5)])
    assert torch.equal(adj, expected)


@pytest.mark.parametrize("topology,extra", [("star_hub", {}), ("chain", {}), ("small_world", dict(ws_k=4, ws_p=0.25)),
                                            ("barabasi_albert", dict(ba_m=2))])
@pytest.mark.parametrize("N", [3, 12, 100])
def test_every_topology_works_for_any_n_and_is_symmetric_without_loops(topology, extra, N):
    adj = draw(topology, N, range(4), **extra)
    assert adj.shape == (4, N, N)
    assert torch.equal(adj, adj.transpose(1, 2))
    assert not adj.diagonal(dim1=1, dim2=2).any()


def test_two_agents_is_the_smallest_swarm():
    assert draw("chain", 2, [1])[0].tolist() == [[False, True], [True, False]]


def test_watts_strogatz_without_rewiring_is_the_ring_lattice():
    adj = draw("small_world", 12, [1, 2], ws_k=4, ws_p=0.0)
    assert (degrees(adj) == 4).all()
    assert adj[0, 0, 1] and adj[0, 0, 2] and adj[0, 0, 11] and adj[0, 0, 10] and not adj[0, 0, 3]


def test_watts_strogatz_rewiring_keeps_edge_count_at_most_and_changes_some_graphs():
    ring = draw("small_world", 12, [0], ws_k=4, ws_p=0.0)[0]
    adj = draw("small_world", 12, range(200), ws_k=4, ws_p=0.25)
    assert (adj.sum((1, 2)) <= ring.sum()).all()           # a rewired edge can collide with an existing one
    assert (adj != ring).flatten(1).any(1).float().mean() > 0.9
    assert 0.8 * ring.sum() < adj.sum((1, 2)).double().mean() <= ring.sum()


def test_watts_strogatz_is_deterministic_for_a_key_and_independent_of_batch_size():
    both = draw("small_world", 12, [5, 6], ws_k=4, ws_p=0.25)
    assert torch.equal(both[1], draw("small_world", 12, [6], ws_k=4, ws_p=0.25)[0])


def test_watts_strogatz_refuses_k_that_wraps_the_ring():
    with pytest.raises(ValueError, match="ws_k"):
        draw("small_world", 4, [1], ws_k=8, ws_p=0.25)


def test_barabasi_albert_has_m_edges_per_arrival_and_is_connected():
    N, m = 30, 2
    adj = draw("barabasi_albert", N, range(20), ba_m=m)
    assert (adj.sum((1, 2)) == 2 * m * (N - m)).all()      # initial star (m edges) + m per later node, both directions
    assert (graphs.diameters(adj) < N).all() and (graphs.reach_all(adj)).all()
    assert (degrees(adj)[:, m + 1:] >= m).all()


def test_barabasi_albert_is_preferential_early_nodes_become_hubs():
    adj = draw("barabasi_albert", 60, range(300), ba_m=2)
    deg = degrees(adj).double()
    assert deg[:, :5].mean() > 2.5 * deg[:, -30:].mean()
    assert deg.max(1).values.mean() > 3 * deg.mean()


def test_barabasi_albert_needs_more_agents_than_m():
    with pytest.raises(ValueError, match="ba_m"):
        draw("barabasi_albert", 2, [1], ba_m=2)


def test_unknown_topology_is_refused():
    with pytest.raises(ValueError, match="topology"):
        graphs.draw_shape("torus", 5, None, None)


@pytest.mark.parametrize("topology,N,expected", [("chain", 7, 6), ("star_hub", 9, 2), ("star_leaf", 4, 2)])
def test_effective_diameter_of_the_named_graphs(topology, N, expected):
    adj = draw(topology, N, [1, 2, 3])
    assert graphs.diameters(adj).tolist() == [expected] * 3


def test_effective_diameter_is_the_longest_finite_shortest_path_and_ignores_unreachable_pairs():
    adj = torch.zeros(2, 6, 6, dtype=torch.bool)
    for i, j in [(0, 1), (1, 2), (2, 3), (4, 5)]:         # a 4-chain and a separate pair
        adj[0, i, j] = adj[0, j, i] = True
    for i, j in [(0, 1)]:
        adj[1, i, j] = adj[1, j, i] = True
    assert graphs.diameters(adj).tolist() == [3, 1]
    assert graphs.reach_all(adj).tolist() == [False, False]


def test_effective_diameter_of_a_single_agent_or_no_edges_is_zero():
    assert graphs.diameters(torch.zeros(1, 3, 3, dtype=torch.bool)).tolist() == [0]


@pytest.mark.parametrize("c,D,T", [(4.0, 2, 8), (1.5, 3, 5), (0.1, 30, 3), (2.0, 0, 1), (0.2, 1, 1), (3.0, 11, 33)])
def test_horizon_is_the_constant_times_the_diameter_rounded_up_and_at_least_one_round(c, D, T):
    assert graphs.horizon_rounds(torch.tensor([D]), c).tolist() == [T]
