"""Communication graphs of a seed run: seeded, key-driven, any N."""
import numpy as np
import pytest

from bazt_lab.core import public as core
from bazt_lab.features.llm_harness.private import graphs

RUNTIME = {"runtime": {"random_edge_probability": 0.5, "ba_m": 2, "ws_k": 4, "ws_rewire_p": 0.3}}
KEY = ("llm_harness", "graph", "x", 7)


def build(topology, n=12, key=KEY, settings=RUNTIME):
    return graphs.build_adjacency(topology, n, settings, key)


@pytest.mark.parametrize("topology", graphs.TOPOLOGIES)
def test_every_topology_is_a_square_0_1_matrix_without_self_loops(topology):
    adj = build(topology)
    assert adj.shape == (12, 12) and set(np.unique(adj)) <= {0, 1} and adj.diagonal().sum() == 0


@pytest.mark.parametrize("topology", graphs.TOPOLOGIES)
def test_same_key_same_graph_other_key_other_graph_for_random_ones(topology):
    assert (build(topology) == build(topology)).all()
    if topology in ("random_directed", "barabasi_albert", "watts_strogatz"):
        assert (build(topology) != build(topology, key=KEY + (1,))).any()


@pytest.mark.parametrize("topology", ["hub_star", "barabasi_albert", "watts_strogatz", "chain", "tree", "wheel"])
def test_these_topologies_are_symmetric(topology):
    adj = build(topology)
    assert (adj == adj.T).all()


def test_hub_star_has_a_hub_and_leaves():
    adj = build("hub_star")
    assert adj[0, 1:].all() and adj[1:, 0].all() and adj[1:, 1:].sum() == 0


def test_random_directed_density_follows_the_edge_probability():
    dense = {"runtime": {**RUNTIME["runtime"], "random_edge_probability": 1.0}}
    assert build("random_directed", settings=dense).sum() == 12 * 11
    none = {"runtime": {**RUNTIME["runtime"], "random_edge_probability": 0.0}}
    assert build("random_directed", settings=none).sum() == 0


def test_barabasi_albert_adds_m_edges_per_new_agent():
    adj = build("barabasi_albert", n=30)
    assert adj.sum() // 2 == 2 + (30 - 3) * 2     # a star of m edges, then m per new agent


def test_watts_strogatz_keeps_ws_k_edges_per_agent_on_average():
    assert build("watts_strogatz", n=20).sum() == 20 * 4


def test_wheel_is_a_hub_plus_a_ring_of_leaves():
    adj = build("wheel", n=8)
    assert adj[0, 1:].all() and (adj[1:, 1:].sum(1) == 2).all()


def test_works_for_any_n():
    for n in (3, 5, 40):
        for t in graphs.TOPOLOGIES:
            assert build(t, n=n).shape == (n, n)


def test_a_missing_runtime_value_names_its_key():
    with pytest.raises(core.Unset, match=r"runtime\.ba_m"):
        graphs.build_adjacency("barabasi_albert", 12, {"runtime": {}}, KEY)


def test_unknown_topology_and_bad_parameters_refuse():
    with pytest.raises(ValueError, match="unknown topology"):
        build("ring")
    bad = {"runtime": {**RUNTIME["runtime"], "ba_m": 12}}
    with pytest.raises(ValueError, match="ba_m"):
        build("barabasi_albert", settings=bad)
