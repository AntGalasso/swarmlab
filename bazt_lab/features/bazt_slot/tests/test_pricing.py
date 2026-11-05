"""The value of each of the eight actions on a hand-built graph, checked against explicit numpy sums.

Graph: 0 -> 1, 0 -> 2, 1 -> 2, 2 -> 0 (strongly connected). T = 4, t = 2, so K = 2 hops remain and 3 rounds are left to
pay for. Every number below is recomputed here from the definitions, not read back from the code under test.
"""
import numpy as np
import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.bazt_slot.private import contagion as ct
from bazt_lab.features.bazt_slot.private import params as P
from bazt_lab.features.bazt_slot.private import pricing as pr
from bazt_lab.features.bazt_slot.tests.testkit import COSTS_MAIN, COSTS_ROBUST, adj_of, make_settings

F64 = torch.float64
K = core.ActionKind
EDGES = [(0, 1), (0, 2), (1, 2), (2, 0)]
A_NP = np.zeros((3, 3))
for i, j in EDGES:
    A_NP[i, j] = 1.0
R_NP = np.array([0.9, 0.2, 0.1])
D, SIGMA, LAM = 0.5, 0.9, 1.0            # katz_decay, sanitize_success, utility_weight
T_ROUND, T_END = 2, 4
HOPS = T_END - T_ROUND                  # 2
ROUNDS_LEFT = T_END - T_ROUND + 1       # 3


def world(r=R_NP, *, t=T_ROUND, T=T_END, isolated=(), degraded=(), limited=(), quarantined=(), sink=0):
    adj = adj_of(1, EDGES, 3)
    def agents(ids):
        m = torch.zeros(1, 3, dtype=torch.bool)
        m[0, list(ids)] = True
        return m
    q = torch.zeros(1, 3, 3, dtype=torch.bool)
    for i, j in quarantined:
        q[0, i, j] = True
    iso = agents(isolated)
    live = adj & ~q & ~iso[:, :, None] & ~iso[:, None, :]
    return pr.World(t=t, T=T, adj=adj, live=live, quarantined=q, active=~iso, isolated=iso, degraded=agents(degraded),
                    limited=agents(limited), r=torch.as_tensor(r, dtype=F64)[None],
                    sink=torch.tensor([sink], dtype=torch.int64))


def numpy_reach(A, r, hops, w=None):
    """H_i = sum_{k=1..hops} d^(k-1) (A^k s)_i with s = (1 - r) w, and R_j = s_j + d H_j^(hops-1)."""
    w = np.ones_like(r) if w is None else w
    s = (1 - r) * w
    def H(h):
        return sum(D ** (k - 1) * np.linalg.matrix_power(A, k) @ s for k in range(1, h + 1)) if h > 0 else 0 * s
    return H(hops), s + D * H(hops - 1), H(hops - 1) if hops > 1 else 0 * s


def numpy_centrality(adj_np, t, T):
    deg = adj_np.sum(1)
    return 0.5 * deg / max(deg.max(), 1) + 0.5 * (1 - t / T)


@pytest.fixture(scope="module")
def p():
    return P.read(make_settings())


@pytest.fixture(scope="module")
def priced(p):
    w = world()
    return pr.price(w, p, COSTS_MAIN, elig=torch.ones(1, 3, dtype=torch.bool), kinds=frozenset(K), topology=True)


@pytest.fixture(scope="module")
def structure():
    lam, node_rel, edge_rel = ct.eigen_drop(torch.as_tensor(A_NP, dtype=F64)[None], 300)
    return node_rel[0].numpy(), edge_rel[0].numpy()


def test_reach_is_the_truncated_katz_sum_with_the_sink_weighted(p):
    w = world(sink=2)
    reach = pr.reach(w, pr.transmission(w, p), p, topology=True)
    weight = np.array([1.0, 1.0, 1.0])            # sink_weight is 1 in the test settings
    H, R, _ = numpy_reach(A_NP, R_NP, HOPS, weight)
    np.testing.assert_allclose(reach.node[0].numpy(), D * H, rtol=1e-12)
    np.testing.assert_allclose(reach.edge[0].numpy(), D * A_NP * R[None, :], rtol=1e-12)
    assert reach.spread[0].item() == pytest.approx(D * float(R_NP @ H), rel=1e-12)


def test_the_protected_sink_weighs_more_in_what_is_at_stake():
    p3 = P.read(make_settings(bazt__sink_weight=3.0))
    w = world(sink=1)
    reach = pr.reach(w, pr.transmission(w, p3), p3, topology=True)
    H, R, _ = numpy_reach(A_NP, R_NP, HOPS, np.array([1.0, 3.0, 1.0]))
    np.testing.assert_allclose(reach.node[0].numpy(), D * H, rtol=1e-12)


def test_isolation_removes_the_agents_own_reach_plus_its_structural_share(p, priced, structure):
    node_rel, _ = structure
    H, R, _ = numpy_reach(A_NP, R_NP, HOPS)
    own = R_NP * D * H
    spread = float((R_NP * D * H).sum())
    gain = own + node_rel * np.maximum(spread - own, 0.0)
    loss = LAM * (1 - R_NP) * p.rho["isolation"] * numpy_centrality(A_NP, T_ROUND, T_END)
    np.testing.assert_allclose(priced[K.ISOLATION].net[0].numpy(), gain - loss, rtol=1e-9)
    assert priced[K.ISOLATION].cost[0].tolist() == pytest.approx([2 * ROUNDS_LEFT, 1 * ROUNDS_LEFT, 1 * ROUNDS_LEFT])


def test_edge_quarantine_values_the_walks_through_the_edge_and_its_structural_share(p, priced, structure):
    _, edge_rel = structure
    H, R, _ = numpy_reach(A_NP, R_NP, HOPS)
    direct = A_NP * (R_NP * D)[:, None] * R[None, :]
    spread = float((R_NP * D * H).sum())
    others = np.maximum(spread - R_NP * D * H, 0.0)[:, None]
    gain = direct + edge_rel * others
    C = numpy_centrality(A_NP, T_ROUND, T_END)
    loss = LAM * (1 - R_NP)[:, None] * p.rho["edge_quarantine"] * (C / A_NP.sum(1))[:, None] * A_NP
    got = priced[K.EDGE_QUARANTINE]
    np.testing.assert_allclose((got.net[0].numpy() * A_NP), (gain - loss) * A_NP, rtol=1e-9)
    assert got.cost[0][A_NP > 0].tolist() == [1.0 * ROUNDS_LEFT] * 4        # one edge, every remaining delivery
    assert not (got.valid[0] & torch.as_tensor(A_NP == 0)).any()                      # only existing edges


def test_sanitize_values_one_round_of_the_edge_and_is_stripped_with_probability_sigma(p, priced):
    H, R, _ = numpy_reach(A_NP, R_NP, HOPS)
    direct = A_NP * (R_NP * D)[:, None] * R[None, :]
    C = numpy_centrality(A_NP, T_ROUND, T_END)
    loss = LAM * (1 - R_NP)[:, None] * p.rho["sanitize"] * (C / A_NP.sum(1))[:, None] * A_NP
    expected = (SIGMA * direct / HOPS - loss) * A_NP
    got = priced[K.SANITIZE]
    np.testing.assert_allclose(got.net[0].numpy() * A_NP, expected, rtol=1e-9)
    assert got.cost[0][A_NP > 0].tolist() == [1.0] * 4 and got.recurrence[0][A_NP > 0].tolist() == [3.0] * 4


def test_delay_removes_the_last_hop_of_the_cascade_behind_the_edge(p, priced):
    """The value of R_j (value behind j over K hops) minus the same over K - 1 hops."""
    _, R_full, _ = numpy_reach(A_NP, R_NP, HOPS)
    _, R_short, _ = numpy_reach(A_NP, R_NP, HOPS - 1)
    direct = A_NP * (R_NP * D)[:, None] * (R_full - R_short)[None, :]
    C = numpy_centrality(A_NP, T_ROUND, T_END)
    loss = LAM * (1 - R_NP)[:, None] * p.rho["delay"] * (C / A_NP.sum(1))[:, None] * A_NP
    np.testing.assert_allclose(priced[K.DELAY].net[0].numpy() * A_NP, (direct - loss) * A_NP, rtol=1e-9)


def test_delaying_the_last_delivery_that_can_matter_removes_all_of_it(p):
    """With one hop left (t = T - 1), a one-round delay pushes the delivery past the end."""
    w = world(t=T_END - 1)
    got = pr.price(w, p, COSTS_MAIN, elig=torch.ones(1, 3, dtype=torch.bool), kinds=frozenset(K), topology=True)
    reach = pr.reach(w, pr.transmission(w, p), p, topology=True)
    C = numpy_centrality(A_NP, T_END - 1, T_END)
    loss = LAM * (1 - R_NP)[:, None] * p.rho["delay"] * (C / A_NP.sum(1))[:, None] * A_NP
    at_stake = R_NP[:, None] * reach.edge[0].numpy()
    np.testing.assert_allclose(got[K.DELAY].net[0].numpy() * A_NP, (at_stake - loss) * A_NP, rtol=1e-9)
    san_loss = LAM * (1 - R_NP)[:, None] * p.rho["sanitize"] * (C / A_NP.sum(1))[:, None] * A_NP
    np.testing.assert_allclose(got[K.SANITIZE].net[0].numpy() * A_NP, (SIGMA * at_stake - san_loss) * A_NP, rtol=1e-9)


def test_rate_limit_and_degradation_remove_a_share_of_the_agents_own_reach(p, priced):
    H, _, _ = numpy_reach(A_NP, R_NP, HOPS)
    own = R_NP * D * H
    C = numpy_centrality(A_NP, T_ROUND, T_END)
    out = A_NP.sum(1)
    efficacy = np.maximum(0.0, 1.0 - p.rate_limit / out)
    np.testing.assert_allclose(priced[K.PRIVILEGE_DEGRADATION].net[0].numpy(),
                               SIGMA * own - LAM * (1 - R_NP) * p.rho["privilege_degradation"] * C, rtol=1e-9)
    rl = priced[K.RATE_LIMIT]
    np.testing.assert_allclose(rl.net[0].numpy(), efficacy * own - LAM * (1 - R_NP) * p.rho["rate_limit"] * C * efficacy,
                               rtol=1e-9)
    assert rl.valid[0].tolist() == [True, False, False]                 # only agent 0 sends to more than 1 receiver
    assert rl.cost[0][0].item() == pytest.approx((2 - p.rate_limit) * ROUNDS_LEFT)
    assert priced[K.PRIVILEGE_DEGRADATION].cost[0].tolist() == pytest.approx([6.0, 3.0, 3.0])


def test_costs_follow_the_cost_condition(p):
    w = world()
    got = pr.price(w, p, COSTS_ROBUST, elig=torch.ones(1, 3, dtype=torch.bool), kinds=frozenset(K), topology=True)
    assert got[K.ISOLATION].cost[0].tolist() == pytest.approx([3 * 2 * ROUNDS_LEFT, 3 * ROUNDS_LEFT, 3 * ROUNDS_LEFT])
    assert got[K.SANITIZE].cost[0][A_NP > 0].tolist() == [2.0] * 4
    assert got[K.PRIVILEGE_DEGRADATION].cost[0].tolist() == pytest.approx([2 * 2 * 3, 2 * 3, 2 * 3])


def test_an_ineligible_agent_is_never_a_candidate_for_a_restriction(p):
    w = world()
    got = pr.price(w, p, COSTS_MAIN, elig=torch.tensor([[True, False, False]]), kinds=frozenset(K), topology=True)
    for kind in (K.ISOLATION, K.PRIVILEGE_DEGRADATION, K.RATE_LIMIT, K.SANITIZE, K.DELAY, K.EDGE_QUARANTINE):
        assert not got[kind].valid[0, 1:].any(), kind          # agents 1 and 2 (rows of the edge kinds too)


def test_only_candidates_with_positive_net_value_are_valid(p, priced):
    for kind, c in priced.items():
        assert (c.net[c.valid] > 0).all(), kind


def test_nothing_is_worth_buying_on_the_last_round_even_without_topology(p):
    for topology in (True, False):
        w = world(t=T_END)
        got = pr.price(w, p, COSTS_MAIN, elig=torch.ones(1, 3, dtype=torch.bool), kinds=frozenset(K), topology=topology)
        assert not any(c.valid.any() for c in got.values())


def test_without_topology_each_agent_stands_for_one_unit_at_stake_and_the_graph_is_ignored(p):
    w = world()
    got = pr.price(w, p, COSTS_MAIN, elig=torch.ones(1, 3, dtype=torch.bool), kinds=frozenset(K), topology=False)
    C = numpy_centrality(A_NP, T_ROUND, T_END)
    np.testing.assert_allclose(got[K.ISOLATION].net[0].numpy(), R_NP * 1.0 - LAM * (1 - R_NP) * p.rho["isolation"] * C,
                               rtol=1e-9)
    # the same value for two agents of the same posterior but different position in the graph
    w2 = world(r=[0.5, 0.5, 0.5])
    g2 = pr.price(w2, p, COSTS_MAIN, elig=torch.ones(1, 3, dtype=torch.bool), kinds=frozenset(K), topology=False)
    gain = g2[K.ISOLATION].net[0] + LAM * 0.5 * p.rho["isolation"] * torch.as_tensor(C)
    assert torch.allclose(gain, torch.full((3,), 0.5, dtype=F64))


def test_the_kinds_argument_removes_actions_from_the_choice(p):
    w = world()
    got = pr.price(w, p, COSTS_MAIN, elig=torch.ones(1, 3, dtype=torch.bool), kinds=frozenset({K.ISOLATION}),
                   topology=True)
    assert set(got) == {K.ISOLATION}


def test_an_isolated_agent_can_only_be_reinstated_and_a_free_one_cannot(p):
    w = world(isolated=[0])
    elig = torch.zeros(1, 3, dtype=torch.bool)
    got = pr.price(w, p, COSTS_MAIN, elig=elig, kinds=frozenset(K), topology=True)
    assert not got[K.ISOLATION].valid[0, 0] and not got[K.PRIVILEGE_DEGRADATION].valid[0, 0]
    free = pr.price(world(), p, COSTS_MAIN, elig=elig, kinds=frozenset(K), topology=True)
    assert not free[K.REINSTATEMENT].valid.any()


def test_reinstating_an_agent_that_is_probably_benign_pays_and_one_that_is_probably_not_does_not(p):
    elig = torch.zeros(1, 3, dtype=torch.bool)
    benign = pr.price(world(r=[0.02, 0.2, 0.1], isolated=[0]), p, COSTS_MAIN, elig=elig, kinds=frozenset(K), topology=True)
    guilty = pr.price(world(r=[0.9, 0.2, 0.1], isolated=[0]), p, COSTS_MAIN, elig=elig, kinds=frozenset(K), topology=True)
    assert benign[K.REINSTATEMENT].valid[0, 0] and not guilty[K.REINSTATEMENT].valid[0, 0]
    still_eligible = pr.price(world(r=[0.02, 0.2, 0.1], isolated=[0]), p, COSTS_MAIN,
                              elig=torch.ones(1, 3, dtype=torch.bool), kinds=frozenset(K), topology=True)
    assert still_eligible[K.REINSTATEMENT].valid[0, 0]            # the net decides, not the eligibility
    # the platform charges a reinstatement per delivery the agent sends again, over the remaining rounds
    assert benign[K.REINSTATEMENT].cost[0, 0].item() == 2 * ROUNDS_LEFT
    # restoring agent 0 (r = 0.02): utility regained minus the reach of the restored agent, on the graph where it sends
    full = np.array(A_NP)
    H, _, _ = numpy_reach(full, np.array([0.02, 0.2, 0.1]), HOPS)
    C = numpy_centrality(full, T_ROUND, T_END)
    expected = LAM * 0.98 * p.rho["isolation"] * C[0] - 0.02 * D * H[0]
    assert benign[K.REINSTATEMENT].net[0, 0].item() == pytest.approx(expected, rel=1e-9)


def test_an_agent_with_a_quarantined_edge_into_it_can_be_reinstated_too(p):
    """Reinstatement lifts every quarantine that touches the agent, in either direction (enforcement rules)."""
    w = world(r=[0.02, 0.2, 0.1], quarantined=[(2, 0)])
    got = pr.price(w, p, COSTS_MAIN, elig=torch.zeros(1, 3, dtype=torch.bool), kinds=frozenset(K), topology=True)
    assert got[K.REINSTATEMENT].valid[0, 0] and not got[K.REINSTATEMENT].valid[0, 1]     # 1 touches no cut edge


def test_a_quarantined_edge_is_no_longer_a_candidate_and_no_longer_carries_reach(p):
    w = world(quarantined=[(0, 1)])
    reach = pr.reach(w, pr.transmission(w, p), p, topology=True)
    assert reach.edge[0, 0, 1].item() == 0.0
    got = pr.price(w, p, COSTS_MAIN, elig=torch.ones(1, 3, dtype=torch.bool), kinds=frozenset(K), topology=True)
    for kind in (K.SANITIZE, K.DELAY, K.EDGE_QUARANTINE):
        assert not got[kind].valid[0, 0, 1]


def test_a_degraded_sender_transmits_less_and_is_not_degraded_again(p):
    base, deg = world(), world(degraded=[0])
    r0 = pr.reach(base, pr.transmission(base, p), p, topology=True)
    r1 = pr.reach(deg, pr.transmission(deg, p), p, topology=True)
    assert r1.edge[0, 0].sum().item() == pytest.approx(r0.edge[0, 0].sum().item() * (1 - SIGMA), rel=1e-9)
    got = pr.price(deg, p, COSTS_MAIN, elig=torch.ones(1, 3, dtype=torch.bool), kinds=frozenset(K), topology=True)
    assert not got[K.PRIVILEGE_DEGRADATION].valid[0, 0]


def test_the_inspection_value_is_the_value_of_information_bounded_by_the_error_it_can_avoid(p):
    """q * min(r * at_stake, (1 - r) * lambda * centrality): settling a true positive saves what is at stake, clearing a
    false positive saves the utility it would have cost. The same form as bazt.actions.score_inspect."""
    w = world()
    reach = pr.reach(w, pr.transmission(w, p), p, topology=True)
    voi = pr.inspect_value(w, p, reach)
    C = numpy_centrality(A_NP, T_ROUND, T_END)
    H, _, _ = numpy_reach(A_NP, R_NP, HOPS)
    expected = p.q_inspect * np.minimum(R_NP * D * H, (1 - R_NP) * LAM * C)
    np.testing.assert_allclose(voi[0].numpy(), expected, rtol=1e-9)
    iso = world(isolated=[1])
    assert pr.inspect_value(iso, p, pr.reach(iso, pr.transmission(iso, p), p, topology=True))[0, 1].item() == 0.0


def test_agent_level_actions_are_priced_on_every_live_delivery_quarantined_edges_included(p):
    """The platform charges an agent-level action on the agent's live deliveries (base edge, both ends active), an edge
    it quarantined earlier included. A price on the open edges only would undershoot the charge and overspend."""
    w = world(quarantined=[(0, 1)])                       # agent 0 sends to 1 (quarantined) and 2
    got = pr.price(w, p, COSTS_MAIN, elig=torch.ones(1, 3, dtype=torch.bool), kinds=frozenset(K), topology=True)
    assert got[K.ISOLATION].cost[0, 0].item() == 2 * ROUNDS_LEFT
    assert got[K.PRIVILEGE_DEGRADATION].cost[0, 0].item() == 2 * ROUNDS_LEFT
    assert got[K.RATE_LIMIT].cost[0, 0].item() == (2 - p.rate_limit) * ROUNDS_LEFT
