"""The value of each of the eight actions on a hand-built graph, checked against explicit numpy sums.

Graph: 0 -> 1, 0 -> 2, 1 -> 2, 2 -> 0 (strongly connected). T = 4, t = 2, so K = 2 hops remain and 3 rounds are left to
pay for. The Katz decay is bazt.katz_decay over the spectral radius of the graph (D115), the radius taken from numpy.
Every number below is recomputed here from the definitions, not read back from the code under test.
"""
import numpy as np
import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.bazt_slot.private import contagion as ct
from bazt_lab.features.bazt_slot.private import params as P
from bazt_lab.features.bazt_slot.private import pricing as pr
from bazt_lab.features.bazt_slot.tests.testkit import (
    COSTS_MAIN,
    COSTS_ROBUST,
    adj_of,
    barabasi_albert,
    make_settings,
    small_world,
    spectral_radius_np,
    star_hub,
)

F64 = torch.float64
K = core.ActionKind
EDGES = [(0, 1), (0, 2), (1, 2), (2, 0)]
A_NP = np.zeros((3, 3))
for i, j in EDGES:
    A_NP[i, j] = 1.0
R_NP = np.array([0.9, 0.2, 0.1])
KATZ, SIGMA, LAM = 0.5, 0.9, 1.0         # bazt.katz_decay, sanitize_success, utility_weight
D = KATZ / spectral_radius_np(A_NP)      # the Katz decay on this graph (D115)
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


def restored_structure(r, quarantined=(), isolated=()):
    """D118 reference from numpy: on the restored graph adj & ~quarantined (every isolated agent sends again), the Perron
    pair from numpy.linalg.eig, the NetShield drop of each node, the first-order NetMelt value v_i u_j / (lam v.u) of
    every pair (masked by the caller to the lifted edges), and `others`: the spread of the active agents that does not
    start at the agent, on the same graph."""
    A = A_NP.copy()
    for i, j in quarantined:
        A[i, j] = 0.0
    vals, right = np.linalg.eig(A)
    lam = float(vals.real.max())
    node_rel, edge_rel = np.zeros(3), np.zeros((3, 3))
    if lam > 1e-9:
        u = np.abs(right[:, vals.real.argmax()].real)
        lvals, left = np.linalg.eig(A.T)
        v = np.abs(left[:, lvals.real.argmax()].real)
        vu = v @ u
        node_rel = np.clip(2 * v * u / vu, 0, 1)
        edge_rel = np.clip(np.outer(v, u) / (lam * vu), 0, 1)
    s = 1 - np.asarray(r)
    H = sum(D ** (k - 1) * np.linalg.matrix_power(A, k) @ s for k in range(1, HOPS + 1))
    active = np.ones(3)
    active[list(isolated)] = 0.0
    own = np.asarray(r) * D * H
    others = np.maximum((active * own).sum() - active * own, 0.0)
    return node_rel, edge_rel, others


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
    node_rel, _, others = restored_structure([0.02, 0.2, 0.1], isolated=[0])
    expected = LAM * 0.98 * p.rho["isolation"] * C[0] - 0.02 * D * H[0] - node_rel[0] * others[0]     # D118 structural share
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


# --------------------------------------------- D115: the decay is relative to each episode's graph, on the grid topologies
GRID = {"star_hub": star_hub(12), "small_world": small_world(30), "barabasi_albert": barabasi_albert(100)}


def grid_world(graphs, r, *, t, T, dtype=F64, decay=None):
    """Every agent active, nothing restricted, sink 0: the transmission matrix is the base graph."""
    adj = torch.as_tensor(np.stack(graphs) > 0)
    B, N, _ = adj.shape
    no = torch.zeros(B, N, dtype=torch.bool)
    return pr.World(t=t, T=T, adj=adj, live=adj.clone(), quarantined=torch.zeros_like(adj), active=~no, isolated=no,
                    degraded=no.clone(), limited=no.clone(), r=torch.as_tensor(np.asarray(r), dtype=dtype),
                    sink=torch.zeros(B, dtype=torch.int64), decay=decay)


def explicit_reach(A, r, d, hops, sink_weight):
    """node = d H, edge = d A_ij R_j with H = sum_{k=1..hops} d^(k-1) A^k s, R = s + d H^(hops-1), s = (1 - r) w."""
    w = np.ones(len(A))
    w[0] = sink_weight
    s = (1 - r) * w
    def H(h):
        return sum(d ** (k - 1) * np.linalg.matrix_power(A, k) @ s for k in range(1, h + 1)) if h > 0 else 0 * s
    return d * H(hops), d * A * (s + d * H(hops - 1))[None, :]


@pytest.fixture(scope="module")
def p09():
    return P.read(make_settings(bazt__katz_decay=0.9, bazt__sink_weight=2.0))


@pytest.mark.parametrize("name", sorted(GRID))
def test_what_is_at_stake_uses_katz_decay_over_the_spectral_radius_of_the_graph(name, p09):
    A = GRID[name]
    r = np.random.default_rng(7).uniform(0.0, 0.6, len(A))
    w = grid_world([A], r[None], t=4, T=10)                                # 6 hops left
    reach = pr.reach(w, pr.transmission(w, p09), p09, topology=True)
    node, edge = explicit_reach(A, r, 0.9 / spectral_radius_np(A), 6, 2.0)
    np.testing.assert_allclose(reach.node[0].numpy(), node, rtol=1e-9)
    np.testing.assert_allclose(reach.edge[0].numpy(), edge, rtol=1e-9, atol=1e-15)


@pytest.mark.parametrize("name", sorted(GRID))
def test_what_is_at_stake_stays_bounded_over_a_long_horizon_at_09(name, p09):
    """d x rho = 0.9 < 1: the reach over 35 hops (a seed run) is below the infinite Katz sum d A (I - d A)^-1 s, and over
    400 hops it is that sum."""
    A = GRID[name]
    r = np.random.default_rng(8).uniform(0.0, 0.6, len(A))
    d = 0.9 / spectral_radius_np(A)
    s = (1 - r) * np.where(np.arange(len(A)) == 0, 2.0, 1.0)
    limit = d * (A @ np.linalg.solve(np.eye(len(A)) - d * A, s))
    seed_run, long_run = (pr.reach(w, pr.transmission(w, p09), p09, topology=True).node[0].numpy()
                          for w in (grid_world([A], r[None], t=1, T=36), grid_world([A], r[None], t=1, T=401)))
    assert (seed_run <= limit * (1 + 1e-9)).all()
    np.testing.assert_allclose(long_run, limit, rtol=1e-9)


def test_a_batch_of_different_graphs_prices_each_episode_on_its_own_radius(p09):
    # N = 12 (the star_hub size of the grid): CPU (MKL) bmm is bitwise batch-independent there; from about N = 30 it
    # differs in the last ulp between batch sizes, for every bmm of the lane (see CONTRACT.md, Compute)
    graphs = [star_hub(12), small_world(12, seed=2), barabasi_albert(12, m=2, seed=3)]
    r = np.random.default_rng(9).uniform(0.0, 0.6, (3, 12))
    w = grid_world(graphs, r, t=3, T=8)                                    # 5 hops left
    reach = pr.reach(w, pr.transmission(w, p09), p09, topology=True)
    for b, A in enumerate(graphs):
        node, edge = explicit_reach(A, r[b], 0.9 / spectral_radius_np(A), 5, 2.0)
        np.testing.assert_allclose(reach.node[b].numpy(), node, rtol=1e-9)
        np.testing.assert_allclose(reach.edge[b].numpy(), edge, rtol=1e-9, atol=1e-15)
        alone = grid_world([A], r[b:b + 1], t=3, T=8)
        one = pr.reach(alone, pr.transmission(alone, p09), p09, topology=True)
        assert torch.equal(reach.node[b], one.node[0]) and torch.equal(reach.edge[b], one.edge[0])
        assert torch.equal(reach.tail[b], one.tail[0])


def test_the_decay_a_world_carries_is_the_one_priced_with(p09):
    """BAZT computes the decay once per episode at reset and hands it to every World; pricing uses it as given."""
    A = GRID["star_hub"]
    r = np.full((1, 12), 0.3)
    given = grid_world([A], r, t=4, T=10, decay=torch.tensor([0.05], dtype=F64))
    reach = pr.reach(given, pr.transmission(given, p09), p09, topology=True)
    node, _ = explicit_reach(A, r[0], 0.05, 6, 2.0)
    np.testing.assert_allclose(reach.node[0].numpy(), node, rtol=1e-12)
    default = grid_world([A], r, t=4, T=10)
    assert pr.decay(default, p09).tolist() == pytest.approx([0.9 / spectral_radius_np(A)], rel=1e-9)


def test_float32_prices_agree_with_float64_on_the_relative_decay(p09):
    A = GRID["barabasi_albert"]
    r = np.random.default_rng(10).uniform(0.0, 0.6, (1, 100))
    got = {}
    for dtype in (F64, torch.float32):
        w = grid_world([A], r, t=1, T=36, dtype=dtype)
        got[dtype] = pr.reach(w, pr.transmission(w, p09), p09, topology=True)
        assert got[dtype].node.dtype == got[dtype].edge.dtype == got[dtype].tail.dtype == dtype
    np.testing.assert_allclose(got[torch.float32].node[0].double().numpy(), got[F64].node[0].numpy(), rtol=1e-4)
    np.testing.assert_allclose(got[torch.float32].edge[0].double().numpy(), got[F64].edge[0].numpy(), rtol=1e-4,
                               atol=1e-6)


def test_reinstatement_prices_the_restored_reach_with_the_same_episode_decay(p09):
    """The graph where the agent sends again is a subgraph of the base graph: the same decay, so d x rho <= katz_decay."""
    A = GRID["star_hub"]
    r = np.full(12, 0.05)
    r[0] = 0.02
    adj = torch.as_tensor(A > 0)[None]
    iso = torch.zeros(1, 12, dtype=torch.bool)
    iso[0, 0] = True
    live = adj & ~iso[:, :, None] & ~iso[:, None, :]
    no = torch.zeros(1, 12, dtype=torch.bool)
    w = pr.World(t=2, T=5, adj=adj, live=live, quarantined=torch.zeros_like(adj), active=~iso, isolated=iso,
                 degraded=no, limited=no.clone(), r=torch.as_tensor(r[None]), sink=torch.zeros(1, dtype=torch.int64))
    got = pr.price(w, p09, COSTS_MAIN, elig=no.clone(), kinds=frozenset({K.REINSTATEMENT}), topology=True)
    d = 0.9 / spectral_radius_np(A)
    s = (1 - r) * np.where(np.arange(12) == 0, 2.0, 1.0)
    H = sum(d ** (k - 1) * np.linalg.matrix_power(A, k) @ s for k in range(1, 4))
    C = numpy_centrality(A, 2, 5)
    vals, vecs = np.linalg.eigh(A)                       # symmetric star: u = v, the hub carries half the Perron mass
    u = np.abs(vecs[:, vals.argmax()])
    node_rel = min(2 * u[0] ** 2 / (u @ u), 1.0)
    own = r * d * H
    others = own.sum() - own[0]                          # D118: the spread of every other agent on the restored graph
    expected = LAM * (1 - r[0]) * p09.rho["isolation"] * C[0] - r[0] * d * H[0] - node_rel * others
    assert node_rel == pytest.approx(1.0)
    assert got[K.REINSTATEMENT].net[0, 0].item() == pytest.approx(expected, rel=1e-9)


# ---------------------------------------- reinstatement after a quarantine: the lifted deliveries carry their risk again
def lifted_world(r, quarantined):
    return world(r=r, quarantined=quarantined)


def numpy_full_reach(r, quarantined):
    """H, R on the graph where every quarantine stays (s over every agent), as `_reinstatement` builds it."""
    A = A_NP.copy()
    for i, j in quarantined:
        A[i, j] = 0.0
    s = 1 - np.asarray(r)
    def H(h):
        return sum(D ** (k - 1) * np.linalg.matrix_power(A, k) @ s for k in range(1, h + 1)) if h > 0 else 0 * s
    return H(HOPS), s + D * H(HOPS - 1)


def test_reinstating_a_sender_whose_edges_are_all_quarantined_prices_the_reach_those_edges_restore(p):
    """Agent 0 (r = 0.9) has both its out-edges quarantined. Lifting them restores its reach H_0 + sum_j q_0j R_j on the
    graph where it sends again; pricing it on the graph without those edges (reach 0) would reinstate it for free."""
    r = [0.9, 0.05, 0.05]
    q = [(0, 1), (0, 2)]
    got = pr.price(lifted_world(r, q), p, COSTS_MAIN, elig=torch.zeros(1, 3, dtype=torch.bool),
                   kinds=frozenset({K.REINSTATEMENT}), topology=True)[K.REINSTATEMENT]
    H, R = numpy_full_reach(r, q)
    restored = H[0] + R[1] + R[2]
    C = numpy_centrality(A_NP, T_ROUND, T_END)
    q_share = 2 / 3                                    # 2 of its 3 edges (out 0->1, 0->2, in 2->0) are lifted
    _, edge_rel, others = restored_structure(r, q)
    structural = sum(edge_rel[i, j] * others[i] for i, j in q)       # 0 here: 1 -> 2 -> 0 alone has no cycle
    expected = LAM * (1 - r[0]) * q_share * p.rho["edge_quarantine"] * C[0] - 1.0 * r[0] * D * restored - structural
    assert got.net[0, 0].item() == pytest.approx(expected, rel=1e-9)
    assert not got.valid[0, 0]


def test_reinstating_the_receiver_of_a_quarantined_edge_prices_the_delivery_it_reopens(p):
    """Reinstatement lifts the quarantines into the agent too: 0 -> 1 reopens, worth r_0 d R_1 (the complement of what
    quarantining it removed). A benign receiver of a likely compromised sender is not reinstated for its utility alone."""
    r = [0.9, 0.05, 0.05]
    q = [(0, 1)]
    got = pr.price(lifted_world(r, q), p, COSTS_MAIN, elig=torch.zeros(1, 3, dtype=torch.bool),
                   kinds=frozenset({K.REINSTATEMENT}), topology=True)[K.REINSTATEMENT]
    _, R = numpy_full_reach(r, q)
    C = numpy_centrality(A_NP, T_ROUND, T_END)
    q_share = 1 / 2                                    # agent 1 has 2 edges (in 0->1, out 1->2), one is lifted
    _, edge_rel, others = restored_structure(r, q)
    assert edge_rel[0, 1] > 0                          # 0 -> 2 -> 0 carries the radius, 0 -> 1 would feed it
    expected = (LAM * (1 - r[1]) * q_share * p.rho["edge_quarantine"] * C[1] - r[0] * D * R[1]
                - edge_rel[0, 1] * others[0])           # D118: the NetMelt share of the reopened edge, its sender's spread
    assert got.net[0, 1].item() == pytest.approx(expected, rel=1e-9)
    assert not got.valid[0, 1]


def test_without_topology_the_reopened_delivery_is_one_unit_at_stake_per_compromised_sender(p):
    r = [0.9, 0.05, 0.05]
    got = pr.price(lifted_world(r, [(0, 1)]), p, COSTS_MAIN, elig=torch.zeros(1, 3, dtype=torch.bool),
                   kinds=frozenset({K.REINSTATEMENT}), topology=False)[K.REINSTATEMENT]
    C = numpy_centrality(A_NP, T_ROUND, T_END)
    expected = LAM * (1 - r[1]) * 0.5 * p.rho["edge_quarantine"] * C[1] - r[0] * 1.0
    assert got.net[0, 1].item() == pytest.approx(expected, rel=1e-9)


def test_an_isolation_that_still_pays_is_not_also_worth_lifting(p):
    """D118: the two nets are complements, so the same world never makes both the isolation of agent 0 and its
    reinstatement worth buying (r_0 = 0.5 made both pay before the structural share was counted)."""
    r = [0.5, 0.6, 0.6]
    iso = pr.price(world(r=r), p, COSTS_MAIN, elig=torch.ones(1, 3, dtype=torch.bool), kinds=frozenset({K.ISOLATION}),
                   topology=True)[K.ISOLATION]
    back = pr.price(world(r=r, isolated=[0]), p, COSTS_MAIN, elig=torch.zeros(1, 3, dtype=torch.bool),
                    kinds=frozenset({K.REINSTATEMENT}), topology=True)[K.REINSTATEMENT]
    assert iso.valid[0, 0]
    assert not back.valid[0, 0]


@pytest.mark.parametrize("r0", [0.1, 0.3, 0.45, 0.5, 0.6, 0.8, 0.95])
def test_isolation_and_reinstatement_are_priced_as_complements(p, r0):
    """Same posterior, same round: lifting the isolation of agent 0 is worth exactly minus isolating it (own reach, the
    NetShield share of the others' spread, the utility), so at most one of the two pays (D118)."""
    r = [r0, 0.6, 0.2]
    iso = pr.price(world(r=r), p, COSTS_MAIN, elig=torch.ones(1, 3, dtype=torch.bool), kinds=frozenset({K.ISOLATION}),
                   topology=True)[K.ISOLATION]
    back = pr.price(world(r=r, isolated=[0]), p, COSTS_MAIN, elig=torch.zeros(1, 3, dtype=torch.bool),
                    kinds=frozenset({K.REINSTATEMENT}), topology=True)[K.REINSTATEMENT]
    assert back.net[0, 0].item() == pytest.approx(-iso.net[0, 0].item(), rel=1e-9, abs=1e-12)
    assert not (iso.valid[0, 0] and back.valid[0, 0])


def test_the_restored_structure_is_priced_per_episode_in_a_batch(p):
    """Three episodes with different restrictions in one batch: each reinstatement net equals its episode priced alone."""
    worlds = [world(r=[0.5, 0.6, 0.2], isolated=[0]), world(r=[0.9, 0.05, 0.05], quarantined=[(0, 1)]),
              world(r=[0.3, 0.3, 0.3])]

    def cat(field):
        return torch.cat([getattr(w, field) for w in worlds])

    batch = pr.World(T_ROUND, T_END, cat("adj"), cat("live"), cat("quarantined"), cat("active"), cat("isolated"),
                     cat("degraded"), cat("limited"), cat("r"), cat("sink"))
    elig = torch.zeros(3, 3, dtype=torch.bool)
    got = pr.price(batch, p, COSTS_MAIN, elig=elig, kinds=frozenset({K.REINSTATEMENT}), topology=True)[K.REINSTATEMENT]
    for b, w in enumerate(worlds):
        one = pr.price(w, p, COSTS_MAIN, elig=elig[:1], kinds=frozenset({K.REINSTATEMENT}), topology=True)[K.REINSTATEMENT]
        assert torch.equal(got.net[b], one.net[0]) and torch.equal(got.valid[b], one.valid[0])


def test_float32_reinstatement_agrees_with_float64(p):
    for r0 in (0.3, 0.6):
        nets = {}
        for dtype in (F64, torch.float32):
            w = world(r=[r0, 0.6, 0.2], isolated=[0])
            w = pr.World(**{**w.__dict__, "r": w.r.to(dtype)})
            got = pr.price(w, p, COSTS_MAIN, elig=torch.zeros(1, 3, dtype=torch.bool), kinds=frozenset({K.REINSTATEMENT}),
                           topology=True)[K.REINSTATEMENT]
            assert got.net.dtype == dtype and got.cost.dtype == F64          # accounting stays float64
            nets[dtype] = got.net[0, 0].item()
        assert nets[torch.float32] == pytest.approx(nets[F64], rel=1e-5, abs=1e-6)
