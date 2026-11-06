"""World construction: what is random and identical across arms (graph, readings, placements, noise), per-episode
parameters, and refusals. A world depends on its (topology, N, seed) keys only, never on the arm or on the batch."""
import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.adaptive import public as adaptive
from bazt_lab.features.mock_engine.private import world as W
from bazt_lab.features.mock_engine.tests.helpers import make_settings, unit


def build(units, settings=None, T=5, topology=None, N=None, family=None):
    s = settings or make_settings()
    u0 = units[0]
    topology, N, family = topology or u0["topology"], N or u0["agents"], family or u0["attack"]
    adj = W.make_graphs(units, s, topology, N)
    return W.build_world(units, s, topology, N, family, adj, T)


def seeds(n, **kw):
    return [unit(seed=s, **kw) for s in range(1, n + 1)]


def test_the_graph_and_readings_do_not_depend_on_the_arm_attack_budget_or_sink():
    a = build([unit(arm="A", attack="none", seed=3, sink=0, topology="small_world", agents=12)])
    b = build([unit(arm="C", attack="insider_pi", seed=3, sink=5, audit=0.1, intervention=0.2, topology="small_world",
                    agents=12)])
    assert torch.equal(a.adj, b.adj) and torch.equal(a.x, b.x)
    for name in ("mon_z", "u_adopt", "u_stay", "u_decoy", "u_strip", "u_aitm", "aud_z", "burst"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    assert a.sink.item() == 0 and b.sink.item() == 5


def test_each_episode_depends_only_on_its_own_keys_not_on_the_batch():
    full = build(seeds(6, topology="barabasi_albert", agents=10, attack="infection"))
    one = build([unit(seed=4, topology="barabasi_albert", agents=10, attack="infection")])
    for name in ("adj", "x", "seed", "decoy", "burst", "mon_z", "u_adopt", "aud_z", "u_strip"):
        assert torch.equal(getattr(full, name)[3:4], getattr(one, name)), name


def test_units_with_the_same_seed_share_one_world_whatever_their_sink():
    w = build([unit(seed=2, sink=k) for k in range(4)])
    for name in ("adj", "x", "mon_z", "u_adopt", "aud_z"):
        t = getattr(w, name)
        assert all(torch.equal(t[0], t[k]) for k in range(4)), name
    assert w.sink.tolist() == [0, 1, 2, 3]


def test_the_readings_do_not_depend_on_the_horizon():
    a, b = build([unit(seed=7)], T=4), build([unit(seed=7)], T=9)
    assert torch.equal(a.x, b.x) and torch.equal(a.decoy, b.decoy) and torch.equal(a.seed, b.seed)


def test_different_seeds_give_different_episodes():
    w = build(seeds(2, topology="small_world", agents=12))
    assert not torch.equal(w.x[0], w.x[1]) and not torch.equal(w.mon_z[0], w.mon_z[1])


def test_private_readings_are_normal_with_the_configured_mean_and_sd():
    w = build(seeds(1500, agents=6))
    assert w.x.dtype == torch.float64
    assert w.x.mean() == pytest.approx(50.0, abs=0.4) and w.x.std() == pytest.approx(10.0, abs=0.3)
    assert torch.allclose(w.mean, w.x.mean(1))


def test_noise_streams_have_the_right_distributions():
    w = build(seeds(800, agents=6), T=6)
    assert w.mon_z.shape == (800, 6, 6) and w.aud_z.shape == (800, 6, 6)
    for z in (w.mon_z, w.aud_z):
        assert z.mean().abs() < 0.05 and z.std() == pytest.approx(1.0, abs=0.03)
    for u in (w.u_adopt, w.u_stay, w.u_decoy, w.u_strip, w.u_aitm):
        assert u.dtype == torch.float64 and 0.0 <= u.min() and u.max() < 1.0 and u.mean() == pytest.approx(0.5, abs=0.02)
    assert not torch.equal(w.u_adopt, w.u_stay) and not torch.equal(w.mon_z, w.aud_z)


def test_rate_limit_priorities_are_keyed_by_episode_and_round_and_not_by_the_batch():
    w = build(seeds(4, agents=6))
    rows = torch.tensor([1, 3])
    u = W.rate_priority(w, rows, 2)
    assert u.shape == (2, 6, 6) and u.dtype == torch.float64 and 0.0 <= u.min() and u.max() < 1.0
    assert torch.equal(u, W.rate_priority(w, torch.tensor([3, 1]), 2).flip(0))
    assert not torch.equal(u, W.rate_priority(w, rows, 3)) and not torch.equal(u[0], u[1])
    one = build([unit(seed=2)])
    assert torch.equal(W.rate_priority(one, torch.tensor([0]), 2)[0], u[0])


def test_metropolis_weights_use_the_base_graph_degrees():
    adj = torch.zeros(1, 4, 4, dtype=torch.bool)
    for i, j in [(0, 1), (0, 2), (0, 3)]:
        adj[0, i, j] = adj[0, j, i] = True
    Wt = W.metropolis(adj)
    assert Wt[0, 0, 1].item() == pytest.approx(1 / (1 + 3)) and Wt[0, 1, 0].item() == pytest.approx(0.25)
    assert Wt[0, 1, 2].item() == 0.0 and (Wt.sum(2) <= 1.0).all() and torch.equal(Wt, Wt.transpose(1, 2))


def test_star_hub_enters_at_the_hub_and_star_leaf_at_a_random_leaf():
    hub = build(seeds(200, topology="star_hub", agents=8, attack="infection"))
    assert hub.seed[:, 0].all() and hub.seed.sum() == 200
    leaf = build(seeds(400, topology="star_leaf", agents=8, attack="infection"))
    assert not leaf.seed[:, 0].any() and (leaf.seed.sum(1) == 1).all()
    assert (leaf.seed.sum(0)[1:] > 20).all()                   # every leaf gets the entry sometimes
    assert not leaf.attackers.any() and (leaf.victim == -1).all()


def test_entry_is_uniform_on_other_topologies():
    w = build(seeds(1200, topology="chain", agents=6, attack="infection"))
    share = w.seed.sum(0).double() / 1200
    assert torch.allclose(share, torch.full((6,), 1 / 6, dtype=torch.float64), atol=0.05)


@pytest.mark.parametrize("family", ["insider_pi", "insider_ta", "insider_ma"])
def test_insiders_are_n_distinct_persistent_attackers(family):
    w = build(seeds(50, agents=10, attack=family))
    assert (w.attackers.sum(1) == 2).all() and torch.equal(w.attackers, w.seed)
    assert (w.victim == -1).all()


def test_the_primary_insider_follows_the_topology_entry_rule():
    hub = build(seeds(50, topology="star_hub", agents=8, attack="insider_pi"))
    assert hub.attackers[:, 0].all()
    leaf = build(seeds(50, topology="star_leaf", agents=8, attack="insider_pi"))
    assert (leaf.attackers[:, 1:].sum(1) >= 1).all()           # the primary insider is a leaf; the other may be the hub


def test_aitm_has_one_victim_and_no_seed():
    w = build(seeds(60, agents=7, attack="aitm"))
    assert not w.seed.any() and not w.attackers.any() and ((w.victim >= 0) & (w.victim < 7)).all()
    assert w.victim.unique().numel() > 3
    hub = build(seeds(10, topology="star_hub", agents=7, attack="aitm"))
    assert (hub.victim == 0).all()


def test_attack_free_has_no_attacker_seed_or_victim():
    w = build(seeds(5, attack="none"))
    assert not w.seed.any() and not w.attackers.any() and (w.victim == -1).all()


def test_decoys_are_benign_low_degree_agents_never_the_hub_or_the_attacker():
    w = build(seeds(300, topology="star_hub", agents=12, attack="infection"))
    assert (w.decoy.sum(1) == 3).all()                         # mock.model.decoys
    assert not (w.decoy & w.seed).any() and not w.decoy[:, 0].any()
    w = build(seeds(300, topology="barabasi_albert", agents=30, attack="insider_pi"))
    assert not (w.decoy & w.attackers).any()
    keep = ~w.attackers
    pool_deg = torch.where(w.decoy, w.deg, torch.zeros_like(w.deg))
    assert pool_deg.max(1).values.double().mean() <= w.deg.masked_select(keep).double().mean()        # low-degree half only


def test_decoys_are_the_same_across_attack_families_unless_the_attacker_is_in_the_way():
    free = build(seeds(150, agents=12, topology="small_world", attack="none"))
    inf = build(seeds(150, agents=12, topology="small_world", attack="infection"))
    same = (free.decoy == inf.decoy).all(1)
    assert same.double().mean() > 0.6
    assert (free.decoy & inf.seed).any(1)[~same].all()


def test_decoys_cannot_exceed_the_pool():
    w = build(seeds(5, agents=3, attack="infection"))
    assert (w.decoy.sum(1) <= 2).all()


def test_decoy_header_burst_is_one_or_two_rounds_from_round_one_by_default():
    w = build(seeds(200, agents=6), T=6)
    assert w.burst[:, 0, :].all()
    assert set(w.burst.sum(1).unique().tolist()) == {1, 2}


def test_burst_span_staggers_the_start_round_by_agent_index():
    w = build(seeds(50, agents=6), settings=make_settings(mock__model__burst_span=3), T=6)
    first = w.burst.int().argmax(1)                            # first burst round (0-based) per agent
    assert torch.equal(first, (torch.arange(6) % 3).expand(50, 6))


def test_sink_comes_from_the_unit_or_a_keyed_draw():
    w = build([unit(sink=3, seed=1), unit(sink=None, seed=2), unit(sink=None, seed=3)])
    assert w.sink[0].item() == 3 and 0 <= w.sink[1].item() < 6
    again = build([unit(sink=None, seed=2)])
    assert again.sink[0] == w.sink[1]
    with pytest.raises(ValueError, match="sink"):
        build([unit(sink=6)])


def test_per_episode_parameters_follow_the_family():
    s = make_settings(runtime__insider_bias=7.0, mock__model__bias=9.0)
    pi = build([unit(attack="insider_pi")], s)
    ta = build([unit(attack="insider_ta")], s)
    ma = build([unit(attack="insider_ma")], s)
    inf = build([unit(attack="infection")], s)
    aitm = build([unit(attack="aitm")], s)
    none = build([unit(attack="none")], s)
    assert [x.shift.item() for x in (pi, ta, ma)] == [7.0] * 3                       # runtime.insider_bias
    assert [x.mu_mal.item() for x in (pi, ta, ma)] == [2.0, 1.5, 1.2]                # mu_inf, ta, ma detectability
    assert inf.shift.item() == 9.0 and inf.mu_mal.item() == 2.0 and inf.rewrite.item() == 0.0   # mock.model.bias
    assert aitm.rewrite.item() == 1.0 and aitm.shift.item() == 9.0
    assert none.shift.item() == 0.0


def test_attack_params_override_per_unit_and_unknown_keys_are_refused():
    w = build([unit(attack="insider_pi", attack_params={"insider_bias": 3.0, "mu_inf": 1.0}),
               unit(attack="insider_pi", seed=2)])
    assert w.shift.tolist() == [3.0, 8.0] and w.mu_mal.tolist() == [1.0, 2.0]
    with pytest.raises(ValueError, match="attack_params"):
        build([unit(attack="insider_pi", attack_params={"bogus": 1})])
    w = build([unit(attack="insider_pi", attack_params={"n_insiders": 3}), unit(attack="insider_pi", seed=2)])
    assert w.attackers.sum(1).tolist() == [3, 2]


def test_attack_params_entry_ranks_place_the_attacker_by_degree_rank():
    w = build([unit(topology="star_hub", agents=7, attack="infection", attack_params={"entry_ranks": [0]}),
               unit(topology="star_hub", agents=7, attack="infection", attack_params={"entry_ranks": [3]})])
    assert w.seed[0].nonzero().item() == 0 and w.seed[1].nonzero().item() == 3      # hub first, then the leaves by index
    w = build([unit(topology="barabasi_albert", agents=12, attack="insider_pi", seed=5,
                    attack_params={"entry_ranks": [1, 2]})], make_settings(runtime__n_insiders=2))
    order = torch.argsort(-w.deg[0], stable=True)
    assert w.attackers[0].nonzero().flatten().tolist() == sorted(order[[1, 2]].tolist())
    w = build([unit(agents=7, attack="aitm", attack_params={"entry_ranks": [2]})])
    assert w.victim.item() == 3                          # chain degrees: inner agents 1..5 (2) rank before the ends (1)
    with pytest.raises(ValueError, match="entry_ranks"):
        build([unit(attack="infection", attack_params={"entry_ranks": [0, 1]})])
    with pytest.raises(ValueError, match="entry_ranks"):
        build([unit(attack="insider_pi", attack_params={"entry_ranks": [1, 1]})])
    with pytest.raises(ValueError, match="entry_ranks"):
        build([unit(attack="aitm", attack_params={"entry_ranks": [6]})], make_settings(), T=4, N=6)


def test_attack_params_schedule_follows_the_adaptive_features_reference():
    w = build([unit(attack="infection", attack_params={"start_frac": 0.4, "rate": 0.5}), unit(attack="infection", seed=2)],
              T=10)
    assert torch.equal(w.attack_on[0], adaptive.attack_rounds(10, 0.4, 0.5))
    assert w.attack_on[1].all() and w.attack_on.shape == (2, 10)


def test_attack_params_detectability_and_potency_weaken_the_payload():
    w = build([unit(attack="insider_pi", attack_params={"detectability": 0.5, "potency": 0.25})])
    assert w.mu_mal.item() == pytest.approx(1.0) and w.shift.item() == pytest.approx(2.0)
    base = build([unit(attack="insider_pi")])
    assert w.d_pay.item() == pytest.approx(0.5 * base.d_pay.item())


def test_auditor_means_follow_the_unit_preset():
    w = build([unit(), unit(seed=2, preset="weak_judge"), unit(seed=3, preset="adaptive",
                                                               attack_params={"auditor_tpr": 0.2})])
    assert w.d_pay[0] > w.d_pay[1] > w.d_pay[2] and (w.d_dec > 0).all()


@pytest.mark.parametrize("family,key", [("insider_pi", "runtime.n_insiders"), ("insider_ta", "runtime.ta_detectability"),
                                        ("insider_ma", "runtime.ma_detectability"), ("insider_pi", "runtime.insider_bias"),
                                        ("infection", "mock.model.bias"), ("aitm", "runtime.aitm_rewrite_strength"),
                                        ("none", "mock.model.reading_sd"), ("none", "mock.model.decoys"),
                                        ("none", "mock.model.burst_span"), ("none", "mock.model.mu_inf")])
def test_a_missing_setting_refuses_with_its_key_name(family, key):
    s = make_settings()
    *path, leaf = key.split(".")
    node = s
    for part in path:
        node = node[part]
    del node[leaf]
    with pytest.raises(core.Unset) as e:
        build([unit(attack=family)], s)
    assert e.value.args[0] == key


def test_batches_must_share_topology_attack_agent_count_and_budget_cell():
    for bad in (unit(topology="star_hub"), unit(attack="infection"), unit(agents=7), unit(audit=0.1),
                unit(intervention=0.1), unit(preset="weak_judge"), unit(cost_condition="robustness")):
        with pytest.raises(ValueError, match="same"):
            W.check_homogeneous([unit(), bad])
    with pytest.raises(ValueError, match="attack"):
        build([unit(attack="worm")], make_settings())
    with pytest.raises(ValueError, match="topology"):
        build([unit(topology="torus")], make_settings())
    with pytest.raises(ValueError, match="empty"):
        W.check_homogeneous([])


def test_more_insiders_than_victims_is_refused():
    with pytest.raises(ValueError, match="n_insiders"):
        build([unit(attack="insider_pi", agents=2)], make_settings(runtime__n_insiders=2))
