"""Contract tests for `bazt/gateway.py` (`EnforcementGateway`). Offline, no
network, no API key. Uses `swarmlab.StepView`-like fakes and `MockAgentPolicy`
worlds."""
from typing import List, Optional, Set

import swarmlab as sl

from bazt.arms import bazt_config
from bazt.actions import ActionKind
from bazt.config import BaztConfig
from bazt.gateway import EnforcementGateway, GatewayPlan
from bazt.roles import assign_roles


class _FakeSV:
    """Minimal stand-in for `swarmlab.StepView` used by `gateway.decide`."""
    def __init__(self, t: int, active: List[int], p_monitor: List[float],
                 roll_scores: List[float]):
        self.t = t
        self.active = active
        self.p_monitor = p_monitor
        self.roll_scores = roll_scores


def _make_gw(bcfg=None, N=6, seed=9001, ingress=None, entry_seed=None,
             topology="small_world"):
    if bcfg is None:
        bcfg = BaztConfig(B=12)
    cfg = sl.Config(N=N, T=8)
    w = sl.make_world(topology, seed, cfg)
    roles = assign_roles(N, topology, seed, w.adj, set(entry_seed or set()), 2)
    ingress_nodes = set(ingress or {r.node_id for r in roles if r.role == "ingress"})
    entry = set(entry_seed or {0})
    gw = EnforcementGateway(cfg, bcfg, topology, seed, w.adj, roles,
                            ingress_nodes, entry)
    return gw, cfg, w, roles


# ------------------------------------------------------------------ construction
def test_gateway_construction_sets_state():
    gw, cfg, w, roles = _make_gw()
    assert gw.N == cfg.N
    assert gw.T == cfg.T
    assert gw.ledger.capacity == 12
    assert gw.node_state == {i: "normal" for i in range(cfg.N)}
    assert gw.blocked_edges == set()
    assert gw.stripped_messages == set()
    assert gw.delivered_count == 0


# ------------------------------------------------------------------ edge_open
def test_edge_open_blocks_quarantined_and_blocked_edges():
    gw, cfg, w, roles = _make_gw()
    assert gw.edge_open(0, 1, 1) is True
    gw.blocked_edges.add((0, 1))
    assert gw.edge_open(0, 1, 1) is False
    gw.quarantined.add(2)
    assert gw.edge_open(2, 3, 1) is False
    assert gw.edge_open(3, 2, 1) is False


# ------------------------------------------------------------------ inspect_oracle (budget-gated ground truth)
def test_inspect_oracle_spends_budget_and_returns_none_when_broke():
    gw, cfg, w, roles = _make_gw(bcfg=BaztConfig(B=1))
    truth = lambda i: i == 0
    # can afford one inspect
    v1 = gw.inspect_oracle(0, 1, truth)
    assert gw.ledger.spent == 1
    assert v1 in (0.98, 0.02, None)
    # now broke
    v2 = gw.inspect_oracle(1, 1, truth)
    assert v2 is None
    assert gw.ledger.spent == 1  # never charged again


def test_inspect_oracle_clean_clears_taint():
    gw, cfg, w, roles = _make_gw(bcfg=BaztConfig(B=5, q_inspect=1.0))
    truth = lambda i: False  # all clean
    # force conclusive clean verdict (q_inspect=1.0 always conclusive)
    v = gw.inspect_oracle(0, 1, truth)
    assert v == 0.02
    # after inspect_clean, lineage value should be 0
    assert gw.lineage.value(0) == 0.0


# ------------------------------------------------------------------ decide
def test_decide_returns_gateway_plan():
    gw, cfg, w, roles = _make_gw(bcfg=BaztConfig(B=12, use_pacing=False))
    sv = _FakeSV(t=2, active=list(range(cfg.N)),
                 p_monitor=[0.1] * cfg.N, roll_scores=[0.5] * cfg.N)
    plan = gw.decide(sv, lambda i: i == 0)
    assert isinstance(plan, GatewayPlan)
    assert all(isinstance(x, int) for x in plan.isolate)
    assert all(isinstance(x, int) for x in plan.sanitize)
    assert all(isinstance(x, int) for x in plan.restrict)
    assert all(isinstance(x, tuple) and len(x) == 2 for x in plan.block_edges)
    assert all(isinstance(x, int) for x in plan.inspect)


def test_decide_never_exceeds_budget():
    gw, cfg, w, roles = _make_gw(bcfg=BaztConfig(B=3, use_pacing=False))
    sv = _FakeSV(t=2, active=list(range(cfg.N)),
                 p_monitor=[0.1] * cfg.N, roll_scores=[0.9] * cfg.N)
    plan = gw.decide(sv, lambda i: False)
    total = gw.ledger.spent
    assert total <= 3
    # no partial charges: ledger never clips


def test_decide_at_final_step_spends_zero():
    gw, cfg, w, roles = _make_gw(bcfg=BaztConfig(B=12))
    sv = _FakeSV(t=cfg.T, active=list(range(cfg.N)),
                 p_monitor=[0.1] * cfg.N, roll_scores=[0.9] * cfg.N)
    plan = gw.decide(sv, lambda i: False)
    assert plan.isolate == [] and plan.sanitize == []
    assert plan.restrict == [] and plan.block_edges == [] and plan.inspect == []
    assert gw.ledger.spent == 0


# ------------------------------------------------------------------ on_deliver
def test_on_deliver_tracks_stripped_messages():
    gw, cfg, w, roles = _make_gw(bcfg=BaztConfig(B=12))
    delivered = [(0, 1, True), (1, 2, False), (2, 3, True)]
    gw.on_deliver(1, delivered)
    assert gw.delivered_count == 3
    assert gw.stripped_messages == {0, 2}  # sources of san=True edges


# ------------------------------------------------------------------ extra_metrics
def test_extra_metrics_returns_dict_with_section15_keys():
    gw, cfg, w, roles = _make_gw(bcfg=BaztConfig(B=12))
    infected = [0, 1, 2, 3]
    m = gw.extra_metrics(infected, w)
    assert isinstance(m, dict)
    for k in ("bazt_budget_spent", "bazt_budget_capacity", "bazt_budget_violations"):
        assert k in m
    assert m["bazt_budget_spent"] == 0.0
    assert m["bazt_budget_capacity"] == 12.0


def test_extra_metrics_unlimited_capacity_minus_one():
    gw, cfg, w, roles = _make_gw(bcfg=BaztConfig(B=12, unlimited=True))
    m = gw.extra_metrics([0], w)
    assert m["bazt_budget_capacity"] == -1.0
