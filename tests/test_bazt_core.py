"""Pure-unit tests of the BAZT building blocks: config, pricing, ledger, evidence, lineage, roles, reach,
optimizer, audit and the filtering-cost channel. No episode is run here."""
import itertools
import math

import pytest

from bazt_testkit import chain_adj
import swarmlab as sl
from bazt import actions as A
from bazt.audit import AUDIT_FIELDS, AuditLog
from bazt.channel import RationaleAgent, make_edge_weight_modifier, sensor_classes
from bazt.config import BaztConfig
from bazt.contagion import ContagionModel
from bazt.evidence import P_NEGATIVE, P_POSITIVE, InspectionState, fused_probability
from bazt.ledger import BudgetLedger, BudgetViolation
from bazt.optimizer import BudgetedOptimizer, Situation
from bazt.provenance import LineageTracker
from bazt.roles import INGRESS, RELAY, WORKER, articulation_points, assign_roles

SL = sl.Config()


# ------------------------------------------------------------------------------------------- config
def test_config_defaults_match_the_spec():
    c = BaztConfig()
    assert (c.budget_mode, c.B, c.K_hard, c.B_soft) == ("pooled", 12, 3, 6)
    assert (c.cost_inspect, c.cost_sanitize, c.cost_restrict, c.cost_block, c.cost_isolate) == (1, 1, 2, 2, 3)
    assert (c.q_inspect, c.inspect_ttl, c.ttl_restrict, c.tau_release, c.quarantine_ttl) == (0.8, 2, 3, 0.3, None)
    assert (c.w_prov, c.delta_hop, c.lam, c.rho_san, c.rho_res) == (1.0, 0.7, 1.0, 0.15, 0.35)
    assert (c.theta0, c.kappa, c.n_privileged) == (0.15, 1.5, 3)
    assert c.strip_degrades_weight and c.stripped_edge_weight == 0.35 and c.probe_score_hook is None
    assert not c.unlimited and c.use_pacing and c.use_topology and c.use_provenance and c.use_utility
    assert not c.binary_actions


def test_config_is_frozen_and_validated():
    with pytest.raises(Exception):
        BaztConfig().B = 3
    for bad in ({"budget_mode": "nope"}, {"cost_isolate": 0}, {"q_inspect": 1.5}, {"ttl_restrict": 0},
                {"quarantine_ttl": 0}, {"stripped_edge_weight": 0.0}, {"B": -1}):
        with pytest.raises(ValueError):
            BaztConfig(**bad)


def test_total_capacity_per_mode():
    assert BaztConfig().total_capacity == 12
    assert BaztConfig(budget_mode="tiered").total_capacity == 3 * 3 + 6 * 1
    assert math.isinf(BaztConfig(unlimited=True).total_capacity)


# ------------------------------------------------------------------------------------------- actions
def test_action_rank_and_costs():
    assert [int(k) for k in A.ActionKind] == [0, 1, 2, 3, 4]
    c = BaztConfig()
    assert [A.cost_of(c, k) for k in A.ActionKind] == [1, 1, 2, 2, 3]


def test_isolate_net_equals_swarmlab_risk_score():
    adj, n = chain_adj(8), 8
    cfg = sl.Config(N=n)
    p = [0.1, 0.9, 0.2, 0.7, 0.05, 0.4, 0.3, 0.6]
    aset, deg, t = set(range(n)), [len(a) for a in adj], 3
    model = ContagionModel(cfg, BaztConfig(), adj, aset, p, cfg.T - t)
    for i in range(n):
        R, H, C = sl.risk_score(i, t, cfg, adj, aset, p, deg, max(deg), cfg.T)
        cand = A.score_isolate(i, p[i], model.H(i), C, BaztConfig(lam=cfg.lam))
        assert cand.net == pytest.approx(R, abs=1e-12)
        assert cand.density == pytest.approx(R / 3)


def test_candidate_formulas():
    c = BaztConfig()
    p, H, C, h_t, ps = 0.8, 2.0, 0.6, 6, 0.9
    assert A.score_sanitize(1, p, H, C, h_t, ps, c).benefit == pytest.approx(ps * p * H / 6)
    assert A.score_restrict(1, p, H, C, h_t, ps, False, c).benefit == pytest.approx(ps * p * H * 3 / 6)
    assert A.score_block(1, 2, p, 1.7, C, 4, c).benefit == pytest.approx(p * 1.7)
    assert A.score_block(1, 2, p, 1.7, C, 4, c).util_cost == pytest.approx(C / 4)
    assert A.score_sanitize(1, p, H, C, h_t, ps, c).util_cost == pytest.approx(0.15 * C)
    assert A.score_restrict(1, p, H, C, h_t, ps, False, c).util_cost == pytest.approx(0.35 * C)
    assert A.score_inspect(1, p, H, C, c).util_cost == 0.0
    # privileged restriction earns a bonus, other kinds do not
    assert (A.score_restrict(1, p, H, C, h_t, ps, True, c).benefit
            > A.score_restrict(1, p, H, C, h_t, ps, False, c).benefit)


def test_inspect_evpi_is_nonnegative_and_bounded():
    c = BaztConfig()
    for p, H, C in itertools.product([0, .05, .3, .5, .9, 1], [0, .5, 3], [0, .4, 1]):
        b = A.score_inspect(0, p, H, C, c).benefit
        assert b >= 0
        assert b <= c.q_inspect * p * H + 1e-12
        assert b <= c.q_inspect * (1 - p) * c.lam * C + 1e-12


def test_horizon_zero_prices_every_forward_action_at_zero():
    """The horizon rule lives in ContagionModel (H = 0 at h_t = 0); the pricing formulas then give 0 benefit."""
    c = BaztConfig()
    H = ContagionModel(SL, c, chain_adj(5), set(range(5)), [.9] * 5, 0).H(2)
    assert H == 0.0
    assert A.score_sanitize(0, .9, H, .5, 0, .9, c).benefit == 0
    assert A.score_restrict(0, .9, H, .5, 0, .9, False, c).benefit == 0
    assert A.score_restrict(0, .9, H, .5, 0, .9, True, c).benefit == 0      # the privilege bonus is horizon-bounded too


def test_no_utility_ablation_drops_the_utility_term():
    c = BaztConfig(use_utility=False)
    cand = A.score_isolate(0, 0.5, 2.0, 0.9, c)
    assert cand.net == cand.benefit == pytest.approx(1.0)


def test_tie_break_key_orders_density_node_peer_rank():
    mk = lambda **k: A.Candidate(**{**dict(kind=A.ActionKind.ISOLATE, node=1, peer=-1, benefit=1, util_cost=0,
                                           net=1, cost=1, density=1.0, p_prime=.5, H=1), **k})
    cs = [mk(node=2), mk(node=1, peer=3), mk(node=1, peer=-1), mk(density=2.0, node=9),
          mk(node=1, peer=-1, kind=A.ActionKind.SANITIZE)]
    order = sorted(cs, key=lambda c: c.sort_key)
    assert order[0].node == 9                                              # highest density
    assert [(c.node, c.peer, int(c.kind)) for c in order[1:]] == [(1, -1, 1), (1, -1, 4), (1, 3, 4), (2, -1, 4)]


# -------------------------------------------------------------------------------------------- ledger
def test_pooled_ledger_never_clips():
    led = BudgetLedger(BaztConfig(B=5))
    led.charge(A.ActionKind.ISOLATE)                      # 3
    led.charge(A.ActionKind.RESTRICT)                     # 5
    assert led.spent == 5 and led.remaining("pool") == 0
    assert not led.can_afford(A.ActionKind.INSPECT)
    with pytest.raises(BudgetViolation):
        led.charge(A.ActionKind.INSPECT)
    assert led.spent == 5                                 # the failed charge left no trace


def test_tiered_pools_are_separate_and_denominated_in_cost_units():
    led = BudgetLedger(BaztConfig(budget_mode="tiered", K_hard=1, B_soft=2))
    assert led.capacity("hard") == 3 and led.capacity("soft") == 2
    led.charge(A.ActionKind.ISOLATE)
    assert not led.can_afford(A.ActionKind.BLOCK_EDGE)    # hard pool empty ...
    assert led.can_afford(A.ActionKind.RESTRICT)          # ... soft pool untouched
    led.charge(A.ActionKind.RESTRICT)
    assert not led.can_afford(A.ActionKind.INSPECT)
    with pytest.raises(BudgetViolation):
        led.charge(A.ActionKind.SANITIZE)
    assert led.pool_of(A.ActionKind.BLOCK_EDGE) == "hard" and led.pool_of(A.ActionKind.SANITIZE) == "soft"


def test_unlimited_ledger_is_infinite_and_z_is_zero():
    led = BudgetLedger(BaztConfig(unlimited=True))
    for _ in range(50):
        led.charge(A.ActionKind.ISOLATE)
    assert led.spent == 150 and led.z() == 0.0 and led.theta() == BaztConfig().theta0


def test_pacing_theta_formula_and_flat_theta():
    cfg = BaztConfig()
    led = BudgetLedger(cfg)
    assert led.z() == 0 and led.theta() == cfg.theta0
    led.charge(A.ActionKind.ISOLATE)                      # 3 / 12
    assert led.z() == pytest.approx(0.25)
    assert led.theta() == pytest.approx(cfg.theta0 * math.exp(cfg.kappa * 0.25))
    flat = BudgetLedger(BaztConfig(use_pacing=False))
    flat.charge(A.ActionKind.ISOLATE)
    assert flat.theta() == cfg.theta0
    assert BudgetLedger(BaztConfig(B=0)).z() == 0.0       # zero capacity: no division by zero


# ------------------------------------------------------------------------------------------ evidence
def test_fusion_is_bit_identical_to_the_monitor_without_provenance_weight():
    for roll in (0.0, 0.37, 1.0, 2.0, 2.9):
        mon = sl.sigmoid(SL.a * roll - SL.b)
        assert fused_probability(SL, BaztConfig(w_prov=0.0), roll, 0.9) == mon
        assert fused_probability(SL, BaztConfig(use_provenance=False), roll, 0.9) == mon
        assert fused_probability(SL, BaztConfig(), roll, 0.0) == mon


def test_taint_raises_the_fused_probability():
    assert fused_probability(SL, BaztConfig(), 1.0, 1.0) > fused_probability(SL, BaztConfig(), 1.0, 0.0)


def test_inspection_verdict_lifetime():
    s = InspectionState(ttl=2)
    assert s.override(4, 1) is None
    assert s.record(4, 3, True) == P_POSITIVE
    assert [s.override(4, t) for t in (3, 4, 5)] == [P_POSITIVE, P_POSITIVE, None]
    s.record(5, 3, False)
    assert s.clean_active(5, 4) and not s.clean_active(5, 5) and not s.clean_active(4, 3)
    assert s.active_nodes(4) == frozenset({4, 5}) and s.active_nodes(5) == frozenset()
    assert (P_NEGATIVE, P_POSITIVE) == (0.02, 0.98)


# ---------------------------------------------------------------------------------------- provenance
def _run_lineage(n=5, steps=6, cfg=None, ingress=frozenset({0}), stripped=False, is_open=None):
    cfg, adj = cfg or BaztConfig(), chain_adj(n)
    tr = LineageTracker(n, ingress, adj, cfg, p_strip=0.9)
    hist = []
    for t in range(1, steps + 1):
        tau = list(tr.advance(t, is_open or (lambda j, i: True), set(range(n))))
        hist.append(tau)
        tr.record_outgoing({i: stripped for i in range(n)})
    return tr, hist


def test_taint_starts_at_ingress_and_decays_across_hops():
    _, hist = _run_lineage()
    assert hist[0] == [1.0, 0, 0, 0, 0]
    d = BaztConfig().delta_hop
    for tau in hist[1:]:
        assert tau[0] <= 1.0
        for k in range(1, 5):
            assert tau[k] <= d ** k + 1e-12               # never more than delta_hop^hops
    last = hist[-1]
    assert last[0] > last[1] > last[2] > last[3] > last[4] > 0


def test_first_hop_is_delta_hop_times_source_taint():
    _, hist = _run_lineage()
    assert hist[1][1] == pytest.approx(0.7 * 1.0)
    assert hist[2][2] == pytest.approx(0.7 * hist[1][1])


def test_taint_is_zero_without_provenance():
    for cfg in (BaztConfig(use_provenance=False), BaztConfig(w_prov=0.0)):
        _, hist = _run_lineage(cfg=cfg)
        assert all(x == 0.0 for tau in hist for x in tau)


def test_stripped_sender_passes_only_the_residual_taint():
    _, plain = _run_lineage(steps=2)
    _, strip = _run_lineage(steps=2, stripped=True)
    assert strip[1][1] == pytest.approx((1 - 0.9) * plain[1][1])


def test_blocked_edge_stops_propagation_and_clear_resets():
    tr, hist = _run_lineage(is_open=lambda j, i: (j, i) != (0, 1), steps=4)
    assert hist[3][1] == 0.0 and hist[3][2] == 0.0
    tr2, _ = _run_lineage(steps=3)
    tr2.clear(1)
    assert tr2.tau[1] == 0.0


def test_lineage_records_message_ids_and_parent_links():
    tr, _ = _run_lineage(steps=3)
    assert tr.parents[(2, 1)] == ((1, 0), (1, 2))          # message (t, src) <- delivered parents at t-1


# -------------------------------------------------------------------------------------------- roles
def test_roles_are_deterministic_and_partition_the_nodes():
    adj = chain_adj(12)
    a = assign_roles("chain", 9001, adj, BaztConfig())
    b = assign_roles("chain", 9001, adj, BaztConfig())
    assert a == b and set(a) == set(range(12))
    assert {i.role for i in a.values()} <= {INGRESS, RELAY, WORKER}
    assert sum(i.privileged for i in a.values()) == 3
    assert sum(i.role == INGRESS for i in a.values()) == 2
    assert a != assign_roles("chain", 9002, adj, BaztConfig())


def test_roles_use_known_entry_points_and_ignore_the_namespace_of_other_streams():
    ids = assign_roles("chain", 9001, chain_adj(12), BaztConfig(), seed_agents=frozenset({4, 7}))
    assert {i for i, a in ids.items() if a.role == INGRESS} == {4, 7}


def test_ingress_may_only_send_to_relays_and_privilege_count_is_capped():
    ids = assign_roles("star_hub", 1, [[1, 2, 3], [0], [0], [0]], BaztConfig(n_privileged=99))
    assert sum(a.privileged for a in ids.values()) == 4
    relays = {i for i, a in ids.items() if a.role == RELAY}
    for a in ids.values():
        if a.role == INGRESS and any(j in relays for j in (1, 2, 3, 0)):
            assert set(a.allowed_out_edges) <= relays or not relays


def test_articulation_points():
    assert articulation_points(chain_adj(5)) == frozenset({1, 2, 3})
    assert articulation_points([[1, 2, 3], [0], [0], [0]]) == frozenset({0})
    assert articulation_points([[1, 2], [0, 2], [0, 1]]) == frozenset()


# ----------------------------------------------------------------------------------------- contagion
def test_reach_reuses_swarmlab_and_respects_open_edges():
    adj, n = chain_adj(6), 6
    p = [0.2] * n
    m = ContagionModel(SL, BaztConfig(), adj, set(range(n)), p, 6)
    assert m.H(0) == sl.reach_H(adj, 0, 6, SL, set(range(n)), p)
    cut = [list(a) for a in adj]
    cut[0] = []                                            # edge 0 -> 1 closed
    assert ContagionModel(SL, BaztConfig(), cut, set(range(n)), p, 6).H(0) == 0.0


def test_reach_is_zero_at_the_horizon_even_without_topology():
    adj = chain_adj(5)
    for cfg in (BaztConfig(), BaztConfig(use_topology=False)):
        m = ContagionModel(SL, cfg, adj, set(range(5)), [0.1] * 5, 0)
        assert m.H(2) == 0.0 and m.H_edge(2, 3) == 0.0
    assert ContagionModel(SL, BaztConfig(use_topology=False), adj, set(range(5)), [0.1] * 5, 4).H(2) == 1.0


def test_edge_reach_excludes_the_sender_and_discounts_the_receiver():
    adj, n = chain_adj(5), 5
    p = [0.0, 0.0, 0.5, 0.0, 0.0]
    aset = set(range(n))
    m = ContagionModel(SL, BaztConfig(), adj, aset, p, 5)
    beyond = sl.reach_H(adj, 3, 4, SL, aset - {2}, p)      # from 3 without routing back through 2
    assert m.H_edge(2, 3) == pytest.approx((1 - p[3]) * (1 + beyond))
    assert aset == set(range(n))                           # the temporary exclusion is undone


# ---------------------------------------------------------------------------------------- optimizer
def _situation(t=2, n=12, p=None, **kw):
    adj = chain_adj(n)
    deg = tuple(len(a) for a in adj)
    base = dict(t=t, T=12, active=tuple(range(n)), p_prime=list(p or [0.05] * n), deg=deg, maxdeg=max(deg),
                privileged=frozenset(), out_adj=adj, restricted=frozenset(), verdict_active=frozenset(),
                inspect=lambda i: None)
    base.update(kw)
    return Situation(**base)


def test_plan_is_deterministic_and_uses_no_rng(monkeypatch):
    p = [0.05] * 12
    p[3] = 0.95
    p[7] = 0.6

    def boom(*a, **k):
        raise AssertionError("the optimizer must not draw random numbers")
    monkeypatch.setattr(sl, "stream_rng", boom)
    monkeypatch.setattr("numpy.random.default_rng", boom)
    plans = []
    for _ in range(2):
        led = BudgetLedger(BaztConfig())
        plans.append(BudgetedOptimizer(BaztConfig(), SL).plan(_situation(p=p), led))
    key = lambda pl: [(d.cand.kind, d.cand.node, d.cand.peer) for d in pl.decisions]
    assert key(plans[0]) == key(plans[1]) and key(plans[0])


def test_plan_spends_nothing_at_the_last_step():
    p = [0.95] * 12
    for cfg in (BaztConfig(), BaztConfig(use_topology=False), BaztConfig(use_utility=False), BaztConfig(binary_actions=True)):
        led = BudgetLedger(cfg)
        plan = BudgetedOptimizer(cfg, SL).plan(_situation(t=12, p=p), led)
        assert plan.decisions == [] and led.spent == 0


def test_plan_respects_the_budget_and_the_density_threshold():
    p = [0.95] * 12
    for B in (0, 2, 5, 12):
        led = BudgetLedger(BaztConfig(B=B))
        plan = BudgetedOptimizer(BaztConfig(B=B), SL).plan(_situation(p=p), led)
        assert led.spent <= B
        assert all(d.cand.net > 0 and d.cand.density >= d.theta - 1e-12 for d in plan.decisions)


def test_binary_arm_generates_only_isolations():
    p = [0.05] * 12
    p[3] = p[8] = 0.95
    cfg = BaztConfig(binary_actions=True)
    plan = BudgetedOptimizer(cfg, SL).plan(_situation(p=p), BudgetLedger(cfg))
    assert plan.decisions and {d.cand.kind for d in plan.decisions} == {A.ActionKind.ISOLATE}
    assert not plan.restrict and not plan.sanitize and not plan.block and not plan.inspected


def test_inspection_phase_updates_p_prime_before_the_second_phase():
    p = [0.05] * 12
    p[5] = 0.5                                             # ambiguous: inspection is worth buying
    seen = []

    def oracle(i):
        seen.append(i)
        return 0.02                                        # verdict: clean
    sit = _situation(p=p, inspect=oracle)
    plan = BudgetedOptimizer(BaztConfig(), SL).plan(sit, BudgetLedger(BaztConfig()))
    assert 5 in seen and sit.p_prime[5] == 0.02
    assert plan.inspected[5] == 0.02
    assert all(d.cand.node != 5 or d.cand.kind == A.ActionKind.INSPECT for d in plan.decisions)   # cleared: no action


def test_inconclusive_inspection_keeps_the_fused_probability():
    p = [0.05] * 12
    p[5] = 0.5
    sit = _situation(p=p, inspect=lambda i: None)
    plan = BudgetedOptimizer(BaztConfig(), SL).plan(sit, BudgetLedger(BaztConfig()))
    assert plan.inspected.get(5, "x") is None and sit.p_prime[5] == 0.5


def test_one_node_level_action_per_node_and_no_block_after_isolation():
    p = [0.95] * 12
    plan = BudgetedOptimizer(BaztConfig(), SL).plan(_situation(p=p), BudgetLedger(BaztConfig(B=60)))
    nodes = [d.cand.node for d in plan.decisions if d.cand.kind in (A.ActionKind.ISOLATE, A.ActionKind.RESTRICT,
                                                                    A.ActionKind.SANITIZE)]
    assert len(nodes) == len(set(nodes))
    assert not [b for b in plan.block if b[0] in plan.isolate]


# ------------------------------------------------------------------------------------------ audit
def test_audit_log_schema_and_recount():
    assert AUDIT_FIELDS == ["t", "action", "target", "cost", "pool", "spent", "remaining", "p_prime", "s_mon",
                            "taint", "H", "net", "density", "theta", "reason"]
    log = AuditLog()
    log.append(t=1, action="ISOLATE", target=3, cost=3, pool="pool", spent=3, remaining=9)
    log.append(t=2, action="BLOCK_EDGE", target="1->2", cost=2, pool="pool", spent=14, remaining=-2)
    log.append(t=2, action="ISOLATE", target=4, cost=3, remaining=None)       # unlimited pool
    assert len(log) == 3 and log.overspent_rows() == 1
    with pytest.raises(KeyError):
        log.append(bogus=1)


def test_audit_csv_roundtrip(tmp_path):
    log = AuditLog()
    log.append(t=1, action="INSPECT", target=2, cost=1)
    path = tmp_path / "a.csv"
    log.to_csv(str(path), extra={"seed": 7})
    head = path.read_text().splitlines()[0].split(",")
    assert head[0] == "seed" and head[1:] == AUDIT_FIELDS


# ----------------------------------------------------------------------------------------- channel
def test_edge_weight_rule_prices_rationale():
    mod = make_edge_weight_modifier(BaztConfig())
    assert mod(0, 1, 2, "a0 t1 est=1.0 sens=hp") == 1.0
    assert mod(0, 1, 2, "a0 t1 est=1.0 sens=lp") == 0.6
    assert mod(0, 1, 2, "a0 t1 est=1.0") == 0.35           # schema-only: rationale stripped


def test_trusted_sanitize_removes_exactly_the_rationale():
    msg = "a3 t4 est=51.2 [REF-7] sens=lp"
    out = sl.trusted_sanitize(msg, 3, 4)
    assert out == "a3 t4 est=51.2" and "sens=" not in out


def test_rationale_agent_appends_the_class_and_classes_are_crn():
    class Echo:
        def step(self, ctxs):
            return [f"a{c.i} t{c.t} est=1.0" for c in ctxs]
    ctxs = [sl.AgentCtx(i, 1, [], None, 1.0, False, False, False, .5, .5, .5, 1, "chain") for i in (0, 1)]
    out = RationaleAgent(Echo(), ["hp", "lp"]).step(ctxs)
    assert out == ["a0 t1 est=1.0 sens=hp", "a1 t1 est=1.0 sens=lp"]
    a, b = sensor_classes("chain", 12, 9001, 0.4), sensor_classes("chain", 12, 9001, 0.4)
    assert a == b and set(a) <= {"hp", "lp"} and a != sensor_classes("chain", 12, 9002, 0.4)
    assert set(sensor_classes("chain", 12, 9001, 0.0)) == {"hp"}
