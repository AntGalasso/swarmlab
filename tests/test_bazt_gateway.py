"""Gateway behaviour: state machine (restrict / release / quarantine / block), the inspection oracle as the only
door to ground truth, audit consistency and the section-15 metric equations."""
import math

import pytest

from bazt_testkit import chain_adj, hot_roll, make_gateway, make_view
from bazt.actions import ActionKind
from bazt.config import BaztConfig
from bazt.metrics import BAZT_METRICS
from bazt.optimizer import Plan


class FixedOptimizer:
    """Test seam: replays prepared plans so the gateway's state machine can be driven directly."""

    def __init__(self, *plans):
        self.plans = list(plans)

    def plan(self, sit, ledger):
        return self.plans.pop(0) if self.plans else Plan()


# ------------------------------------------------------------------------------------ decisions
def test_decide_stays_within_budget_and_logs_every_decision():
    gw = make_gateway()
    plan = gw.decide(make_view(2, roll=hot_roll(hot=(3, 8))), lambda i: i in (3, 8))
    assert plan.decisions
    rows = gw.audit.rows()
    assert len(rows) == len(plan.decisions)
    assert gw.ledger.spent <= 12 and rows[-1]["spent"] == gw.ledger.spent
    assert [r["action"] for r in rows] == [d.cand.kind.name for d in plan.decisions]
    assert all(r["remaining"] >= 0 and r["cost"] > 0 for r in rows)


def test_nothing_is_bought_at_the_last_step():
    gw = make_gateway()
    plan = gw.decide(make_view(12, roll=hot_roll(hot=(3, 8), level=2.9)), lambda i: True)
    assert plan.decisions == [] and gw.ledger.spent == 0


def test_without_provenance_weight_p_prime_is_the_monitor_probability_bit_for_bit():
    gw = make_gateway(BaztConfig(w_prov=0.0, q_inspect=0.0))
    view = make_view(2, roll=hot_roll(hot=(3, 8), level=1.7))
    plan = gw.decide(view, lambda i: True)
    assert plan.decisions
    for d in plan.decisions:
        assert d.cand.p_prime == view.p_monitor[d.cand.node]


def test_probe_hook_replaces_the_rolling_score_of_chosen_nodes():
    calls = []

    def probe(i, t, view):
        calls.append((i, t))
        return 2.9 if i == 5 else None
    gw = make_gateway(BaztConfig(probe_score_hook=probe, q_inspect=0.0))
    gw.decide(make_view(2, roll=[0.0] * 12), lambda i: False)
    rows = [r for r in gw.audit.rows() if str(r["target"]) == "5"]
    assert rows and all(r["s_mon"] == 2.9 for r in rows) and (5, 2) in calls


# ----------------------------------------------------------------------------- ground-truth firewall
def test_oracle_is_gated_by_q_inspect_and_never_reads_truth_when_inconclusive():
    spy = []
    gw0 = make_gateway(BaztConfig(q_inspect=0.0))
    assert all(gw0.inspect_oracle(i, 1, lambda n: spy.append(n) or True) is None for i in range(12))
    assert spy == []
    gw1 = make_gateway(BaztConfig(q_inspect=1.0))
    assert gw1.inspect_oracle(4, 1, lambda n: n == 4) is True
    assert gw1.inspect_oracle(5, 1, lambda n: n == 4) is False


def test_truth_is_read_only_through_paid_conclusive_inspections():
    for q in (0.0, 0.5, 1.0):
        reads = []
        gw = make_gateway(BaztConfig(q_inspect=q))
        plan = gw.decide(make_view(2, roll=hot_roll(hot=(3, 8), level=1.7)), lambda i: reads.append(i) or (i == 3))
        paid = [d for d in plan.decisions if d.cand.kind == ActionKind.INSPECT]
        conclusive = [i for i, v in plan.inspected.items() if v is not None]
        assert sorted(reads) == sorted(conclusive) and len(reads) <= len(paid)
        if q == 0.0:
            assert reads == [] and not conclusive


def test_oracle_draws_are_precomputed_so_inspecting_one_node_never_shifts_another():
    a, b = make_gateway(BaztConfig(q_inspect=0.5)), make_gateway(BaztConfig(q_inspect=0.5))
    first = [a.inspect_oracle(i, 3, lambda n: True) for i in range(12)]
    for i in (7, 2, 11):                                    # consult in a different order
        b.inspect_oracle(i, 3, lambda n: True)
    assert [b.inspect_oracle(i, 3, lambda n: True) for i in range(12)] == first


def test_clean_verdict_clears_the_lineage_of_the_inspected_node():
    gw = make_gateway(BaztConfig(q_inspect=1.0), seed_agents=frozenset({0}))
    plan = gw.decide(make_view(1, roll=hot_roll(hot=(0,), level=0.2)), lambda i: False)
    assert plan.inspected.get(0) == 0.02
    assert gw.lineage.tau[0] == 0.0 and gw.inspection.clean_active(0, 2)


# ----------------------------------------------------------------------------------- state machine
def test_restriction_lasts_ttl_steps_then_expires():
    gw = make_gateway()
    gw.optimizer = FixedOptimizer(Plan(restrict=[3]))
    gw.decide(make_view(2, roll=hot_roll(hot=(3,))), lambda i: False)
    assert [3 in gw.restricted(t) for t in (2, 3, 4, 5)] == [True, True, True, False]
    for t in (3, 4):
        gw.decide(make_view(t, roll=hot_roll(hot=(3,))), lambda i: False)
        assert 3 in gw.restricted(t)
    gw.decide(make_view(5, roll=hot_roll(hot=(3,))), lambda i: False)
    assert 3 not in gw.restricted(5)


def test_restriction_is_released_early_when_evidence_fades():
    gw = make_gateway()
    gw.optimizer = FixedOptimizer(Plan(restrict=[3]))
    gw.decide(make_view(2, roll=hot_roll(hot=(3,))), lambda i: False)
    gw.decide(make_view(3, roll=[0.0] * 12), lambda i: False)         # p' ~ 0.12 < tau_release
    assert 3 not in gw.restricted(3)
    assert any(r["action"] == "RELEASE" and r["reason"] == "p_prime<tau_release" for r in gw.audit.rows())


def test_quarantine_is_permanent_by_default_and_ttl_bound_when_asked():
    perm = make_gateway()
    perm.optimizer = FixedOptimizer(Plan(isolate=[3]))
    perm.decide(make_view(2, roll=hot_roll()), lambda i: False)
    assert all(perm.pop_released(t) == [] for t in range(3, 13))
    ttl = make_gateway(BaztConfig(quarantine_ttl=2))
    ttl.optimizer = FixedOptimizer(Plan(isolate=[3]))
    ttl.decide(make_view(2, roll=hot_roll()), lambda i: False)
    assert [ttl.pop_released(t) for t in (3, 4, 5, 6)] == [[], [], [3], []]   # inactive at steps 3 and 4
    assert any(r["action"] == "RELEASE" and r["reason"] == "quarantine_ttl" for r in ttl.audit.rows())


def test_blocked_edges_are_directed_and_leave_the_send_graph():
    gw = make_gateway()
    gw.optimizer = FixedOptimizer(Plan(block=[(3, 4)]))
    gw.decide(make_view(2, roll=hot_roll()), lambda i: False)
    assert not gw.edge_open(3, 4, 3) and gw.edge_open(4, 3, 3) and gw.edge_open(3, 2, 3)
    gw.on_deliver(2, {3: False})
    assert [(j, i) for j, i, _ in gw._log.delivered[2]] == [(3, 2)]


def test_quarantined_receivers_are_not_counted_as_delivered():
    gw = make_gateway()
    gw.optimizer = FixedOptimizer(Plan(isolate=[2]))
    gw.decide(make_view(2, roll=hot_roll()), lambda i: False)
    gw.on_deliver(2, {3: False})
    assert [i for _, i, _ in gw._log.delivered[2]] == [4]


# ------------------------------------------------------------------------------------- §15 metrics
def _manual_gateway():
    return make_gateway(n=3, T=3, adj=chain_adj(3))        # E_dir = 4, capacity (T-1)*E_dir = 8


def test_metric_keys_cover_section_15():
    gw = _manual_gateway()
    m = gw.extra_metrics([[False] * 3] * 3)
    assert set(m) == set(BAZT_METRICS)


def test_harm_exposure_and_throughput_equations():
    gw = _manual_gateway()
    gw.on_deliver(1, {0: False, 1: False})                 # edges 0->1, 1->0, 1->2
    gw.on_deliver(2, {2: True})                            # edge 2->1, sender stripped
    series = [[True, False, False], [False, False, True], [False, False, False]]
    m = gw.extra_metrics(series)
    assert m["message_throughput"] == pytest.approx(4 / 8)
    assert m["harm_exposure_auc"] == pytest.approx(1 / 8)  # only 0->1: infected sender, NOT stripped
    gw2 = _manual_gateway()
    gw2.on_deliver(1, {0: True})
    assert gw2.extra_metrics(series)["harm_exposure_auc"] == 0.0      # stripped infected sender: no exposure


def test_spend_split_false_action_rate_and_recovery_time():
    gw = _manual_gateway()
    for kind in (ActionKind.ISOLATE, ActionKind.INSPECT, ActionKind.SANITIZE):
        gw.ledger.charge(kind)
    gw._log.actions += [(1, ActionKind.ISOLATE, 0), (2, ActionKind.SANITIZE, 1)]
    series = [[True, False, False], [True, False, False], [False, False, False]]
    m = gw.extra_metrics(series)
    assert (m["enforcement_spend"], m["spend_isolate"], m["spend_inspect"], m["spend_sanitize"]) == (5, 3, 1, 1)
    assert m["false_action_rate"] == pytest.approx(0.5)    # node 1 was never infected when sanitized; node 0 was
    assert m["recovery_time"] == 2.0                       # first action t=1, first clean step t=3
    never = [[True, False, False]] * 3
    assert gw.extra_metrics(never)["recovery_time"] == 3.0  # censored: T - t_first + 1


def test_no_action_gives_zero_false_rate_and_nan_recovery():
    m = _manual_gateway().extra_metrics([[True, False, False]] * 3)
    assert m["false_action_rate"] == 0.0 and math.isnan(m["recovery_time"])
    assert m["degraded_node_step_frac"] == 0.0


def test_degraded_node_step_fraction_counts_restricted_sanitized_and_isolated():
    gw = _manual_gateway()
    gw.optimizer = FixedOptimizer(Plan(restrict=[0], sanitize=[1], isolate=[2]), Plan())
    gw.decide(make_view(1, n=3, T=3, adj=chain_adj(3), roll=[0.0] * 3), lambda i: False)
    gw.decide(make_view(2, n=3, T=3, adj=chain_adj(3), roll=[2.5, 0.0, 0.0], active=(0, 1)), lambda i: False)
    m = gw.extra_metrics([[False] * 3] * 3)
    assert m["degraded_node_step_frac"] == pytest.approx((3 + 2) / 9)   # t1: {0,1,2}; t2: {0 restricted, 2 isolated}


def test_budget_violation_recount_flags_a_corrupted_ledger_row():
    gw = _manual_gateway()
    gw.audit.append(t=1, action="ISOLATE", target=0, cost=3, pool="pool", spent=13, remaining=-1)
    assert gw.extra_metrics([[False] * 3] * 3)["budget_violations"] == 1


def test_cost_per_prevented_compromise_is_paired_and_nan_when_nothing_was_prevented():
    from bazt.metrics import add_paired_metrics
    keys = dict(topology="chain", regime="decoy", seed=1, N=10)
    rows = [dict(policy="no_defense", final_ever_infected_rate=0.5, enforcement_spend=0.0, **keys),
            dict(policy="a", final_ever_infected_rate=0.3, enforcement_spend=8.0, **keys),
            dict(policy="b", final_ever_infected_rate=0.5, enforcement_spend=8.0, **keys)]
    add_paired_metrics(rows)
    assert rows[1]["cost_per_prevented_compromise"] == pytest.approx(8.0 / 2)   # two nodes spared
    assert math.isnan(rows[0]["cost_per_prevented_compromise"]) and math.isnan(rows[2]["cost_per_prevented_compromise"])
