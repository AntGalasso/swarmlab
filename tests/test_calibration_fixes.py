"""Regression tests for the review findings C1-C6, M2, M3, m3, m8 (calibration direction and delivery accounting)."""
from __future__ import annotations

import inspect
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiments as ex
import swarmlab as sl
from bazt import calibration
from bazt.conventional_zta import ZtaConfig, conventional_bazt_config
from tests.test_experiments_suite import chain_adj, gw_for, view_at

CFG_PATH = "configs/sprint_preregistered.json"


# ------------------------------------------------------------------ M2: a denied edge is not a delivery
@pytest.mark.parametrize("arm", ["arm_B", "arm_C", "gsafeguard_standin"])
def test_blocked_edges_are_not_logged_as_delivered(arm):
    g = gw_for(arm, zcfg=ZtaConfig(tau_alpha=0.5, rl_offset=1.0))
    roll = [0.0] * 12
    roll[5] = 2.5
    plan = g.decide(view_at(t=3, roll=roll), lambda i: False)
    assert plan.block, "precondition: the scenario must block something"
    g.on_deliver(3, {i: False for i in range(12)})
    delivered = {(s, r) for s, r, _st in g._log_delivered[3]}          # noqa: SLF001
    for edge in plan.block:
        assert tuple(edge) not in delivered, f"{arm}: blocked {edge} counted as delivered"


def test_unauthenticated_edge_is_not_logged_as_delivered():
    adj = chain_adj(12)
    recv = [sorted(set(a) | ({0} if i == 11 else set())) for i, a in enumerate(adj)]   # extra 0 -> 11 edge
    z = ZtaConfig()
    g = gw_for("arm_B", zcfg=z, adj=adj)
    from bazt.conventional_zta import make_conventional_gateway
    g = make_conventional_gateway("arm_B", z, conventional_bazt_config(z, None, None), sl.Config(N=12, T=12),
                                  "chain", 9001, adj, recv)
    g.on_deliver(3, {0: False})
    assert (0, 11) not in {(s, r) for s, r, _ in g._log_delivered[3]}   # noqa: SLF001


def test_blocking_lowers_message_throughput_in_a_real_episode():
    cfg = sl.Config(N=12, T=8)
    from dataclasses import replace
    from bazt.arms import RunContext, build_episode
    z = ZtaConfig(tau_alpha=0.05)
    b = conventional_bazt_config(z, None, None)
    w = sl.make_world("chain", 9001, cfg)
    ctx = RunContext(bcfg=b, agent_factory=lambda c, _w: sl.MockAgentPolicy(c), zcfg=z)
    s = build_episode("arm_B", w, cfg, ctx)
    hooks = replace(s.hooks, suppress_patient_zero=True, recv_from=None)
    r = sl.run_episode(w, s.policy, s.cfg, s.agent, frozenset(), hooks=hooks)
    assert r["intervention_spend"] > 0
    assert r["message_throughput"] < 1.0


# ------------------------------------------------------------------------- m3 / C1 / C2: _bisect
def test_bisect_picks_most_aggressive_feasible_when_small_is_aggressive():
    cands = [0.1, 0.2, 0.3, 0.4, 0.5]
    rate = {0.1: 0.5, 0.2: 0.1, 0.3: 0.02, 0.4: 0.01, 0.5: 0.0}              # deny falls as the value rises
    v, r = calibration._bisect(cands, rate.__getitem__, 0.02, aggressive_high=False)
    assert (v, r) == (0.3, 0.02)


def test_bisect_picks_most_aggressive_feasible_when_large_is_aggressive():
    cands = [0.1, 0.2, 0.3, 0.4, 0.5]
    rate = {0.1: 0.0, 0.2: 0.01, 0.3: 0.02, 0.4: 0.1, 0.5: 0.5}              # deny rises with the value
    v, r = calibration._bisect(cands, rate.__getitem__, 0.02, aggressive_high=True)
    assert (v, r) == (0.3, 0.02)


@pytest.mark.parametrize("aggressive_high,expected", [(True, 0.1), (False, 0.5)])
def test_bisect_fallback_is_the_most_permissive_end_of_the_arms_own_direction(aggressive_high, expected):
    v, _ = calibration._bisect([0.1, 0.3, 0.5], lambda x: 0.9, 0.02, aggressive_high=aggressive_high)
    assert v == expected


# ------------------------------------------------- C1/C2/C3/C5: the real calibration, direction from data
@pytest.fixture(scope="module")
def tuned(tmp_path_factory):
    config = ex.load_config(CFG_PATH)
    out = tmp_path_factory.mktemp("cal") / "t.json"
    return calibration.calibrate_thresholds(config, "mock", "tune", str(out), verbose=False,
                                            seeds=list(range(9001, 9007)), topologies=["chain", "small_world"])


def test_arm_c_offset_is_not_frozen_at_the_most_permissive_grid_end(tuned):
    assert tuned["thresholds"]["arm_C"]["rl_offset"] > -0.6


def test_arm_b_tau_is_not_frozen_at_the_most_permissive_grid_end(tuned):
    assert tuned["thresholds"]["arm_B"]["tau_alpha"] < 0.99


def test_every_calibrated_arm_meets_alpha(tuned):
    for arm, rate in tuned["measured_benign_deny_rate"].items():
        assert rate <= tuned["alpha_clean_fpr"] + 1e-9, arm


# ------------------------------------------------------------------------------------------ m8
def test_conventional_bazt_config_has_no_unused_parameter():
    assert "total_deliveries" not in inspect.signature(conventional_bazt_config).parameters


def test_edge_open_has_no_dead_until_is_none_clause():
    from bazt import conventional_zta
    assert "(until is None or t <= until)" not in inspect.getsource(conventional_zta.ConventionalGateway.edge_open)


# ------------------------------------------------------------------ m1: LLM calibration uses 3 seeds
def test_llm_platform_without_explicit_seeds_calibrates_on_three_seeds(tmp_path):
    config = dict(ex.load_config(CFG_PATH))
    config["splits"] = {**config.get("splits", {}), "tune": [9001, 9010]}
    doc = calibration.calibrate_thresholds(config, "llm", "tune", str(tmp_path / "l.json"), verbose=False,
                                           topologies=["chain"])
    assert doc["n_seeds"] == 3 and doc["calibration_seeds"] == [9001, 9003]


def test_mock_platform_without_explicit_seeds_uses_the_whole_tune_split(tmp_path):
    config = dict(ex.load_config(CFG_PATH))
    config["splits"] = {**config.get("splits", {}), "tune": [9001, 9004]}
    doc = calibration.calibrate_thresholds(config, "mock", "tune", str(tmp_path / "m.json"), verbose=False,
                                           topologies=["chain"])
    assert doc["n_seeds"] == 4


# ------------------------------------------- C6: the shipped thresholds must let arm_C actually intervene
SHIPPED_THRESHOLDS = os.path.join("results", "experiments", "frozen_thresholds.json")


def test_arm_c_with_shipped_frozen_thresholds_intervenes_on_the_infection_scenario():
    config = ex.load_config(CFG_PATH)
    th = calibration.load_thresholds(SHIPPED_THRESHOLDS)["thresholds"]
    spend = []
    for seed in (1001, 1002, 1003):
        cell = {"platform": "mock", "stage": "mock_frontier", "split": "main", "scenario": "infection",
                "topology": "barabasi_albert", "N": 12, "arm": "arm_C", "audit_pct": None, "int_pct": None,
                "seed": seed}
        spend.append(ex.run_episode_cell(cell, config, th)["intervention_spend"])
    assert sum(spend) > 0, f"arm_C never intervened with the shipped thresholds: {th['arm_C']}"


def test_shipped_frozen_thresholds_are_not_the_permissive_grid_ends():
    src = inspect.getsource(calibration.calibrate_thresholds)
    # grid ends are local to calibrate_thresholds; fail loudly if they are edited so this guard is not stale
    assert "np.linspace(0.05, 0.99, 20)" in src and "np.linspace(-0.6, 0.8, 15)" in src
    th = calibration.load_thresholds(SHIPPED_THRESHOLDS)["thresholds"]
    assert th["arm_B"]["tau_alpha"] < 0.99
    assert th["arm_C"]["rl_offset"] > -0.6
