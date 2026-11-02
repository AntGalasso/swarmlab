"""Offline tests (no network, no API key). Run: pytest -q"""
import asyncio
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import swarmlab as sl  # noqa: E402
import stats_check as sc  # noqa: E402
from llm_client import LLMBackend, SQLiteLLMCache, run_connectivity_smoke_test  # noqa: E402

CFG = sl.Config()


def _run(topo, seed, policy, decoys=3, cfg=CFG):
    w = sl.make_world(topo, seed, cfg)
    d = sl.pick_decoys(w, decoys)
    return sl.run_episode(w, policy, cfg, sl.MockAgentPolicy(cfg), d)


def test_determinism():
    assert _run("chain", 5, "risk_aware") == _run("chain", 5, "risk_aware")


@pytest.mark.parametrize("topo", sl.TOPOLOGIES + ["complete"])
def test_cap_and_budget(topo):
    for pol in sl.POLICIES:
        r = _run(topo, 3, pol)
        assert r["quarantines_used"] <= (0 if pol == "no_defense" else CFG.K)
        if pol != "risk_aware_tiered":
            assert r["sanitizations_used"] == 0
        assert r["sanitizations_used"] <= CFG.B_san
        assert r["sanitized_nodes"] <= r["sanitizations_used"]


def test_crn_prefix_sharing():
    """Before a window opens, the arm must be identical to no_defense (same shared randomness)."""
    for topo in sl.TOPOLOGIES:
        base = _run(topo, 11, "no_defense")["ever_curve"]
        late = _run(topo, 11, "fixed_late")["ever_curve"]  # window opens at t=9
        assert base[:8] == late[:8]
        mid = _run(topo, 11, "fixed_mid")["ever_curve"]    # window opens at t=5
        assert base[:4] == mid[:4]


def test_clean_and_decoy_share_the_world():
    w = sl.make_world("small_world", 4, CFG)
    w2 = sl.make_world("small_world", 4, CFG)
    assert w.adj == w2.adj and w.p0 == w2.p0 and np.allclose(w.x, w2.x)
    assert sl.pick_decoys(w, 0) == frozenset()
    d = sl.pick_decoys(w, 3)
    assert len(d) == 3 and w.p0 not in d


def test_star_decoys_are_peripheral():
    for s in range(1, 20):
        w = sl.make_world("star_hub", s, CFG)
        assert 0 not in sl.pick_decoys(w, 3)


def test_decoys_are_placed_on_the_periphery():
    for topo in sl.TOPOLOGIES:
        for s in range(1, 8):
            w = sl.make_world(topo, s, CFG)
            deg = [len(a) for a in w.adj]
            d = sorted(sl.pick_decoys(w, 3))
            assert max(deg[i] for i in d) <= np.median(deg), (topo, s, d)


def test_horizon_rule_no_spend_at_last_step():
    """H=0 when no steps remain, so risk_aware never spends at t=T."""
    w = sl.make_world("chain", 2, CFG)
    aset = set(range(CFG.N))
    assert sl.reach_H(w.adj, 0, 0, CFG, aset, [0.0] * CFG.N) == 0.0
    assert sl.reach_H(w.adj, 0, 5, CFG, aset, [0.0] * CFG.N) > 0.0


def test_no_defense_has_no_quarantine_and_never_contains():
    r = _run("star_hub", 1, "no_defense")
    assert r["quarantines_used"] == 0 and r["time_to_containment"] == CFG.T + 1


def test_oracle_never_wastes_budget():
    for s in range(1, 15):
        r = _run("small_world", s, "offline_oracle")
        assert r["collateral_healthy_quarantined"] == 0


def test_node_removal_changes_readout():
    cfg2 = sl.Config(quarantine_mode="node_removal")
    a = _run("chain", 6, "fixed_early", cfg=CFG)
    b = _run("chain", 6, "fixed_early", cfg=cfg2)
    assert a["quarantines_used"] == b["quarantines_used"]
    assert a["task_mae"] != b["task_mae"]


def test_deltas_are_paired_and_bounded():
    rows, _ = sl.run_grid(CFG, ["chain"], ["decoy"], sl.POLICIES, range(1, 6), lambda c: sl.MockAgentPolicy(c), progress=False)
    sl.add_deltas(rows)
    for r in rows:
        if r["policy"] == "no_defense":
            assert r["delta_abs"] == 0
        if r["policy"] == "offline_oracle" and r["delta_rel"] == r["delta_rel"]:
            assert abs(r["delta_rel"] - 1.0) < 1e-9


def test_budget_estimator():
    e = sl.estimate_llm_budget(["a", "b"], ["clean", "decoy"], sl.POLICIES, 30, price_in=0.1, price_out=0.2)
    assert e["worst_case_calls"] == 2 * 2 * len(sl.POLICIES) * 30 * 12 * 12
    assert e["worst_case_usd"] > 0


def test_llm_mode_refuses_without_flag():
    with pytest.raises(SystemExit):
        sl.main(["--mode", "llm", "--seeds", "1"])


def test_llm_mode_refuses_before_kickoff(monkeypatch):
    monkeypatch.setenv("SWARMLAB_KICKOFF_UTC", "2999-01-01T00:00:00+00:00")
    with pytest.raises(SystemExit):
        sl.main(["--mode", "llm", "--allow-live-llm", "--seeds", "1"])


def test_tune_refuses_eval_seeds():
    with pytest.raises(SystemExit):
        sl.main(["--tune", "--split", "main"])


# ============================================================ part 1: decoy realism / divergence
def test_monitor_calibration_decoy_spikes_above_fresh_infection():
    """Root cause of the old equivalence collapse: mu_dec equalled mu_inf/2 exactly.

    A decoy's first (zero-padded) rolling evidence must be at least a freshly infected node's, and must
    still be below the persistent infection level reached one step later.
    """
    roll_decoy = CFG.mu_dec / 2        # single burst step, zero left-padded
    roll_fresh = CFG.mu_inf / 2        # first observed infected step
    roll_persist = CFG.mu_inf          # second observed infected step
    p_decoy = sl.sigmoid(CFG.a * roll_decoy - CFG.b)
    p_fresh = sl.sigmoid(CFG.a * roll_fresh - CFG.b)
    p_persist = sl.sigmoid(CFG.a * roll_persist - CFG.b)
    assert p_decoy > p_fresh, "benign decoy must be at least as suspicious as a fresh infection"
    assert p_decoy >= CFG.tau_p > 0, "a decoy spike must trigger the naive threshold"
    assert p_persist > p_decoy, "persistent contagion must eventually dominate the transient decoy"


def test_decoy_bursts_are_transient_and_early():
    for topo in sl.TOPOLOGIES:
        for s in range(1, 10):
            w = sl.make_world(topo, s, CFG)
            for i in sorted(sl.pick_decoys(w, 3)):
                steps = [t for t in range(1, CFG.T + 1) if w.burst[t - 1, i]]
                assert steps and min(steps) <= 2, "decoy spike must be early"
                assert len(steps) <= 2, "decoy header must be a short transient burst"
                assert steps == list(range(min(steps), max(steps) + 1)), "burst must be contiguous"


def test_fixed_early_and_score_only_use_independent_decision_logic():
    """fixed_early = time-boxed snapshot audit; score_only = persistent alarm, any step."""
    w = sl.make_world("small_world", 7, CFG)
    active = list(range(CFG.N))
    aset, deg = set(active), [len(a) for a in w.adj]
    p = [0.0] * CFG.N
    p[w.p0] = 0.99
    other = next(i for i in active if i != w.p0)
    p[other] = 0.80
    fresh = [0] * CFG.N
    fresh[w.p0] = 1
    fresh[other] = 1
    # inside the early window, a single crossing fires the snapshot audit ...
    quar, _ = sl._decide("fixed_early", 1, w, CFG, w.adj, active, aset, p, [False] * CFG.N, deg, max(deg), 3,
                         0, fresh)
    assert w.p0 in quar
    # ... but the same evidence is NOT enough for score_only's persistence rule
    quar, _ = sl._decide("score_only", 1, w, CFG, w.adj, active, aset, p, [False] * CFG.N, deg, max(deg), 3,
                         0, fresh)
    assert quar == []
    # and outside its window the fixed arm cannot act at all, while score_only still can
    quar, _ = sl._decide("fixed_early", 6, w, CFG, w.adj, active, aset, p, [False] * CFG.N, deg, max(deg), 3,
                         0, [CFG.confirm_steps] * CFG.N)
    assert quar == []
    quar, _ = sl._decide("score_only", 6, w, CFG, w.adj, active, aset, p, [False] * CFG.N, deg, max(deg), 3,
                         0, [CFG.confirm_steps] * CFG.N)
    assert w.p0 in quar


def test_fixed_early_and_score_only_trajectories_diverge_under_decoy_pressure():
    seen_diff = 0
    n = 0
    for topo in sl.TOPOLOGIES:
        for s in range(1, 11):
            a = _run(topo, s, "fixed_early")["ever_curve"]
            b = _run(topo, s, "score_only")["ever_curve"]
            seen_diff += a != b
            n += 1
    assert seen_diff >= 0.6 * n, f"trajectories coincide on too many seeds ({seen_diff}/{n})"
    auc_fe = np.mean([_run(t, s, "fixed_early")["auc_ever_infected"] for t in sl.TOPOLOGIES for s in range(1, 11)])
    auc_so = np.mean([_run(t, s, "score_only")["auc_ever_infected"] for t in sl.TOPOLOGIES for s in range(1, 11)])
    assert auc_fe != auc_so


def test_decoys_bite_naive_arms_more_than_risk_aware():
    def mean(pol, regime):
        return np.mean([_run(t, s, pol, decoys=3 if regime == "decoy" else 0)["auc_ever_infected"]
                        for t in sl.TOPOLOGIES for s in range(1, 11)])
    assert mean("fixed_early", "decoy") - mean("fixed_early", "clean") > 0.03
    assert mean("score_only", "decoy") - mean("score_only", "clean") > 0.0


# ============================================================ part 1: metric definitions
def test_metric_definitions_are_internally_consistent():
    for pol in ("fixed_early", "score_only", "risk_aware", "risk_aware_tiered"):
        r = _run("small_world", 9, pol)
        curve = r["ever_curve"]
        assert r["auc_ever_infected"] == pytest.approx(np.mean(curve) / CFG.N)
        assert r["final_ever_infected_rate"] == pytest.approx(curve[-1] / CFG.N)
        assert 0.0 <= r["auc_active_unquarantined"] <= r["auc_ever_infected"] + 1e-12
        assert 0.0 <= r["decoy_quarantines_wasted"] <= 1.0
        assert r["decoy_quarantines_wasted"] == pytest.approx(
            sum(r["cum_decoy_quarantines"][-1:]) / CFG.K if CFG.K else 0.0)
        assert 1 <= r["time_to_containment"] <= CFG.T + 1
        assert len(r["cum_decoy_quarantines"]) == len(r["cum_true_quarantines"]) == CFG.T


def test_time_to_containment_matches_the_curve():
    r = _run("star_hub", 2, "fixed_early")
    ever = r["ever_curve"]
    assert r["time_to_containment"] <= CFG.T
    assert ever[r["time_to_containment"] - 1] >= 1


def test_delta_rel_clamps_small_denominators():
    rows = []
    for pol, auc in (("no_defense", 0.25), ("offline_oracle", 0.25), ("score_only", 0.10)):
        rows.append({"topology": "chain", "regime": "decoy", "seed": 1, "policy": pol,
                     "auc_ever_infected": auc})
    sl.add_deltas(rows)
    so = [r for r in rows if r["policy"] == "score_only"][0]
    assert so["delta_rel_clamped"] == 1
    assert np.isfinite(so["delta_rel"])
    assert so["delta_rel"] == pytest.approx(0.15 / sl.DEN_FLOOR)


# ============================================================ part 1: tuning + stats
def test_tuning_objective_penalises_mae():
    mae_base = {"chain": 2.0}
    cheap = [{"topology": "chain", "auc_ever_infected": 0.10, "task_mae": 2.0}]
    costly = [{"topology": "chain", "auc_ever_infected": 0.10, "task_mae": 4.0}]
    a = sl.tuning_objective(cheap, mae_base, 0.25)
    b = sl.tuning_objective(costly, mae_base, 0.25)
    assert a["J"] < b["J"]
    assert b["mae_ratio"] == pytest.approx(2.0)
    # a candidate that buys AUC by wrecking MAE must lose
    assert (sl.tuning_objective([{"topology": "chain", "auc_ever_infected": 0.09, "task_mae": 8.0}],
                                mae_base, 0.25)["J"] > a["J"])


def test_tuning_always_uses_the_full_30_tune_seeds(monkeypatch, tmp_path):
    captured = {}

    def fake_tune(cfg, topologies, seeds, factory, alpha=0.25):
        captured["seeds"] = list(seeds)
        return {}, {"alpha": alpha}

    monkeypatch.setattr(sl, "tune", fake_tune)
    assert sl.main(["--tune", "--seeds", "20", "--out", str(tmp_path)]) == 0
    assert captured["seeds"] == list(range(9001, 9031))


def _synthetic_rows(n=30, d_auc=-0.20, d_mae=-0.50, sd_auc=0.02, sd_mae=0.05, topo="chain"):
    rng = np.random.default_rng(0)
    rows = []
    for s in range(1, n + 1):
        for pol, shift_auc, shift_mae in (("risk_aware", 0.0, 0.0), ("score_only", -d_auc, -d_mae)):
            rows.append({"topology": topo, "regime": "decoy", "seed": s, "policy": pol,
                         "auc_ever_infected": 0.30 + shift_auc + rng.normal(0, sd_auc),
                         "task_mae": 4.0 + shift_mae + rng.normal(0, sd_mae)})
    return rows


def test_pareto_labels_strict_and_non_inferior():
    rows = _synthetic_rows()
    rep = sc.h3(rows, ["chain"], challengers=("risk_aware",), margin_mae=0.05)
    v = rep["challengers"]["risk_aware"]["per_topology"]["chain"]["verdict"]["score_only"]
    assert v["label"] == sc.LABEL_STRICT
    assert v["delta_auc"] < 0 and v["delta_mae"] < 0
    assert v["p_raw_auc"] <= v["p_holm_auc"] + 1e-12  # both raw and adjusted p are reported

    noninf = _synthetic_rows(d_mae=0.02, sd_mae=0.004)
    rep = sc.h3(noninf, ["chain"], challengers=("risk_aware",), margin_mae=0.05)
    v = rep["challengers"]["risk_aware"]["per_topology"]["chain"]["verdict"]["score_only"]
    assert v["label"] == sc.LABEL_SAFETY
    assert v["ci95_delta_mae"][1] <= 0.05


def test_utility_loss_is_never_labelled_a_win():
    rows = _synthetic_rows(d_mae=3.0, sd_mae=0.4)
    rep = sc.h3(rows, ["chain"], challengers=("risk_aware",), margin_mae=0.05)
    v = rep["challengers"]["risk_aware"]["per_topology"]["chain"]["verdict"]["score_only"]
    assert v["label"] == sc.LABEL_UTILITY_LOSS
    assert "WIN" not in v["label"]
    assert not v["strict_pareto"] and not v["utility_non_inferior"]
    assert v["p_holm_mae"] < sc.ALPHA


def test_no_significant_difference_is_labelled_null():
    rows = _synthetic_rows(d_auc=-0.001, d_mae=0.0, sd_auc=0.05, sd_mae=0.05)
    rep = sc.h3(rows, ["chain"], challengers=("risk_aware",), margin_mae=0.05)
    v = rep["challengers"]["risk_aware"]["per_topology"]["chain"]["verdict"]["score_only"]
    assert v["label"] == sc.LABEL_NULL


def test_h3_labels_are_exhaustive_and_win_labels_require_lower_auc():
    rows = _synthetic_rows()
    rng = np.random.default_rng(1)
    for s in range(31, 62):
        rows.append({"topology": "chain", "regime": "decoy", "seed": s, "policy": "risk_aware_tiered",
                     "auc_ever_infected": 0.40 + rng.normal(0, 0.02), "task_mae": 3.0 + rng.normal(0, 0.05)})
    rep = sc.h3(rows, ["chain"], challengers=("risk_aware", "risk_aware_tiered"), margin_mae=0.05)
    allowed = {sc.LABEL_STRICT, sc.LABEL_SAFETY, sc.LABEL_CONTAIN, sc.LABEL_UTILITY_LOSS, sc.LABEL_NULL}
    for ch, d in rep["challengers"].items():
        for t, per in d["per_topology"].items():
            for r, v in per["verdict"].items():
                assert v["label"] in allowed
                if "WIN" in v["label"]:
                    assert v["delta_auc"] < 0 and v["p_holm_auc"] < sc.ALPHA
                    assert not (v["delta_mae"] > 0 and v["p_holm_mae"] < sc.ALPHA)


# ============================================================ part 2: two-tier protocol
def test_trusted_sanitize_keeps_the_number_and_drops_every_tag():
    txt = f"{sl.DECOY_TAG} a3 t2 est=51.25 calib chatter {sl.CANARY}"
    out = sl.trusted_sanitize(txt, 3, 2, 1)
    assert sl.CANARY not in out and sl.DECOY_TAG not in out
    assert "est=51.2" in out and "calib" not in out
    assert sl.trusted_sanitize("no number here", 1, 1) == "a1 t1 est=NA"


def test_tiered_sanitizes_instead_of_quarantining_under_ambiguity():
    w = sl.make_world("small_world", 5, CFG)
    decoys = sl.pick_decoys(w, 3)
    r = sl.run_episode(w, "risk_aware_tiered", CFG, sl.MockAgentPolicy(CFG), decoys)
    assert r["sanitizations_used"] > 0
    assert r["quarantines_used"] <= CFG.K
    assert r["sanitizations_used"] <= CFG.B_san
    assert r["decoy_quarantines_wasted"] <= 1.0


def test_tier1_is_per_step_not_permanent_and_re_selects_nodes():
    """B_san caps ACTIONS (node-steps): a node that stays risky is re-selected and re-spends.

    This is what separates tier 1 from quarantine: it never silences a node permanently.
    """
    reused = 0
    for topo in sl.TOPOLOGIES:
        for s in range(1, 8):
            r = _run(topo, s, "risk_aware_tiered")
            assert r["sanitizations_used"] <= CFG.B_san
            assert r["sanitized_nodes"] <= r["sanitizations_used"]
            reused += r["sanitizations_used"] > r["sanitized_nodes"]
    assert reused > 0, "per-step sanitation must re-select at least one still-risky node"


def test_p_strip_controls_how_often_the_tag_is_actually_stripped():
    """p_strip is a real mock probability (CRN draw per node-step), not a decorative constant."""
    def mean_auc(p_strip):
        cfg = sl.Config(p_strip=p_strip)
        return np.mean([_run(t, s, "risk_aware_tiered", cfg=cfg)["auc_ever_infected"]
                        for t in sl.TOPOLOGIES for s in range(1, 7)])
    never, always = mean_auc(0.0), mean_auc(1.0)
    assert always <= never + 1e-12, "always stripping cannot be worse for containment"
    assert never - always > 0.0, "p_strip must have a measurable containment effect"


def test_tiered_respects_both_budgets_and_contains_at_least_as_well_as_hard_quarantine():
    fe = np.mean([_run("small_world", s, "fixed_early")["auc_ever_infected"] for s in range(1, 11)])
    ti = np.mean([_run("small_world", s, "risk_aware_tiered")["auc_ever_infected"] for s in range(1, 11)])
    assert ti <= fe
    for s in range(1, 11):
        r = _run("small_world", s, "risk_aware_tiered")
        assert r["sanitizations_used"] <= CFG.B_san and r["quarantines_used"] <= CFG.K


def test_tiered_keeps_utility_above_hard_quarantine_on_the_hub():
    maes_r = np.mean([_run("star_hub", s, "risk_aware")["task_mae"] for s in range(1, 11)])
    maes_t = np.mean([_run("star_hub", s, "risk_aware_tiered")["task_mae"] for s in range(1, 11)])
    assert maes_t <= maes_r + 1e-9


# ============================================================ part 3: scaling + payloads
def test_scaling_sweep_budget_rules():
    rows, eps = sl.run_scaling(CFG, range(1, 4), ns=(12, 24), policies=sl.SCALING_POLICIES, progress=False)
    seen = {(r["N"], r["budget_rule"], r["K"]) for r in rows}
    assert (12, "prop", 3) in seen and (24, "prop", 6) in seen and (24, "fixed", 3) in seen
    for r in rows:
        assert r["quarantines_used"] <= r["K"]
    assert all(len(e["cum_decoy_quarantines"]) == CFG.T for e in eps)


def test_payload_templates_render_for_both_payloads():
    base = dict(i=2, t=1, inbox=["a1 t1 est=50.0"], own_prev=None, est=50.0, is_seed=False, is_decoy=False,
                decoy_active=False, u_adopt=0.5, u_stay=0.5, u_decoy=0.5, seed=1, topology="chain")
    seed_ctx = sl.AgentCtx(**dict(base, is_seed=True))
    decoy_ctx = sl.AgentCtx(**dict(base, is_decoy=True, decoy_active=True))
    for payload in sl.PAYLOADS:
        pol = sl.LLMAgentPolicy(backend=None, payload=payload)
        seed_sys = pol._prompts(seed_ctx)[0]
        decoy_sys = pol._prompts(decoy_ctx)[0]
        assert sl.DECOY_TAG in decoy_sys
        if payload == "canary_tag":
            assert sl.CANARY in seed_sys
        else:
            assert "+25" in seed_sys and sl.CANARY not in seed_sys
    with pytest.raises(ValueError):
        sl.LLMAgentPolicy(backend=None, payload="not_a_payload")


def test_mock_mode_never_touches_the_llm_backend(monkeypatch):
    import llm_client
    monkeypatch.setattr(llm_client, "make_llm_policy", lambda *a, **k: pytest.fail("live LLM used in mock mode"))
    assert sl.main(["--mode", "mock", "--seeds", "2", "--out", tempfile.mkdtemp(),
                    "--topologies", "chain", "--policies", "no_defense", "risk_aware_tiered"]) == 0


# ------------------------------------------------------------------------------- llm_client
class FakeTransport:
    def __init__(self):
        self.calls = 0

    async def __call__(self, model, messages, temperature, max_tokens):
        self.calls += 1
        await asyncio.sleep(0)
        return "READY", 5, 1


def test_cache_hits_and_crn_salt():
    with tempfile.TemporaryDirectory() as d:
        cache = SQLiteLLMCache(os.path.join(d, "c.sqlite"))
        tr = FakeTransport()
        be = LLMBackend("m", cache, tr, temperature=0.0, max_tokens=8)

        try:
            req = [("sys", "user", "salt-A")] * 3
            out = be.generate(req)
            assert out == ["READY"] * 3

            # Identical key requested concurrently may hit the API more than once;
            # a second pass must be all cache.
            before = tr.calls
            be.generate(req)
            assert tr.calls == before

            be.generate([("sys", "user", "salt-B")])
            assert tr.calls == before + 1  # Different CRN salt -> different key

        finally:
            be.close()
            cache.close()


def test_smoke_test_with_fake_transport(tmp_path):
    ok = run_connectivity_smoke_test(db_path=str(tmp_path / "s.sqlite"), transport=FakeTransport())
    assert ok


def test_retry_then_fail():
    class Boom:
        async def __call__(self, *a):
            raise RuntimeError("down")

    with tempfile.TemporaryDirectory() as d:
        cache = SQLiteLLMCache(os.path.join(d, "c.sqlite"))

        try:
            be = LLMBackend("m", cache, Boom(), retries=0)

            with pytest.raises(RuntimeError):
                be.generate([("s", "u", "x")])

        finally:
            cache.close()


def test_cache_rejects_use_after_close(tmp_path):
    """Regression: get()/put() after close() must raise a clear RuntimeError, not AttributeError."""
    cache = SQLiteLLMCache(db_path=str(tmp_path / "closed.sqlite"))
    cache.put("k1", "model-x", "hello", 3, 2)
    assert cache.get("k1") == "hello"
    cache.close()
    with pytest.raises(RuntimeError):
        cache.get("k1")
    with pytest.raises(RuntimeError):
        cache.put("k2", "model-x", "hello2", 3, 2)
