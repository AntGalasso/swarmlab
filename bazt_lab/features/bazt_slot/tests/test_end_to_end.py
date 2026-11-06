"""BAZT and BAZT_upstream through the real MockPlatform on a tiny setting (every setting is a TEST value)."""
import math

import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.bazt_slot import public as slot
from bazt_lab.features.mock_engine import public as mock

ARMS = {a.name: a for a in slot.ARMS}
CAP = 0.2


def settings(mode="open_loop"):
    s = core.load_settings(profile="sprint")
    test = {
        "runtime.n_insiders": 2, "runtime.rate_limit_per_identity": 1, "runtime.horizon_constant": 3.0,
        "runtime.eps_correct": 2.0, "runtime.ba_m": 2, "runtime.ws_k": 4, "runtime.ws_rewire_p": 0.25,
        "runtime.sanitize_success": 0.9, "runtime.insider_bias": 8.0, "runtime.ta_detectability": 1.5,
        "runtime.ma_detectability": 1.2, "runtime.aitm_rewrite_strength": 1.0, "runtime.recovery_margin": 0.05,
        "mock.model.bias": 8.0, "mock.model.burst_span": 1, "mock.auditor.binormal_method": "shared_threshold",
        "mock.batch_size": 2, "calibration.mode": mode, "costs.robustness.deny": 1,
        "conventional.c.judge_score_bins": {"mock": [[0.0, 0.0], [2.0, 4.0], [math.inf, 12.0]]},
        "conventional.c.presence_window_rounds": 3,
        "bazt_upstream.verdict_probability": 0.98, "bazt.katz_decay": 0.5, "bazt.eigen_iterations": 50, "bazt.sink_weight": 2.0, "bazt.utility_weight": 1.0,
        "bazt.utility_loss.sanitize": 0.15, "bazt.utility_loss.delay": 0.1, "bazt.utility_loss.edge_quarantine": 0.3,
        "bazt.utility_loss.rate_limit": 0.3, "bazt.utility_loss.privilege_degradation": 0.35,
        "bazt.utility_loss.isolation": 1.0, "bazt.flag_logodds": 4.0, "bazt.clear_logodds": 3.0,
        "bazt.memory_decay": 0.8, "bazt.prior": 0.1, "bazt.inspect_conclusive": 0.8,
    }
    for k, v in test.items():
        core.put(s, k, v)
    return s


def unit(arm, seed, attack="insider_pi", audit=CAP, intervention=CAP, **extra):
    u = dict(block="e2e", platform="mock", arm=arm, audit=audit, intervention=intervention, topology="star_hub",
             attack=attack, agents=6, seed=seed, sink=0, **extra)
    u["id"] = f"{arm}-{attack}-{seed}-{audit}-{intervention}"
    return u


class Rows:
    """Rows function: a summary row per episode; keeps the traces."""

    def __init__(self):
        self.traces = []

    def __call__(self, trace, settings):
        self.traces.append(trace)
        return [{"unit_id": u["id"], "spend_audit": float(trace.spend_audit[b].sum()),
                 "spend_intervention": float(trace.spend_intervention[b].sum()),
                 "violations": int(trace.violations[b]), "correct": trace.correct[b].tolist(),
                 "infected": trace.infected[b].tolist()} for b, u in enumerate(trace.unit_ids)]


def thresholds_for(name, s, tmp_path):
    arm = ARMS[name]
    if not arm.calibrated:
        return {}
    units = [unit(name, 9000 + k, "none", "uncapped", "uncapped") for k in range(1, 4)]
    return mock.MockPlatform(Rows()).calibrate(units, arm, s, core.CheckpointStore(tmp_path / "cal"))


@pytest.mark.parametrize("mode", ["open_loop", "closed_loop"])
@pytest.mark.parametrize("name", ["BAZT", "BAZT_upstream"])
def test_the_arm_runs_on_the_real_mock_platform_within_its_budgets(name, mode, tmp_path):
    s = settings(mode)
    th = thresholds_for(name, s, tmp_path)
    rec = Rows()
    units = [unit(name, k, thresholds=th) for k in range(1, 5)]
    rows = mock.MockPlatform(rec).run_batch(units, ARMS[name], s, core.CheckpointStore(tmp_path / "run"), core.NullTracker())
    assert [r["unit_id"] for r in rows] == [u["id"] for u in units]
    assert all(r["violations"] == 0 for r in rows)
    assert sum(r["spend_intervention"] + r["spend_audit"] for r in rows) > 0          # it does act
    for trace in rec.traces:
        deliveries = trace.attempts.sum((1, 2, 3)).to(torch.float64)       # edges x rounds
        assert (trace.spend_intervention.sum(1) <= CAP * deliveries + 1e-9).all()
        assert (trace.spend_audit.sum(1) <= CAP * deliveries + 1e-9).all()


@pytest.mark.parametrize("mode", ["open_loop", "closed_loop"])
@pytest.mark.parametrize("name", ["BAZT", "BAZT_upstream"])
def test_a_run_resumed_after_a_crash_gives_the_same_rows(name, mode, tmp_path):
    s = settings(mode)
    th = thresholds_for(name, s, tmp_path)
    units = [unit(name, k, thresholds=th) for k in range(1, 5)]
    full = mock.MockPlatform(Rows()).run_batch(units, ARMS[name], s, core.CheckpointStore(tmp_path / "a"), core.NullTracker())
    store = core.CheckpointStore(tmp_path / "b")
    for u, row in zip(units[:2], full[:2], strict=True):                  # the first chunk was saved before the crash
        store.append_rows(u["id"], [row])
        store.mark_done(u["id"])
    rest = mock.MockPlatform(Rows()).run_batch(units, ARMS[name], s, store, core.NullTracker())
    assert rest == full[2:]


# ------------------------------------------------------------------------------------ the real LLM platform (fake API)
LLM_COSTS = {k.value: 1.0 for k in core.ActionKind}


def llm_settings():
    s = settings()
    for k, v in {"llm.api": "fake", "llm.ma_correctness": ["rule"], "llm.ma_judge_model": "fake/judge",
                 "llm.conversations_per_cell": 4, "llm.calibration_conversations_per_cell": 4,
                 "gnn_train.graphs_per_setting": 3, "budgets.alpha": 0.1, "calibration.mode": "open_loop",
                 "costs.main": LLM_COSTS, "costs.robustness": LLM_COSTS, "runtime.llm_seeds": [1, 2],
                 "runtime.llm_calibration_seeds": [101, 102], "runtime.random_edge_probability": 1.0,
                 "runtime.llm_sink_positions": {"random_directed": [0, 1]},
                 "conventional.c.judge_score_bins": {"llm": [[-5.0, 0.0], [-1.0, 4.0], [math.inf, 12.0]]}}.items():
        core.put(s, k, v)
    return s


@pytest.mark.parametrize("name", ["BAZT", "BAZT_upstream"])
def test_the_arm_runs_on_the_real_llm_platform_with_the_budget_spread_over_the_seed_run(name, tmp_path):
    from bazt_lab.features.llm_harness import public as harness
    s = llm_settings()
    u = {"id": f"{name}-llm", "block": "b", "platform": "llm", "arm": name, "audit": CAP, "intervention": CAP,
         "backbone": "fake/b", "topology": "random_directed", "attack": "insider_pi", "task": "mmlu", "seed": 1,
         "conversations": 4, "agents": 5, "rounds": 4, "sanitize_mode": "redaction", "thresholds": {}}
    cal = {**u, "id": f"{name}-cal", "attack": "none", "audit": "uncapped", "intervention": "uncapped"}
    platform = harness.LLMPlatform()
    th = platform.calibrate([cal], ARMS[name], s, core.CheckpointStore(tmp_path / "cal"))
    rows = platform.run_batch([{**u, "thresholds": th}], ARMS[name], s, core.CheckpointStore(tmp_path / "run"),
                              core.NullTracker())
    assert len(rows) == 4
    assert all(r["violations"] == 0 for r in rows)                      # the platform never had to reject a round
    assert sum(r["spend_intervention_total"] for r in rows) > 0         # it acts
    if name == "BAZT":      # the audit budget is paced over the whole seed run, not spent in the first conversation
        assert sum(1 for r in rows if r["spend_audit_total"] > 0) >= 2


# ------------------------------------------------------------------ D123: a clearing audit lifts an isolated agent
def _lifted(trace):
    """Number of (episode, agent) reinstatements in the trace: active again after a round of isolation."""
    a = trace.active
    return int((a[:, 1:] & ~a[:, :-1]).sum())


def _isolated_audit_run(tmp_path, tag, *, dtype="float64", batch=64, audit="uncapped", name="BAZT_no_budget"):
    s = settings()
    core.put(s, "compute.dtype", dtype)
    core.put(s, "mock.batch_size", batch)
    th = thresholds_for(name, s, tmp_path / tag)
    rec = Rows()
    units = [dict(unit(name, k, thresholds=th, audit=audit, intervention="uncapped"), topology="barabasi_albert", agents=8)
             for k in range(1, 7)]
    rows = mock.MockPlatform(rec).run_batch(units, ARMS[name], s, core.CheckpointStore(tmp_path / tag / "run"), core.NullTracker())
    return rows, rec.traces


def test_on_the_real_mock_a_clearing_audit_lifts_isolated_agents(tmp_path):
    _, traces = _isolated_audit_run(tmp_path, "a")
    assert sum(int((~t.active).any()) for t in traces) > 0                                  # agents do get isolated
    assert sum(_lifted(t) for t in traces) > 0                                              # and audits now clear some


def test_the_lifts_do_not_depend_on_the_batch_or_the_precision_class(tmp_path):
    lifts = {bs: sum(_lifted(t) for t in _isolated_audit_run(tmp_path, f"b{bs}", batch=bs)[1]) for bs in (1, 3, 64)}
    assert len(set(lifts.values())) == 1 and next(iter(lifts.values())) > 0
    rows_a, _ = _isolated_audit_run(tmp_path, "f1", batch=1)
    rows_b, _ = _isolated_audit_run(tmp_path, "f2", batch=64)
    assert rows_a == rows_b


def test_the_audit_cap_holds_exactly_with_the_isolated_channel(tmp_path):
    _, traces = _isolated_audit_run(tmp_path, "c", audit=CAP, name="BAZT")
    for trace in traces:
        deliveries = trace.attempts.sum((1, 2, 3)).to(torch.float64)
        assert (trace.spend_audit.sum(1) <= CAP * deliveries + 1e-9).all()
        assert (trace.spend_audit.sum(1) == torch.floor(trace.spend_audit.sum(1))).all()          # whole units at cost 1


def test_float32_isolated_audits_still_lift_and_accounting_stays_float64(tmp_path):
    _, t32 = _isolated_audit_run(tmp_path, "d32", dtype="float32")
    assert sum(_lifted(t) for t in t32) > 0 and all(t.spend_audit.dtype == torch.float64 for t in t32)
