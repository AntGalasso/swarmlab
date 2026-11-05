"""MockPlatform: batching by horizon, progressive checkpoints, resume, calibration, determinism, refusals."""
import json
import math
import time

import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.metrics import public as metrics
from bazt_lab.features.mock_engine import public as mock
from bazt_lab.features.mock_engine.private import graphs
from bazt_lab.features.mock_engine.private import world as W
from bazt_lab.features.mock_engine.tests.helpers import FakeStore, Fn, Scripted, fake_rows, make_settings, spec, unit

Actions = core.Actions


class Recorder:
    """A rows function that keeps the traces it was given (and returns the fake rows)."""

    def __init__(self):
        self.traces = []

    def __call__(self, trace, settings):
        self.traces.append(trace)
        return fake_rows(trace, settings)


def platform(rows=None):
    return mock.MockPlatform(rows_fn=rows or fake_rows)


def units_of(n=4, **kw):
    return [unit(arm="scripted", seed=s, sink=s % 6, **kw) for s in range(1, n + 1)]


def deny_all(o):
    return Actions(deny=o.attempts.clone())


def test_the_platform_satisfies_the_core_contract_and_the_public_surface():
    p = mock.MockPlatform()
    assert isinstance(p, core.Platform) and p.name == "mock"
    assert mock.TOPOLOGIES == ("star_hub", "star_leaf", "chain", "small_world", "barabasi_albert")
    assert "insider_pi" in mock.ATTACKS and "none" in mock.ATTACKS


def test_run_batch_appends_one_row_per_unit_marks_it_done_and_returns_the_rows_in_input_order():
    store, units = FakeStore(), units_of(5, attack="infection")
    rows = platform().run_batch(units, spec(), make_settings(), store, core.NullTracker())
    assert [r["unit_id"] for r in rows] == [u["id"] for u in units]
    assert all(store.is_done(u["id"]) and len(store.rows[u["id"]]) == 1 for u in units)
    assert all(r["n_rounds"] == 15 for r in rows)                                          # 3.0 x diameter 5


def test_units_already_done_are_skipped_and_a_partial_unit_is_not_written_twice():
    store, units = FakeStore(), units_of(4)
    store.mark_done(units[0]["id"])
    store.append_rows(units[1]["id"], [{"unit_id": units[1]["id"], "kept": True}])           # crashed before mark_done
    seen = []
    p = platform(lambda t, s: (seen.append(len(t.unit_ids)), fake_rows(t, s))[1])
    rows = p.run_batch(units, spec(), make_settings(), store, core.NullTracker())
    assert seen == [3] and [r["unit_id"] for r in rows] == [u["id"] for u in units[1:]]
    assert store.rows[units[1]["id"]] == [{"unit_id": units[1]["id"], "kept": True}]         # not duplicated
    assert all(store.is_done(u["id"]) for u in units) and units[0]["id"] not in store.rows
    assert platform().run_batch(units, spec(), make_settings(), store, core.NullTracker()) == []


def test_rows_are_checkpointed_chunk_by_chunk_so_a_crash_keeps_the_finished_chunks():
    store, units = FakeStore(), units_of(5)
    built = []

    def factory():
        built.append(1)
        if len(built) == 3:
            raise RuntimeError("simulated crash")
        return Scripted()
    with pytest.raises(RuntimeError, match="crash"):
        platform().run_batch(units, spec(factory), make_settings(mock__batch_size=2), store, core.NullTracker())
    assert [u["id"] for u in units if store.is_done(u["id"])] == [u["id"] for u in units[:4]]
    assert not store.is_done(units[4]["id"])
    platform().run_batch(units, spec(), make_settings(mock__batch_size=2), store, core.NullTracker())
    assert all(store.is_done(u["id"]) for u in units) and all(len(store.rows[u["id"]]) == 1 for u in units)


def test_each_episode_runs_for_its_own_horizon_and_the_defender_sees_that_length():
    rec, ts = Recorder(), []

    class Probe(Scripted):
        def reset(self, ctx):
            super().reset(ctx)
            ts.append((ctx.T, ctx.B))
    units = [unit(arm="scripted", topology="small_world", agents=20, seed=s, sink=s % 20) for s in range(1, 41)]
    s = make_settings(runtime__ws_rewire_p=0.5)
    rows = platform(rec).run_batch(units, spec(Probe), s, FakeStore(), core.NullTracker())
    diam = graphs.diameters(W.make_graphs(units, s, "small_world", 20))
    want = (3 * diam).tolist()
    assert len(set(want)) > 1 and [r["n_rounds"] for r in rows] == want
    assert sorted(t for t, _ in ts) == sorted(set(want)) and sum(b for _, b in ts) == 40
    assert all(tr.correct.shape[1] == tr.infected.shape[1] == tr.spend_audit.shape[1] for tr in rec.traces)


def test_results_do_not_depend_on_the_batch_size_or_the_chunking():
    units = [unit(arm="scripted", topology="small_world", attack="infection", agents=8, seed=s, sink=s % 8,
                  audit=0.2, intervention=0.2) for s in range(1, 25)]
    s = make_settings()
    arm = core.ArmSpec("scripted", lambda: Fn(lambda o: Actions(deny=o.attempts & (o.monitor > 0.6)[:, :, None])))
    whole = mock.MockPlatform().run_batch(units, arm, s, FakeStore(), core.NullTracker())
    singles = [mock.MockPlatform().run_batch([u], arm, s, FakeStore(), core.NullTracker())[0] for u in units]
    chunked = mock.MockPlatform().run_batch(units, arm, {**s, "mock": {**s["mock"], "batch_size": 5}}, FakeStore(),
                                            core.NullTracker())
    dump = lambda rows: [json.dumps(r, sort_keys=True) for r in rows]            # noqa: E731  (NaN-safe comparison)
    assert dump(whole) == dump(singles) == dump(chunked)
    assert any(r["spend_intervention_total"] > 0 for r in whole)


def test_a_large_batch_gives_each_unit_the_row_it_has_alone():
    units = [unit(arm="scripted", topology="barabasi_albert", attack="insider_pi", agents=9, seed=s, sink=s % 9)
             for s in range(1, 257)]
    s = make_settings()
    arm = core.ArmSpec("scripted", Scripted)
    big = mock.MockPlatform().run_batch(units, arm, s, FakeStore(), core.NullTracker())
    for k in (0, 17, 255):
        one = mock.MockPlatform().run_batch([units[k]], arm, s, FakeStore(), core.NullTracker())[0]
        assert json.dumps(one, sort_keys=True) == json.dumps(big[k], sort_keys=True)


def test_arm_independent_draws_are_shared_across_arms():
    s = make_settings()
    units = [unit(arm="X", topology="small_world", attack="insider_ta", agents=10, seed=sd, sink=sd % 10) for sd in range(1, 9)]
    rec_a, rec_b = Recorder(), Recorder()
    run = lambda rec, arm_name, factory: platform(rec).run_batch(                  # noqa: E731
        [{**u, "arm": arm_name, "id": u["id"] + arm_name} for u in units], spec(factory, name=arm_name), s, FakeStore(),
        core.NullTracker())
    run(rec_a, "A", Scripted)
    run(rec_b, "B", lambda: Fn(deny_all))
    for ta, tb in zip(rec_a.traces, rec_b.traces, strict=True):
        assert torch.equal(ta.attackers, tb.attackers) and torch.equal(ta.sink, tb.sink)
        assert torch.equal(ta.infected[:, 0], tb.infected[:, 0]) and torch.equal(ta.correct[:, 0], tb.correct[:, 0])
        assert torch.equal(ta.attempts, tb.attempts)
        assert not torch.equal(ta.delivered, tb.delivered)


def test_the_defender_gets_a_context_built_from_the_unit_and_the_settings():
    got = []

    class Peek(Scripted):
        def reset(self, ctx):
            super().reset(ctx)
            got.append(ctx)
    units = [unit(arm="scripted", seed=s, sink=s, audit=0.1, intervention="uncapped", cost_condition="robustness",
                  thresholds={"judge": 1.5}) for s in range(1, 4)]
    platform().run_batch(units, spec(Peek), make_settings(), FakeStore(), core.NullTracker())
    ctx = got[0]
    assert (ctx.B, ctx.N, ctx.arm) == (3, 6, "scripted") and ctx.unit_ids == tuple(u["id"] for u in units)
    assert ctx.audit_cap.tolist() == pytest.approx([0.1 * 10 * ctx.T] * 3) and torch.isinf(ctx.intervention_cap).all()
    assert ctx.costs["sanitize"] == 2 and ctx.costs["reinstatement"] == 5 and ctx.thresholds == {"judge": 1.5}
    assert ctx.sink.tolist() == [1, 2, 3] and ctx.adj.dtype == torch.bool and ctx.adj.shape == (3, 6, 6)
    assert torch.equal(ctx.arm_stream("x", 0).get_state(), core.stream("x", "scripted", units[0]["id"]).get_state())
    assert not torch.equal(ctx.arm_stream("x", 0).get_state(), ctx.arm_stream("x", 1).get_state())


def test_main_costs_are_the_default_cost_condition():
    got = []

    class Peek(Scripted):
        def reset(self, ctx):
            got.append(ctx.costs)
    platform().run_batch(units_of(2), spec(Peek), make_settings(), FakeStore(), core.NullTracker())
    assert set(got[0].values()) == {1.0}


def test_units_of_one_batch_must_agree_on_their_thresholds():
    units = units_of(2)
    units[0]["thresholds"], units[1]["thresholds"] = {"judge": 1.0}, {"judge": 2.0}
    with pytest.raises(ValueError, match="thresholds"):
        platform().run_batch(units, spec(), make_settings(), FakeStore(), core.NullTracker())


def test_refusals_name_what_is_wrong():
    p, s, store, t = platform(), make_settings(), FakeStore(), core.NullTracker()
    with pytest.raises(ValueError, match="empty"):
        p.run_batch([], spec(), s, store, t)
    with pytest.raises(ValueError, match="arm"):
        p.run_batch(units_of(2), spec(name="other"), s, store, t)
    with pytest.raises(ValueError, match="platform"):
        p.run_batch(units_of(2), spec(platforms=("llm",)), s, store, t)
    bad = units_of(2)
    del bad[0]["id"]
    with pytest.raises(ValueError, match="id"):
        p.run_batch(bad, spec(), s, store, t)
    with pytest.raises(ValueError, match="audit"):
        p.run_batch(units_of(2, audit=1.5), spec(), s, store, t)
    with pytest.raises(ValueError, match="batch_size"):
        p.run_batch(units_of(2), spec(), make_settings(mock__batch_size=0), store, t)
    s = make_settings()
    del s["runtime"]["horizon_constant"]
    with pytest.raises(core.Unset) as e:
        p.run_batch(units_of(2), spec(), s, store, t)
    assert e.value.args[0] == "runtime.horizon_constant"
    assert not store.rows


def test_attack_traces_carry_the_clean_accuracy_and_the_flag_matrix_for_the_metrics():
    rec = Recorder()
    units = units_of(3, attack="insider_pi", topology="star_hub")
    platform(rec).run_batch(units, spec(), make_settings(runtime__eps_correct=8.0), FakeStore(), core.NullTracker())
    m = rec.traces[0].meta
    assert m["clean_accuracy"].shape == (3,) and m["clean_accuracy"].dtype == torch.float64
    assert ((0 <= m["clean_accuracy"]) & (m["clean_accuracy"] <= 1)).all() and m["clean_accuracy"].mean() > 0.5
    assert m["flagged"].shape == rec.traces[0].infected.shape and m["flagged"].dtype == torch.bool
    rec = Recorder()
    platform(rec).run_batch(units_of(2), spec(), make_settings(), FakeStore(), core.NullTracker())
    assert rec.traces[0].attack_free and "clean_accuracy" not in rec.traces[0].meta


def test_the_clean_reference_is_the_same_world_without_the_attack():
    rec = Recorder()
    units = [unit(arm="scripted", seed=s, sink=0, attack="infection", topology="chain", agents=6) for s in range(1, 6)]
    s = make_settings(runtime__eps_correct=1e9)                      # everything is correct: accuracy 1 in the clean run
    platform(rec).run_batch(units, spec(), s, FakeStore(), core.NullTracker())
    assert rec.traces[0].meta["clean_accuracy"].tolist() == [1.0] * 5


def test_the_real_metrics_accept_the_trace_of_every_family_and_budget_cell():
    s = make_settings()
    for family in mock.ATTACKS:
        for audit, interv in (("uncapped", "uncapped"), (0.1, 0.05)):
            units = [unit(arm="scripted", seed=sd, sink=sd % 8, attack=family, topology="barabasi_albert", agents=8,
                          audit=audit, intervention=interv) for sd in range(1, 5)]
            arm = core.ArmSpec("scripted", lambda: Fn(lambda o: Actions(deny=o.attempts & (o.monitor > 0.5)[:, :, None])))
            rows = mock.MockPlatform().run_batch(units, arm, s, FakeStore(), core.NullTracker())
            assert len(rows) == 4 and set(metrics.COLUMNS) <= set(rows[0])
            assert rows[0]["attack"] == family and rows[0]["attack_free"] == (family == "none")


# ------------------------------------------------------------------------------------------------- calibration
class Reporter(Scripted):
    """Reports statistics: a uniform draw per delivery from the arm's own stream, an agent-level one (the monitor) and one
    that is NaN for half of the deliveries."""

    def __init__(self):
        super().__init__()
        self.name = "scripted"

    def decide(self, obs):
        B, N = obs.active.shape
        g = self.ctx.arm_stream(f"stat|{obs.t}", 0)
        edge = core.uniforms(g, (B, N, N))
        stat = {"edge": edge, "agent": obs.monitor.clone()}
        stat["partial"] = torch.where((edge > 0.5) & obs.attempts, edge, torch.full_like(edge, math.nan))
        return Actions(statistics=stat)


def calibration_units(n=20, **kw):
    return [unit(arm="scripted", seed=9000 + s, sink=None, attack="none", topology="small_world", agents=10, **kw)
            for s in range(1, n + 1)]


def test_calibration_returns_the_one_minus_alpha_quantile_of_each_statistic_over_the_live_deliveries():
    """An edge statistic is pooled over the live deliveries; an agent statistic stands for the agent's outgoing live
    deliveries (the deny rate of the spec is a rate over deliveries), so it enters the pool once per delivery."""
    s = make_settings()
    seen = []

    class Collect(Reporter):
        def decide(self, obs):
            a = super().decide(obs)
            seen.append((a.statistics["edge"][obs.attempts].clone(), a.statistics["agent"].clone(), obs.attempts.clone()))
            return a
    store = FakeStore()
    th = platform().calibrate(calibration_units(), spec(Collect), s, store)
    edge = torch.cat([e for e, _, _ in seen]).sort().values
    n = edge.numel()
    k = math.ceil(round(n * 0.98, 9))
    assert th["edge"] == edge[k - 1].item() and set(th) == {"edge", "agent", "partial"}
    assert 0.02 - 1.0 / n <= (edge > th["edge"]).double().mean().item() <= 0.02
    agent = torch.cat([a.flatten().repeat_interleave(att.sum(2).flatten()) for _, a, att in seen]).sort().values
    assert th["agent"] == agent[math.ceil(round(agent.numel() * 0.98, 9)) - 1].item()
    # an agent score is repeated once per outgoing delivery (at most N - 1 = 9), so ties move the rate in blocks of that size
    assert 0.02 - 10.0 / agent.numel() <= (agent > th["agent"]).double().mean().item() <= 0.02
    assert all(isinstance(v, float) for v in th.values())


def test_an_agent_statistic_is_weighted_by_the_deliveries_the_agent_sends():
    """On a star of 100 agents the hub sends half of the deliveries but is 1% of the agents: the deny rate of the spec is a
    rate over deliveries, so a hub score above every leaf score puts the threshold at the hub's score, not below it."""
    class Hub(Scripted):
        def decide(self, obs):
            B, N = obs.active.shape
            score = torch.arange(N, dtype=torch.float64) / N
            score[0] = 9.0
            return Actions(statistics={"agent": score.expand(B, N).clone()})
    units = [unit(arm="scripted", seed=9000 + s, sink=None, topology="star_hub", agents=100) for s in range(1, 4)]
    assert platform().calibrate(units, spec(Hub), make_settings(), FakeStore()) == {"agent": 9.0}


def test_calibration_ignores_unchecked_deliveries_and_runs_the_arm_without_thresholds():
    got = []

    class Probe(Reporter):
        def reset(self, ctx):
            super().reset(ctx)
            got.append((dict(ctx.thresholds), ctx.audit_cap.isinf().all().item(), ctx.intervention_cap.isinf().all().item()))
    units = calibration_units(6)
    for u in units:
        u["thresholds"] = {}
    th = platform().calibrate(units, spec(Probe), make_settings(), FakeStore())
    assert got and all(g == ({}, True, True) for g in got)
    assert 0.5 < th["partial"] < 1                                                   # NaN entries never enter the pool


def test_calibration_writes_one_row_per_unit_and_does_not_repeat_finished_units():
    store, units = FakeStore(), calibration_units(4)
    store.mark_done(units[0]["id"])
    platform().calibrate(units, spec(Reporter), make_settings(), store)
    assert all(store.is_done(u["id"]) for u in units) and units[0]["id"] not in store.rows
    assert all(len(store.rows[u["id"]]) == 1 for u in units[1:])
    platform().calibrate(units, spec(Reporter), make_settings(), store)              # a rerun recomputes and appends nothing
    assert all(len(store.rows[u["id"]]) == 1 for u in units[1:])


def test_calibration_is_attack_free_uncapped_and_needs_statistics():
    p, s = platform(), make_settings()
    with pytest.raises(ValueError, match="attack-free"):
        p.calibrate([unit(arm="scripted", attack="infection", sink=None)], spec(Reporter), s, FakeStore())
    with pytest.raises(ValueError, match="uncapped"):
        p.calibrate(calibration_units(2, audit=0.1), spec(Reporter), s, FakeStore())
    with pytest.raises(ValueError, match="statistics"):
        p.calibrate(calibration_units(2), spec(Scripted), s, FakeStore())
    s = make_settings()
    del s["budgets"]["alpha"]
    with pytest.raises(core.Unset) as e:
        p.calibrate(calibration_units(2), spec(Reporter), s, FakeStore())
    assert e.value.args[0] == "budgets.alpha"


def test_calibration_pools_all_units_and_does_not_depend_on_chunking():
    units = calibration_units(12)
    a = platform().calibrate(units, spec(Reporter), make_settings(), FakeStore())
    b = platform().calibrate(units, spec(Reporter), make_settings(mock__batch_size=5), FakeStore())
    assert set(a) == set(b)
    assert a["agent"] == b["agent"]                      # a keyed monitor statistic: identical whatever the chunking


# --------------------------------------------------------------------------------------------------- speed
def test_a_batch_of_512_episodes_at_12_agents_runs_in_well_under_a_second():
    units = [unit(arm="scripted", topology="small_world", attack="infection", agents=12, seed=sd, sink=sd % 12,
                  audit=0.2, intervention=0.2) for sd in range(1, 513)]
    s = make_settings(runtime__horizon_constant=1.0)                    # T = diameter (2..4): the batch of the frontier
    arm = core.ArmSpec("scripted", lambda: Fn(lambda o: Actions(deny=o.attempts & (o.monitor > 0.6)[:, :, None])))
    p = mock.MockPlatform()
    p.run_batch(units[:8], arm, s, FakeStore(), core.NullTracker())     # warm-up
    times = []
    for _ in range(3):                                                  # best of 3: a busy machine must not fail the test
        t0 = time.perf_counter()
        rows = p.run_batch(units, arm, s, FakeStore(), core.NullTracker())
        times.append(time.perf_counter() - t0)
    dt = min(times)
    assert len(rows) == 512 and dt < 1.0, dt


def test_one_hundred_agents_work():
    units = [unit(arm="scripted", topology="barabasi_albert", attack="infection", agents=100, seed=sd, sink=sd % 100,
                  audit=0.1, intervention=0.1) for sd in range(1, 9)]
    s = make_settings(runtime__horizon_constant=2.0, runtime__ba_m=2)
    arm = core.ArmSpec("scripted", lambda: Fn(lambda o: Actions(deny=o.attempts & (o.monitor > 0.6)[:, :, None])))
    rows = mock.MockPlatform().run_batch(units, arm, s, FakeStore(), core.NullTracker())
    assert len(rows) == 8 and all(r["n_agents"] == 100 for r in rows) and all(r["n_rounds"] >= 2 for r in rows)


# ------------------------------------------------------------------------------------------ integration requests
def test_the_context_names_the_platform_and_stacks_a_timing_schedule_only_when_every_unit_has_one():
    got = []

    class Peek(Scripted):
        def reset(self, ctx):
            super().reset(ctx)
            got.append(ctx)
    units = units_of(2)                                                    # chain of 6: T = 15
    platform().run_batch(units, spec(Peek), make_settings(), FakeStore(), core.NullTracker())
    assert got[0].platform == "mock" and got[0].timing_schedule is None
    for k, u in enumerate(units):
        u["timing_schedule"] = [t % (k + 2) == 0 for t in range(15)]
    platform().run_batch(units, spec(Peek), make_settings(), FakeStore(), core.NullTracker())
    ts = got[1].timing_schedule
    assert ts.dtype == torch.bool and tuple(ts.shape) == (2, 15) and ts[1].tolist() == units[1]["timing_schedule"]
    del units[0]["timing_schedule"]
    platform().run_batch(units, spec(Peek), make_settings(), FakeStore(), core.NullTracker())
    assert got[2].timing_schedule is None


def test_nan_statistics_do_not_move_the_calibrated_quantile():
    class Nan(Scripted):
        def decide(self, obs):
            B, N = obs.active.shape
            v = torch.full((B, N, N), math.nan, dtype=torch.float64)
            v[:, 0] = torch.linspace(0, 1, N, dtype=torch.float64)
            return Actions(statistics={"s": v})
    th = platform().calibrate(calibration_units(6), spec(Nan), make_settings(), FakeStore())
    assert 0.0 <= th["s"] <= 1.0 and not math.isnan(th["s"])


def test_an_over_request_for_audits_grants_a_keyed_random_subset_not_the_lowest_agents():
    scores = []

    class Ask(Scripted):
        def decide(self, obs):
            if obs.t == 1:
                s = obs.audit(torch.ones_like(obs.active))
                scores.append(s)
            return Actions()
    units = [unit(arm="scripted", topology="chain", agents=10, seed=sd, sink=0, audit=0.0165) for sd in range(1, 41)]
    s = make_settings(runtime__horizon_constant=1.0)
    platform().run_batch(units, spec(Ask), s, FakeStore(), core.NullTracker())
    got = ~torch.isnan(scores[0])
    assert (got.sum(1) == got.sum(1)[0]).all() and 0 < got.sum(1)[0] < 10
    assert got[:, 5:].any() and len({tuple(r.tolist()) for r in got}) > 5          # not always the first agents
    scores.clear()
    platform().run_batch(units[:1], spec(Ask), s, FakeStore(), core.NullTracker())
    assert torch.equal(~torch.isnan(scores[0])[0], got[0])


def test_tracker_rows_of_one_table_all_carry_the_same_keys():
    logged = []

    class T(core.NullTracker):
        def log_rows(self, table, rows):
            logged.append((table, rows))
    rows_fn = lambda tr, s: [{"unit_id": u["id"], **({"extra": 1} if k == 0 else {})} for k, u in enumerate(tr.unit_ids)]  # noqa: E731
    platform(rows_fn).run_batch(units_of(3), spec(), make_settings(), FakeStore(), T())
    keys = {tuple(r) for _, rows in logged for r in rows}
    assert len(keys) == 1 and logged[0][1][1]["extra"] is None


def test_the_deny_cost_is_required_per_cost_condition_and_named_when_missing():
    s = make_settings()
    del s["costs"]["robustness"]["deny"]
    platform().run_batch(units_of(2), spec(), s, FakeStore(), core.NullTracker())          # main works
    with pytest.raises(core.Unset) as e:
        platform().run_batch(units_of(2, cost_condition="robustness"), spec(), s, FakeStore(), core.NullTracker())
    assert e.value.args[0] == "costs.robustness.deny"


def test_a_real_store_gets_one_batch_file_per_chunk_and_a_rerun_skips_every_finished_unit(tmp_path):
    units = units_of(6)
    store = core.CheckpointStore(tmp_path)
    platform().run_batch(units, spec(), make_settings(mock__batch_size=3), store, core.NullTracker())
    files = sorted(p.name for p in (tmp_path / "units").glob("*.jsonl"))
    assert len(files) == 2 and all(f.startswith("batch-") for f in files)
    lines = [json.loads(line) for f in files for line in (tmp_path / "units" / f).read_text().splitlines()]
    assert sorted(r["unit_id"] for r in lines) == sorted(u["id"] for u in units)
    again = core.CheckpointStore(tmp_path)
    assert all(again.is_done(u["id"]) for u in units)
    assert platform().run_batch(units, spec(), make_settings(mock__batch_size=3), again, core.NullTracker()) == []
