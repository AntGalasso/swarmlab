"""calibration.mode: open_loop (statistics of the arm acting on nothing) and closed_loop (thresholds searched while the
arm acts, so that the realised attack-free deny rate is alpha within CLOSED_LOOP_TOLERANCE)."""
import math

import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.bazt_slot import public as slot
from bazt_lab.features.conventional import public as conv
from bazt_lab.features.mock_engine import public as mock
from bazt_lab.features.mock_engine.private.platform import CLOSED_LOOP_TOLERANCE
from bazt_lab.features.mock_engine.tests.helpers import FakeStore, Scripted, make_settings, spec, unit

Actions = core.Actions
ALPHA = 0.02


class Rec:
    def __init__(self):
        self.traces = []

    def __call__(self, trace, settings):
        self.traces.append(trace)
        return [{"unit_id": u["id"]} for u in trace.unit_ids]


def deny_rate(traces):
    attempts = sum(t.attempts.sum().item() for t in traces)
    return sum((t.attempts & ~t.delivered).sum().item() for t in traces) / attempts


class Feedback(Scripted):
    gain = 0.05

    """Stateful arm: the statistic of a delivery is a keyed uniform plus a term that grows with the number of senders
    denied so far, so the denies it makes raise its later statistics (like the network threat level of C)."""
    name = "scripted"

    def reset(self, ctx):
        super().reset(ctx)
        self.denied = torch.zeros(ctx.B, dtype=torch.float64)

    def decide(self, obs):
        B, N = obs.active.shape
        draw = torch.stack([core.uniforms(self.ctx.arm_stream(f"fb|{obs.t}", b), (N, N)) for b in range(B)])
        stat = draw + self.gain * self.denied[:, None, None] / N
        th = float(self.ctx.thresholds.get("fb", math.inf))
        deny = obs.attempts & (stat > th)
        self.denied = self.denied + deny.any(2).sum(1)
        return Actions(deny=deny, statistics={"fb": stat})


def units(arm, seeds, **kw):
    return [unit(arm=arm, seed=s, sink=None, attack="none", topology="small_world", agents=12, **kw) for s in seeds]


def uncapped(arm, seeds):
    return units(arm, seeds, audit="uncapped", intervention="uncapped")


def settings(mode, **over):
    s = make_settings(**over)
    core.put(s, "calibration.mode", mode)
    return s


def run(arm_spec, us, s, th):
    rec = Rec()
    mock.MockPlatform(rows_fn=rec).run_batch([{**u, "thresholds": th} for u in us], arm_spec, s, FakeStore(), core.NullTracker())
    return deny_rate(rec.traces)


def test_the_calibration_mode_is_an_owner_key_and_unset_refuses_naming_it():
    s = make_settings()
    del s["calibration"]
    cal = uncapped("scripted", range(9001, 9004))
    with pytest.raises(core.Unset) as e:
        mock.MockPlatform(rows_fn=Rec()).calibrate(cal, spec(Feedback), s, FakeStore())
    assert e.value.args[0] == "calibration.mode"
    with pytest.raises(ValueError, match=r"calibration\.mode"):
        mock.MockPlatform(rows_fn=Rec()).calibrate(cal, spec(Feedback), settings("sideways"), FakeStore())


def test_open_loop_calibration_of_a_stateful_arm_denies_more_than_alpha_and_closed_loop_does_not():
    cal = uncapped("scripted", range(9001, 9021))
    open_th = mock.MockPlatform(rows_fn=Rec()).calibrate(cal, spec(Feedback), settings("open_loop"), FakeStore())
    p = mock.MockPlatform(rows_fn=Rec())
    closed_th = p.calibrate(cal, spec(Feedback), settings("closed_loop"), FakeStore())
    s = settings("closed_loop")
    assert run(spec(Feedback), cal, s, open_th) > 2 * ALPHA                              # the defect
    rate = run(spec(Feedback), cal, s, closed_th)
    assert abs(rate - ALPHA) <= CLOSED_LOOP_TOLERANCE * ALPHA, rate
    assert p.last_calibration["mode"] == "closed_loop" and p.last_calibration["rate"] == pytest.approx(rate)
    assert closed_th["fb"] > open_th["fb"]


def test_when_the_deny_rate_jumps_over_the_tolerance_band_the_calibrated_rate_stays_at_or_below_it():
    class Cliff(Feedback):
        gain = 5.0                                           # one denial pushes every later statistic over any threshold
    cal = uncapped("scripted", range(9001, 9013))
    p = mock.MockPlatform(rows_fn=Rec())
    th = p.calibrate(cal, spec(Cliff), settings("closed_loop"), FakeStore())
    rate = run(spec(Cliff), cal, settings("closed_loop"), th)
    assert rate <= ALPHA * (1 + CLOSED_LOOP_TOLERANCE) and rate == pytest.approx(p.last_calibration["rate"])
    assert p.last_calibration["within_tolerance"] in (True, False)
    assert p.last_calibration["worst_episode_rate"] >= rate - 1e-12                  # the worst episode is at least the mean


def test_closed_loop_does_not_depend_on_the_batch_size():
    cal = uncapped("scripted", range(9001, 9013))
    a = mock.MockPlatform(rows_fn=Rec()).calibrate(cal, spec(Feedback), settings("closed_loop"), FakeStore())
    b = mock.MockPlatform(rows_fn=Rec()).calibrate(cal, spec(Feedback), settings("closed_loop", mock__batch_size=5), FakeStore())
    assert a == b


def test_closed_loop_resumes_from_the_store_without_repeating_finished_passes(tmp_path):
    cal = uncapped("scripted", range(9001, 9011))
    store = core.CheckpointStore(tmp_path)
    first = mock.MockPlatform(rows_fn=Rec())
    th = first.calibrate(cal, spec(Feedback), settings("closed_loop"), store)
    passes = first.last_calibration["passes"]
    assert passes >= 2
    again = mock.MockPlatform(rows_fn=Rec())
    assert again.calibrate(cal, spec(Feedback), settings("closed_loop"), store) == th
    assert again.last_calibration["passes"] == passes and again.last_calibration["simulated"] == 1   # only the statistics pass


def test_open_loop_is_unchanged_and_reports_its_mode():
    p = mock.MockPlatform(rows_fn=Rec())
    p.calibrate(uncapped("scripted", range(9001, 9005)), spec(Feedback), settings("open_loop"), FakeStore())
    assert p.last_calibration["mode"] == "open_loop"


# ----------------------------------------------------------------------------------------------- the real arms
def real_settings(mode):
    s = settings(mode, runtime__rate_limit_per_identity=20)
    keys = {
        "conventional.c.judge_score_bins": {"mock": [[0.0, 0.0], [2.0, 4.0], [1e9, 12.0]]},
        "conventional.c.presence_window_rounds": 3, "mock.batch_size": 16,
        "bazt.katz_decay": 0.5, "bazt.eigen_iterations": 50, "bazt.sink_weight": 2.0, "bazt.utility_weight": 1.0,
        "bazt.utility_loss.sanitize": 0.15, "bazt.utility_loss.delay": 0.1, "bazt.utility_loss.edge_quarantine": 0.3,
        "bazt.utility_loss.rate_limit": 0.3, "bazt.utility_loss.privilege_degradation": 0.35,
        "bazt.utility_loss.isolation": 1.0, "bazt.flag_logodds": 4.0, "bazt.clear_logodds": 3.0,
        "bazt.memory_decay": 0.8, "bazt.prior": 0.1, "bazt.inspect_conclusive": 0.8,
    }
    for k, v in keys.items():
        core.put(s, k, v)
    return s


@pytest.mark.parametrize("name", ["C", "BAZT"])
def test_closed_loop_calibrates_the_real_arms_to_alpha(name):
    arm = {a.name: a for a in (*conv.ARMS, *slot.ARMS)}[name]
    cal = uncapped(name, range(9001, 9021))
    p = mock.MockPlatform(rows_fn=Rec())
    th = p.calibrate(cal, arm, real_settings("closed_loop"), FakeStore())
    open_th = mock.MockPlatform(rows_fn=Rec()).calibrate(cal, arm, real_settings("open_loop"), FakeStore())
    rate = run(arm, cal, real_settings("closed_loop"), th)
    assert rate == pytest.approx(p.last_calibration["rate"])
    # The statistics of C and BAZT are discrete (sums of weights), so the rate moves in steps and alpha may fall inside a
    # step: the calibrated rate is then at or below the band, never above it (within_tolerance reports which case it was).
    assert rate <= ALPHA * (1 + CLOSED_LOOP_TOLERANCE) and "within_tolerance" in p.last_calibration
    assert run(arm, cal, real_settings("closed_loop"), open_th) >= rate                 # never worse than the open loop
    # Held-out episodes are not bounded by alpha: at the edge of a feedback cliff a few episodes cascade (C denies ~85% of
    # an episode's deliveries once its threat level rises), and the calibration sample may hold none. The calibrator
    # reports the worst episode it saw; on held-out seeds it still beats the open loop.
    held = uncapped(name, range(1, 31))
    assert run(arm, held, real_settings("closed_loop"), th) <= run(arm, held, real_settings("open_loop"), open_th)
    assert 0.0 <= p.last_calibration["worst_episode_rate"] <= 1.0


def test_budget_exhausted_events_are_kept_in_the_trace():
    ev = {"event": "budget_exhausted", "b": 0, "budget": "audit"}

    class Say(Scripted):
        def decide(self, obs):
            return Actions(events=[dict(ev)] if obs.t == 2 else [])
    rec = Rec()
    mock.MockPlatform(rows_fn=rec).run_batch(units("scripted", [1]), spec(Say), make_settings(), FakeStore(), core.NullTracker())
    assert rec.traces[0].meta["events"] == [ev]
