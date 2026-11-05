"""MatchedTiming: an arm that may act only at the rounds of a schedule (spec 2.7, D32)."""
import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.references.private.allocation import OracleAlloc, RandomAlloc
from bazt_lab.features.references.private.legacy import RandomCapped, RiskAware, ScoreOnly
from bazt_lab.features.references.private.timing import MatchedTiming, schedule_from_rounds
from bazt_lab.features.references.tests.synth import FakePlatform, make_ctx, star
from bazt_lab.features.references.tests.test_legacy import SETTINGS, const_monitor, isolated_at

SCHEDULE = [[2, 4]]


def with_schedule(ctx, rounds):
    """What the platform will do once core's EpisodeContext carries the schedule of the unit."""
    object.__setattr__(ctx, "timing_schedule", schedule_from_rounds(rounds, ctx.T))
    return ctx


def test_schedule_from_rounds_is_a_bool_mask_over_one_based_rounds():
    m = schedule_from_rounds([[1, 3], [], [4]], 4)
    assert m.dtype == torch.bool and m.tolist() == [[True, False, True, False], [False] * 4, [False, False, False, True]]
    with pytest.raises(ValueError, match="round"):
        schedule_from_rounds([[0]], 4)
    with pytest.raises(ValueError, match="round"):
        schedule_from_rounds([[5]], 4)


def test_the_arm_acts_at_scheduled_rounds_only():
    fp = FakePlatform(with_schedule(make_ctx(star(6), 6, settings=SETTINGS, arm="m"), SCHEDULE))
    fp.run(MatchedTiming(RandomCapped(own_timing=False)))
    assert list(isolated_at(fp)) == [2, 4]                               # one random agent at each scheduled round


def test_the_inner_arm_keeps_observing_between_scheduled_rounds():
    # agent 1 crosses the threshold at t=1, 2, 3: score_only (2 consecutive crossings) is ripe by t=2, but may only
    # act at t=3 when the schedule says so; acting at t=1 alone would not have been possible
    ctx = with_schedule(make_ctx(star(4), 3, settings=SETTINGS, thresholds={"monitor": 0.5}), [[3]])
    fp = FakePlatform(ctx, monitor=const_monitor([0.1, 0.9, 0.1, 0.1]))
    fp.run(MatchedTiming(ScoreOnly()))
    assert isolated_at(fp) == {3: [1]}
    early = with_schedule(make_ctx(star(4), 3, settings=SETTINGS, thresholds={"monitor": 0.5}), [[1]])
    fp = FakePlatform(early, monitor=const_monitor([0.1, 0.9, 0.1, 0.1]))
    fp.run(MatchedTiming(ScoreOnly()))
    assert not isolated_at(fp)                                           # not ripe at t=1, and never scheduled again


def test_statistics_are_reported_at_every_round_so_calibration_is_unaffected():
    fp = FakePlatform(with_schedule(make_ctx(star(4), 4, settings=SETTINGS), SCHEDULE), monitor=const_monitor(
        [0.9, 0.5, 0.5, 0.5]))
    fp.run(MatchedTiming(RiskAware()))
    assert all("risk" in r.actions.statistics for r in fp.log)


def test_audits_outside_the_schedule_are_not_performed_and_not_paid():
    # cap 4 over T = 8: cumulative allowance floor(4 t / 8); unscheduled rounds carry their allowance forward
    ctx = with_schedule(make_ctx(star(6), 8, audit_cap=4, arm="m"), [[4, 8]])
    fp = FakePlatform(ctx)
    fp.run(MatchedTiming(RandomAlloc()))
    assert [r.audited.sum().item() for r in fp.log] == [0, 0, 0, 2, 0, 0, 0, 2]
    assert fp.rem_audit.item() == 0.0


def test_the_platforms_observation_is_not_modified():
    ctx = with_schedule(make_ctx(star(6), 2, audit_cap=4), [[2]])
    fp = FakePlatform(ctx)
    d = MatchedTiming(RandomAlloc())
    d.reset(ctx)
    r = fp.step(d, 1)
    assert not r.audited.any()                                           # round 1 is not scheduled
    direct = r.obs.audit(torch.tensor([[True, False, False, False, False, False]]))   # the platform's own callable
    assert not direct[0, 0].isnan() and fp.rem_audit.item() == 3.0
    assert fp.step(d, 2).audited.any()


def test_a_missing_schedule_is_refused_not_replaced_by_acting_everywhere():
    with pytest.raises(ValueError, match="timing_schedule"):
        MatchedTiming(RandomCapped(own_timing=False)).reset(make_ctx(star(6), 4, settings=SETTINGS))


def test_a_threshold_driven_arm_without_schedule_or_thresholds_is_a_calibration_run_and_only_collects():
    # calibration runs the uncapped arm attack-free: no BAZT schedule exists and no threshold is set yet
    fp = FakePlatform(make_ctx(star(4), 3, settings=SETTINGS), monitor=const_monitor([0.9, 0.5, 0.5, 0.5]))
    fp.run(MatchedTiming(RiskAware()))
    assert all(not r.actions.isolation.any() and "risk" in r.actions.statistics for r in fp.log)
    with pytest.raises(ValueError, match="timing_schedule"):             # but with thresholds a schedule is required
        MatchedTiming(RiskAware()).reset(make_ctx(star(4), 3, settings=SETTINGS, thresholds={"risk": 0.1}))


def test_a_schedule_given_to_the_constructor_is_used_when_the_context_has_none():
    ctx = make_ctx(star(6), 4, settings=SETTINGS)
    fp = FakePlatform(ctx)
    fp.run(MatchedTiming(RandomCapped(own_timing=False), schedule=schedule_from_rounds([[3]], 4)))
    assert list(isolated_at(fp)) == [3]


def test_a_schedule_of_the_wrong_shape_is_refused():
    ctx = make_ctx(star(6), 4, B=2, settings=SETTINGS)
    with pytest.raises(ValueError, match="shape"):
        MatchedTiming(RandomCapped(own_timing=False), schedule=schedule_from_rounds([[1]], 4)).reset(ctx)
    with pytest.raises(ValueError, match="shape"):
        MatchedTiming(RandomCapped(own_timing=False), schedule=schedule_from_rounds([[1], [1]], 5)).reset(ctx)


def test_each_episode_follows_its_own_schedule():
    ctx = with_schedule(make_ctx(star(6), 4, B=2, settings=SETTINGS, unit_ids=("a", "b"), arm="m"), [[1], [2, 3]])
    fp = FakePlatform(ctx)
    fp.run(MatchedTiming(RandomCapped(own_timing=False)))
    got = [[r.t for r in fp.log if r.actions.isolation[b].any()] for b in range(2)]
    assert got == [[1], [2, 3]]


def test_name_and_truth_requirement_follow_the_inner_arm():
    d = MatchedTiming(RandomCapped(own_timing=False))
    assert d.name == "random_capped_matched" and d.needs_truth is False
    assert MatchedTiming(OracleAlloc()).needs_truth is True


def test_state_includes_the_inner_state_and_resumes_exactly(tmp_path):
    mon = lambda t: torch.tensor([[0.1, 0.9, 0.1, 0.1]], dtype=torch.float64)  # noqa: E731

    def world():
        return with_schedule(make_ctx(star(4), 5, settings=SETTINGS, thresholds={"monitor": 0.5}), [[3, 5]])

    ref = FakePlatform(world(), monitor=mon)
    ref.run(MatchedTiming(ScoreOnly()))
    part = FakePlatform(world(), monitor=mon)
    d1 = MatchedTiming(ScoreOnly())
    d1.reset(part.ctx)
    for t in (1, 2):
        part.step(d1, t)
    store = core.CheckpointStore(tmp_path)
    store.save_state("u", d1.state_dict())
    assert store.load_state("u")["inner.alarm_run"].tolist() == [[0, 2, 0, 0]]
    d2 = MatchedTiming(ScoreOnly())
    d2.reset(part.ctx)
    d2.load_state_dict(store.load_state("u"))
    for t in (3, 4, 5):
        part.step(d2, t)
    for a, b in zip(ref.log, part.log, strict=True):
        assert torch.equal(a.actions.isolation, b.actions.isolation)
