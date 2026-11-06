"""The feature's public surface: the arms the runner registers, and their flags."""
import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.references import public as refs
from bazt_lab.features.references.tests.synth import FakePlatform, make_ctx, star
from bazt_lab.features.references.tests.test_legacy import SETTINGS

BASES = ["fixed_early", "fixed_mid", "fixed_late", "score_only", "centrality_only", "random_capped", "risk_aware",
         "risk_aware_tiered"]
BY_NAME = {a.name: a for a in refs.ARMS}


def test_the_feature_exports_every_reference_arm_of_the_spec_once():
    names = [a.name for a in refs.ARMS]
    assert len(names) == len(set(names))
    assert set(names) == {"random_alloc", "oracle_alloc", "bra_greedy", "GNN_standin", *BASES,
                          *(f"{b}_matched" for b in BASES)}
    assert all(isinstance(a, core.ArmSpec) for a in refs.ARMS)


def test_only_the_oracle_needs_truth():
    assert [a.name for a in refs.ARMS if a.needs_truth] == ["oracle_alloc"]


def test_platform_scopes():
    for name in ("random_alloc", "oracle_alloc", "bra_greedy"):
        assert BY_NAME[name].platforms == ("mock", "llm")
    for name in ("GNN_standin", *BASES, *(f"{b}_matched" for b in BASES)):
        assert BY_NAME[name].platforms == ("mock",)


def test_calibrated_flags_follow_the_arms_that_act_on_a_threshold():
    calibrated = {a.name for a in refs.ARMS if a.calibrated}
    thresholded = {"random_alloc", "bra_greedy", "GNN_standin", "fixed_early", "fixed_mid", "fixed_late", "score_only",
                   "risk_aware", "risk_aware_tiered"}
    assert calibrated == thresholded | {f"{b}_matched" for b in thresholded if b in BASES}


def test_every_factory_builds_a_fresh_defender_with_the_arms_name():
    for spec in refs.ARMS:
        d1, d2 = spec.factory(), spec.factory()
        assert isinstance(d1, core.Defender) and d1 is not d2 and d1.name == spec.name
        assert bool(getattr(d1, "needs_truth", False)) == spec.needs_truth


@pytest.mark.parametrize("spec", refs.ARMS, ids=lambda a: a.name)
def test_every_arm_runs_a_short_episode_through_its_public_factory(spec):
    ctx = make_ctx(star(5), 4, B=2, settings=SETTINGS, thresholds={}, intervention_cap=20, audit_cap=8, arm=spec.name)
    object.__setattr__(ctx, "timing_schedule", refs.schedule_from_rounds([[1, 3], [2]], 4))
    mon = lambda t: torch.tensor([[0.1, 0.9, 0.2, 0.7, 0.3]] * 2, dtype=torch.float64)  # noqa: E731
    fp = FakePlatform(ctx, monitor=mon, infected=lambda t: torch.tensor([[False, True, False, False, False]] * 2))
    fp.run(spec.factory())                                             # the platform asserts no budget violation
    assert len(fp.log) == 4
