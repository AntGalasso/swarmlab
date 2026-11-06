"""Drift guard: the adaptive feature's best-response table fingerprint against the runner's mock_adaptive block identity,
on the real experiments/profiles.toml.

adaptive/private/run.py lists by hand what its fingerprint ignores (RUN_TIME, LLM_ONLY, OTHER_BLOCKS, PER_CELL), mirroring
the lists of runner/private/identity.py. Its keys must be a subset of the runner's view: a key in the fingerprint that the
runner ignores makes a relaunch the runner accepts (another --workers or --device, for example) refuse the saved table
(round-3 R3-M1). These tests fail as soon as the two lists drift apart. They read private modules on purpose; production
code of the adaptive feature never imports the runner's private identity.
"""
import copy

import pytest

from bazt_lab.core import public as core
from bazt_lab.features.adaptive.private import run as adaptive_run
from bazt_lab.features.runner.private import identity

BLOCK = "mock_adaptive"
PROFILES = ("sprint", "research")
PROBE = ["drift probe"]


def real_settings(profile):
    return core.load_settings(core.SETTINGS_PATH, profile)


def changed(settings, key):
    probe = copy.deepcopy(settings)
    core.put(probe, key, PROBE)
    return probe


def fingerprint_keys(settings) -> list:
    """The settings keys the fingerprint depends on, found by changing each one (not by reading adaptive's lists)."""
    base = adaptive_run.fingerprint(settings)
    return [k for k in identity.flatten(settings) if adaptive_run.fingerprint(changed(settings, k)) != base]


def runner_ignored_prefixes() -> list:
    """What the runner's mock_adaptive view leaves out (identity.block_view for this block on the mock platform)."""
    other_blocks = [f"mock.{x}" for x in identity.SCOPED_MOCK if f"mock_{x}" != BLOCK]
    return [*identity.IGNORED, *identity.MOCK_ONLY_IGNORED, *other_blocks]


def probe_keys(settings, prefix) -> list:
    """Keys that stand for everything under `prefix`: the key itself when it is a set leaf or absent, and a new key under
    it when it is (or may be) a table, so an ignored section the profile does not set yet is covered too."""
    flat = identity.flatten(settings)
    if prefix in flat:
        return [prefix]
    is_table = any(k.startswith(prefix + ".") for k in flat)
    return [f"{prefix}.drift_probe"] if is_table else [prefix, f"{prefix}.drift_probe"]


@pytest.mark.parametrize("profile", PROFILES)
def test_every_key_of_the_adaptive_fingerprint_is_in_the_runners_mock_adaptive_identity(profile):
    s = real_settings(profile)
    view = identity.block_view(s, BLOCK, "mock")
    keys = fingerprint_keys(s)
    assert keys, "the fingerprint depends on no setting: the probe is vacuous"
    outside = [k for k in keys if k not in view]
    assert not outside, (f"the adaptive fingerprint depends on {outside}, which the runner's {BLOCK} identity ignores: "
                         f"a relaunch the runner accepts would refuse the saved best responses")


@pytest.mark.parametrize("profile", PROFILES)
def test_no_key_the_runners_view_ignores_changes_the_adaptive_fingerprint(profile):
    s = real_settings(profile)
    base = adaptive_run.fingerprint(s)
    drifted = []
    for prefix in runner_ignored_prefixes():
        for key in probe_keys(s, prefix):
            probe = changed(s, key)
            assert key not in identity.block_view(probe, BLOCK, "mock"), f"the runner does not ignore {key}"
            if adaptive_run.fingerprint(probe) != base:
                drifted.append(key)
    assert not drifted, (f"the runner's {BLOCK} identity ignores {drifted}, but the adaptive fingerprint depends on "
                         f"them (adaptive/private/run.py RUN_TIME, LLM_ONLY, OTHER_BLOCKS mirror the runner's lists)")


def test_the_guard_sees_the_round3_drift():
    """Not vacuous: with compute.* back in the fingerprint (the R3-M1 state), both checks fail."""
    s = real_settings("sprint")
    original = adaptive_run.IGNORED
    try:
        adaptive_run.IGNORED = tuple(k for k in original if k != "compute")
        view = identity.block_view(s, BLOCK, "mock")
        assert [k for k in fingerprint_keys(s) if k not in view and k.startswith("compute.")]
        assert adaptive_run.fingerprint(changed(s, "compute.drift_probe")) != adaptive_run.fingerprint(s)
    finally:
        adaptive_run.IGNORED = original
