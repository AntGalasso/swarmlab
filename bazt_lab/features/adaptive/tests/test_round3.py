"""Third fix round (R3-M1): the fingerprint holds the settings that change a best response on the mock platform and
nothing the runner's block identity ignores (compute, sync, tracking, keys only the LLM platform reads)."""
import copy

import pytest

from bazt_lab.core import public as core
from bazt_lab.features.adaptive.private import run as rn
from bazt_lab.features.adaptive.tests.fakes import FakePlatform, arm_of, make_settings, unit, units_of

N = 6
# What a real launch carries besides the search grids (test values, not experiment choices).
LAUNCH = {
    "mock.batch_size": 40, "mock.model.beta": 0.3, "costs.main.sanitize": 1, "bazt_upstream.verdict_probability": 0.98,
    "compute.device": "cpu", "compute.compile": False, "compute.deterministic": True, "compute.dtype": "float64",
    "compute.workers": 1, "llm.allow_live": False, "gnn.embedding.batch_size": 32,
    "conventional.c.judge_score_bins.mock": [[0.0, 1.0], [9.0, 2.0]],
    "conventional.c.judge_score_bins.llm": [[-1.0, 1.0], [0.0, 2.0]],
}
# --workers, --device, --compile, the other compute keys and what only the LLM platform reads.
RUN_TIME = [{"compute.workers": 2}, {"compute.device": "cuda"}, {"compute.compile": True},
            {"compute.deterministic": False}, {"compute.dtype": "float32"},
            {"conventional.c.judge_score_bins.llm": [[-2.0, 1.0]]}, {"llm.allow_live": True},
            {"gnn.embedding.batch_size": 7}]
# Values the mock episodes (and so the best responses) depend on.
MATTERS = [{"mock.model.beta": 0.4}, {"costs.main.sanitize": 2}, {"conventional.c.judge_score_bins.mock": [[1.0, 1.0]]},
           {"bazt_upstream.verdict_probability": 0.9}]


def asr(arm, u):
    return 0.9 if u["attack_params"] == {"entry_ranks": [2], "start_frac": 0.2} else 0.3


def cells():
    return units_of(unit(agents=N), (1, 2), range(N))


def launch(tmp_path, over=None):
    platform = FakePlatform(asr)
    settings = make_settings({**LAUNCH, **(over or {})})
    return rn.run_search(cells(), arm_of, platform, settings, tmp_path), platform, settings


@pytest.mark.parametrize("over", RUN_TIME, ids=lambda o: next(iter(o)))
def test_a_relaunch_with_other_run_time_or_llm_only_settings_reuses_the_table(tmp_path, over):
    first, _, _ = launch(tmp_path)
    again, platform, settings = launch(tmp_path, over)
    assert platform.calls == []                                     # nothing searched again
    loaded = rn.load_best_responses(tmp_path, settings)
    for u in cells():
        assert again.choice(u) == loaded.choice(u) == first.choice(u)


@pytest.mark.parametrize("over", MATTERS, ids=lambda o: next(iter(o)))
def test_a_relaunch_with_a_setting_that_changes_episodes_is_refused(tmp_path, over):
    launch(tmp_path)
    with pytest.raises(ValueError, match="other settings"):
        launch(tmp_path, over)
    with pytest.raises(ValueError, match="other settings"):
        rn.load_best_responses(tmp_path, make_settings({**LAUNCH, **over}))


def _leaves(tree, prefix=""):
    for key, value in tree.items():
        if isinstance(value, dict):
            yield from _leaves(value, f"{prefix}{key}.")
        else:
            yield f"{prefix}{key}", value


def _changed(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if isinstance(value, list):
        return [*value, 999]
    return f"{value}_other"


@pytest.mark.parametrize("section", ["compute", "sync", "tracking"])
def test_no_key_of_the_real_compute_sync_or_tracking_section_enters_the_fingerprint(section):
    real = core.load_settings()
    base = rn.fingerprint(real)
    leaves = list(_leaves(real[section], f"{section}."))
    assert leaves
    for key, value in leaves:
        changed = copy.deepcopy(real)
        core.put(changed, key, _changed(value))
        assert rn.fingerprint(changed) == base, key


@pytest.mark.parametrize("key", ["conventional.c.judge_score_bins.llm", "gnn.embedding.batch_size", "compute.workers"])
def test_an_ignored_key_set_alone_where_the_profile_has_no_table_yet_leaves_the_fingerprint_as_it_was(key):
    """The runner's identity is made of the set leaves, so an ignored key set inside a table the profile does not have
    yet (judge_score_bins.llm before the mock bins are set) changes nothing there; the fingerprint must agree, not keep
    the empty table the ignored key leaves behind (found by tests/test_adaptive_fingerprint_identity.py)."""
    real = core.load_settings()
    changed = copy.deepcopy(real)
    core.put(changed, key, [[-1.0, 1.0]])
    assert rn.fingerprint(changed) == rn.fingerprint(real)
