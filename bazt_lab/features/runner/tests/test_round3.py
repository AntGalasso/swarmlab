"""Blocks the LLM platform cannot run, the table headers of profiles.toml, the threshold identity of every calibrated
arm, and the LLM sizes of `run-all --dry-run`."""
import pytest

from bazt_lab.core import public as core
from bazt_lab.features.runner import public as rx
from bazt_lab.features.runner.private import blocks, identity, needs, runall, status
from bazt_lab.features.runner.tests import fakes
from bazt_lab.features.runner.tests.test_commands import LLM, MOCK, blocks_run, run, sets
from bazt_lab.features.runner.tests.test_launch import LLM as LLM_LAUNCH
from bazt_lab.features.runner.tests.test_launch import MOCK_B, calibrate_then, launch, past_kickoff

ADAPTIVE_CHECK = ["llm.adaptive_check.arms=['B']", "llm.adaptive_check.families=['insider_pi']",
                  "llm.adaptive_check.condition='gray_box'", "llm.adaptive_check.tasks=['mmlu']"]


class ConditionRefusing(fakes.FakePlatform):
    """Refuses a unit that carries an adaptive condition, as the LLM harness does (llm_harness CONTRACT.md)."""

    def run_batch(self, units, arm, settings, store, tracker):
        for u in units:
            if u.get("condition") is not None:
                raise ValueError(f"unit {u['id']} carries the adaptive condition {u['condition']!r}, which the LLM "
                                 f"harness does not implement")
        return super().run_batch(units, arm, settings, store, tracker)


def handed_conditions(platform):
    units = [u for _, us in platform.batches for u in us] + [u for _, us, _ in platform.probes for u in us]
    return {u.get("condition") for u in units} - {None}


# ============================================================================ blocks the platform cannot run
def test_run_all_skips_the_llm_adaptive_check_before_anything_runs_and_goes_on(tmp_path, capsys, monkeypatch):
    past_kickoff(monkeypatch)
    llm, mock = ConditionRefusing("llm", adaptive_conditions=()), fakes.FakePlatform("mock")
    argv = ["run-all", "--allow-live-llm", "--results-root", str(tmp_path)] + sets(MOCK, LLM, ADAPTIVE_CHECK)
    assert run(argv, fakes.wiring({"llm": llm, "mock": mock})) == 0
    out = capsys.readouterr().out
    assert blocks_run(out) == ["llm_calibration", "mock_calibration", "llm_headline", "llm_capped", "mock_frontier"]
    line = next(ln for ln in out.splitlines() if ln.startswith("llm_adaptive_check: skipped: "))
    assert "the LLM platform has no adaptive condition yet" in line and "'gray_box'" in line
    assert "D110" in line and "gray-box best response only" in line
    assert handed_conditions(llm) == set() and not (tmp_path / "sprint" / "llm_adaptive_check").exists()


@pytest.mark.parametrize("flag", [[], ["--allow-live-llm"]])
def test_a_launch_of_the_llm_adaptive_check_is_refused_with_the_same_reason_before_any_write(tmp_path, monkeypatch,
                                                                                             flag):
    past_kickoff(monkeypatch)
    s = fakes.settings(LLM, ADAPTIVE_CHECK)
    llm = ConditionRefusing("llm", adaptive_conditions=())
    reason = dict(runall.plan(s, tmp_path, True, fakes.wiring({"llm": llm}))[1])["llm_adaptive_check"]
    with pytest.raises(SystemExit) as refused:
        run(["--run", "--block", "llm_adaptive_check", *flag, "--results-root", str(tmp_path)] + sets(LLM, ADAPTIVE_CHECK),
            fakes.wiring({"llm": llm}))
    assert refused.value.code == f"Refusing llm_adaptive_check: {reason}"
    assert llm.probes == [] and llm.batches == [] and not (tmp_path / "sprint" / "llm_adaptive_check").exists()


def test_status_shows_why_the_llm_adaptive_check_cannot_run(tmp_path, capsys):
    none = fakes.wiring({"llm": fakes.FakePlatform("llm", adaptive_conditions=())})
    run(["status", "--results-root", str(tmp_path)] + sets(MOCK, LLM, ADAPTIVE_CHECK), none)
    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("llm_adaptive_check"))
    assert "not runnable: the LLM platform has no adaptive condition yet" in line and "D110" in line
    run(["status", "--results-root", str(tmp_path)] + sets(MOCK, LLM), fakes.wiring())
    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("llm_adaptive_check"))
    assert "scope not set" in line and "not runnable" not in line


def test_what_the_platform_cannot_run_is_asked_of_the_units_condition_not_of_the_block_name():
    s = fakes.settings(LLM, ADAPTIVE_CHECK)

    def labelled(settings, strict):
        return rx.UnitList({**u, "condition": "white_box"} for u in rx.units_llm_headline(settings, strict))
    other = rx.Block("llm_other", "llm", "evaluate", "an LLM block whose units carry a condition", labelled)
    assert "'white_box'" in blocks.unrunnable(other, s, fakes.wiring())
    assert blocks.unrunnable(rx.BLOCKS["llm_headline"], s, fakes.wiring()) is None
    mock_adaptive = rx.Block("mock_x", "mock", "evaluate", "", rx.BLOCKS["mock_adaptive"].units, adaptive=True)
    assert blocks.unrunnable(mock_adaptive, fakes.settings(MOCK), fakes.wiring()) is None   # the mock's conditions: adaptive feature



MOCK_ADAPTIVE = ["mock.adaptive.arms=['B']", "mock.adaptive.cells=[[10, 10]]",
                 "mock.adaptive.conditions=['gray_box', 'white_box']"]


class Asked:
    """A wiring's platform builder that records which platforms were built (asked for), over fakes.wiring."""

    def __init__(self, platforms):
        self.inner, self.built = fakes.wiring(platforms), []

    def wiring(self):
        def platform(name):
            self.built.append(name)
            return self.inner.platform(name)
        return rx.Wiring(arm=self.inner.arm, platform=platform, tracker=self.inner.tracker)


def test_what_a_platform_runs_is_asked_of_the_platform_through_the_wiring():
    s = fakes.settings(MOCK, LLM, ADAPTIVE_CHECK, MOCK_ADAPTIVE)
    runs_it = fakes.wiring({"llm": fakes.FakePlatform("llm", adaptive_conditions=("gray_box",))})
    assert blocks.unrunnable(rx.BLOCKS["llm_adaptive_check"], s, runs_it) is None
    gray_only = fakes.wiring({"mock": fakes.FakePlatform("mock", adaptive_conditions=("gray_box",))})
    reason = blocks.unrunnable(rx.BLOCKS["mock_adaptive"], s, gray_only)
    assert reason == "the MOCK platform runs the adaptive conditions 'gray_box' only (its units carry condition 'white_box')"
    assert blocks.unrunnable(rx.BLOCKS["mock_adaptive"], s, fakes.wiring()) is None


def test_a_platform_is_built_to_be_asked_only_for_units_that_carry_a_condition_and_a_missing_one_is_left_to_the_launch():
    s = fakes.settings(MOCK, LLM, ADAPTIVE_CHECK, MOCK_ADAPTIVE)
    asked = Asked({"llm": fakes.FakePlatform("llm", adaptive_conditions=())})
    for name in ("llm_headline", "llm_capped", "mock_frontier", "mock_calibration"):
        assert blocks.unrunnable(rx.BLOCKS[name], s, asked.wiring()) is None
    assert asked.built == []
    assert "'gray_box'" in blocks.unrunnable(rx.BLOCKS["llm_adaptive_check"], s, asked.wiring())
    assert blocks.unrunnable(rx.BLOCKS["mock_adaptive"], s, asked.wiring()) is None      # the mock feature is not built
    assert asked.built == ["llm", "mock"]


def test_the_mock_grid_lists_its_conditions_from_its_axes():
    s = fakes.settings(MOCK, MOCK_ADAPTIVE)
    grid = rx.BLOCKS["mock_adaptive"].units(s, False)
    assert grid.conditions() == ["gray_box", "white_box"] == list(dict.fromkeys(u["condition"] for u in grid))
    assert rx.BLOCKS["mock_frontier"].units(s, False).conditions() == []


# ============================================================================ table headers of profiles.toml
def test_a_table_header_is_read_whatever_its_trailing_comment_holds(tmp_path):
    path = tmp_path / "profiles.toml"
    path.write_text("\n".join([
        "[bazt]                       # every value is the owner's",
        "# katz_decay = <float>       # (0, 1]: decay",
        "[bazt.utility_loss]          # >= 0: utility lost per affected delivery",
        "# sanitize = <float>",
        "[estimates.by_arm.\"12\"]      # seconds = cpu time",
        "# \"A\" = <float>",
        "# [analysis.extra_metrics]   # OPEN: metric = [kind, rows]",
        "# protected_sink_harm = [\"binary\", \"attacked\"]",
        "  [indented . table]         # k = v",
        "# leaf = <int>",
        "values = [",
        "    [1]",
        "]",
        "# after_array = <int>",
    ]), encoding="utf-8")
    assert set(status.options(path)) == {"bazt.katz_decay", "bazt.utility_loss.sanitize", "estimates.by_arm.12.A",
                                         "analysis.extra_metrics.protected_sink_harm", "indented.table.leaf",
                                         "indented.table.after_array"}


def test_the_profile_lists_the_utility_losses_and_the_extra_metrics_under_their_own_tables():
    found = status.options()
    actions = ("sanitize", "delay", "edge_quarantine", "rate_limit", "privilege_degradation", "isolation")
    assert {f"bazt.utility_loss.{a}" for a in actions} <= set(found)
    assert "analysis.extra_metrics.protected_sink_harm" in found
    assert not {f"bazt.{a}" for a in actions} & set(found) and "analysis.pair_unit.protected_sink_harm" not in found


def test_a_utility_loss_or_extra_metric_the_owner_set_is_no_longer_listed(tmp_path, capsys):
    run(["status", "--results-root", str(tmp_path)] + sets(MOCK), fakes.wiring())
    out = capsys.readouterr().out
    assert "bazt.utility_loss.sanitize" in out and "analysis.extra_metrics.protected_sink_harm" in out
    owner = [('bazt.utility_loss={ "sanitize" = 0.1, "delay" = 0.1, "edge_quarantine" = 0.2, "rate_limit" = 0.1, '
              '"privilege_degradation" = 0.2, "isolation" = 0.5 }'),
             'analysis.extra_metrics={ "protected_sink_harm" = ["binary", "attacked"] }']
    run(["status", "--results-root", str(tmp_path)] + sets(MOCK, owner), fakes.wiring())
    out = capsys.readouterr().out
    assert "bazt.utility_loss." not in out and "analysis.extra_metrics." not in out
    assert "bazt.sanitize" not in out and "analysis.pair_unit.protected_sink_harm" not in out


# ============================================================================ threshold identity of every arm
# What each calibrated arm reads while it decides, from the arm features' code (bazt_slot params.py and
# adapters/upstream.py, conventional arm_b.py and arm_c.py, references legacy.py `keys`, gnn arm.py deploy_family).
ARM_KEYS = {
    "mock": {"B": ["runtime.rate_limit_per_identity", "costs.main.deny"],
             "C": ["conventional.c.presence_window_rounds", "conventional.c.judge_score_bins.mock"],
             "BAZT": ["bazt.katz_decay", "bazt.utility_loss.sanitize", "runtime.sanitize_success"],
             "BAZT_upstream": ["bazt_upstream.verdict_probability", "bazt.prior", "runtime.sanitize_success"],
             "score_only": ["mock.legacy.confirm_steps"],
             "risk_aware": ["mock.legacy.lam", "mock.legacy.gamma", "mock.legacy.dmax"],
             "risk_aware_tiered": ["mock.legacy.tau_san", "mock.legacy.tau_central", "mock.legacy.tau_confirm"],
             "random_alloc": ["costs.main.inspect"], "bra_greedy": ["costs.main.isolation"]},
    "llm": {"B": ["runtime.rate_limit_per_identity", "llm.judge.provider"],
            "C": ["conventional.c.presence_window_rounds", "conventional.c.judge_score_bins.llm"],
            "BAZT": ["bazt.katz_decay", "bazt.utility_loss.isolation", "runtime.sanitize_success"],
            "BAZT_upstream": ["bazt_upstream.verdict_probability", "bazt.prior"],
            "GNN_recalibrated": ["gnn.cut", "gnn.carry_flags", "gnn.fallback_family", "gnn_train.families",
                                 "gnn.embedding.model"]},
}
SET = ["bazt.katz_decay=0.1", "bazt.prior=0.1", "bazt.utility_loss.sanitize=0.1", "bazt.utility_loss.isolation=0.5",
       "mock.legacy.confirm_steps=2", "mock.legacy.lam=1.0", "mock.legacy.gamma=0.7", "mock.legacy.dmax=3",
       "mock.legacy.tau_san=0.05", "mock.legacy.tau_central=0.75", "mock.legacy.tau_confirm=0.85",
       "gnn.fallback_family='pi_same_task'", "runtime.sanitize_success=0.9"]


def _changed(value):
    return value + 1 if isinstance(value, (int, float)) and not isinstance(value, bool) else ["changed", value]


@pytest.mark.parametrize("platform", ["mock", "llm"])
def test_every_setting_a_calibrated_arm_reads_is_part_of_its_platforms_threshold_identity(platform):
    s = fakes.settings(MOCK, LLM, SET)
    view = identity.threshold_view(s, platform)
    for arm, keys in ARM_KEYS[platform].items():
        for key in keys + needs.required_keys(s, platform, [arm], f"{platform}_calibration"):
            assert key in view, f"{arm} reads {key}, which the {platform} thresholds would not remember"
            *path, leaf = key.split(".")
            other = core.get(s, key)
            node = s
            for part in path:
                node = node[part]
            node[leaf] = _changed(other)
            assert identity.threshold_view(s, platform) != view, key
            node[leaf] = other


def test_scope_and_configuration_keys_of_those_sections_leave_the_thresholds_valid():
    s = fakes.settings(MOCK, LLM, SET)
    views = {p: identity.threshold_view(s, p) for p in ("mock", "llm")}
    later = fakes.settings(MOCK, LLM, SET, ["mock.legacy.arms=['score_only']", "mock.legacy.cells=[[10, 10]]",
                                            "gnn_train.logs_dir='elsewhere'", "gnn_train.keep_checkpoints=1"])
    assert {p: identity.threshold_view(later, p) for p in views} == views


def test_a_fresh_evaluation_block_refuses_thresholds_calibrated_under_another_upstream_verdict_probability(tmp_path):
    s = fakes.settings(*MOCK_B)
    calibrate_then(tmp_path, s)
    changed = fakes.settings(*MOCK_B, ["bazt_upstream.verdict_probability=0.9"])
    with pytest.raises(SystemExit, match=r"Refusing mock_frontier: .*thresholds.*bazt_upstream\.verdict_probability"):
        launch(changed, "mock_frontier", tmp_path)



# What changes the results of an LLM calibration besides the arms' keys (llm_harness platform.py _environment and
# adapters/live.py, llm_data sampling.py): the backend, judge and items (llm.api: "fake" is a rehearsal), the pinned
# provider of each backbone (it also salts the request cache key) and the pinned data (revisions, vendored digests).
LLM_RESULT_KEYS = {"llm.api": "'fake'", "llm.providers": "{ bb1 = 'p1', bb2 = 'p2' }",
                   "llm_data.revisions.mmlu": "'0123456789abcdef0123456789abcdef01234567'",
                   "llm_data.vendored_sha256.injecagent": "'" + "0" * 64 + "'"}


@pytest.mark.parametrize("key", list(LLM_RESULT_KEYS))
def test_the_llm_thresholds_remember_the_api_the_providers_and_the_data_and_the_mock_ones_do_not(key):
    base = fakes.settings(MOCK, LLM)
    changed = fakes.settings(MOCK, LLM, [f"{key}={LLM_RESULT_KEYS[key]}"])
    assert identity.threshold_view(changed, "llm") != identity.threshold_view(base, "llm"), key
    assert identity.threshold_view(changed, "mock") == identity.threshold_view(base, "mock"), key


def test_thresholds_calibrated_on_the_fake_api_are_refused_by_a_live_evaluation_block_in_the_same_root(tmp_path,
                                                                                                      monkeypatch):
    past_kickoff(monkeypatch)
    rehearsal = fakes.settings(*LLM_LAUNCH, ["llm.api='fake'"])
    launch(rehearsal, "llm_calibration", tmp_path, platform_name="llm", allow_live_llm=True)
    launch(rehearsal, "llm_headline", tmp_path, platform_name="llm", allow_live_llm=True)     # the rehearsal reuses them
    live = fakes.FakePlatform("llm")
    with pytest.raises(SystemExit, match=r"Refusing llm_capped: the thresholds in .* were calibrated under other "
                                         r"settings: llm\.api \('fake' -> '<unset>'\)"):
        launch(fakes.settings(*LLM_LAUNCH), "llm_capped", tmp_path, platform=live, platform_name="llm",
               allow_live_llm=True)
    assert live.batches == [] and not (tmp_path / "sprint" / "llm_capped" / "units").exists()


# ============================================================================ LLM sizes of a dry run
PRICED = ["estimates.usd_per_million_tokens.bb1=[0.1, 0.2]", "estimates.usd_per_million_tokens.bb2=[0.1, 0.2]"]


def test_run_all_dry_run_sizes_the_llm_blocks_without_the_live_flag_and_runs_nothing(tmp_path, capsys):
    llm, mock = fakes.FakePlatform("llm"), fakes.FakePlatform("mock")
    argv = ["run-all", "--dry-run", "--results-root", str(tmp_path)] + sets(MOCK, LLM, PRICED)
    assert run(argv, fakes.wiring({"llm": llm, "mock": mock})) == 0
    out = capsys.readouterr().out
    assert "need --allow-live-llm" not in out
    for name in ("llm_calibration", "llm_headline", "llm_capped"):
        row = next(ln for ln in out.splitlines() if ln.startswith(f"{name} "))
        assert int(row.split()[3].replace(",", "")) > 0, row        # the LLM calls column
    assert "LLM calls: " in out and " h at " in out
    note = next(ln for ln in out.splitlines() if "--allow-live-llm" in ln)
    assert "llm_headline" in note and "kickoff" in note
    assert llm.batches == llm.probes == llm.calibrations == [] and mock.batches == mock.calibrations == []
    assert not any(tmp_path.iterdir())


def test_without_the_live_flag_a_real_run_all_still_skips_every_llm_block(tmp_path, capsys):
    llm = fakes.FakePlatform("llm")
    argv = ["run-all", "--results-root", str(tmp_path)] + sets(MOCK, LLM)
    assert run(argv, fakes.wiring({"llm": llm, "mock": fakes.FakePlatform("mock")})) == 0
    assert blocks_run(capsys.readouterr().out) == ["mock_calibration", "mock_frontier"]
    assert llm.batches == llm.probes == llm.calibrations == []
