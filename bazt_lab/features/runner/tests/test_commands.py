"""`run-all` and `status`: the two commands a person types."""
import json
from pathlib import Path

import pytest

from bazt_lab.core import public as core
from bazt_lab.features.runner import public as rx
from bazt_lab.features.runner.tests import fakes
from bazt_lab.features.runner.tests.test_launch import past_kickoff

MOCK = fakes.RUNTIME_MOCK + fakes.TINY_MOCK + ["mock.allocation_arms=[]", "budgets.levels_percent=[10]"]
LLM = fakes.RUNTIME_LLM + fakes.TINY_LLM


def sets(*groups):
    return [x for g in groups for o in g for x in ("--set", o)]


def run(argv, wiring=None, **kw):
    return rx.main(argv, wiring=wiring or fakes.wiring(**kw))


def blocks_run(out):
    return [line.split()[1] for line in out.splitlines() if line.startswith("== ")]


# ====================================================================================================== run-all
def test_run_all_runs_calibrations_before_evaluations_and_says_why_it_skipped_the_rest(tmp_path, capsys):
    plat = fakes.FakePlatform("mock")
    assert run(["run-all", "--results-root", str(tmp_path)] + sets(MOCK), fakes.wiring({"mock": plat})) == 0
    out = capsys.readouterr().out
    assert blocks_run(out) == ["mock_calibration", "mock_frontier"]
    assert "mock_ablations: skipped: scope not set (mock.ablations.arms)" in out
    assert "mock_heldout: skipped: the analysis is not frozen" in out
    assert "llm_headline: skipped: LLM blocks need --allow-live-llm" in out
    assert "analysis: skipped" in out
    assert plat.calibrations and plat.batches


def test_run_all_stops_at_the_first_refusal_with_one_line(tmp_path, capsys):
    bare = [o for o in MOCK if not o.startswith("runtime.eps_correct")]
    plat = fakes.FakePlatform("mock")
    assert run(["run-all", "--results-root", str(tmp_path)] + sets(bare), fakes.wiring({"mock": plat})) == 1
    captured = capsys.readouterr()
    assert blocks_run(captured.out) == ["mock_calibration"]
    lines = [ln for ln in captured.err.splitlines() if ln.strip()]
    assert len(lines) == 1 and lines[0].startswith("Refusing mock_calibration:") and "eps_correct" in lines[0]
    assert plat.batches == [] and plat.calibrations == []


def test_run_all_resumes_after_a_crash_and_repeats_nothing(tmp_path, capsys):
    argv = ["run-all", "--results-root", str(tmp_path)] + sets(MOCK)
    crashing = fakes.FakePlatform("mock", crash_on=lambda u: u["seed"] == 2 and u["sink"] == 1)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run(argv, fakes.wiring({"mock": crashing}))
    capsys.readouterr()
    healthy = fakes.FakePlatform("mock")
    assert run(argv, fakes.wiring({"mock": healthy})) == 0
    assert healthy.calibrations == []                                  # the calibration was complete
    first = {u["id"] for _, us in crashing.batches for u in us}
    assert "already done" in capsys.readouterr().out and len(first) < len(
        {u["id"] for _, us in healthy.batches for u in us} | first)


def test_run_all_dry_run_plans_and_runs_nothing(tmp_path, capsys):
    plat = fakes.FakePlatform("mock")
    argv = ["run-all", "--dry-run", "--results-root", str(tmp_path)] + sets(MOCK)
    assert run(argv, fakes.wiring({"mock": plat})) == 0
    out = capsys.readouterr().out
    assert "mock_calibration" in out and "mock_frontier" in out and "mock_ablations: skipped" in out
    assert plat.batches == [] and plat.calibrations == [] and not any(tmp_path.iterdir())


def test_run_all_runs_the_heldout_block_only_when_frozen(tmp_path, capsys):
    (tmp_path / "ANALYSIS_FROZEN").write_text("frozen")
    assert run(["run-all", "--results-root", str(tmp_path)] + sets(MOCK), fakes.wiring(
        {"mock": fakes.FakePlatform("mock")})) == 0
    assert blocks_run(capsys.readouterr().out) == ["mock_calibration", "mock_frontier", "mock_heldout"]


def test_run_all_runs_llm_blocks_only_with_the_live_flag(tmp_path, capsys, monkeypatch):
    past_kickoff(monkeypatch)
    platforms = {"llm": fakes.FakePlatform("llm"), "mock": fakes.FakePlatform("mock")}
    argv = ["run-all", "--allow-live-llm", "--results-root", str(tmp_path)] + sets(MOCK, LLM)
    assert run(argv, fakes.wiring(platforms)) == 0
    out = capsys.readouterr().out
    assert blocks_run(out) == ["llm_calibration", "mock_calibration", "llm_headline", "llm_capped", "mock_frontier"]
    assert "gnn_train: skipped: scope not set (gnn_train.backbone)" in out


def test_run_all_finishes_with_the_analysis_when_its_settings_are_set(tmp_path, capsys):
    analysis = fakes.FakeAnalysis()
    assert run(["run-all", "--results-root", str(tmp_path)] + sets(MOCK), fakes.wiring(
        {"mock": fakes.FakePlatform("mock")}, analysis=analysis)) == 0
    assert analysis.calls == [tmp_path / "sprint"] and "analysis: report written" in capsys.readouterr().out
    unset = fakes.FakeAnalysis(missing=["analysis.challengers"])
    run(["run-all", "--results-root", str(tmp_path)] + sets(MOCK), fakes.wiring(
        {"mock": fakes.FakePlatform("mock")}, analysis=unset))
    assert "analysis: skipped: analysis settings not set (analysis.challengers)" in capsys.readouterr().out
    assert unset.calls == []


# ====================================================================================================== status
def test_status_shows_progress_the_last_error_and_what_is_still_unset(tmp_path, capsys, monkeypatch):
    for name in ("HF_TOKEN", "WANDB_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_not_to_be_printed_1234567890abcdef")
    plat = fakes.FakePlatform("mock")
    with pytest.raises(SystemExit, match="no calibrated thresholds"):
        run(["--run", "--block", "mock_frontier", "--results-root", str(tmp_path)] + sets(MOCK, [
            "mock.calibrated_arms=['B']", "mock.uncapped_arms=['A', 'B']", "mock.capped_arms=['B']"]),
            fakes.wiring({"mock": plat}))
    run(["--run", "--block", "mock_calibration", "--results-root", str(tmp_path)] + sets(MOCK), fakes.wiring(
        {"mock": plat}))
    capsys.readouterr()
    assert run(["status", "--results-root", str(tmp_path)] + sets(MOCK), fakes.wiring({"mock": plat})) == 0
    out = capsys.readouterr().out
    cal = next(ln for ln in out.splitlines() if ln.startswith("mock_calibration"))
    done, total = cal.split()[2].split("/")
    assert done == total and int(done) > 0
    frontier = next(ln for ln in out.splitlines() if ln.startswith("mock_frontier"))
    assert "0/" in frontier and "Refusing mock_frontier: no calibrated thresholds" in out
    assert "unset" in out.lower() and "mock.ablations.arms" in out and "calibration.mode" not in out
    assert "HF_TOKEN" in out and "hf_not_to_be_printed" not in out
    assert next(ln for ln in out.splitlines() if "WANDB_API_KEY" in ln).split()[-1] == "missing"
    assert next(ln for ln in out.splitlines() if "HF_TOKEN" in ln).split()[-1] == "present"


def test_status_lists_each_owner_decision_with_its_options_from_the_settings_file(tmp_path, capsys):
    bare = fakes.settings(fakes.TINY_MOCK)                      # nothing of the owner's is set
    assert bare is not None
    assert run(["status", "--results-root", str(tmp_path)] + sets(fakes.TINY_MOCK), fakes.wiring()) == 0
    out = capsys.readouterr().out
    assert "mock_calibration" in out and "calibration.mode" in out
    assert "closed_loop" in out and "open_loop" in out
    assert "gnn.fallback_family" in out and "pi_same_task" in out


def test_status_never_expands_the_mock_grid(tmp_path, monkeypatch, capsys):
    from bazt_lab.features.runner.private import units as units_mod

    def no_expansion(self):
        raise AssertionError("status must count arithmetically")
    monkeypatch.setattr(units_mod.MockGrid, "__iter__", no_expansion)
    assert run(["status", "--results-root", str(tmp_path)], fakes.wiring()) == 0
    assert "mock_frontier" in capsys.readouterr().out


# ======================================================================================================= sync
def test_sync_runs_beside_each_block_and_stops_with_it(tmp_path):
    sync = fakes.FakeSync()
    argv = ["--run", "--block", "mock_calibration", "--block", "mock_frontier", "--results-root", str(tmp_path)]
    run(argv + sets(MOCK, ["sync.enabled=true"]), fakes.wiring({"mock": fakes.FakePlatform("mock")}, sync=sync))
    # the profile folder is what is mirrored, so two profiles in one repository stay apart (<profile>/<block>/...)
    assert [paths for _, paths in sync.started] == [[tmp_path / "sprint"], [tmp_path / "sprint"]]
    assert core.get(sync.started[0][0], "sync.enabled") is True and sync.stopped == 2


def test_the_gnn_artifacts_are_synced_during_gnn_training(tmp_path, monkeypatch):
    past_kickoff(monkeypatch)
    sync = fakes.FakeSync()
    argv = ["--run", "--block", "gnn_train", "--results-root", str(tmp_path)]
    run(argv + sets(LLM, ["sync.enabled=true", "gnn_train.backbone='bb1'", "gnn.artifacts_dir='ckpt'",
                                  "gnn_train.logs_dir='logs'"]) +
        ["--allow-live-llm"], fakes.wiring({"llm": fakes.FakePlatform("llm")}, sync=sync, gnn=fakes.FakeGnn()))
    ((_, paths),) = sync.started
    assert paths == [tmp_path / "sprint", Path("ckpt"), Path("logs")]          # as the gnn feature reads them


def test_sync_is_stopped_after_a_crash_and_run_all_starts_one_per_block(tmp_path):
    sync = fakes.FakeSync()
    crashing = fakes.FakePlatform("mock", crash_on=lambda u: True)
    argv = ["run-all", "--results-root", str(tmp_path)] + sets(MOCK, ["sync.enabled=true"])
    with pytest.raises(RuntimeError):
        run(argv, fakes.wiring({"mock": crashing}, sync=sync))
    assert len(sync.started) == 1 and sync.stopped == 1
    run(argv, fakes.wiring({"mock": fakes.FakePlatform("mock")}, sync=sync))
    assert len(sync.started) == 3 and sync.stopped == 3                   # calibration (rerun) and frontier


def test_sync_is_off_unless_enabled_and_never_started_for_list_or_dry_run(tmp_path):
    sync = fakes.FakeSync()
    argv = ["--run", "--block", "mock_calibration", "--results-root", str(tmp_path)] + sets(MOCK)
    run(argv, fakes.wiring({"mock": fakes.FakePlatform("mock")}, sync=sync))
    run(["--dry-run"] + sets(MOCK, ["sync.enabled=true"]), fakes.wiring(sync=sync))
    run(["--list"] + sets(MOCK, ["sync.enabled=true"]), fakes.wiring(sync=sync))
    assert sync.started == []


def test_sync_enabled_without_the_sync_feature_is_refused_before_anything_is_written(tmp_path):
    argv = ["--run", "--block", "mock_calibration", "--results-root", str(tmp_path)] + sets(MOCK, ["sync.enabled=true"])
    with pytest.raises(SystemExit, match=r"Refusing mock_calibration: feature sync is not built"):
        run(argv, fakes.wiring({"mock": fakes.FakePlatform("mock")}))
    assert not (tmp_path / "sprint" / "mock_calibration").exists()


@pytest.mark.parametrize("failure, message", [(RuntimeError("sync.repo_id must be owner/name"), "sync.repo_id"),
                                              (core.Unset("sync.every_minutes"), "sync.every_minutes")])
def test_a_sync_that_cannot_start_stops_the_launch_like_a_tracking_refusal(tmp_path, failure, message):
    class Refusing(fakes.FakeSync):
        def start(self, settings, paths):
            raise failure

    plat = fakes.FakePlatform("mock")
    argv = ["--run", "--block", "mock_calibration", "--results-root", str(tmp_path)] + sets(MOCK, ["sync.enabled=true"])
    with pytest.raises(SystemExit, match=rf"Refusing mock_calibration: .*{message}"):
        run(argv, fakes.wiring({"mock": plat}, sync=Refusing()))
    assert plat.calibrations == [] and not list(tmp_path.rglob("manifest-*.json"))


# ============================================================================== platforms that are not configured
def test_run_all_skips_a_platform_none_of_whose_run_time_variables_is_set(tmp_path, capsys, monkeypatch):
    past_kickoff(monkeypatch)
    argv = ["run-all", "--allow-live-llm", "--results-root", str(tmp_path)] + sets(LLM)
    assert run(argv, fakes.wiring({"llm": fakes.FakePlatform("llm")})) == 0
    out = capsys.readouterr().out
    assert blocks_run(out) == ["llm_calibration", "llm_headline", "llm_capped"]
    assert "mock_frontier: skipped: no mock run-time variable is set (runtime.horizon_constant)" in out


def test_the_profile_holds_the_sync_and_checkpoint_keys():
    s = rx.load_settings()
    assert core.get(s, "gnn_train.keep_checkpoints") == 5
    assert not core.is_set(s, "sync.enabled") and not core.is_set(s, "sync.repo_id")
    assert [core.get(s, f"sync.{k}") for k in ("repo_type", "private", "every_minutes", "include", "exclude", "priority",
                                              "final_push_timeout_seconds")] == [
        "dataset", True, 10, [], [".launch.lock"], "below_normal", 600]
    assert json.dumps(core.get(s, "tracking.flush_rows")) == "3600"


def test_sync_failures_are_printed_at_the_end_of_the_block(tmp_path, capsys):
    sync = fakes.FakeSync()
    sync.summary = {"commits": 3, "failures": {"commit": 2}}
    argv = ["--run", "--block", "mock_calibration", "--results-root", str(tmp_path)]
    run(argv + sets(MOCK, ["sync.enabled=true"]), fakes.wiring({"mock": fakes.FakePlatform("mock")}, sync=sync))
    assert "mock_calibration: sync reported failures {'commit': 2} (3 commits)" in capsys.readouterr().out
    sync.summary = {"commits": 3, "failures": {}}
    run(argv + sets(MOCK, ["sync.enabled=true"]), fakes.wiring({"mock": fakes.FakePlatform("mock")}, sync=sync))
    assert "sync reported" not in capsys.readouterr().out
