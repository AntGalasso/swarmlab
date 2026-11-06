"""Run order, per-platform identity, honest progress, the full list of unset decisions, LLM probes, sync paths,
atomic writes and the smaller launch rules."""
import json
import os
from pathlib import Path

import pytest

from bazt_lab.core import public as core
from bazt_lab.features.runner import public as rx
from bazt_lab.features.runner.private import files, runall
from bazt_lab.features.runner.tests import fakes
from bazt_lab.features.runner.tests.test_commands import LLM, MOCK, run, sets
from bazt_lab.features.runner.tests.test_launch import (
    MOCK_A,
    MOCK_B,
    block_dir,
    calibrate_then,
    launch,
    past_kickoff,
    without,
)

GNN = ["gnn.cut=0.5", "gnn.carry_flags=true", "gnn.fallback_family='pi_same_task'"]


# ============================================================================================ run-all order
def test_run_all_trains_the_gnn_before_any_calibration(tmp_path):
    s = fakes.settings(MOCK, LLM, ["gnn_train.backbone='bb1'"])
    run_, _ = runall.plan(s, tmp_path, True, fakes.wiring())
    assert run_.index("gnn_train") < run_.index("llm_calibration") < run_.index("llm_headline")
    assert run_.index("gnn_train") < run_.index("mock_calibration")


def test_a_calibration_of_the_recalibrated_gnn_needs_its_checkpoints_first(tmp_path, monkeypatch):
    past_kickoff(monkeypatch)
    s = fakes.settings(LLM, GNN, ["llm.calibrated_arms=['B', 'GNN_recalibrated']"])
    plat = fakes.FakePlatform("llm")
    wiring = fakes.wiring({"llm": plat}, gnn=fakes.FakeGnn(finished=False))
    with pytest.raises(SystemExit, match=r"Refusing llm_calibration: .*GNN checkpoint.*gnn_train"):
        rx.run_block(s, "llm_calibration", True, wiring=wiring, results_root=tmp_path)
    assert plat.calibrations == [] and not any(tmp_path.iterdir())
    ok = fakes.wiring({"llm": plat}, gnn=fakes.FakeGnn())
    rx.run_block(s, "llm_calibration", True, wiring=ok, results_root=tmp_path)
    assert {arm for arm, _ in plat.calibrations} == {"B", "GNN_recalibrated"}


# =================================================================================== identity per platform
MOCK_ONLY_AFTER = ["runtime.horizon_constant=9", "runtime.eps_correct=2.0",
                   "runtime.insider_bias=1.0", "mock.model.beta=0.9"]
LLM_ONLY_AFTER = ["runtime.llm_calibration_seeds=[101, 102]", "runtime.random_edge_probability=0.3",
                  "conventional.c.judge_score_bins.llm=[0.0, 5.0]", "runtime.llm_seeds=[1, 2]"]


def test_llm_keys_do_not_invalidate_mock_thresholds_or_rows(tmp_path):
    s = fakes.settings(*MOCK_B)
    calibrate_then(tmp_path, s)
    launch(s, "mock_frontier", tmp_path)
    again = fakes.settings(*MOCK_B, LLM_ONLY_AFTER)
    assert launch(again, "mock_frontier", tmp_path).batches == []


def test_mock_keys_do_not_invalidate_llm_thresholds_or_rows(tmp_path, monkeypatch):
    past_kickoff(monkeypatch)
    s = fakes.settings(LLM)
    launch(s, "llm_calibration", tmp_path, platform_name="llm", allow_live_llm=True)
    launch(s, "llm_headline", tmp_path, platform_name="llm", allow_live_llm=True)
    again = fakes.settings(LLM, MOCK_ONLY_AFTER, ["conventional.c.judge_score_bins.mock=[0.0, 9.0]"])
    assert launch(again, "llm_headline", tmp_path, platform_name="llm", allow_live_llm=True).batches == []


# ======================================================================================= honest progress
class BatchPlatform(fakes.FakePlatform):
    """Commits a whole batch with one marker, as the mock engine does."""

    def run_batch(self, units, arm, settings, store, tracker):
        if store.root.name == "preflight-probe":
            return super().run_batch(units, arm, settings, store, tracker)
        self.batches.append((arm.name, [dict(u) for u in units]))
        store.append_batch({u["id"]: [{"unit": u, "accuracy": 1.0}] for u in units if not store.is_done(u["id"])})
        return []


def line_of(out, block):
    return next(ln for ln in out.splitlines() if ln.startswith(block))


def test_status_counts_units_committed_in_batches(tmp_path, capsys):
    argv = ["--run", "--block", "mock_frontier", "--results-root", str(tmp_path)] + sets(MOCK, [
        "mock.uncapped_arms=['A']", "mock.capped_arms=[]"])
    run(argv, fakes.wiring({"mock": BatchPlatform("mock")}))
    capsys.readouterr()
    run(["status", "--results-root", str(tmp_path)] + sets(MOCK, ["mock.uncapped_arms=['A']", "mock.capped_arms=[]"]),
        fakes.wiring())
    done, total = line_of(capsys.readouterr().out, "mock_frontier").split()[2].split("/")
    assert done == total and int(done) == 12


def test_status_counts_calibrated_cells_of_a_calibration_block(tmp_path, capsys):
    s = fakes.settings(*MOCK_B)
    calibrate_then(tmp_path, s)
    capsys.readouterr()
    argv = ["status", "--results-root", str(tmp_path)] + sets(*MOCK_B)
    run(argv, fakes.wiring())
    cells = len(rx.units_mock_calibration(s, True).cells())
    assert line_of(capsys.readouterr().out, "mock_calibration").split()[2] == f"{cells}/{cells}"


def test_status_writes_nothing(tmp_path):
    run(["status", "--results-root", str(tmp_path)] + sets(MOCK), fakes.wiring())
    assert not any(tmp_path.iterdir())


# ====================================================================================== unset decisions
def unset_groups(out):
    groups, current = {}, None
    section = False
    for ln in out.splitlines():
        if ln.startswith("owner decisions still unset"):
            section = True
        elif section and ln and not ln.startswith(" "):
            break
        elif section and ln.startswith("  ") and not ln.startswith("    "):
            current = ln.strip()
            groups[current] = []
        elif section and current and ln.startswith("    "):
            groups[current].append(ln.split()[0])
    return groups


def test_status_lists_every_key_a_block_will_demand_under_the_block(tmp_path, capsys):
    run(["status", "--results-root", str(tmp_path)] + sets(MOCK, LLM), fakes.wiring())
    groups = unset_groups(capsys.readouterr().out)
    assert any(k.startswith("bazt.") for k in groups["mock_calibration"])
    assert "costs.robustness.deny" in groups["mock_costs_robustness"]
    assert "mock.legacy.arms" in groups["mock_legacy"] and "mock.legacy.tau_san" in groups["mock_legacy"]
    assert "gnn_train.backbone" in groups["gnn_train"]
    assert any(k.startswith("analysis.") for k in groups["analysis"])
    assert "llm.backbones" not in groups["mock_frontier"] and "bazt.katz_decay" not in groups["llm_adaptive_check"]


def test_a_key_that_is_set_is_not_listed(tmp_path, capsys):
    run(["status", "--results-root", str(tmp_path)] + sets(MOCK, ["bazt.katz_decay=0.5"]), fakes.wiring())
    out = capsys.readouterr().out
    assert "bazt.katz_decay" not in out and "bazt.prior" in out


# ======================================================================================== LLM probes
def test_an_llm_block_runs_one_tiny_fake_api_conversation_per_arm_and_attack_family(tmp_path, monkeypatch):
    past_kickoff(monkeypatch)
    plat = fakes.FakePlatform("llm")
    s = fakes.settings(LLM, ["llm.headline_arms=['A', 'B']"])
    launch(s, "llm_calibration", tmp_path, platform=plat, platform_name="llm", allow_live_llm=True)
    plat = fakes.FakePlatform("llm")
    launch(s, "llm_headline", tmp_path, platform=plat, platform_name="llm", allow_live_llm=True)
    assert {(arm, u["attack"]) for arm, units, _ in plat.probes for u in units} == {
        ("A", "insider_pi"), ("A", "none"), ("B", "insider_pi"), ("B", "none")}
    assert all(len(units) == 1 and units[0]["conversations"] == 1 for _, units, _ in plat.probes)
    assert {core.get(st, "llm.api") for st in plat.probe_settings} == {"fake"}
    assert {core.get(st, "llm.allow_live") for st in plat.probe_settings} == {False}
    assert all(len(units) == 1 for _, units in plat.batches)               # the probes are not part of the run


def test_an_llm_probe_that_meets_an_unset_key_refuses_before_the_first_unit(tmp_path, monkeypatch):
    past_kickoff(monkeypatch)
    s = fakes.settings(LLM)
    launch(s, "llm_calibration", tmp_path, platform_name="llm", allow_live_llm=True)
    plat = fakes.FakePlatform("llm")
    plat.probe_raises = core.Unset("bazt.katz_decay")
    with pytest.raises(SystemExit, match=r"Refusing llm_headline: .*bazt\.katz_decay"):
        launch(s, "llm_headline", tmp_path, platform=plat, platform_name="llm", allow_live_llm=True)
    assert plat.batches == [] and not block_dir(tmp_path, "llm_headline").exists()


# ============================================================================================== sync
def test_the_gnn_logs_are_synced_during_gnn_training_too(tmp_path, monkeypatch):
    past_kickoff(monkeypatch)
    sync = fakes.FakeSync()
    s = fakes.settings(LLM, ["sync.enabled=true", "gnn_train.backbone='bb1'", "gnn.artifacts_dir='ckpt'",
                              "gnn_train.logs_dir='logs'"])
    rx.run_block(s, "gnn_train", True, wiring=fakes.wiring({"llm": fakes.FakePlatform("llm")}, sync=sync,
                                                            gnn=fakes.FakeGnn()), results_root=tmp_path)
    ((_, paths),) = sync.started
    assert paths == [tmp_path / "sprint", Path("ckpt"), Path("logs")]


# ============================================================================================ atomic writes
def test_small_files_are_replaced_with_the_cores_retrying_replace(tmp_path, monkeypatch):
    real, calls = os.replace, []

    def flaky(src, dst):
        calls.append(1)
        if len(calls) == 1:
            raise PermissionError("held by a reader")
        real(src, dst)
    monkeypatch.setattr(os, "replace", flaky)
    files.write_json(tmp_path / "x.json", {"a": 1})
    assert json.loads((tmp_path / "x.json").read_text()) == {"a": 1} and len(calls) == 2


# ======================================================================================= smaller rules
def test_settings_are_pinned_only_after_the_tracker_started(tmp_path):
    class Bad(fakes.RecordingTracker):
        def start(self, *a):
            raise RuntimeError("wandb is down")

    s = fakes.settings(*MOCK_A, ["tracking.enabled=true"])
    with pytest.raises(SystemExit, match="wandb is down"):
        launch(s, "mock_frontier", tmp_path, tracker=Bad())
    assert not (block_dir(tmp_path, "mock_frontier") / "settings.json").exists()
    other = fakes.settings(*MOCK_A, ["mock.main_seeds=[1, 3]"])
    assert launch(other, "mock_frontier", tmp_path).batches                 # nothing was pinned by the failed launch


def test_bazt_upstream_audits_so_it_needs_the_judge_keys():
    s = fakes.settings(*MOCK_A)
    assert "BAZT_upstream" in core.get(s, "llm.judge_arms")
    from bazt_lab.features.runner.private import needs
    assert "mock.auditor.binormal_method" in needs.required_keys(s, "mock", {"BAZT_upstream"}, "mock_frontier")


def test_a_busy_block_is_refused_without_a_trace_in_the_last_errors(tmp_path):
    out = block_dir(tmp_path, "mock_frontier")
    out.mkdir(parents=True)
    with files.exclusive(out / ".launch.lock", "held"), pytest.raises(SystemExit, match="another launch"):
        run(["--run", "--block", "mock_frontier", "--results-root", str(tmp_path)] + sets(MOCK_A[0], MOCK_A[1],
            MOCK_A[2]), fakes.wiring({"mock": fakes.FakePlatform("mock")}))
    assert not (tmp_path / "sprint" / "last_errors.json").exists()


# ================================================================================================== --device
def probe_settings(tmp_path, argv_extra, command=("--run", "--block", "mock_frontier"), groups=(MOCK,)):
    plat = fakes.FakePlatform("mock")
    run([*command, "--results-root", str(tmp_path), *argv_extra] + sets(*groups, ["mock.uncapped_arms=['A']",
                                                                                 "mock.capped_arms=[]"]),
        fakes.wiring({"mock": plat}))
    return plat.probe_settings


def test_device_and_compile_flags_are_shortcuts_for_the_compute_settings(tmp_path):
    seen = probe_settings(tmp_path, ["--device", "cuda:1", "--compile"])
    assert {core.get(st, "compute.device") for st in seen} == {"cuda:1"}
    assert {core.get(st, "compute.compile") for st in seen} == {True}
    plain = probe_settings(tmp_path / "plain", [])
    assert {core.get(st, "compute.device") for st in plain} == {"cpu"} and {
        core.get(st, "compute.compile") for st in plain} == {False}


def test_run_all_takes_the_same_flags(tmp_path):
    seen = probe_settings(tmp_path, ["--device", "cuda"], command=("run-all",))
    assert {core.get(st, "compute.device") for st in seen} == {"cuda"}


@pytest.mark.parametrize("bad", ["gpu", "cuda:", "cuda:x", "CUDA"])
def test_a_device_that_is_not_cpu_or_cuda_is_refused(tmp_path, bad):
    with pytest.raises(SystemExit):
        run(["--run", "--block", "mock_frontier", "--device", bad, "--results-root", str(tmp_path)] + sets(MOCK),
            fakes.wiring())


def test_the_help_shows_the_device_flags(capsys):
    for command in ([], ["run-all"]):
        with pytest.raises(SystemExit):
            run([*command, "--help"], fakes.wiring())
        out = capsys.readouterr().out
        assert "--device" in out and "cuda:N" in out and "--compile" in out


def test_the_device_is_not_part_of_any_identity(tmp_path):
    s_cpu, s_gpu = fakes.settings(*MOCK_B), fakes.settings(*MOCK_B, ["compute.device='cuda'", "compute.compile=true"])
    calibrate_then(tmp_path, s_cpu)
    launch(s_cpu, "mock_frontier", tmp_path)
    assert launch(s_gpu, "mock_frontier", tmp_path).batches == []             # same rows, other device: accepted
    assert calibrate_then(tmp_path, s_gpu).calibrations == []                  # same thresholds


def test_status_prints_the_active_device(tmp_path, capsys):
    run(["status", "--device", "cuda:2", "--results-root", str(tmp_path)] + sets(MOCK), fakes.wiring())
    assert "device cuda:2" in capsys.readouterr().out
    run(["status", "--results-root", str(tmp_path)] + sets(MOCK), fakes.wiring())
    assert "device cpu" in capsys.readouterr().out


def test_bazt_upstream_resolves_its_verdict_probability_in_preflight(tmp_path):
    s = fakes.settings(*MOCK_A, ["mock.uncapped_arms=['BAZT_upstream']", "mock.calibrated_arms=['BAZT_upstream']"])
    assert core.is_set(s, "bazt_upstream.verdict_probability")
    with pytest.raises(SystemExit, match=r"Refusing mock_calibration: .*bazt_upstream\.verdict_probability"):
        launch(without(s, "bazt_upstream.verdict_probability"), "mock_calibration", tmp_path)
    assert not any(tmp_path.iterdir())


# ============================================================================== sanitize_success on the LLM side
def test_sanitize_success_is_part_of_the_llm_identity_and_required_for_bazt_arms(tmp_path, monkeypatch):
    past_kickoff(monkeypatch)
    from bazt_lab.features.runner.private import needs
    s = fakes.settings(LLM, ["runtime.sanitize_success=0.9", "llm.headline_arms=['A']"])
    launch(s, "llm_calibration", tmp_path, platform_name="llm", allow_live_llm=True)
    launch(s, "llm_headline", tmp_path, platform_name="llm", allow_live_llm=True)
    changed = fakes.settings(LLM, ["runtime.sanitize_success=0.5", "llm.headline_arms=['A']"])
    with pytest.raises(SystemExit, match=r"Refusing llm_headline: .*runtime\.sanitize_success"):
        launch(changed, "llm_headline", tmp_path, platform_name="llm", allow_live_llm=True)
    with pytest.raises(SystemExit, match=r"Refusing llm_calibration: .*runtime\.sanitize_success"):
        launch(changed, "llm_calibration", tmp_path, platform_name="llm", allow_live_llm=True)
    assert "runtime.sanitize_success" in needs.required_keys(s, "llm", {"BAZT"}, "llm_headline")
    assert "runtime.sanitize_success" in needs.required_keys(s, "llm", {"BAZT_upstream"}, "llm_headline")
    assert "runtime.sanitize_success" not in needs.required_keys(s, "llm", {"A", "B", "C"}, "llm_headline")


def test_an_llm_block_with_bazt_arms_refuses_without_sanitize_success(tmp_path, monkeypatch):
    past_kickoff(monkeypatch)
    s = fakes.settings(LLM, ["llm.calibrated_arms=['B', 'BAZT']"])        # the LLM test values carry no sanitize_success
    assert not core.is_set(s, "runtime.sanitize_success")
    with pytest.raises(SystemExit, match=r"Refusing llm_calibration: .*runtime\.sanitize_success"):
        launch(s, "llm_calibration", tmp_path, platform_name="llm", allow_live_llm=True)
    assert not any(tmp_path.iterdir())
