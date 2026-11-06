"""scripts/e2e_common.py, e2e_mock.py and e2e_llm_fake.py.

The checks are tested on files written here. The two scripts run end to end against scripts/tests/fake_launcher.py,
which wires fake platforms into the real runner, with real process kills; planted defects (FAKE_BUG) must be caught.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import e2e_common as e2e
import e2e_llm_fake
import e2e_mock
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LAUNCHER = str(HERE / "fake_launcher.py")
NAN = math.nan


def good_row(**changes) -> dict:
    row = dict.fromkeys(e2e.columns(), 0.5)
    row.update(arm="B", audit="uncapped", intervention="uncapped", attack="infection", agents=4, attack_free=False,
               n_agents=4, sink=1, violations=0, exhausted_step=0, spend_audit_total=1.0,
               spend_intervention_total=2.0, spend_total=3.0, time_to_containment=2.0)
    row.update(accuracy_clean=NAN, fpr_message=NAN, fpr_agent=NAN)
    row.update(changes)
    return row


def write_unit(units: Path, uid: str, rows: list, done: bool = True, tail: str = "") -> None:
    units.mkdir(parents=True, exist_ok=True)
    (units / f"{uid}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows) + tail, encoding="utf-8")
    if done:
        (units / f"{uid}.done").write_bytes(b"")


# ------------------------------------------------------------------------------------------- test values
PROFILES = """
active_profile = "sprint"
[runtime]
# n_insiders = <int>
# horizon_constant = <float>           # mock T
ba_m = 2
[runtime.llm_sink_positions]
# random_directed = [<int>]
[llm]
# provider = "x"
"""


def test_documented_run_time_variables_are_the_commented_keys_of_the_runtime_table():
    assert e2e.documented_runtime_keys(PROFILES) == ["n_insiders", "horizon_constant"]
    assert e2e.documented_runtime_keys("[llm]\n# x = 1\n") == []


def test_every_run_time_variable_of_profiles_toml_has_a_test_value():
    text = (REPO / "experiments" / "profiles.toml").read_text(encoding="utf-8")
    assert e2e.documented_runtime_keys(text), "profiles.toml documents no run-time variable"
    assert e2e.missing_test_values(text) == []
    assert e2e.missing_test_values(PROFILES.replace("n_insiders", "brand_new_variable")) == ["brand_new_variable"]


def test_runtime_sets_carry_every_test_value_under_the_runtime_table():
    assert e2e.runtime_sets() == {f"runtime.{k}": v for k, v in e2e.RUNTIME_TEST_VALUES.items()}
    changed = e2e.runtime_sets(llm_seeds=[7])
    assert changed["runtime.llm_seeds"] == [7] and e2e.RUNTIME_TEST_VALUES["llm_seeds"] == [1, 2]
    with pytest.raises(KeyError):
        e2e.runtime_sets(not_a_run_time_variable=1)


def test_overrides_are_rendered_as_toml_values_after_set():
    args = e2e.set_args({"a.b": 3, "c": [1, 2], "d": "x", "e": True, "f": 0.5, "g.h": ["p", "q"]})
    assert args == ["--set", "a.b=3", "--set", "c=[1, 2]", "--set", 'd="x"', "--set", "e=true", "--set", "f=0.5",
                    "--set", 'g.h=["p", "q"]']


def test_the_child_environment_has_empty_secrets_and_no_inherited_kickoff_override_unless_asked():
    base = {"PATH": "p", "HF_TOKEN": "real", "OPENROUTER_API_KEY": "real", "WANDB_API_KEY": "real"}
    env = e2e.child_env(base)
    assert all(env[name] == "" for name in e2e.SECRETS) and env["PATH"] == "p"
    assert "SWARMLAB_KICKOFF_UTC" not in env
    assert e2e.child_env({"SWARMLAB_KICKOFF_UTC": "2999-01-01T00:00:00+00:00"})[
        "SWARMLAB_KICKOFF_UTC"] == "2999-01-01T00:00:00+00:00"
    assert e2e.child_env(base, kickoff_override="2000-01-01T00:00:00+00:00")[
        "SWARMLAB_KICKOFF_UTC"] == "2000-01-01T00:00:00+00:00"
    assert base["HF_TOKEN"] == "real"


# ------------------------------------------------------------------------------------------------- rows
def test_a_valid_row_has_no_problem():
    assert e2e.check_row(good_row(), e2e.columns(), "B") == []
    clean = good_row(attack="none", attack_free=True, asr_final=NAN, accuracy_clean=0.9, accuracy_attack=NAN, nrp=NAN,
                     time_to_containment=NAN, fpr_message=0.1, fpr_agent=0.0, n_attackers=0.0)
    assert e2e.check_row(clean, e2e.columns(), "B") == []


@pytest.mark.parametrize("changes, text", [
    ({"asr_final": 1.5}, "asr_final"),
    ({"nrp": -0.1}, "nrp"),
    ({"spend_total": -1.0}, "spend_total"),
    ({"spend_total": NAN}, "spend_total"),
    ({"violations": math.inf}, "violations"),
    ({"arm": "C"}, "arm"),
    ({"n_agents": 5}, "n_agents"),
    ({"sink": 4}, "sink"),
    ({"violations": 1}, "uncapped"),
    ({"exhausted_step": 2}, "uncapped"),
    ({"attack": "none"}, "attack_free"),
    ({"accuracy_attack": NAN}, "accuracy_attack"),
    ({"accuracy_clean": 0.9}, "accuracy_clean"),
    ({"time_to_containment": -1.0}, "time_to_containment"),
])
def test_a_row_that_breaks_an_invariant_is_reported(changes, text):
    problems = e2e.check_row(good_row(**changes), e2e.columns(), "B")
    assert any(text in p for p in problems), problems


def test_a_missing_column_is_reported_and_arm_a_must_cost_nothing():
    row = good_row()
    del row["nrp"]
    assert any("nrp" in p for p in e2e.check_row(row, e2e.columns(), "B"))
    assert any("spend_total" in p for p in e2e.check_row(good_row(arm="A"), e2e.columns(), "A"))
    assert e2e.check_row(good_row(arm="A", spend_total=0.0, spend_audit_total=0.0, spend_intervention_total=0.0),
                         e2e.columns(), "A") == []


def test_an_attack_free_row_must_not_report_an_asr():
    row = good_row(attack="none", attack_free=True, accuracy_clean=0.9, accuracy_attack=NAN, nrp=NAN)
    assert any("asr_final" in p for p in e2e.check_row(row, e2e.columns(), "B"))


def test_an_attack_free_row_has_no_attacker():
    clean = dict(attack="none", attack_free=True, asr_final=NAN, accuracy_clean=0.9, accuracy_attack=NAN, nrp=NAN,
                 time_to_containment=NAN, fpr_message=0.1, fpr_agent=0.0)
    assert e2e.check_row(good_row(n_attackers=0.0, **clean), e2e.columns(), "B") == []
    assert any("n_attackers" in p for p in e2e.check_row(good_row(n_attackers=1.0, **clean), e2e.columns(), "B"))


def test_arms_that_share_a_scenario_share_its_ground_truth():
    rows = [good_row(arm=a, topology="chain", attack="infection", seed=1, sink=2, n_attackers=1.0) for a in "ABC"]
    rows += [good_row(arm="B", topology="chain", attack="infection", seed=2, sink=2, n_attackers=2.0)]
    assert e2e.check_scenarios(rows) == []
    rows[2]["n_attackers"] = 2.0
    (problem,) = e2e.check_scenarios(rows)
    assert "n_attackers" in problem and "chain" in problem and "seed=1" in problem
    rows[2].update(n_attackers=1.0, n_agents=5)
    assert any("n_agents" in p for p in e2e.check_scenarios(rows))


# ------------------------------------------------------------------------------------------------ files
def test_a_finished_block_has_one_done_unit_file_per_unit_with_the_right_rows(tmp_path):
    units = tmp_path / "units"
    for i in range(3):
        write_unit(units, f"B-{i}", [good_row(), good_row(sink=2)])
    assert e2e.check_block(tmp_path, 3, 2) == []
    assert any("expected 4" in p for p in e2e.check_block(tmp_path, 4, 2))
    assert any("2 rows, expected 3" in p for p in e2e.check_block(tmp_path, 3, 3))


def test_unfinished_torn_and_stray_files_are_reported(tmp_path):
    units = tmp_path / "units"
    write_unit(units, "B-0", [good_row()])
    write_unit(units, "B-1", [good_row()], done=False)
    write_unit(units, "B-2", [good_row()], tail='{"arm": "half')
    (units / "B-3.done").write_bytes(b"")
    (units / "B-4.jsonl.tmp").write_text("x")
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "B-1.pt").write_bytes(b"x")
    problems = " | ".join(e2e.check_block(tmp_path, 3, 1))
    for text in ("B-1.jsonl: not marked done", "B-2.jsonl: 1 line", "B-3.done has no rows", "B-4.jsonl.tmp",
                 "B-1.pt"):
        assert text in problems, (text, problems)


def test_every_row_of_every_unit_file_is_checked_against_the_arm_in_its_file_name(tmp_path):
    units = tmp_path / "units"
    write_unit(units, "B-0", [good_row()])
    write_unit(units, "C-1", [good_row(arm="C")])
    assert e2e.check_rows(units, e2e.columns()) == []
    write_unit(units, "B-2", [good_row(asr_final=1.5)])
    write_unit(units, "C-3", [good_row(arm="B")])
    problems = e2e.check_rows(units, e2e.columns())
    assert any(p.startswith("B-2.jsonl:") and "asr_final" in p for p in problems)
    assert any(p.startswith("C-3.jsonl:") and "arm" in p for p in problems)


def test_resume_must_keep_finished_files_and_continue_unfinished_ones_after_their_last_row():
    done_before = {"B-0"}
    before = {"B-0.jsonl": b'{"a": 1}\n', "B-1.jsonl": b'{"a": 1}\n{"b": 2'}
    good = {"B-0.jsonl": b'{"a": 1}\n', "B-1.jsonl": b'{"a": 1}\n{"b": 2}\n'}
    assert e2e.check_resume(before, {"B-0.jsonl": b'{"a": 1}\n', "B-1.jsonl": b'{"a": 1}\n{"c": 3}\n'}, done_before) == []
    assert any("changed" in p for p in e2e.check_resume(before, {**good, "B-0.jsonl": b'{"a": 1}\n\n'}, done_before))
    assert any("rewritten" in p or "prefix" in p for p in e2e.check_resume(
        before, {**good, "B-1.jsonl": b'{"z": 9}\n'}, done_before))
    assert any("lost" in p for p in e2e.check_resume(before, {"B-0.jsonl": b'{"a": 1}\n'}, done_before))


def test_thresholds_file_must_hold_every_cell_with_finite_values(tmp_path):
    path = tmp_path / "thresholds.json"
    cell = {"arm": "B", "topology": "chain", "agents": 4, "backbone": None, "task": None,
            "thresholds": {"deny_score": 0.5}}
    assert any("missing" in p for p in e2e.check_thresholds(path, 1, 0.02))
    path.write_text(json.dumps({"alpha": 0.02, "cells": [cell]}))
    assert e2e.check_thresholds(path, 1, 0.02) == []
    assert any("expected 2" in p for p in e2e.check_thresholds(path, 2, 0.02))
    assert any("alpha" in p for p in e2e.check_thresholds(path, 1, 0.05))
    path.write_text(json.dumps({"alpha": 0.02, "cells": [cell, dict(cell)]}))
    assert any("twice" in p for p in e2e.check_thresholds(path, 2, 0.02))
    path.write_text(json.dumps({"alpha": 0.02, "cells": [{**cell, "thresholds": {"deny_score": "NaN"}}]}))
    assert any("deny_score" in p for p in e2e.check_thresholds(path, 1, 0.02))
    path.write_text(json.dumps({"alpha": 0.02, "cells": [{**cell, "thresholds": {}}]}))
    assert any("no threshold" in p for p in e2e.check_thresholds(path, 1, 0.02))


def manifest(block="mock_frontier", units=10, **changes) -> dict:
    m = {"block": block, "profile": "sprint", "units": units, "threads": 4,
         "pre_registered": {"ni_margin_accuracy": 0.01, "rank_agreement_threshold": 0.5},
         "settings": {"runtime": dict(e2e.RUNTIME_TEST_VALUES)}}
    m.update(changes)
    return m


def test_every_launch_leaves_a_manifest_that_records_the_test_values(tmp_path):
    for i in range(2):
        (tmp_path / f"manifest-{i}.json").write_text(json.dumps(manifest()))
    assert e2e.check_manifests(tmp_path, "mock_frontier", launches=2, units=10) == []
    assert any("3" in p for p in e2e.check_manifests(tmp_path, "mock_frontier", launches=3, units=10))
    assert any("units" in p for p in e2e.check_manifests(tmp_path, "mock_frontier", launches=2, units=11))
    (tmp_path / "manifest-1.json").write_text(json.dumps(manifest(pre_registered={})))
    assert any("pre_registered" in p for p in e2e.check_manifests(tmp_path, "mock_frontier", launches=2, units=10))


def test_manifests_are_compared_with_the_run_time_values_the_plan_set(tmp_path):
    runtime = {**e2e.RUNTIME_TEST_VALUES, "llm_seeds": [1], "ni_margin_accuracy": 0.02}
    (tmp_path / "manifest-0.json").write_text(json.dumps(manifest(
        settings={"runtime": runtime}, pre_registered={"ni_margin_accuracy": 0.02, "rank_agreement_threshold": 0.5})))
    assert e2e.check_manifests(tmp_path, "mock_frontier", 1, 10, runtime=runtime) == []
    problems = e2e.check_manifests(tmp_path, "mock_frontier", 1, 10)
    assert any("run-time" in p for p in problems) and any("pre_registered" in p for p in problems)


# --------------------------------------------------------------------------------------------- progress
def test_progress_counts_started_and_done_units(tmp_path):
    units = tmp_path / "units"
    assert e2e.progress(units) == e2e.Progress(started=0, done=0)
    write_unit(units, "A-1", [good_row()])
    write_unit(units, "A-2", [good_row()], done=False)
    (units / "A-3.jsonl").write_text("")
    assert e2e.progress(units) == e2e.Progress(started=2, done=1)


# --------------------------------------------------------------------------------------------- analysis
def analysis_session(tmp_path, **plan_changes) -> e2e.Session:
    plan = e2e.Plan("t", {"runtime.n_insiders": 3}, [], [], **plan_changes)
    return e2e.Session(plan, Path("launcher.py"), tmp_path / "results", "sprint", {}, 1, tmp_path)


def test_the_analysis_gets_the_profile_the_run_time_values_and_its_own_test_values(tmp_path):
    session = analysis_session(tmp_path, analysis_sets={"analysis.baseline": "C"})
    (session.results_root / "sprint").mkdir(parents=True)
    report = e2e.Report()
    result = e2e.stage_analysis(session, report, "fake_analysis:echo")
    assert result.pop("meta") == {"platforms": {}}
    assert result == {"results_dir": str(session.results_root / "sprint"), "profile": "sprint", "n_insiders": 3,
                      "baseline": "C"}
    assert report.failures == []


def test_an_analysis_that_returns_nothing_is_a_failure(tmp_path):
    session = analysis_session(tmp_path)
    (session.results_root / "sprint").mkdir(parents=True)
    report = e2e.Report()
    e2e.stage_analysis(session, report, "fake_analysis:analyze")
    assert len(report.failures) == 1 and "analysis returned" in report.failures[0]


def test_an_analysis_that_needs_a_setting_stops_the_run_naming_the_key(tmp_path):
    session = analysis_session(tmp_path)
    (session.results_root / "sprint").mkdir(parents=True)
    with pytest.raises(e2e.Abort, match=r"analysis\.never_set_anywhere"):
        e2e.stage_analysis(session, e2e.Report(), "fake_analysis:needs_setting")


# ------------------------------------------------------------------------- the plans against the runner
def args_of(module, *extra):
    return module.parser().parse_args(["--launcher", LAUNCHER, *extra])


def test_the_mock_plan_counts_the_units_the_runner_expands():
    from bazt_lab.features.runner import public as rx
    for extra in ([], ["--quick"], ["--seeds", "3", "--agents", "5"]):
        plan = e2e_mock.build_plan(args_of(e2e_mock, *extra))
        settings = rx.load_settings(overrides=[a for a in e2e.set_args(plan.sets) if a != "--set"])
        by_block = {s.block: s for s in plan.evaluations}
        assert by_block["mock_frontier"].units == len(rx.units_mock_frontier(settings, True)), extra
        (calibration,) = plan.calibrations
        assert calibration.cells == len(rx.units_mock_calibration(settings, True).cells()), extra


def test_the_llm_plan_counts_the_units_the_runner_expands():
    from bazt_lab.features.runner import public as rx
    for extra in ([], ["--with-gnn"]):
        plan = e2e_llm_fake.build_plan(args_of(e2e_llm_fake, *extra))
        settings = rx.load_settings(overrides=[a for a in e2e.set_args(plan.sets) if a != "--set"])
        by_block = {s.block: s for s in plan.evaluations}
        assert by_block["llm_headline"].units == len(rx.units_llm_headline(settings, True)), extra
        assert by_block["llm_capped"].units == len(rx.units_llm_capped(settings, True)), extra
        assert by_block["llm_headline"].rows_per_unit == rx.units_llm_headline(settings, True)[0]["conversations"]
        (calibration,) = plan.calibrations
        assert calibration.cells == len(rx.units_llm_calibration(settings, True).cells()), extra
        assert ("gnn_train" in by_block) is bool(extra)


def test_the_llm_plan_always_selects_the_fake_api_and_the_plan_never_holds_a_secret():
    plan = e2e_llm_fake.build_plan(args_of(e2e_llm_fake))
    assert plan.sets["llm.api"] == "fake"
    assert not any("KEY" in k.upper() or "TOKEN" in k.upper() for k in plan.sets)
    custom = e2e_llm_fake.build_plan(args_of(e2e_llm_fake, "--fake-set", 'llm.client="stub"'))
    assert custom.sets["llm.client"] == "stub" and "llm.api" not in custom.sets


# ------------------------------------------------------------------------------- end to end, fake launcher
QUICK = ["--quick", "--agents", "3", "--seeds", "1", "--timeout", "240", "--analysis-entry", "fake_analysis:analyze"]


def run_e2e(module, capsys, *extra) -> tuple[int, str]:
    code = module.main(["--launcher", LAUNCHER, *QUICK, *extra])
    return code, capsys.readouterr().out


def test_mock_smoke_run_passes_on_a_launcher_that_follows_the_contract(capsys, monkeypatch):
    monkeypatch.delenv("FAKE_BUG", raising=False)
    code, out = run_e2e(e2e_mock, capsys)
    assert code == 0, out
    for stage in ("list", "dry run", "mock_calibration", "mock_frontier: killed", "mock_frontier: resumed",
                  "mock_frontier: second launch", "analysis"):
        assert stage in out, (stage, out)
    assert "FAIL" not in out and "e2e mock: passed" in out


@pytest.mark.parametrize("bug, text", [("duplicate", "rows, expected"), ("range", "asr_final")])
def test_mock_smoke_run_catches_a_planted_defect(capsys, monkeypatch, bug, text):
    monkeypatch.setenv("FAKE_BUG", bug)
    code, out = run_e2e(e2e_mock, capsys)
    assert code == 1 and "FAIL" in out and text in out, out


def test_llm_smoke_run_passes_with_the_explicit_kickoff_override(capsys, monkeypatch):
    monkeypatch.delenv("FAKE_BUG", raising=False)
    code, out = run_e2e(e2e_llm_fake, capsys, "--kickoff-override", "2000-01-01T00:00:00+00:00")
    assert code == 0, out
    assert "llm_headline: killed" in out and "llm_capped" in out and "e2e llm-fake: passed" in out
    assert "part-written" in out and "resume path not exercised" not in out, out


def test_llm_smoke_run_catches_a_torn_line_after_the_restart(capsys, monkeypatch):
    monkeypatch.setenv("FAKE_BUG", "torn")
    code, out = run_e2e(e2e_llm_fake, capsys, "--kickoff-override", "2000-01-01T00:00:00+00:00")
    assert code == 1 and "not JSON" in out, out


def test_llm_smoke_run_stops_with_code_3_when_the_kickoff_guard_refuses(capsys, monkeypatch):
    monkeypatch.setenv("SWARMLAB_KICKOFF_UTC", "2999-01-01T00:00:00+00:00")
    code, out = run_e2e(e2e_llm_fake, capsys)
    assert code == 3 and "kickoff" in out and "--kickoff-override" in out, out
    assert "llm_calibration" not in os.listdir(REPO / "results") if (REPO / "results").is_dir() else True


def test_a_launcher_that_does_not_exist_is_a_usage_error(capsys):
    assert e2e_mock.main(["--launcher", str(HERE / "nope.py"), *QUICK]) == 2
    assert "launcher" in capsys.readouterr().out


def test_the_scripts_are_runnable_by_path():
    import subprocess
    for script in ("e2e_mock.py", "e2e_llm_fake.py"):
        done = subprocess.run([sys.executable, str(REPO / "scripts" / script), "--help"], capture_output=True, text=True,
                              check=False)
        assert done.returncode == 0 and "--launcher" in done.stdout, done.stderr


def test_every_open_arm_decision_of_profiles_toml_has_a_test_value():
    text = (REPO / "experiments" / "profiles.toml").read_text(encoding="utf-8")
    assert e2e.documented_arm_keys(text), "profiles.toml documents no arm decision"
    assert e2e.missing_arm_values(text) == []
    assert e2e.missing_arm_values(text.replace("# katz_decay", "# brand_new_key")) == ["bazt.brand_new_key"]


def test_every_text_test_value_is_one_of_the_options_its_profiles_toml_comment_lists():
    text = (REPO / "experiments" / "profiles.toml").read_text(encoding="utf-8")
    options = e2e.documented_options(text)
    assert options["gnn.fallback_family"] == {"pi_same_task", "none"}
    assert options["analysis.rank_agreement_statistic"] == {"spearman", "kendall"}
    assert e2e.invalid_choices(text) == []
    wrong = text.replace('"pi_same_task" (the insider_pi', '"other_family" (the insider_pi')
    assert any("gnn.fallback_family" in p for p in e2e.invalid_choices(wrong))


def write_batch(units: Path, digest: str, rows: list, ids: list, done: bool = True) -> None:
    units.mkdir(parents=True, exist_ok=True)
    (units / f"batch-{digest}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    if done:
        (units / f"batch-{digest}.done").write_text(json.dumps(ids), encoding="utf-8")


def test_a_batch_rows_file_commits_all_the_units_its_done_list_names(tmp_path):
    units = tmp_path / "units"
    write_batch(units, "aa", [good_row(), good_row(arm="C")], ["B-1", "C-2"])
    write_batch(units, "bb", [good_row()], ["B-3"])
    assert e2e.check_block(tmp_path, 3, 1) == []
    assert any("2 unit files" in p or "3 unit files" in p for p in e2e.check_block(tmp_path, 4, 1))
    write_batch(units, "cc", [good_row()], ["B-4", "B-5"])
    assert any("batch-cc.jsonl: 1 rows, expected 2" in p for p in e2e.check_block(tmp_path, 5, 1))
    assert e2e.progress(units) == e2e.Progress(started=3, done=5)


def test_rows_of_a_batch_are_checked_against_their_own_arm(tmp_path):
    units = tmp_path / "units"
    write_batch(units, "aa", [good_row(arm="B"), good_row(arm="C")], ["B-1", "C-2"])
    assert e2e.check_rows(units, e2e.columns()) == []
    write_batch(units, "bb", [good_row(asr_final=1.5)], ["B-3"])
    assert any(p.startswith("batch-bb.jsonl:") and "asr_final" in p for p in e2e.check_rows(units, e2e.columns()))


def test_an_orphan_batch_rows_file_is_not_a_committed_unit(tmp_path):
    units = tmp_path / "units"
    write_batch(units, "aa", [good_row()], ["B-1"], done=False)
    assert any("not marked done" in p for p in e2e.check_block(tmp_path, 0, 1))
    assert e2e.progress(units).done == 0 and e2e.progress(units).rows == 1


# ------------------------------------------------------------------------- a resumed run equals a straight run
def test_two_runs_are_compared_episode_by_episode_ignoring_timings():
    a = [good_row(seed=1, latency=1.0), good_row(seed=2, asr_final=NAN)]
    b = [good_row(seed=2, asr_final=NAN), good_row(seed=1, latency=9.0)]
    assert e2e.compare_runs(a, b) == []
    changed = [good_row(seed=1), good_row(seed=2, asr_final=0.9)]
    (problem,) = e2e.compare_runs(a, changed)
    assert "asr_final" in problem and "seed" not in problem.split("differ")[0].split(":")[1]
    assert any("only in the resumed" in p for p in e2e.compare_runs(a + [good_row(seed=3)], b))
    assert any("only in the straight" in p for p in e2e.compare_runs(a, b + [good_row(seed=3)]))


def test_llm_smoke_run_catches_rows_that_a_resume_changes(capsys, monkeypatch):
    monkeypatch.setenv("FAKE_BUG", "resume_shift")
    code, out = run_e2e(e2e_llm_fake, capsys, "--kickoff-override", "2000-01-01T00:00:00+00:00")
    assert code == 1 and "uninterrupted" in out and "benign_delivery_rate" in out, out


def test_mock_smoke_run_catches_arms_that_disagree_on_the_ground_truth(capsys, monkeypatch):
    monkeypatch.setenv("FAKE_BUG", "crn")
    code, out = run_e2e(e2e_mock, capsys)
    assert code == 1 and "n_attackers" in out and "share its ground truth" in out, out


def test_rows_of_one_episode_without_a_conversation_field_are_compared_in_order():
    a = [good_row(seed=1, asr_final=0.1), good_row(seed=1, asr_final=0.2)]
    assert e2e.compare_runs(a, [dict(r) for r in a]) == []
    b = [good_row(seed=1, asr_final=0.1), good_row(seed=1, asr_final=0.9)]
    assert len(e2e.compare_runs(a, b)) == 1


def test_estimates_are_not_decisions_and_need_no_test_value():
    text = '[estimates]\n# B = 1\n# note = "x"\n[estimates.mock_seconds_per_episode_by_arm."12"]\n# C = 2\n[bazt]\n# katz_decay = <float>\n'
    assert e2e.documented_arm_keys(text) == ["bazt.katz_decay"]


def test_the_compute_values_are_the_reference_ones_and_the_dtype_is_the_profiles():
    """D117: every run computes in compute.dtype of profiles.toml (float32); the e2e runs that dtype, not a pinned one
    (check_gpu.py keeps its CPU float64 combination as the reference float32 is compared against)."""
    assert (e2e.ARM_TEST_VALUES["compute.device"], e2e.ARM_TEST_VALUES["compute.compile"],
            e2e.ARM_TEST_VALUES["compute.deterministic"], e2e.ARM_TEST_VALUES["compute.workers"]) == (
        "cpu", False, True, 1)
    import tomllib

    from bazt_lab.features.runner import public as rx
    profile_dtype = tomllib.loads((REPO / "experiments" / "profiles.toml").read_text(encoding="utf-8"))["compute"][
        "dtype"]
    for module, extra in ((e2e_mock, ["--quick", "--workers", "2"]), (e2e_llm_fake, ["--with-gnn"])):
        plan = module.build_plan(args_of(module, *extra))
        assert "compute.dtype" not in plan.sets, module.__name__
        settings = rx.load_settings(overrides=e2e.override_items(plan.sets))
        assert settings["compute"]["dtype"] == profile_dtype, module.__name__


def test_every_repo_global_path_setting_is_moved_into_the_temporary_directory(tmp_path):
    sets = e2e.isolated_paths(tmp_path)
    assert set(sets) == {"gnn_train.logs_dir", "gnn.artifacts_dir"}
    assert all(Path(v).is_relative_to(tmp_path) for v in sets.values())
    text = (REPO / "experiments" / "profiles.toml").read_text(encoding="utf-8")
    import re
    in_profile = set(re.findall(r'^(\w+_dir)\s*=\s*"results/', text, re.M))
    assert in_profile == {"logs_dir", "artifacts_dir"}, "a new results path in profiles.toml needs isolating in the e2e"


def test_a_worker_shard_folder_below_units_is_read_like_the_flat_layout(tmp_path):
    units = tmp_path / "units"
    write_batch(units / "w0", "aa", [good_row(), good_row(arm="C")], ["B-1", "C-2"])
    write_batch(units / "w1", "bb", [good_row()], ["B-3"])
    assert e2e.check_block(tmp_path, 3, 1) == []
    assert e2e.progress(units) == e2e.Progress(started=2, done=3)
    assert set(e2e.snapshot(units)) == {"w0/batch-aa.jsonl", "w1/batch-bb.jsonl"}
    assert len(e2e.all_rows(units)) == 3 and e2e.check_rows(units, e2e.columns()) == []


def test_llm_containment_reads_the_headline_block_where_the_reference_arm_runs():
    assert e2e.ANALYSIS_TEST_VALUES["analysis.blocks"]["containment"]["llm"] == ["llm_capped", "llm_headline"]


# ------------------------------------------------------------------- committed rows (core.committed_row_files)
def test_rows_are_read_from_committed_files_only_in_units_and_every_shard(tmp_path):
    units = tmp_path / "units"
    write_batch(units / "w0", "aa", [good_row(), good_row(arm="C")], ["B-1", "C-2"])
    write_batch(units / "w1", "bb", [good_row()], ["B-3"])
    write_batch(units / "w2", "cc", [good_row(asr_final=1.5)], ["B-4"], done=False)    # an orphan in a stale shard
    write_unit(units, "B-5", [good_row(asr_final=1.5)], done=False)                     # a unit still running
    assert len(e2e.all_rows(units)) == 3
    assert e2e.check_rows(units, e2e.columns()) == []
    assert e2e.committed_rows(tmp_path) == 3
    assert e2e.committed_rows(tmp_path / "no_such_block") == 0


def test_left_over_temporary_and_state_files_are_found_in_every_shard_and_listed_once(tmp_path):
    units = tmp_path / "units"
    write_batch(units / "w0", "aa", [good_row()], ["B-1"])
    (units / "w0" / "batch-bb.jsonl.tmp").write_text("x")
    (tmp_path / "state" / "w1").mkdir(parents=True)
    (tmp_path / "state" / "w1" / "B-1.pt").write_bytes(b"x")
    (tmp_path / "state" / "w1" / "B-1.pt.tmp").write_bytes(b"x")
    (tmp_path / "thresholds.json.tmp").write_text("x")
    problems = e2e.check_block(tmp_path, 1, 1)
    for name in ("units/w0/batch-bb.jsonl.tmp", "state/w1/B-1.pt", "state/w1/B-1.pt.tmp", "thresholds.json.tmp"):
        assert problems.count(f"left over: {name}") == 1, (name, problems)
    assert len(problems) == 4, problems


def test_a_temporary_file_the_kill_left_is_tolerated_but_one_made_after_it_is_not(tmp_path):
    """The store writes a batch to <name>.jsonl.tmp and renames it; a kill in between leaves the .tmp, which nothing
    reads, and a resume that regroups its batches never rewrites it. Only the .tmp files that existed right after the
    kill are crash artifacts; a new one after the last launch is a real leftover."""
    units = tmp_path / "units"
    write_batch(units / "w1", "aa", [good_row()], ["B-1"])
    (units / "w1" / "batch-killed.jsonl.tmp").write_text("x")
    assert e2e.tmp_files(tmp_path) == {"units/w1/batch-killed.jsonl.tmp"}
    assert e2e.check_block(tmp_path, 1, 1, stale_tmp=e2e.tmp_files(tmp_path)) == []
    (units / "w1" / "batch-new.jsonl.tmp").write_text("x")
    problems = e2e.check_block(tmp_path, 1, 1, stale_tmp={"units/w1/batch-killed.jsonl.tmp"})
    assert problems == ["left over: units/w1/batch-new.jsonl.tmp"], problems


def test_a_restart_may_remove_a_batch_rows_file_that_has_no_done_list():
    """CheckpointStore.append_batch removes the orphans of its write folder before writing: their units are not done
    and run again in another batch. A unit's part-written file must still be continued, never dropped."""
    before = {"w0/batch-aa.jsonl": b'{"a": 1}\n', "w0/batch-bb.jsonl": b'{"b": 1}\n', "B-1.jsonl": b'{"c": 1}\n'}
    done_before = {"batch-aa"}
    after = {"w0/batch-aa.jsonl": b'{"a": 1}\n', "w0/batch-cc.jsonl": b'{"b": 1}\n',
             "B-1.jsonl": b'{"c": 1}\n{"c": 2}\n'}
    assert e2e.check_resume(before, after, done_before) == []
    assert any("B-1.jsonl: lost" in p for p in e2e.check_resume(
        before, {k: v for k, v in after.items() if k != "B-1.jsonl"}, done_before))
    assert any("batch-aa.jsonl: lost" in p for p in e2e.check_resume(
        before, {k: v for k, v in after.items() if k != "w0/batch-aa.jsonl"}, done_before))


# --------------------------------------------------------------------- the analysis read what the run wrote
def platform_meta(rows: int, blocks: list, missing: list = ()) -> dict:
    return {"rows": rows, "blocks": list(blocks), "blocks_missing": list(missing), "uncommitted_files": 0}


def shard_rows(results: Path) -> None:
    """mock_frontier with three committed rows in two worker shards and an orphan in a third (not a result)."""
    units = results / "mock_frontier" / "units"
    write_batch(units / "w0", "aa", [good_row(), good_row(arm="C")], ["B-1", "C-2"])
    write_batch(units / "w1", "bb", [good_row()], ["B-3"])
    write_batch(units / "w2", "cc", [good_row()], ["B-4"], done=False)


def test_the_analysis_must_have_read_every_committed_row_of_the_blocks_it_reads(tmp_path):
    shard_rows(tmp_path)
    llm = platform_meta(0, ["llm_capped"], ["llm_capped"])
    good = {"meta": {"platforms": {"mock": platform_meta(3, ["mock_frontier"]), "llm": llm}}}
    assert e2e.check_analysis(good, tmp_path, ["mock_frontier"]) == []
    blind = {"meta": {"platforms": {"mock": platform_meta(0, ["mock_frontier"], ["mock_frontier"]), "llm": llm}},
             "hypotheses": {"H-budget": {"mock": {"status": "no_data", "blocks_missing": ["mock_frontier"]}}}}
    problems = e2e.check_analysis(blind, tmp_path, ["mock_frontier"])
    assert any(p.startswith("mock:") and "read 0 rows" in p and "hold 3" in p for p in problems), problems
    assert any("mock_frontier" in p and "blocks_missing" in p for p in problems), problems
    assert any("H-budget" in p and "no_data" in p for p in problems), problems
    with_orphan = {"meta": {"platforms": {"mock": platform_meta(4, ["mock_frontier"]), "llm": llm}}}
    assert any("read 4 rows" in p for p in e2e.check_analysis(with_orphan, tmp_path, ["mock_frontier"]))
    assert any("llm_headline" in p and "reads" in p for p in e2e.check_analysis(good, tmp_path, ["llm_headline"]))
    assert any("meta.platforms" in p for p in e2e.check_analysis({"rows": 3}, tmp_path, ["mock_frontier"]))


def shard_session(tmp_path) -> e2e.Session:
    plan = e2e.Plan("t", {"runtime.n_insiders": 3}, [], [e2e.Evaluation("mock_frontier", 3, 1)],
                    analysis_sets={"analysis.blocks": e2e.ANALYSIS_TEST_VALUES["analysis.blocks"]})
    return e2e.Session(plan, Path("launcher.py"), tmp_path / "results", "sprint", {}, 1, tmp_path)


def test_the_analysis_stage_fails_an_analysis_that_reads_no_row_of_the_worker_shards(tmp_path, capsys):
    session = shard_session(tmp_path)
    shard_rows(session.results_root / "sprint")
    report = e2e.Report()
    e2e.stage_analysis(session, report, "fake_analysis:analyze")
    assert report.failures == []
    capsys.readouterr()
    report = e2e.Report()
    e2e.stage_analysis(session, report, "fake_analysis:top_folder_only")
    out = capsys.readouterr().out
    assert len(report.failures) == 1 and "committed row" in report.failures[0], report.failures
    assert "mock: the analysis read 0 rows, the committed rows files of mock_frontier hold 3" in out, out
    assert "mock_frontier: in mock.blocks_missing" in out, out


def test_the_analysis_stage_fails_a_full_report_of_an_empty_results_directory(tmp_path, capsys):
    session = shard_session(tmp_path)
    (session.results_root / "sprint").mkdir(parents=True)
    report = e2e.Report()
    e2e.stage_analysis(session, report, "fake_analysis:top_folder_only")
    out = capsys.readouterr().out
    assert len(report.failures) == 1 and "mock_frontier: in mock.blocks_missing" in out, out


# ------------------------------------------------------------------- a kill that cuts a unit in the middle
def test_a_part_written_unit_holds_some_but_not_all_of_its_rows_and_no_done_marker(tmp_path):
    units = tmp_path / "units"
    write_unit(units, "A-1", [good_row()], done=False)                      # 1 of 2 rows: cut in the middle
    write_unit(units, "A-2", [good_row(), good_row()], done=False)          # every row, done marker not yet written
    write_unit(units, "A-3", [good_row()])                                  # done
    write_unit(units, "A-4", [], done=False, tail='{"arm": "ha')            # a torn first line: no row yet
    write_batch(units, "aa", [good_row()], ["B-1", "B-2"], done=False)      # an orphan batch is not a unit
    assert e2e.part_written(units, 2) == ["A-1.jsonl"]
    assert e2e.part_written(units, 1) == []
    assert e2e.part_written(tmp_path / "no_such_units", 2) == []


def test_a_kill_that_cuts_no_unit_is_followed_by_another_launch_and_kill(tmp_path, monkeypatch):
    session = analysis_session(tmp_path)
    units = session.block_dir("llm_headline") / "units"
    launches = []

    def run_and_kill(block, n_units, rows_per_unit, cut_at=3):
        launches.append(block)
        n = len(launches)
        write_unit(units, f"A-{n}", [good_row(), good_row()])                   # whole units only ...
        if n == cut_at:
            write_unit(units, f"B-{n}", [good_row()], done=False)               # ... until this kill cuts one
        return e2e.Progress(started=n, done=n), True

    monkeypatch.setattr(session, "run_and_kill", run_and_kill)
    found = session.interrupt(e2e.Evaluation("llm_headline", 10, 2, interrupt=True))
    assert (found.killed, found.launches, found.cut) == (True, 3, ("B-3.jsonl",))
    launches.clear()
    import shutil
    shutil.rmtree(units)
    monkeypatch.setattr(session, "run_and_kill", lambda *a: run_and_kill(*a, cut_at=99))
    found = session.interrupt(e2e.Evaluation("llm_headline", 10, 2, interrupt=True))
    assert (found.launches, found.cut) == (e2e.KILL_ATTEMPTS, ())
    launches.clear()
    found = session.interrupt(e2e.Evaluation("mock_frontier", 10, 1, interrupt=True))   # whole batches: one kill
    assert found.launches == 1


def test_the_stage_fails_when_no_kill_left_a_unit_part_written(tmp_path, monkeypatch):
    session = analysis_session(tmp_path)
    monkeypatch.setattr(e2e.Session, "interrupt",
                        lambda self, ev: e2e.Interruption(e2e.Progress(1, 1), True, e2e.KILL_ATTEMPTS, ()))
    monkeypatch.setattr(e2e.Session, "run_block", lambda self, block: None)
    report = e2e.Report()
    e2e.stage_evaluation(session, report, e2e.Evaluation("llm_headline", 2, 2, interrupt=True, metrics=False))
    assert any("resume path not exercised" in f for f in report.failures), report.failures


def test_a_frozen_launcher_writes_nothing_until_it_is_let_go(tmp_path):
    import subprocess
    import time
    out = tmp_path / "ticks"
    code = ("import sys, time\nwith open(sys.argv[1], 'a') as f:\n    while True:\n"
            "        f.write('x\\n'); f.flush(); time.sleep(0.005)\n")
    proc = subprocess.Popen([sys.executable, "-c", code, str(out)], start_new_session=True)

    def grows(seconds: float) -> bool:
        start = out.stat().st_size if out.exists() else 0
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if out.exists() and out.stat().st_size > start:
                return True
            time.sleep(0.01)
        return False

    try:
        assert grows(30)
        assert e2e.freeze(proc)
        time.sleep(0.05)
        assert not grows(0.4)
        assert e2e.thaw(proc)
        assert grows(10)
    finally:
        e2e.kill_tree(proc)
        proc.wait()


def test_on_posix_the_kill_and_the_freeze_take_the_launchers_whole_process_group(monkeypatch):
    calls = []
    monkeypatch.setattr(e2e, "WINDOWS", False)
    monkeypatch.setattr(e2e.os, "killpg", lambda pid, sig: calls.append((pid, sig)), raising=False)
    for name, number in (("SIGKILL", 9), ("SIGSTOP", 19), ("SIGCONT", 18)):
        monkeypatch.setattr(e2e.signal, name, number, raising=False)

    class Proc:
        pid = 4321

        def kill(self):
            calls.append("kill")

    e2e.kill_tree(Proc())
    assert calls[0] == (4321, 9)
    calls.clear()
    assert e2e.freeze(Proc()) and e2e.thaw(Proc())
    assert calls == [(4321, 19), (4321, 18)]


def test_the_launcher_starts_in_its_own_process_group(tmp_path, monkeypatch):
    seen = {}

    class Finished:
        pid, returncode = 1, 0

        def __init__(self, argv, **kwargs):
            seen.update(kwargs)

        def poll(self):
            return 0

        def wait(self):
            return 0

    monkeypatch.setattr(e2e.subprocess, "Popen", Finished)
    session = analysis_session(tmp_path)
    assert session.run_and_kill("mock_frontier", 4, 1) == (e2e.Progress(0, 0), False)
    assert seen.get("start_new_session") is True


def test_a_block_that_finishes_without_a_cut_is_started_over_on_a_fresh_block_folder(tmp_path, monkeypatch):
    """Under load the remaining units can all finish inside the grace period of a later attempt. The attempts so far
    are then useless (nothing left to cut): the block is wiped and the kill starts again, within KILL_ATTEMPTS."""
    session = shard_session(tmp_path)
    evaluation = e2e.Evaluation("mock_frontier", 3, 2, interrupt=True)
    units = session.block_dir("mock_frontier") / "units"
    seen_before = []
    script = iter(["cut_missed", "finished", "cut"])

    def fake(self, block, n_units, rows_per_unit):
        seen_before.append(sorted(p.name for p in units.rglob("*")) if units.is_dir() else [])
        step = next(script)
        if step == "cut_missed":                         # killed, but every unit was complete
            for i in range(3):
                write_unit(units, f"B-{i}", [good_row(), good_row()])
            return e2e.progress(units), True
        if step == "finished":                           # not killed: the rest finished within the grace period
            return e2e.progress(units), False
        write_unit(units, "B-0", [good_row()], done=False)
        return e2e.progress(units), True

    monkeypatch.setattr(e2e.Session, "run_and_kill", fake)
    kill = session.interrupt(evaluation)
    assert kill.killed and kill.cut == ("B-0.jsonl",), kill
    assert kill.launches == 1, kill                     # launches counted since the last wipe: the manifests on disk
    assert seen_before[-1] == [], seen_before           # the last attempt ran on a wiped block folder
