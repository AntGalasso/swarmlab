"""WandbTracker against a recording stand-in for the wandb module (fakes.py): what it asks of wandb, what it refuses,
what it reports. The real library, offline, is exercised in test_offline_run.py."""
from __future__ import annotations

import traceback

import numpy as np
import pytest

from bazt_lab.core import public as core
from bazt_lab.features.tracking.adapters import env, wandb_tracker
from bazt_lab.features.tracking.adapters.wandb_tracker import STEP_KEY, WandbTracker
from bazt_lab.features.tracking.private.clean import clean_config
from bazt_lab.features.tracking.private.errors import TrackingError
from bazt_lab.features.tracking.tests.fakes import FakeWandb

PAST, FUTURE = "2000-01-01T00:00:00+00:00", "2999-01-01T00:00:00+00:00"
NOT_A_KEY = "not-a-real-key-for-tests"


def cfg(mode="offline", **tracking):
    tracking = {"flush_rows": 1, **tracking}
    return {"profile": "sprint", "tracking": {"project": "bazt-experiments", "mode": mode, **tracking},
            "llm": {"agents": 12}, "manifest": {"git_commit": "abc123"}}


@pytest.fixture
def wb(monkeypatch):
    fake = FakeWandb()
    monkeypatch.setattr(wandb_tracker, "_wandb", lambda: fake)
    return fake


def started(wb, config=None, resume_id=None, block="mock_main"):
    tracker = WandbTracker()
    tracker.start(block, config or cfg(), resume_id)
    return tracker, wb.runs[-1]


# ----------------------------------------------------------------------------------------------------- start
def test_the_tracker_satisfies_the_core_contract_and_builds_without_arguments(monkeypatch):
    monkeypatch.setattr(wandb_tracker, "_wandb", lambda: pytest.fail("building a tracker must not import wandb"))
    assert isinstance(WandbTracker(), core.Tracker)


def test_one_run_per_block_grouped_by_profile_with_the_block_as_job_type(wb):
    tracker = WandbTracker()
    run_id = tracker.start("mock_main", cfg(), None)
    assert len(wb.inits) == 1
    kwargs = wb.inits[0]
    assert (kwargs["project"], kwargs["group"], kwargs["job_type"]) == ("bazt-experiments", "sprint", "mock_main")
    assert kwargs["mode"] == "offline" and kwargs["id"] is None and kwargs["resume"] == "allow"
    assert not kwargs.get("entity")
    assert run_id == wb.runs[0].id


def test_a_saved_id_is_passed_back_and_returned_so_the_run_resumes(wb):
    tracker = WandbTracker()
    assert tracker.start("mock_main", cfg(), "abc12345") == "abc12345"
    assert wb.inits[0]["id"] == "abc12345" and wb.inits[0]["resume"] == "allow"


def test_the_entity_is_passed_when_set(wb):
    started(wb, cfg(entity="vittoria"))
    assert wb.inits[0]["entity"] == "vittoria"


def test_the_config_is_the_resolved_settings_without_secrets_or_periods(wb):
    config = cfg()
    config["llm"]["hf_token"] = "x y"
    config["estimates"] = {"price": {"deepseek/deepseek-v4.1-flash": [0.1, 0.5]}}
    started(wb, config)
    sent = wb.inits[0]["config"]
    assert sent == clean_config(config)
    assert "hf_token" not in sent["llm"] and "deepseek/deepseek-v4_1-flash" in sent["estimates"]["price"]
    assert sent["manifest"] == {"git_commit": "abc123"}


def test_the_runner_step_becomes_the_x_axis_of_every_metric(wb):
    _, run = started(wb)
    assert run.metrics == [(STEP_KEY, None), ("*", STEP_KEY)]


def test_no_code_and_no_git_state_leave_the_machine(wb):
    started(wb)
    kwargs = wb.inits[0]
    assert kwargs["save_code"] is False and kwargs["settings"]["disable_git"] is True


def test_a_wandb_directory_from_settings_is_created_and_used(wb, tmp_path):
    target = tmp_path / "runs" / "wandb"
    started(wb, cfg(dir=str(target)))
    assert wb.inits[0]["dir"] == str(target) and target.is_dir()


def test_start_twice_is_refused_instead_of_returning_the_same_run(wb):
    tracker, _ = started(wb)
    with pytest.raises(TrackingError, match="already"):
        tracker.start("other", cfg(), None)
    assert len(wb.inits) == 1


def test_a_missing_setting_raises_core_unset_naming_the_key(wb):
    config = cfg()
    del config["tracking"]["project"]
    with pytest.raises(core.Unset) as error:
        WandbTracker().start("mock_main", config, None)
    assert error.value.args[0] == "tracking.project" and not wb.inits


def test_the_profile_is_required_for_the_group(wb):
    config = cfg()
    del config["profile"]
    with pytest.raises(core.Unset) as error:
        WandbTracker().start("mock_main", config, None)
    assert error.value.args[0] == "profile"


# --------------------------------------------------------------------------------- modes, key, kickoff
def test_online_without_the_key_is_a_clear_error_before_wandb_is_touched(wb, monkeypatch):
    monkeypatch.setenv("SWARMLAB_KICKOFF_UTC", PAST)
    with pytest.raises(TrackingError, match="WANDB_API_KEY"):
        WandbTracker().start("mock_main", cfg("online", entity="team"), None)
    assert not wb.inits


def test_online_without_an_entity_names_the_setting(wb, monkeypatch):
    monkeypatch.setenv("SWARMLAB_KICKOFF_UTC", PAST)
    monkeypatch.setenv("WANDB_API_KEY", NOT_A_KEY)
    with pytest.raises(TrackingError, match=r"tracking\.entity"):
        WandbTracker().start("mock_main", cfg("online"), None)
    assert not wb.inits


def test_online_after_the_kickoff_with_key_and_entity_goes_online(wb, monkeypatch):
    monkeypatch.setenv("SWARMLAB_KICKOFF_UTC", PAST)
    monkeypatch.setenv("WANDB_API_KEY", NOT_A_KEY)
    tracker, _ = started(wb, cfg("online", entity="team"))
    assert wb.inits[0]["mode"] == "online" and tracker.mode == "online"


def test_online_before_the_kickoff_runs_offline_needs_no_key_and_says_so(wb, monkeypatch, capsys):
    monkeypatch.setenv("SWARMLAB_KICKOFF_UTC", FUTURE)
    tracker, _ = started(wb, cfg("online"))
    assert wb.inits[0]["mode"] == "offline" and tracker.mode == "offline"
    err = capsys.readouterr().err
    assert "offline" in err and "kickoff" in err


def test_online_before_the_kickoff_is_honoured_when_the_setting_allows_it(wb, monkeypatch):
    monkeypatch.setenv("SWARMLAB_KICKOFF_UTC", FUTURE)
    monkeypatch.setenv("WANDB_API_KEY", NOT_A_KEY)
    started(wb, cfg("online", entity="team", allow_online_before_kickoff=True))
    assert wb.inits[0]["mode"] == "online"


def test_the_key_comes_from_the_dotenv_file_when_the_process_has_none(wb, monkeypatch, tmp_path):
    monkeypatch.setenv("SWARMLAB_KICKOFF_UTC", PAST)
    dotenv = tmp_path / "present.env"
    dotenv.write_text(f"WANDB_API_KEY={NOT_A_KEY}\n", encoding="utf-8")
    monkeypatch.setattr(env, "ENV_PATH", dotenv)
    started(wb, cfg("online", entity="team"))
    assert wb.inits[0]["mode"] == "online"


def test_a_kickoff_time_that_is_not_iso_is_an_error_not_a_guess(wb, monkeypatch):
    monkeypatch.setenv("SWARMLAB_KICKOFF_UTC", "tomorrow")
    with pytest.raises(TrackingError, match="SWARMLAB_KICKOFF_UTC"):
        WandbTracker().start("mock_main", cfg("online", entity="team"), None)


def test_disabled_mode_starts_a_noop_run_and_returns_the_saved_id(wb):
    tracker = WandbTracker()
    assert tracker.start("mock_main", cfg("disabled"), "abc12345") == "abc12345"
    assert wb.inits[0]["mode"] == "disabled" and tracker.mode == "disabled"


@pytest.mark.parametrize("mode, extra", [("shared", {}), ("online", {"entity": "team"}), ("online", {}),
                                         ("offline", {"flush_rows": 0}), ("offline", {"flush_rows": True})])
def test_constructor_settings_refuse_before_anything_is_written(wb, monkeypatch, tmp_path, mode, extra):
    monkeypatch.setenv("SWARMLAB_KICKOFF_UTC", PAST)          # online: no key, or no entity
    target = tmp_path / "never"
    with pytest.raises(TrackingError):
        WandbTracker(cfg(mode, dir=str(target), **extra))
    assert not wb.inits and not target.exists()


def test_the_flush_size_is_required_and_names_the_key(wb):
    config = cfg()
    del config["tracking"]["flush_rows"]
    with pytest.raises(core.Unset) as error:
        WandbTracker(config)
    assert error.value.args[0] == "tracking.flush_rows"


def test_constructor_settings_are_checked_before_any_start(wb, monkeypatch):
    monkeypatch.setenv("SWARMLAB_KICKOFF_UTC", PAST)
    with pytest.raises(TrackingError, match="WANDB_API_KEY"):
        WandbTracker(cfg("online", entity="team"))
    with pytest.raises(core.Unset):
        WandbTracker({"tracking": {"mode": "offline"}})
    assert not wb.inits
    tracker = WandbTracker(cfg())
    tracker.start("mock_main", cfg(), None)
    assert wb.inits[0]["project"] == "bazt-experiments"


# ---------------------------------------------------------------------------- start failures are never hidden
def test_a_wandb_failure_at_start_is_raised_as_tracking_error_and_chained(wb):
    wb.fail["init"] = ConnectionError("no route to host")
    with pytest.raises(TrackingError, match="no route to host") as error:
        WandbTracker().start("mock_main", cfg(), None)
    assert isinstance(error.value.__cause__, ConnectionError)


def test_the_api_key_never_appears_in_an_error(wb, monkeypatch):
    monkeypatch.setenv("SWARMLAB_KICKOFF_UTC", PAST)
    monkeypatch.setenv("WANDB_API_KEY", NOT_A_KEY)
    wb.fail["init"] = RuntimeError(f"rejected {NOT_A_KEY} by server")
    with pytest.raises(TrackingError) as error:
        WandbTracker().start("mock_main", cfg("online", entity="team"), None)
    assert NOT_A_KEY not in str(error.value) and "rejected" in str(error.value)


def test_the_api_key_does_not_reach_the_traceback_through_the_chained_error(wb, monkeypatch):
    monkeypatch.setenv("SWARMLAB_KICKOFF_UTC", PAST)
    monkeypatch.setenv("WANDB_API_KEY", NOT_A_KEY)
    wb.fail["init"] = RuntimeError(f"rejected {NOT_A_KEY} by server")
    with pytest.raises(TrackingError) as error:
        WandbTracker().start("mock_main", cfg("online", entity="team"), None)
    shown = "".join(traceback.format_exception(error.value))          # what an uncaught error prints
    assert NOT_A_KEY not in shown and "rejected" in shown


def test_the_api_key_in_a_chained_cause_does_not_reach_the_traceback(wb, monkeypatch):
    monkeypatch.setenv("SWARMLAB_KICKOFF_UTC", PAST)
    monkeypatch.setenv("WANDB_API_KEY", NOT_A_KEY)

    def refuse(**kwargs):
        try:
            raise ValueError(f"HTTP 401 for Authorization: Basic {NOT_A_KEY}")
        except ValueError as inner:
            raise RuntimeError("could not authenticate") from inner

    wb.init = refuse
    with pytest.raises(TrackingError, match="could not authenticate") as error:
        WandbTracker().start("mock_main", cfg("online", entity="team"), None)
    assert NOT_A_KEY not in "".join(traceback.format_exception(error.value))


def test_a_cause_without_the_key_stays_chained(wb):
    wb.fail["init"] = ConnectionError("no route")
    with pytest.raises(TrackingError) as error:
        WandbTracker().start("mock_main", cfg(), None)
    assert isinstance(error.value.__cause__, ConnectionError)


def test_a_run_that_cannot_be_set_up_is_closed_and_the_start_fails(wb):
    wb.fail["define_metric"] = RuntimeError("bad metric")
    tracker = WandbTracker()
    with pytest.raises(TrackingError, match="bad metric"):
        tracker.start("mock_main", cfg(), None)
    assert wb.runs[0].finished == 1
    del wb.fail["define_metric"]
    tracker.start("mock_main", cfg(), None)                   # the failed start left no half-started run behind
    assert len(wb.inits) == 2


# ----------------------------------------------------------------------------------------------------- log
def test_log_sends_the_metrics_with_the_runner_step_as_a_metric(wb):
    tracker, run = started(wb)
    tracker.log({"runner/units_done": 512, "runner/batches": 1}, step=np.int64(512))
    assert run.logs == [{"runner/units_done": 512, "runner/batches": 1, STEP_KEY: 512}]
    assert type(run.logs[0][STEP_KEY]) is int


def test_a_failing_log_is_reported_once_and_the_run_goes_on(wb, capsys):
    tracker, run = started(wb)
    wb.fail["log"] = ConnectionError("network down")
    for step in (1, 2, 3):
        tracker.log({"x": 1}, step=step)                  # must not raise
    err = capsys.readouterr().err
    assert err.count("network down") == 1 and "log" in err
    assert tracker.failures == {"log": 3}
    wb.fail.clear()
    tracker.log({"x": 1}, step=4)
    assert run.logs == [{"x": 1, STEP_KEY: 4}]


def test_using_the_tracker_before_start_is_reported_not_raised(wb, capsys):
    tracker = WandbTracker()
    tracker.log({"x": 1}, step=1)
    tracker.log_rows("t", [{"a": 1}])
    tracker.artifact("nowhere", "n", "k")
    assert "not started" in capsys.readouterr().err
    assert tracker.failures == {"log": 1, "log_rows[t]": 1, "artifact": 1}


# ------------------------------------------------------------------------------------------------ log_rows
def test_rows_are_logged_as_one_table_part_per_flush_with_the_first_columns(wb):
    tracker, run = started(wb)
    tracker.log_rows("episodes", [{"unit": "u1", "asr": 0.5}, {"unit": "u2", "asr": float("nan"), "extra": [1, 2]}])
    tracker.log_rows("episodes", [{"unit": "u3", "asr": np.float64(0.25), "extra": None}])
    assert [t.columns for t in wb.tables] == [["unit", "asr", "extra"]] * 2
    assert wb.tables[0].allow_mixed_types is True
    first = run.logs[0]["episodes"]
    assert first[0] == "table" and first[1][0] == ["u1", 0.5, None] and first[1][1][2] == "[1, 2]"
    assert run.logs[1]["episodes_part1"] == ("table", [["u3", 0.25, None]])
    assert [list(e) for e in run.logs] == [["episodes"], ["episodes_part1"]]


def test_rows_wait_in_memory_until_the_flush_size_is_reached(wb):
    tracker, run = started(wb, cfg(flush_rows=4))
    tracker.log_rows("t", [{"a": 1}, {"a": 2}, {"a": 3}])
    assert run.logs == [] and wb.tables == []
    tracker.log_rows("t", [{"a": 4}, {"a": 5}])
    assert [e["t"][1] for e in run.logs] == [[[1], [2], [3], [4], [5]]]


def test_finish_flushes_the_rows_still_waiting_before_it_ends_the_run(wb):
    tracker, run = started(wb, cfg(flush_rows=100))
    tracker.log_rows("a", [{"x": 1}])
    tracker.log_rows("b", [{"y": 2}, {"y": 3}])
    tracker.finish()
    assert {k: v[1] for e in run.logs for k, v in e.items()} == {"a": [[1]], "b": [[2], [3]]}
    assert run.finished == 1


def test_many_rows_cost_wandb_one_table_per_flush_not_one_per_call(wb):
    tracker, run = started(wb, cfg(flush_rows=360))
    for i in range(1000):
        tracker.log_rows("t", [{"i": i}])
    tracker.finish()
    assert len(wb.tables) == 3 and sum(len(e[k][1]) for e in run.logs for k in e) == 1000


def test_a_row_may_lack_a_column_and_gets_none(wb):
    tracker, run = started(wb)
    tracker.log_rows("t", [{"a": 1, "b": 2}])
    tracker.log_rows("t", [{"a": 3}])
    assert run.logs[1]["t_part1"][1] == [[3, None]]


def test_a_column_the_table_does_not_have_is_dropped_and_reported_once(wb, capsys):
    tracker, run = started(wb)
    tracker.log_rows("t", [{"a": 1}])
    tracker.log_rows("t", [{"a": 2, "late": 9}])
    tracker.log_rows("t", [{"a": 3, "late": 9}])
    err = capsys.readouterr().err
    assert err.count("late") == 1 and "dropped" in err
    assert [next(iter(e.values()))[1] for e in run.logs] == [[[1]], [[2]], [[3]]]
    assert tracker.failures == {"log_rows[t]": 2}


def test_two_table_names_are_two_tables(wb):
    tracker, run = started(wb)
    tracker.log_rows("a", [{"x": 1}])
    tracker.log_rows("b", [{"y": 2}])
    assert [t.columns for t in wb.tables] == [["x"], ["y"]]


def test_no_rows_logs_nothing(wb):
    tracker, run = started(wb)
    tracker.log_rows("t", [])
    assert run.logs == [] and wb.tables == []


def test_a_flush_larger_than_the_wandb_row_limit_is_split_into_numbered_parts(monkeypatch):
    fake = FakeWandb(max_rows=3)
    monkeypatch.setattr(wandb_tracker, "_wandb", lambda: fake)
    tracker, run = started(fake, cfg(flush_rows=7))
    tracker.log_rows("t", [{"i": i} for i in range(5)])
    tracker.log_rows("t", [{"i": 5}, {"i": 6}])
    assert [list(e) for e in run.logs] == [["t"], ["t_part1"], ["t_part2"]]
    assert [e[next(iter(e))][1] for e in run.logs] == [[[0], [1], [2]], [[3], [4], [5]], [[6]]]


def test_a_failing_flush_is_reported_and_does_not_stop_the_experiment(wb, capsys):
    tracker, run = started(wb)
    wb.fail["log"] = TypeError("incompatible types")
    tracker.log_rows("t", [{"a": 1}])
    assert "incompatible types" in capsys.readouterr().err and tracker.failures == {"log_rows[t]": 1}
    wb.fail.clear()
    tracker.log_rows("t", [{"a": 2}])
    assert next(iter(run.logs[0].values()))[1] == [[2]]          # the failed rows are not retried in a loop


def test_disabled_runs_build_no_tables(wb):
    tracker, run = started(wb, cfg("disabled"))
    tracker.log_rows("t", [{"a": 1}])
    tracker.finish()
    assert wb.tables == [] and run.logs == []


# ------------------------------------------------------------------------------------------------ artifact
def test_a_file_becomes_an_artifact_with_a_name_wandb_accepts(wb, tmp_path):
    path = tmp_path / "manifest-1.json"
    path.write_text("{}", encoding="utf-8")
    tracker, run = started(wb)
    tracker.artifact(str(path), "mock main/manifest", "manifest")
    (art,) = run.artifacts
    assert (art.name, art.type, art.files, art.dirs) == ("mock_main_manifest", "manifest", [str(path)], [])


def test_a_directory_becomes_an_artifact_too(wb, tmp_path):
    tracker, run = started(wb)
    tracker.artifact(str(tmp_path), "results", "results")
    assert run.artifacts[0].dirs == [str(tmp_path)] and run.artifacts[0].files == []


def test_an_artifact_that_cannot_be_logged_is_reported_and_the_run_goes_on(wb, tmp_path, capsys):
    tracker, run = started(wb)
    tracker.artifact(str(tmp_path / "missing.json"), "m", "manifest")
    assert "missing.json" in capsys.readouterr().err and tracker.failures == {"artifact": 1}
    assert run.artifacts == []


def test_disabled_runs_build_no_artifacts(wb, tmp_path):
    path = tmp_path / "m.json"
    path.write_text("{}", encoding="utf-8")
    tracker, run = started(wb, cfg("disabled"))
    tracker.artifact(str(path), "m", "manifest")
    assert run.artifacts == []


# ------------------------------------------------------------------------------------------------- finish
def test_finish_ends_the_run_once(wb):
    tracker, run = started(wb)
    tracker.finish()
    tracker.finish()
    assert run.finished == 1


def test_finish_without_a_start_does_nothing(wb):
    WandbTracker().finish()


def test_finish_prints_how_many_operations_failed(wb, capsys):
    tracker, run = started(wb)
    wb.fail["log"] = ConnectionError("down")
    tracker.log({"x": 1}, step=1)
    tracker.log({"x": 1}, step=2)
    capsys.readouterr()
    tracker.finish()
    err = capsys.readouterr().err
    assert "log: 2" in err and "incomplete" in err


def test_a_clean_run_finishes_silently(wb, capsys):
    tracker, run = started(wb)
    tracker.log({"x": 1}, step=1)
    capsys.readouterr()
    tracker.finish()
    assert capsys.readouterr().err == ""


def test_a_failing_finish_is_reported_not_raised(wb, capsys):
    tracker, run = started(wb)
    wb.fail["finish"] = ConnectionError("upload failed")
    tracker.finish()
    assert "upload failed" in capsys.readouterr().err and tracker.failures == {"finish": 1}


def test_after_finish_the_tracker_can_start_a_new_run(wb):
    tracker, _ = started(wb)
    tracker.finish()
    tracker.start("calibration", cfg(), None)
    assert len(wb.inits) == 2
