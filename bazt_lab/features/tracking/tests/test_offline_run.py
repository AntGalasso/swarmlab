"""The real wandb library, with no network: an offline run in a temp directory read back from its own log file, and a
disabled run. One module-scoped session does the slow part (starting wandb-core) once; each test reads its result.

The environment says WANDB_MODE=disabled while the tracker asks for offline: the explicit mode must win, otherwise no
offline run would exist. The settings ask for "online" before the kickoff: it must end up offline without a key.
"""
from __future__ import annotations

import json
from collections import namedtuple

import pytest

from bazt_lab.features.tracking.adapters import env, offline_log
from bazt_lab.features.tracking.adapters.wandb_tracker import STEP_KEY, WandbTracker

FUTURE = "2999-01-01T00:00:00+00:00"
Session = namedtuple("Session", "run_id dirs disabled_id disabled_dirs disabled_resumed root")


def settings(root, mode):
    return {"profile": "sprint", "tracking": {"project": "bazt-test", "mode": mode, "dir": str(root), "flush_rows": 2},
            "llm": {"agents": 12, "hf_token": "not a credential"},
            "estimates": {"price": {"meta-llama/llama-3.3-70b-instruct": [0.1, 0.32]}},
            "manifest": {"git_commit": "abc123", "started_at": "2026-11-05T12:00:00+00:00"}}


def launch(root, run_id, steps):
    tracker = WandbTracker(settings(root, "online"))
    got = tracker.start("mock_main", settings(root, "online"), run_id)
    for step in steps:
        tracker.log({"runner/units_done": step}, step=step)
    return tracker, got


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    root = tmp_path_factory.mktemp("offline")
    patch = pytest.MonkeyPatch()
    patch.setenv("WANDB_MODE", "disabled")
    patch.setenv("WANDB__SERVICE_WAIT", "300")
    patch.setenv("SWARMLAB_KICKOFF_UTC", FUTURE)
    patch.delenv("WANDB_API_KEY", raising=False)
    patch.setattr(env, "ENV_PATH", root / "absent.env")
    try:
        manifest = root / "manifest-1.json"
        manifest.write_text("{}", encoding="utf-8")
        first, run_id = launch(root, None, [512, 1024])
        first.log_rows("episodes", [{"unit": "u1", "asr": 0.5}, {"unit": "u2", "asr": float("nan")}])
        first.log_rows("episodes", [{"unit": "u3", "asr": 0.25}])
        first.artifact(str(manifest), "mock_main-manifest", "manifest")
        first.finish()
        second, again = launch(root, run_id, [1536])
        assert again == run_id
        second.finish()

        quiet = root / "disabled"
        tracker = WandbTracker(settings(quiet, "disabled"))
        disabled_id = tracker.start("mock_main", settings(quiet, "disabled"), None)
        tracker.log({"x": 1}, step=1)
        tracker.log_rows("t", [{"a": 1}])
        tracker.artifact(str(manifest), "m", "manifest")
        tracker.finish()
        disabled_resumed = tracker.start("mock_main", settings(quiet, "disabled"), "abc12345")
        tracker.finish()
        yield Session(run_id, sorted((root / "wandb").glob(f"offline-run-*-{run_id}")), disabled_id,
                      list(quiet.rglob("offline-run-*")), disabled_resumed, root)
    finally:
        patch.undo()


def history(run_dir):
    return [{".".join(i.nested_key) or i.key: i.value_json for i in record.history.item}
            for record in offline_log.records(run_dir) if record.WhichOneof("record_type") == "history"]


def of_kind(run_dir, kind):
    return [getattr(r, kind) for r in offline_log.records(run_dir) if r.WhichOneof("record_type") == kind]


def test_each_launch_leaves_one_offline_run_and_the_second_reuses_the_id(session):
    assert len(session.dirs) == 2 and session.run_id


def test_the_environment_cannot_override_the_explicit_mode(session):
    assert all((d / f"run-{session.run_id}.wandb").is_file() for d in session.dirs)


def test_the_run_record_carries_project_group_job_type_and_config(session):
    run = of_kind(session.dirs[0], "run")[0]
    assert (run.run_id, run.project, run.run_group, run.job_type) == (session.run_id, "bazt-test", "sprint", "mock_main")
    config = {u.key: json.loads(u.value_json) for u in run.config.update}
    assert config["llm"] == {"agents": 12}
    assert config["estimates"] == {"price": {"meta-llama/llama-3_3-70b-instruct": [0.1, 0.32]}}
    assert config["manifest"]["git_commit"] == "abc123"
    assert "hf_token" not in json.dumps(config)


def test_no_git_state_is_recorded(session):
    run = of_kind(session.dirs[0], "run")[0]
    assert not run.git.remote_url and not run.git.commit


def test_the_runner_step_is_the_x_axis_of_every_metric(session):
    metrics = of_kind(session.dirs[0], "metric")
    assert [m.name for m in metrics if m.name] == [STEP_KEY]
    assert [(m.glob_name, m.step_metric) for m in metrics if m.glob_name] == [("*", STEP_KEY)]


def test_metrics_are_logged_with_the_runner_step_in_launch_order(session):
    first = [h for h in history(session.dirs[0]) if "runner/units_done" in h]
    assert [(h["runner/units_done"], h[STEP_KEY]) for h in first] == [("512", "512"), ("1024", "1024")]
    second = [h for h in history(session.dirs[1]) if "runner/units_done" in h]
    assert [(h["runner/units_done"], h[STEP_KEY]) for h in second] == [("1536", "1536")]


def test_rows_are_logged_as_one_table_part_per_flush_and_the_rest_at_finish(session):
    files = sorted((session.dirs[0] / "files" / "media" / "table").glob("episodes*.table.json"))
    assert len(files) == 2
    parts = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    assert all(part["columns"] == ["unit", "asr"] for part in parts)
    assert sorted(r[0] for part in parts for r in part["data"]) == ["u1", "u2", "u3"]
    assert sorted(len(part["data"]) for part in parts) == [1, 2]       # flush_rows=2 flushed u1, u2; finish flushed u3


def test_the_manifest_is_logged_as_an_artifact(session):
    arts = [a for a in of_kind(session.dirs[0], "artifact") if a.type == "manifest"]
    assert [a.name for a in arts] == ["mock_main-manifest"]
    assert [c.path for c in arts[0].manifest.contents] == ["manifest-1.json"]


def test_a_disabled_run_returns_an_id_and_writes_no_offline_run(session):
    assert session.disabled_id and session.disabled_dirs == []


def test_a_disabled_run_hands_back_the_saved_id_so_a_later_launch_can_resume_it(session):
    assert session.disabled_resumed == "abc12345"
