"""The real child process, with the fake uploader (no network): priority, final push, a dying child, the token,
refusals. Each test starts its own child; they take a few seconds each (interpreter start plus imports)."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import pytest

from bazt_lab.core import public as core
from bazt_lab.features.sync import public as sync
from bazt_lab.features.sync.adapters import process

CANARY = "hf_not-a-real-token-for-tests"
FAKE = "bazt_lab.features.sync.tests.fakes:make_uploader"


def settings(**sync_keys):
    base = {"enabled": True, "repo_id": "me/r", "repo_type": "dataset", "private": True, "every_minutes": 0.01,
            "include": [], "exclude": [], "priority": "below_normal", "final_push_timeout_seconds": 300}
    return {"sync": {**base, **sync_keys}}


@pytest.fixture
def fake(monkeypatch, tmp_path):
    log = tmp_path / "events.jsonl"
    monkeypatch.setenv("HF_TOKEN", CANARY)
    monkeypatch.setenv("BAZT_SYNC_FAKE_LOG", str(log))
    monkeypatch.setattr(process, "UPLOADER", FAKE)
    return log


def events(log):
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []


def commits(log):
    return [e["files"] for e in events(log) if e["event"] == "commit"]


def wait_for(condition, seconds=180):
    end = time.time() + seconds
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.1)
    return False


def folder(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    (root / "rows.jsonl").write_text("{}\n", encoding="utf-8")
    return root


# ------------------------------------------------------------------------------------------- the priority
@pytest.mark.parametrize("level", ["below_normal", "idle"])
def test_the_child_is_started_with_a_low_os_priority(level):
    windows = process.launch_options("nt", level)
    posix = process.launch_options("posix", level)
    assert windows == {"creationflags": {"below_normal": 0x00004000, "idle": 0x00000040}[level]}
    assert set(posix) == {"preexec_fn"}


def test_the_posix_child_lowers_its_own_nice_value(monkeypatch):
    seen = []
    monkeypatch.setattr(os, "nice", lambda n: seen.append(n) or n, raising=False)
    process.launch_options("posix", "below_normal")["preexec_fn"]()
    process.launch_options("posix", "idle")["preexec_fn"]()
    assert seen == [10, 19]


def test_the_running_child_really_has_a_priority_below_normal(fake, tmp_path):
    psutil = pytest.importorskip("psutil")
    handle = sync.start(settings(every_minutes=60), [folder(tmp_path)])
    try:
        child = psutil.Process(handle.pid)
        if os.name == "nt":
            assert child.nice() == psutil.BELOW_NORMAL_PRIORITY_CLASS
        else:
            assert child.nice() >= 10
        assert psutil.Process().nice() != child.nice()
    finally:
        handle.stop()


# ------------------------------------------------------------------------------------------- pushing
def test_the_child_pushes_by_itself_and_stop_does_a_final_push(fake, tmp_path):
    root = folder(tmp_path)
    handle = sync.start(settings(), [root])
    assert wait_for(lambda: commits(fake) == [["results/rows.jsonl"]])
    (root / "late.json").write_text("{}", encoding="utf-8")
    handle.stop(), handle.stop()                                    # stop twice is fine
    assert commits(fake)[-1] == ["results/late.json"]
    pids = {e["pid"] for e in events(fake)}
    assert len(pids) == 1 and os.getpid() not in pids               # uploads ran in the child, not here
    assert handle.summary["commits"] >= 2 and handle.summary["failures"] == {}


def test_nothing_is_uploaded_before_the_first_interval_except_by_stop(fake, tmp_path):
    root = folder(tmp_path)
    handle = sync.start(settings(every_minutes=60), [root])
    time.sleep(1)
    assert commits(fake) == []
    handle.stop()
    assert commits(fake) == [["results/rows.jsonl"]]


# ------------------------------------------------------------------------------------------- failures
def test_a_child_that_dies_is_reported_once_and_the_parent_carries_on(fake, tmp_path, capsys):
    handle = sync.start(settings(every_minutes=60), [folder(tmp_path)])
    os.kill(handle.pid, signal.SIGTERM) if os.name != "nt" else subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(handle.pid)], capture_output=True, check=False)
    assert wait_for(lambda: not handle.alive)
    assert wait_for(lambda: handle.reported == 1)
    assert "uploader process exited" in capsys.readouterr().err
    handle.stop()                                                   # does not raise, does not hang
    handle.stop()
    assert handle.summary is None and handle.reported == 1


def test_upload_failures_are_counted_in_the_summary_and_never_stop_the_child(fake, tmp_path, monkeypatch, capfd):
    monkeypatch.setenv("BAZT_SYNC_FAKE_FAIL", "1")
    handle = sync.start(settings(), [folder(tmp_path)])
    time.sleep(2)
    assert handle.alive                                             # the failing commits did not end the child
    handle.stop()
    assert handle.summary["commits"] == 0 and handle.summary["failures"]["commit"] >= 1
    assert "upload operation(s) failed during the run" in capfd.readouterr().err   # visible without reading summary


def test_the_token_is_in_no_output_of_the_child_or_the_parent(fake, tmp_path, monkeypatch, capfd):
    monkeypatch.setenv("BAZT_SYNC_FAKE_FAIL", "1")                  # the fake puts the token in its error text
    handle = sync.start(settings(), [folder(tmp_path)])
    time.sleep(2)
    handle.stop()
    out = capfd.readouterr()
    assert CANARY not in out.out and CANARY not in out.err and "[redacted]" in out.err
    assert CANARY not in json.dumps(handle.summary)


def test_the_token_is_not_on_the_command_line_of_the_child(fake, tmp_path):
    psutil = pytest.importorskip("psutil")
    handle = sync.start(settings(every_minutes=60), [folder(tmp_path)])
    try:
        assert CANARY not in " ".join(psutil.Process(handle.pid).cmdline())
    finally:
        handle.stop()


# ------------------------------------------------------------------------------------------- refusals
def test_disabled_sync_starts_nothing(fake, tmp_path):
    handle = sync.start({"sync": {"enabled": False}}, [folder(tmp_path)])
    assert handle.pid is None and not handle.alive
    handle.stop()


def test_enabled_unset_names_the_key():
    with pytest.raises(core.Unset) as error:
        sync.start({}, ["x"])
    assert error.value.args[0] == "sync.enabled"


def test_enabled_without_a_repo_id_refuses_before_a_child_is_started(fake, tmp_path, monkeypatch):
    s = settings()
    del s["sync"]["repo_id"]
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("no child may be started"))
    with pytest.raises(core.Unset) as error:
        sync.start(s, [folder(tmp_path)])
    assert error.value.args[0] == "sync.repo_id"


def test_enabled_without_a_token_refuses_naming_the_variable(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("no child may be started"))
    with pytest.raises(sync.SyncError, match="HF_TOKEN") as error:
        sync.start(settings(), [folder(tmp_path)])
    assert "profiles.toml" in str(error.value) and ".env" in str(error.value)


@pytest.mark.parametrize("paths", [[], ["a/same", "b/same"]])
def test_no_paths_or_two_paths_with_one_name_are_refused(fake, tmp_path, paths):
    with pytest.raises(sync.SyncError, match="path"):
        sync.start(settings(), [tmp_path / p for p in paths])


def test_a_child_that_cannot_start_is_a_refusal_not_a_silent_skip(fake, tmp_path, monkeypatch):
    def broken(*a, **k):
        raise OSError("no such interpreter")

    monkeypatch.setattr(subprocess, "Popen", broken)
    with pytest.raises(sync.SyncError, match="no such interpreter"):
        sync.start(settings(), [folder(tmp_path)])


# ------------------------------------------------------------------------------------------- parent cost
def test_the_parent_spends_almost_no_cpu_while_the_child_works(fake, tmp_path):
    root = folder(tmp_path)
    before = time.process_time()
    handle = sync.start(settings(every_minutes=0.02), [root])
    time.sleep(4)
    handle.stop()
    spent = time.process_time() - before
    assert spent < 1.0, f"parent CPU {spent:.3f}s over 4s of uploading"


def test_the_package_stays_importable_without_huggingface_hub_in_the_parent():
    code = "import sys; import bazt_lab.features.sync.public; sys.exit('huggingface_hub' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], cwd=os.getcwd(), check=False).returncode == 0
