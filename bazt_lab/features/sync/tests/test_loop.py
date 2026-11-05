"""The upload loop that runs inside the child process, driven in-process with a recording uploader."""
from __future__ import annotations

import threading

from bazt_lab.features.sync.private import loop
from bazt_lab.features.sync.private.settings import SyncSettings
from bazt_lab.features.sync.tests.fakes import Recorder

CANARY = "hf_not-a-real-token-for-tests"


def sync_settings(**kw):
    base = dict(repo_id="me/r", repo_type="dataset", private=True, every_minutes=0.0005, include=(), exclude=(),
                priority="idle", final_push_timeout_seconds=5.0)
    return SyncSettings(**{**base, **kw})


def write(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path


def make(tmp_path, uploader, **kw):
    lines = []
    uploads = loop.Uploads(sync_settings(**kw), [tmp_path / "r"], uploader, report=lines.append)
    return uploads, lines


def test_one_push_makes_one_commit_with_every_changed_file(tmp_path):
    write(tmp_path / "r" / "a.json"), write(tmp_path / "r" / "b.json")
    rec = Recorder()
    uploads, _ = make(tmp_path, rec)
    uploads.push()
    assert rec.commits == [["r/a.json", "r/b.json"]] and rec.repos == 1


def test_a_later_push_uploads_only_what_changed_and_nothing_when_nothing_did(tmp_path):
    a = write(tmp_path / "r" / "a.json", "1")
    rec = Recorder()
    uploads, _ = make(tmp_path, rec)
    uploads.push()
    uploads.push()
    assert rec.commits == [["r/a.json"]]
    write(tmp_path / "r" / "b.json")
    uploads.push()
    a.write_text("changed content", encoding="utf-8")
    uploads.push()
    assert rec.commits == [["r/a.json"], ["r/b.json"], ["r/a.json"]]


def test_tmp_files_and_excluded_files_stay_local(tmp_path):
    write(tmp_path / "r" / "a.json"), write(tmp_path / "r" / "a.json.tmp"), write(tmp_path / "r" / ".launch.lock")
    rec = Recorder()
    uploads, _ = make(tmp_path, rec, exclude=(".launch.lock",))
    uploads.push()
    assert rec.commits == [["r/a.json"]]


def test_a_failed_commit_is_reported_once_and_retried_at_the_next_push(tmp_path):
    write(tmp_path / "r" / "a.json")
    rec = Recorder()
    rec.fail = ConnectionError("network down")
    uploads, lines = make(tmp_path, rec)
    for _ in range(3):
        uploads.push()                                           # must not raise
    assert len(lines) == 1 and "network down" in lines[0] and "disk" in lines[0]
    assert uploads.failures == {"commit": 3}
    rec.fail = None
    uploads.push()
    assert rec.commits == [["r/a.json"]]                         # nothing was marked uploaded by the failures


def test_a_repo_that_cannot_be_created_is_retried_before_the_next_commit(tmp_path):
    write(tmp_path / "r" / "a.json")

    class Flaky(Recorder):
        def ensure_repo(self):
            super().ensure_repo()
            if self.repos == 1:
                raise ConnectionError("no route")

    rec = Flaky()
    uploads, lines = make(tmp_path, rec)
    uploads.push()
    assert rec.commits == [] and uploads.failures == {"repo": 1}
    uploads.push()
    assert rec.commits == [["r/a.json"]] and rec.repos == 2


def test_the_token_never_appears_in_what_is_reported(tmp_path):
    write(tmp_path / "r" / "a.json")
    rec = Recorder()
    rec.fail = RuntimeError(f"401 for {CANARY}")
    uploads, lines = make(tmp_path, rec)
    uploads.redact = lambda text: text.replace(CANARY, "[redacted]")
    uploads.push()
    assert CANARY not in lines[0] and "[redacted]" in lines[0]


def test_run_pushes_every_interval_and_once_more_when_stopped(tmp_path):
    write(tmp_path / "r" / "a.json")
    rec = Recorder()
    uploads, _ = make(tmp_path, rec)
    stop = threading.Event()
    worker = threading.Thread(target=loop.run, args=(uploads, stop), daemon=True)
    worker.start()
    for _ in range(200):
        if rec.commits:
            break
        stop.wait(0.05)
    assert rec.commits == [["r/a.json"]]                         # pushed by the interval, stop not yet set
    write(tmp_path / "r" / "late.json")                          # written after the last interval
    stop.set()
    worker.join(10)
    assert not worker.is_alive() and rec.commits[-1] == ["r/late.json"]


def test_a_file_that_grows_during_the_push_is_uploaded_up_to_the_size_seen_at_listing(tmp_path):
    f = write(tmp_path / "r" / "rows.jsonl", "line1\n")
    rec = Recorder()
    uploads, _ = make(tmp_path, rec)
    entries = uploads.pending()
    f.write_bytes(b"line1\nline2\n")                             # appended after the listing
    uploads.commit(entries)
    assert rec.contents == [{"r/rows.jsonl": b"line1\n"}]
    uploads.push()                                               # the growth is the next push's change
    assert rec.commits[-1] == ["r/rows.jsonl"] and rec.contents[-1]["r/rows.jsonl"] == b"line1\nline2\n"
