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


def test_a_push_is_split_into_batches_so_it_never_holds_thousands_of_files(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "COMMIT_FILES", 2)
    for name in "abcde":
        write(tmp_path / "r" / f"{name}.json")
    rec = Recorder()
    uploads, _ = make(tmp_path, rec)
    uploads.push()
    assert rec.commits == [["r/a.json", "r/b.json"], ["r/c.json", "r/d.json"], ["r/e.json"]]
    assert uploads.commits == 3
    uploads.push()
    assert len(rec.commits) == 3


def test_a_failed_batch_keeps_the_earlier_batches_marked_and_retries_only_the_rest(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "COMMIT_FILES", 2)
    for name in "abcd":
        write(tmp_path / "r" / f"{name}.json")

    class FailsSecond(Recorder):
        def commit(self, files, message):
            if len(self.commits) == 1 and self.fail is None and not getattr(self, "armed", False):
                self.armed = True
                self.fail = ConnectionError("down")
            return super().commit(files, message)

    rec = FailsSecond()
    uploads, _ = make(tmp_path, rec)
    uploads.push()
    assert rec.commits == [["r/a.json", "r/b.json"]] and uploads.failures == {"commit": 1}
    rec.fail = None
    uploads.push()
    assert rec.commits == [["r/a.json", "r/b.json"], ["r/c.json", "r/d.json"]]


def test_a_file_the_uploader_skipped_is_not_marked_uploaded_and_is_retried(tmp_path):
    write(tmp_path / "r" / "a.json"), write(tmp_path / "r" / "b.json")

    class SkipsB(Recorder):
        skip = True

        def commit(self, files, message):
            kept = [f for f in files if not (self.skip and f.repo_path == "r/b.json")]
            super().commit(kept, message)
            return kept

    rec = SkipsB()
    uploads, _ = make(tmp_path, rec)
    uploads.push()
    rec.skip = False
    uploads.push()
    assert rec.commits == [["r/a.json"], ["r/b.json"]] and uploads.failures == {}


def test_a_listing_that_fails_is_reported_and_does_not_end_the_loop(tmp_path, monkeypatch):
    rec = Recorder()
    uploads, lines = make(tmp_path, rec)
    real = loop.plan.snapshot
    monkeypatch.setattr(loop.plan, "snapshot", lambda *a: (_ for _ in ()).throw(PermissionError("denied")))
    uploads.push()                                               # must not raise
    assert uploads.failures == {"scan": 1} and "denied" in lines[0]
    monkeypatch.setattr(loop.plan, "snapshot", real)
    write(tmp_path / "r" / "a.json")
    uploads.push()
    assert rec.commits == [["r/a.json"]]


def test_the_child_still_does_its_final_push_when_the_listing_failed_in_between(tmp_path, monkeypatch):
    write(tmp_path / "r" / "a.json")
    rec = Recorder()
    uploads, _ = make(tmp_path, rec)
    real = loop.plan.snapshot
    calls = []

    def flaky(*args):
        calls.append(1)
        if len(calls) == 1:
            raise PermissionError("denied")
        return real(*args)

    monkeypatch.setattr(loop.plan, "snapshot", flaky)
    stop = threading.Event()
    stop.set()
    loop.run(uploads, stop)                                      # the interval push fails; the loop still ends cleanly
    assert uploads.failures == {"scan": 1}
