"""The REAL HubUploader.commit (real files, real huggingface_hub operation objects) with only create_commit faked:
the experiment must be able to replace or delete a file while it is being uploaded, a push never holds more than one
batch of files, and what is uploaded is always one consistent version of each file."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from bazt_lab.features.sync.adapters import env, hub
from bazt_lab.features.sync.private import plan

CONFIG = {"settings": {"repo_id": "me/r", "repo_type": "dataset", "private": True}}


class FakeApi:
    """create_commit reads every staged file the way the real one does, after running `during` (the experiment
    working while the push is in flight)."""

    def __init__(self, during=None):
        self.during = during
        self.commits = []

    def create_repo(self, *args, **kwargs):
        pass

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        if self.during:
            self.during()
        self.commits.append({op.path_in_repo: Path(op.path_or_fileobj).read_bytes() for op in operations})


@pytest.fixture
def uploader(monkeypatch):
    monkeypatch.setenv(env.VARIABLE, "hf_not-a-real-token-for-tests")
    up = hub.make_uploader(CONFIG)
    up.api = FakeApi()
    return up


def listed(root, *names):
    return plan.snapshot([root], [], [])


def write(path, data=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_the_files_of_a_push_are_uploaded_with_their_content(uploader, tmp_path):
    write(tmp_path / "r" / "a.json", b'{"a": 1}')
    write(tmp_path / "r" / "sub" / "b.bin", bytes(range(256)))
    done = uploader.commit(plan.snapshot([tmp_path / "r"], [], []), "m")
    assert uploader.api.commits == [{"r/a.json": b'{"a": 1}', "r/sub/b.bin": bytes(range(256))}]
    assert sorted(e.repo_path for e in done) == ["r/a.json", "r/sub/b.bin"]


def test_an_empty_file_is_uploaded(uploader, tmp_path):
    write(tmp_path / "r" / "u.done", b"")
    uploader.commit(plan.snapshot([tmp_path / "r"], [], []), "m")
    assert uploader.api.commits == [{"r/u.done": b""}]


def test_the_experiment_can_replace_and_delete_listed_files_while_the_push_is_in_flight(uploader, tmp_path):
    state, gone = write(tmp_path / "r" / "state.json", b"old state"), write(tmp_path / "r" / "u1.pt", b"checkpoint")
    entries = plan.snapshot([tmp_path / "r"], [], [])

    def experiment():
        fresh = write(tmp_path / "r" / "state.json.tmp", b"new state, longer")
        os.replace(fresh, state)                        # CheckpointStore / thresholds atomic rewrite
        gone.unlink()                                   # mark_done, epoch rotation

    uploader.api.during = experiment
    uploader.commit(entries, "m")                       # PermissionError in the experiment would raise out of here
    assert state.read_bytes() == b"new state, longer" and not gone.exists()
    assert uploader.api.commits == [{"r/state.json": b"old state", "r/u1.pt": b"checkpoint"}]


def test_a_file_replaced_by_a_larger_one_after_listing_uploads_the_whole_new_version(uploader, tmp_path):
    f = write(tmp_path / "r" / "thresholds.json", b'{"cell1": 0.5}')
    entries = plan.snapshot([tmp_path / "r"], [], [])
    write(tmp_path / "r" / "new", b'{"cell1": 0.5, "cell2": 0.7}')
    os.replace(tmp_path / "r" / "new", f)
    done = uploader.commit(entries, "m")
    assert uploader.api.commits == [{"r/thresholds.json": b'{"cell1": 0.5, "cell2": 0.7}'}]
    assert plan.changed(plan.snapshot([tmp_path / "r"], [], []), plan.stamps(done)) == []   # not uploaded twice


def test_a_file_appended_to_during_the_push_is_uploaded_up_to_what_was_there_at_open(uploader, tmp_path):
    f = write(tmp_path / "r" / "rows.jsonl", b'{"a": 1}\n')
    entries = plan.snapshot([tmp_path / "r"], [], [])

    def experiment():
        with f.open("ab") as fh:
            fh.write(b'{"b": 2}\n')

    uploader.api.during = experiment
    done = uploader.commit(entries, "m")
    assert uploader.api.commits == [{"r/rows.jsonl": b'{"a": 1}\n'}]
    assert [e.size for e in done] == [9]
    assert [e.repo_path for e in plan.changed(plan.snapshot([tmp_path / "r"], [], []), plan.stamps(done))] == [
        "r/rows.jsonl"]                                  # the appended row goes in the next push


def test_a_file_that_vanished_after_listing_is_skipped_and_the_rest_is_still_committed(uploader, tmp_path):
    write(tmp_path / "r" / "a.json", b"a")
    gone = write(tmp_path / "r" / "b.json", b"b")
    entries = plan.snapshot([tmp_path / "r"], [], [])
    gone.unlink()
    done = uploader.commit(entries, "m")
    assert uploader.api.commits == [{"r/a.json": b"a"}] and [e.repo_path for e in done] == ["r/a.json"]


def test_nothing_readable_means_no_commit(uploader, tmp_path):
    gone = write(tmp_path / "r" / "b.json", b"b")
    entries = plan.snapshot([tmp_path / "r"], [], [])
    gone.unlink()
    assert uploader.commit(entries, "m") == [] and uploader.api.commits == []


def test_every_handle_is_closed_when_the_commit_fails(uploader, tmp_path):
    f = write(tmp_path / "r" / "a.json", b"a")
    entries = plan.snapshot([tmp_path / "r"], [], [])

    def refuse():
        raise ConnectionError("down")

    uploader.api.during = refuse
    with pytest.raises(ConnectionError):
        uploader.commit(entries, "m")
    f.unlink()                                           # fails with PermissionError on Windows if a handle leaked
    assert not f.exists()
