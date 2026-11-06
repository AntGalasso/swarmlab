"""Core contracts: settings, streams, checkpoints."""
import sys

import pytest
import torch

from bazt_lab.core import public as core


def test_profiles_and_overrides():
    s = core.load_settings(profile="research", overrides=["llm.conversations_per_cell=80"])
    assert s["profile"] == "research"
    assert core.get(s, "llm.conversations_per_cell") == 80
    with pytest.raises(core.Unset):
        core.get(s, "runtime.n_insiders")


def test_streams_are_keyed_not_positional():
    a = core.uniforms(core.stream("adopt", "chain", 1), (3, 4))
    b = core.uniforms(core.stream("adopt", "chain", 1), (3, 4))
    c = core.uniforms(core.stream("adopt", "chain", 11), (3, 4))
    assert torch.equal(a, b) and not torch.equal(a, c) and a.dtype == torch.float64
    assert core.seed_of("x", 1, 12) != core.seed_of("x", 11, 2)


def test_batch_uniforms_do_not_depend_on_batch_size():
    keys = [("adopt", "chain", s) for s in range(5)]
    full = core.batch_uniforms(keys, (2, 3))
    assert torch.equal(full[3], core.batch_uniforms(keys[3:4], (2, 3))[0])


def test_checkpoint_progressive_rows_state_and_done(tmp_path):
    store = core.CheckpointStore(tmp_path)
    store.append_rows("u1", [{"a": 1}])
    store.append_rows("u1", [{"a": 2}])
    (tmp_path / "units" / "u1.jsonl").open("a").write('{"torn": ')   # crash mid-write
    assert store.resume_rows("u1") == [{"a": 1}, {"a": 2}]
    g = core.stream("k", 1)
    store.save_state("u1", {"gen": g.get_state(), "step": 3})
    state = store.load_state("u1")
    assert state["step"] == 3 and torch.equal(state["gen"], g.get_state())
    assert not store.is_done("u1")
    store.mark_done("u1")
    assert store.is_done("u1") and store.load_state("u1") is None


def test_rows_appended_after_a_torn_line_are_kept(tmp_path):
    store = core.CheckpointStore(tmp_path)
    store.append_rows("u", [{"i": 0}])
    with (tmp_path / "units" / "u.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"i": 1, "x"')                       # crash in the middle of a write
    assert store.resume_rows("u") == [{"i": 0}]
    store.append_rows("u", [{"i": 1}, {"i": 2}])
    assert store.resume_rows("u") == [{"i": 0}, {"i": 1}, {"i": 2}]


def test_append_without_resume_repairs_a_torn_tail(tmp_path):
    store = core.CheckpointStore(tmp_path)
    store.append_rows("u", [{"i": 0}])
    with (tmp_path / "units" / "u.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"i": 1')
    store.append_rows("u", [{"i": 1}])
    assert store.resume_rows("u") == [{"i": 0}, {"i": 1}]


# ------------------------------------------------------------------------------------------------ batch API
def _all_rows(root):
    import json
    return [json.loads(line) for path in sorted((root / "units").glob("*.jsonl")) for line in path.read_text().splitlines()]


def test_append_batch_writes_one_file_with_one_fsync_and_marks_every_unit_done(tmp_path, monkeypatch):
    import os
    store = core.CheckpointStore(tmp_path)
    calls = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real(fd))[1])
    store.append_batch({f"u{k}": [{"unit_id": f"u{k}", "v": k}] for k in range(5)})
    assert len(calls) == 1
    files = sorted(p.name for p in (tmp_path / "units").iterdir())
    assert len([f for f in files if f.endswith(".jsonl")]) == 1 and len([f for f in files if f.endswith(".done")]) == 1
    assert all(store.is_done(f"u{k}") for k in range(5)) and not store.is_done("u9")
    assert sorted(r["v"] for r in _all_rows(tmp_path)) == list(range(5))


def test_the_done_index_is_loaded_at_construction_from_batch_and_per_unit_markers(tmp_path):
    store = core.CheckpointStore(tmp_path)
    store.append_batch({"a": [{"x": 1}], "b": [{"x": 2}]})
    store.append_rows("old", [{"x": 3}])
    store.mark_done("old")
    fresh = core.CheckpointStore(tmp_path)
    assert fresh.is_done("a") and fresh.is_done("b") and fresh.is_done("old") and not fresh.is_done("c")


def test_a_crash_between_the_rows_and_the_done_write_leaves_no_duplicate_rows_after_the_rerun(tmp_path, monkeypatch):
    from bazt_lab.core.private import checkpoint
    store = core.CheckpointStore(tmp_path)
    monkeypatch.setattr(checkpoint, "_atomic_write_bytes", lambda *a: (_ for _ in ()).throw(RuntimeError("crash")))
    with pytest.raises(RuntimeError):
        store.append_batch({f"u{k}": [{"unit_id": f"u{k}"}] for k in range(4)})
    monkeypatch.undo()
    resumed = core.CheckpointStore(tmp_path)
    assert not any(resumed.is_done(f"u{k}") for k in range(4))
    resumed.append_batch({f"u{k}": [{"unit_id": f"u{k}"}] for k in range(2)})          # a different batch composition
    resumed.append_batch({f"u{k}": [{"unit_id": f"u{k}"}] for k in range(2, 4)})
    ids = sorted(r["unit_id"] for r in _all_rows(tmp_path))
    assert ids == ["u0", "u1", "u2", "u3"] and all(resumed.is_done(f"u{k}") for k in range(4))


def test_the_same_batch_rerun_after_a_crash_replaces_its_rows(tmp_path, monkeypatch):
    from bazt_lab.core.private import checkpoint
    batch = {f"u{k}": [{"unit_id": f"u{k}"}] for k in range(3)}
    monkeypatch.setattr(checkpoint, "_atomic_write_bytes", lambda *a: (_ for _ in ()).throw(RuntimeError("crash")))
    with pytest.raises(RuntimeError):
        core.CheckpointStore(tmp_path).append_batch(batch)
    monkeypatch.undo()
    core.CheckpointStore(tmp_path).append_batch(batch)
    assert len(_all_rows(tmp_path)) == 3


def test_per_unit_api_and_resume_rows_still_work_next_to_batches(tmp_path):
    store = core.CheckpointStore(tmp_path)
    store.append_batch({"a": [{"i": 0}]})
    store.append_rows("u", [{"i": 1}])
    assert store.resume_rows("u") == [{"i": 1}] and store.resume_rows("a") == [] and not store.is_done("u")
    store.mark_done("u")
    assert store.is_done("u") and store.is_done("a")


def test_batch_states_are_removed_for_the_units_marked_done(tmp_path):
    store = core.CheckpointStore(tmp_path)
    store.save_state("a", {"step": 1})
    store.append_batch({"a": [{"i": 0}], "b": []})
    assert store.load_state("a") is None and store.is_done("b")


def test_shards_write_apart_and_every_store_sees_every_shards_done_units(tmp_path):
    w0 = core.CheckpointStore(tmp_path, shard="w0")
    w1 = core.CheckpointStore(tmp_path, shard="w1")
    w0.append_batch({"a": [{"i": 0}], "b": [{"i": 1}]})
    w1.append_batch({"c": [{"i": 2}]})
    assert sorted(p.parent.name for p in (tmp_path / "units").rglob("*.jsonl")) == ["w0", "w1"]
    fresh = core.CheckpointStore(tmp_path, shard="w2")            # a later launch with another worker count
    assert all(fresh.is_done(u) for u in "abc") and fresh.done_count() == 3
    w1.refresh()
    assert w1.is_done("a")


def test_a_shard_clears_only_its_own_orphans(tmp_path):
    w0 = core.CheckpointStore(tmp_path, shard="w0")
    (tmp_path / "units" / "w1").mkdir(parents=True)
    other = tmp_path / "units" / "w1" / "batch-0123456789abcdef.jsonl"
    other.write_text('{"i": 9}\n', encoding="utf-8")                # another worker's batch, not committed yet
    w0.append_batch({"a": [{"i": 0}]})
    assert other.exists()


def test_a_shard_name_must_be_a_plain_folder_name(tmp_path):
    with pytest.raises(ValueError):
        core.CheckpointStore(tmp_path, shard="../x")


def test_rows_of_returns_the_rows_of_a_unit_saved_alone_or_in_a_batch(tmp_path):
    store = core.CheckpointStore(tmp_path)
    store.append_batch({"a": [{"id": "a", "v": 1}], "b": [{"id": "b", "v": 2}]})
    store.append_rows("c", [{"id": "c", "v": 3}])
    store.mark_done("c")
    for s in (store, core.CheckpointStore(tmp_path)):
        assert s.rows_of("a") == [{"id": "a", "v": 1}] and s.rows_of("b") == [{"id": "b", "v": 2}]
        assert s.rows_of("c") == [{"id": "c", "v": 3}] and s.rows_of("zzz") == []


# ------------------------------------------------------------------------------------------------ shard readers (R3-C2)
def _batch_done_file(root, shard):
    return next((root / "units" / shard).glob("batch-*.done"))


def _fail_reads_of(monkeypatch, target, error, times=1):
    """Make reading `target` raise `error` for the next `times` reads (what another process holding the file does)."""
    import pathlib
    real, left = pathlib.Path.read_text, {"n": times}

    def read_text(self, *args, **kwargs):
        if pathlib.Path(self) == target and left["n"] > 0:
            left["n"] -= 1
            raise error
        return real(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", read_text)


@pytest.mark.parametrize("error", [PermissionError(13, "Permission denied"), FileNotFoundError(2, "No such file")],
                         ids=["held-by-another-process", "gone-after-listing"])
def test_refresh_skips_a_done_list_it_cannot_read_now_and_loads_it_on_a_later_call(tmp_path, monkeypatch, error):
    w0, w1 = core.CheckpointStore(tmp_path, shard="w0"), core.CheckpointStore(tmp_path, shard="w1")
    w0.append_batch({"a": [{"id": "a"}], "b": [{"id": "b"}]})
    _fail_reads_of(monkeypatch, _batch_done_file(tmp_path, "w0"), error)
    assert w1.refresh() == 1                                       # one done list left for a later call, nothing raised
    assert not w1.is_done("a") and not w1.is_done("b")            # never done from a marker that was not read
    assert w1.refresh() == 0
    assert w1.is_done("a") and w1.is_done("b") and w1.rows_of("a") == [{"id": "a"}]


def test_refresh_skips_a_torn_done_list_until_it_is_whole(tmp_path):
    w0, w1 = core.CheckpointStore(tmp_path, shard="w0"), core.CheckpointStore(tmp_path, shard="w1")
    w0.append_batch({"a": [{"id": "a"}], "b": [{"id": "b"}]})
    done = _batch_done_file(tmp_path, "w0")
    whole = done.read_text(encoding="utf-8")
    done.write_text(whole[: len(whole) // 2], encoding="utf-8")   # a partial write, as a non-atomic copy would leave
    reader = core.CheckpointStore(tmp_path, shard="w2")
    assert not reader.is_done("a") and reader.done_count() == 0
    done.write_text(whole, encoding="utf-8")
    assert reader.refresh() == 0 and reader.is_done("a") and reader.is_done("b")
    assert w1.refresh() == 0 and w1.is_done("a")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows sharing rules")
def test_refresh_survives_a_done_list_still_held_by_the_rename_that_created_it(tmp_path):
    """The real failure: os.replace (MoveFileEx) holds the file it renames with DELETE access, and Python's open() does
    not share delete access, so a reader that lists the new name before the rename's handle closes gets errno 13."""
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                                wintypes.DWORD, wintypes.HANDLE]
    w0, w1 = core.CheckpointStore(tmp_path, shard="w0"), core.CheckpointStore(tmp_path, shard="w1")
    w0.append_batch({"a": [{"id": "a"}]})
    handle = k32.CreateFileW(str(_batch_done_file(tmp_path, "w0")), 0x00010000, 0x7, None, 3, 0, None)  # DELETE, share all
    assert handle not in (None, wintypes.HANDLE(-1).value)
    try:
        with pytest.raises(PermissionError):
            _batch_done_file(tmp_path, "w0").read_text(encoding="utf-8")     # the condition the workers hit
        assert w1.refresh() == 1 and not w1.is_done("a")
        core.CheckpointStore(tmp_path, shard="w2")                            # a new store does not crash either
    finally:
        k32.CloseHandle(handle)
    assert w1.refresh() == 0 and w1.is_done("a")


def test_a_second_refresh_reads_only_the_done_lists_that_are_new(tmp_path, monkeypatch):
    import pathlib
    w0, w1 = core.CheckpointStore(tmp_path, shard="w0"), core.CheckpointStore(tmp_path, shard="w1")
    for k in range(3):
        w0.append_batch({f"a{k}": [{"id": f"a{k}"}]})
        w1.append_batch({f"b{k}": [{"id": f"b{k}"}]})
    w0.mark_done("solo")
    reads, real = [], pathlib.Path.read_text
    monkeypatch.setattr(pathlib.Path, "read_text", lambda self, *a, **k: (reads.append(pathlib.Path(self).name), real(self, *a, **k))[1])
    reader = core.CheckpointStore(tmp_path, shard="w2")
    assert len([r for r in reads if r.endswith(".done")]) == 6 and reader.done_count() == 7
    reads.clear()
    reader.refresh()
    assert reads == []                                             # nothing new: no done list read again
    w1.append_batch({"b9": [{"id": "b9"}]})
    w1.refresh()
    assert _done_name(["b9"]) not in reads and w1.is_done("a0")    # a store never reads back the done list it wrote
    reads.clear()
    reader.refresh()
    assert reads == [_done_name(["b9"])] and reader.is_done("b9") and reader.done_count() == 8
    assert all(reader.is_done(u) for u in ["a0", "a1", "a2", "b0", "b1", "b2", "solo"])


def _done_name(ids):
    import hashlib
    return "batch-" + hashlib.blake2b(chr(10).join(ids).encode(), digest_size=8).hexdigest() + ".done"


def test_an_unreadable_done_list_of_the_own_shard_does_not_make_its_rows_an_orphan(tmp_path, monkeypatch):
    first = core.CheckpointStore(tmp_path, shard="w0")
    first.append_batch({"a": [{"id": "a"}]})
    rows = next((tmp_path / "units" / "w0").glob("batch-*.jsonl"))
    _fail_reads_of(monkeypatch, _batch_done_file(tmp_path, "w0"), PermissionError(13, "Permission denied"))
    restarted = core.CheckpointStore(tmp_path, shard="w0")         # a relaunch while a scanner holds the done list
    restarted.append_batch({"b": [{"id": "b"}]})                   # clears orphans before writing
    assert rows.exists()                                           # a committed batch is never removed as an orphan
    restarted.refresh()
    assert restarted.is_done("a") and restarted.is_done("b")


def test_an_empty_batch_writes_nothing(tmp_path):
    store = core.CheckpointStore(tmp_path, shard="w0")
    store.append_batch({})
    assert list((tmp_path / "units" / "w0").iterdir()) == [] and store.done_count() == 0


def test_a_per_unit_id_shaped_like_a_batch_file_is_refused(tmp_path):
    """units/batch-*.jsonl and batch-*.done belong to the batch API: a per-unit file with such a name would be removed as
    an orphan or read as a done list."""
    store = core.CheckpointStore(tmp_path)
    for write in (lambda: store.append_rows("batch-0123456789abcdef", [{"i": 0}]), lambda: store.mark_done("batch-x")):
        with pytest.raises(ValueError, match="batch-"):
            write()
    assert list((tmp_path / "units").iterdir()) == []


# ------------------------------------------------------------------------------------------ committed rows reader (R3)
def _rows_in(files):
    import json
    return sorted(json.loads(line)["id"] for path in files for line in path.read_text(encoding="utf-8").splitlines())


def test_committed_row_files_are_the_rows_files_of_units_and_every_shard_with_their_done_marker(tmp_path):
    """The store's done rule, across a single-writer start, a resume with two workers, and a stale shard: <stem>.jsonl is
    committed when <stem>.done is in the same folder (a batch done list, batch-<16 hex>.done, or a unit's marker)."""
    top = core.CheckpointStore(tmp_path)
    top.append_batch({"a": [{"id": "a"}], "b": [{"id": "b"}]})
    top.append_rows("u", [{"id": "u"}])
    top.mark_done("u")
    w0, w1, w2 = (core.CheckpointStore(tmp_path, shard=f"w{k}") for k in range(3))
    w0.append_batch({"c": [{"id": "c"}]})
    w1.append_batch({"d": [{"id": "d"}]})
    w1.append_rows("v", [{"id": "v"}])
    w1.mark_done("v")
    w2.append_batch({"e": [{"id": "e"}]})
    next((tmp_path / "units" / "w2").glob("batch-*.done")).unlink()        # an orphan: cut before its done list
    w2.append_rows("f", [{"id": "f"}])                                      # a unit still running (or abandoned)
    (tmp_path / "units" / "w0" / "batch-0123456789abcdef.jsonl.tmp").write_text('{"id": "t"}\n', encoding="utf-8")
    (tmp_path / "units" / "README").write_text("a dotless file is not a shard\n", encoding="utf-8")
    files, uncommitted = core.committed_row_files(tmp_path)
    assert _rows_in(files) == ["a", "b", "c", "d", "u", "v"] and uncommitted == 2
    folders = [p.parent.name for p in files]
    assert folders == sorted(folders, key=lambda f: (f != "units", f))       # units/ first, then the shards in name order
    assert core.committed_row_files(tmp_path) == (files, uncommitted)        # a fixed order


def test_the_rows_of_a_batch_whose_done_list_cannot_be_read_yet_are_committed(tmp_path, monkeypatch):
    """The done list exists, so the batch is committed (the store's orphan rule); refresh only waits to read its ids."""
    w0 = core.CheckpointStore(tmp_path, shard="w0")
    w0.append_batch({"a": [{"id": "a"}]})
    _fail_reads_of(monkeypatch, _batch_done_file(tmp_path, "w0"), PermissionError(13, "Permission denied"), times=5)
    files, uncommitted = core.committed_row_files(tmp_path)
    assert _rows_in(files) == ["a"] and uncommitted == 0


def test_a_recursive_glob_would_read_orphans_and_unfinished_unit_files(tmp_path):
    """Why the reader exists: units/**/*.jsonl also matches rows that are not results."""
    w0 = core.CheckpointStore(tmp_path, shard="w0")
    w0.append_batch({"a": [{"id": "a"}]})
    next((tmp_path / "units" / "w0").glob("batch-*.done")).unlink()
    w0.append_rows("b", [{"id": "b"}])
    assert _rows_in((tmp_path / "units").rglob("*.jsonl")) == ["a", "b"]
    assert core.committed_row_files(tmp_path) == ([], 2)


def test_d123_the_observation_has_an_optional_isolated_audit_channel_that_defaults_to_none():
    import dataclasses
    fields = {f.name: f for f in dataclasses.fields(core.Observation)}
    assert fields["audit_isolated"].default is None
