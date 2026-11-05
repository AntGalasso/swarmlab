"""Core contracts: settings, streams, checkpoints."""
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
