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
