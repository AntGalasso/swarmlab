"""Training: the upstream loop (Adam, BCE, cosine schedule, best validated epoch kept), resumable per epoch."""
import copy
from pathlib import Path

import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.gnn.private import epochs
from bazt_lab.features.gnn.private.checkpoint import load_checkpoint
from bazt_lab.features.gnn.private.embedding import CachedEmbedder
from bazt_lab.features.gnn.private.graph import Example, conversation_to_example
from bazt_lab.features.gnn.private.logs import log_conversations
from bazt_lab.features.gnn.private.train import (
    TrainingDataIncomplete,
    checkpoint_path,
    evaluate,
    split_examples,
    train_family,
    train_unit,
)
from bazt_lab.features.gnn.tests.conftest import FakeEncoder, make_records

FAMILY = "insider_pi__mmlu"


def records(n=24, **kw):
    return make_records(n, n_agents=6, rounds=4, p=0.8, k=1, **kw)


def unit(uid="GNN-aaa"):
    return {"block": "gnn_train", "platform": "llm", "arm": "GNN", "attack": "insider_pi", "task": "mmlu", "id": uid,
            "edge_probability": 0.6, "attackers": 1, "backbone": "test-backbone", "conversations": 6, "agents": 6,
            "rounds": 4}


# ------------------------------------------------------------------------------------------ split_examples
def test_the_sequential_split_is_the_upstream_positional_split():
    items = list(range(10))
    assert split_examples(items, "sequential", 0.8, FAMILY, 7) == (list(range(8)), [8, 9])
    assert split_examples(list(range(7)), "sequential", 0.8, FAMILY, 7) == (list(range(5)), [5, 6])   # int(7 * 0.8)


def test_the_shuffled_split_partitions_the_items_and_depends_only_on_its_key():
    items = list(range(20))
    train, val = split_examples(items, "shuffled", 0.75, FAMILY, 7)
    assert len(train) == 15 and sorted(train + val) == items
    assert (train, val) == split_examples(items, "shuffled", 0.75, FAMILY, 7)
    assert train != split_examples(items, "shuffled", 0.75, FAMILY, 8)[0]
    assert train != items[:15]


def test_a_split_that_leaves_one_side_empty_is_refused():
    with pytest.raises(ValueError, match="train_fraction"):
        split_examples([1, 2, 3], "sequential", 0.2, FAMILY, 7)          # int(3 * 0.2) = 0 training graphs
    with pytest.raises(ValueError, match="validation"):
        split_examples([1, 2, 3], "sequential", 1.0, FAMILY, 7)


def test_an_unknown_split_is_refused():
    with pytest.raises(ValueError, match="split"):
        split_examples([1, 2], "random", 0.5, FAMILY, 7)


# ---------------------------------------------------------------------------------------------- evaluate
class ConstantLogit(torch.nn.Module):
    def forward(self, x, edge_index, edge_attr):
        return torch.full((x.shape[0], 1), 2.0)


def graph(y):
    n = len(y)
    return Example(torch.tensor([[0], [1]]), torch.zeros(1, 2, 3), torch.tensor(y)), n


def test_validation_averages_the_loss_over_graphs_and_counts_accuracy_over_nodes():
    # Every node gets logit 2 (probability 0.88, predicted attacker). Per-node loss: y=1 -> log(1+e^-2) = 0.126928,
    # y=0 -> log(1+e^2) = 2.126928. Graph A y=[1,0,0,0]: loss 1.626928, 1 of 4 right. Graph B y=[1,1]: loss 0.126928,
    # 2 of 2 right. Loss = mean of graph losses = 0.876928; accuracy = 3 of 6 nodes = 0.5 (not the graph mean 0.625).
    loss, acc = evaluate(ConstantLogit().eval(), [graph([1, 0, 0, 0])[0], graph([1, 1])[0]])
    assert loss == pytest.approx(0.8769280110, abs=1e-6) and acc == pytest.approx(0.5)


# ---------------------------------------------------------------------------------------------- training
def test_training_learns_to_flag_the_attacker_and_keeps_the_best_validated_epoch(settings):
    recs = records()
    meta = train_family(settings, FAMILY, recs, embed=CachedEmbedder(FakeEncoder()))
    history = meta["history"]
    assert [h["epoch"] for h in history] == list(range(6))
    assert meta["best_val_acc"] == max(h["val_acc"] for h in history) >= 0.9
    first_best = next(h["epoch"] for h in history if h["val_acc"] == meta["best_val_acc"])
    assert meta["best_epoch"] == first_best                                  # a later tie does not replace it

    model, stored = load_checkpoint(checkpoint_path(settings, FAMILY))
    assert stored["best_epoch"] == meta["best_epoch"] and not model.training
    embed = CachedEmbedder(FakeEncoder())
    examples = [conversation_to_example(r, embed) for r in recs]
    _, val = split_examples(examples, "shuffled", 0.75, FAMILY, 7)
    val_loss, val_acc = evaluate(model, val)
    assert val_acc == meta["best_val_acc"] and val_loss == pytest.approx(meta["best_val_loss"], abs=1e-5)


def test_the_checkpoint_records_what_it_was_trained_with(settings):
    meta = train_family(settings, FAMILY, records(), embed=CachedEmbedder(FakeEncoder()))
    assert meta["arch"] == {"in_channels": 16, "hidden_channels": 16, "out_channels": 1, "heads": 2,
                            "edge_dim": [4, 16], "num_layers": 2, "dropout": 0.2, "aggr_type": "mean"}
    assert meta["family"] == FAMILY and meta["backbone"] == "test-backbone"
    assert (meta["n_train"], meta["n_val"]) == (18, 6)
    assert meta["embedding"] == {"model": "fake/minilm", "revision": "rev0"}
    assert meta["hyperparameters"]["lr"] == 0.01 and meta["hyperparameters"]["train_node_index"] == "target"
    assert meta["torch"] == torch.__version__


def test_the_same_settings_and_data_train_the_same_model(settings, tmp_path):
    other = copy.deepcopy(settings)
    other["gnn"]["artifacts_dir"] = str(tmp_path / "other")
    a = train_family(settings, FAMILY, records(), embed=CachedEmbedder(FakeEncoder()))
    b = train_family(other, FAMILY, records(), embed=CachedEmbedder(FakeEncoder()))
    assert a["history"] == b["history"]
    wa = load_checkpoint(checkpoint_path(settings, FAMILY))[0].state_dict()
    wb = load_checkpoint(checkpoint_path(other, FAMILY))[0].state_dict()
    assert all(torch.equal(wa[k], wb[k]) for k in wa)


def test_the_seed_changes_the_run(settings, tmp_path):
    other = copy.deepcopy(settings)
    other["gnn"]["artifacts_dir"] = str(tmp_path / "other")
    other["gnn_train"]["seed"] = 8
    a = train_family(settings, FAMILY, records(), embed=CachedEmbedder(FakeEncoder()))
    b = train_family(other, FAMILY, records(), embed=CachedEmbedder(FakeEncoder()))
    assert a["history"][0]["train_loss"] != b["history"][0]["train_loss"]


def test_the_training_node_index_picks_the_feature_the_model_trains_on(settings, tmp_path):
    other = copy.deepcopy(settings)
    other["gnn"]["artifacts_dir"] = str(tmp_path / "other")
    other["gnn_train"]["train_node_index"] = "source"
    a = train_family(settings, FAMILY, records(), embed=CachedEmbedder(FakeEncoder()))
    b = train_family(other, FAMILY, records(), embed=CachedEmbedder(FakeEncoder()))
    assert a["history"][0]["train_loss"] != b["history"][0]["train_loss"]
    assert b["hyperparameters"]["train_node_index"] == "source"


@pytest.mark.parametrize("key,value,match", [
    ("train_node_index", "both", "gnn_train.train_node_index"),
    ("split", "random", "gnn_train.split"),
    ("epochs", 0, "gnn_train.epochs"),
    ("batch_size", 0, "gnn_train.batch_size"),
])
def test_an_invalid_training_setting_is_refused_before_any_embedding(settings, key, value, match):
    settings["gnn_train"][key] = value
    enc = FakeEncoder()
    with pytest.raises(ValueError, match=match):
        train_family(settings, FAMILY, records(), embed=CachedEmbedder(enc))
    assert enc.calls == []


def test_a_missing_training_setting_is_named(settings):
    del settings["gnn_train"]["lr"]
    with pytest.raises(core.Unset, match=r"gnn_train\.lr"):
        train_family(settings, FAMILY, records(), embed=CachedEmbedder(FakeEncoder()))


def test_a_malformed_record_refuses_the_training_before_any_embedding(settings):
    recs = records()
    recs[5]["attackers"] = [42]
    enc = FakeEncoder()
    with pytest.raises(ValueError, match=rf"{recs[5]['id']}.*attacker"):
        train_family(settings, FAMILY, recs, embed=CachedEmbedder(enc))
    assert enc.calls == []


# ------------------------------------------------------------------------------------------------- resume
def uninterrupted(settings, tmp_path):
    clean = copy.deepcopy(settings)
    clean["gnn"]["artifacts_dir"] = str(tmp_path / "clean")
    return train_family(clean, FAMILY, records(), embed=CachedEmbedder(FakeEncoder()))


def test_a_crash_between_epochs_resumes_to_the_uninterrupted_result(settings, tmp_path, monkeypatch):
    expected = uninterrupted(settings, tmp_path)
    real, calls = epochs.save_epoch, {"n": 0}

    def crashing(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 4:                                   # epoch 3 trained, its state never saved
            raise RuntimeError("simulated crash")
        return real(*args, **kwargs)

    monkeypatch.setattr(epochs, "save_epoch", crashing)
    with pytest.raises(RuntimeError, match="simulated crash"):
        train_family(settings, FAMILY, records(), embed=CachedEmbedder(FakeEncoder()))
    monkeypatch.undo()

    resumed = train_family(settings, FAMILY, records(), embed=CachedEmbedder(FakeEncoder()))
    assert resumed["history"] == expected["history"]
    assert (resumed["best_epoch"], resumed["best_val_acc"]) == (expected["best_epoch"], expected["best_val_acc"])
    a = load_checkpoint(checkpoint_path(settings, FAMILY))[0].state_dict()
    b = load_checkpoint(checkpoint_path({**settings, "gnn": {**settings["gnn"], "artifacts_dir": str(tmp_path / "clean")}},
                                        FAMILY))[0].state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_the_epoch_log_has_each_epoch_once_even_when_a_crash_lands_between_state_and_row(settings, monkeypatch):
    real, calls = core.CheckpointStore.append_rows, {"n": 0}

    def crashing(self, uid, rows):
        calls["n"] += 1
        if calls["n"] == 3:                                   # epoch 2: state saved, its row not
            raise RuntimeError("simulated crash")
        real(self, uid, rows)

    monkeypatch.setattr(core.CheckpointStore, "append_rows", crashing)
    with pytest.raises(RuntimeError, match="simulated crash"):
        train_family(settings, FAMILY, records(), embed=CachedEmbedder(FakeEncoder()))
    monkeypatch.undo()
    meta = train_family(settings, FAMILY, records(), embed=CachedEmbedder(FakeEncoder()))
    log = core.CheckpointStore(Path(settings["gnn"]["artifacts_dir"]) / "train_state").resume_rows(meta["unit_id"])
    assert [row["epoch"] for row in log] == list(range(6))
    assert log == meta["history"]


def test_a_finished_training_is_not_repeated(settings):
    enc = FakeEncoder()
    first = train_family(settings, FAMILY, records(), embed=CachedEmbedder(enc))
    calls = len(enc.calls)
    again = train_family(settings, FAMILY, records(), embed=CachedEmbedder(enc))
    assert again == first and len(enc.calls) == calls


def test_other_data_or_hyperparameters_are_a_new_training(settings):
    first = train_family(settings, FAMILY, records(), embed=CachedEmbedder(FakeEncoder()))
    settings["gnn_train"]["lr"] = 0.02
    second = train_family(settings, FAMILY, records(), embed=CachedEmbedder(FakeEncoder()))
    assert second["unit_id"] != first["unit_id"] and second["history"] != first["history"]
    third = train_family(settings, FAMILY, records(seed=1), embed=CachedEmbedder(FakeEncoder()))
    assert third["unit_id"] != second["unit_id"]


def test_a_finished_training_whose_checkpoint_was_deleted_is_reported_not_silently_redone(settings):
    train_family(settings, FAMILY, records(), embed=CachedEmbedder(FakeEncoder()))
    checkpoint_path(settings, FAMILY).unlink()
    with pytest.raises(RuntimeError, match="checkpoint"):
        train_family(settings, FAMILY, records(), embed=CachedEmbedder(FakeEncoder()))


# ------------------------------------------------------------------------------------------------ train_unit
def log_all(settings, recs):
    for i in range(0, len(recs), 6):
        log_conversations(settings, unit(f"GNN-{i:03d}"), recs[i:i + 6])


def test_train_unit_trains_the_units_family_from_its_logs(settings):
    recs = records()                                          # 6 conversations x 2 edge probabilities x 2 attacker counts
    log_all(settings, recs)
    meta = train_unit(unit(), settings, embed=CachedEmbedder(FakeEncoder()))
    assert meta["family"] == FAMILY and meta["n_train"] + meta["n_val"] == 24
    assert checkpoint_path(settings, FAMILY).is_file()


def test_train_unit_waits_for_the_whole_family(settings):
    log_all(settings, records()[:18])
    with pytest.raises(TrainingDataIncomplete) as e:
        train_unit(unit(), settings, embed=CachedEmbedder(FakeEncoder()))
    assert (e.value.have, e.value.need) == (18, 24) and FAMILY in str(e.value)
    assert not checkpoint_path(settings, FAMILY).exists()


def test_train_unit_refuses_more_conversations_than_the_setting_asks_for(settings):
    log_all(settings, records(30))
    with pytest.raises(ValueError, match=r"30.*24"):
        train_unit(unit(), settings, embed=CachedEmbedder(FakeEncoder()))


@pytest.mark.parametrize("change,message", [
    (lambda r: r.update(backbone="other-model"), "backbone"),
    (lambda r: r.update(turns=r["turns"][:2]), "rounds"),
    (lambda r: r.update(adj=[[0, 1], [1, 0]], attackers=[0], turns=[["a", "b"]] * 4), "agents"),
])
def test_train_unit_refuses_logs_that_do_not_match_the_training_setting(settings, change, message):
    recs = records()
    change(recs[3])
    log_all(settings, recs)
    with pytest.raises(ValueError, match=rf"{recs[3]['id']}.*{message}"):
        train_unit(unit(), settings, embed=CachedEmbedder(FakeEncoder()))


def test_train_unit_refuses_a_family_outside_the_training_families(settings):
    u = {**unit(), "task": "csqa"}
    with pytest.raises(ValueError, match="insider_pi__csqa"):
        train_unit(u, settings, embed=CachedEmbedder(FakeEncoder()))


def test_train_unit_is_repeatable(settings):
    log_all(settings, records())
    enc = FakeEncoder()
    first = train_unit(unit(), settings, embed=CachedEmbedder(enc))
    calls = len(enc.calls)
    assert train_unit(unit("GNN-006"), settings, embed=CachedEmbedder(enc)) == first and len(enc.calls) == calls


# ------------------------------------------------------------------------------- per-epoch checkpoints
def epoch_files(settings, meta):
    return sorted(p.name for p in epochs.epoch_dir(settings, meta["unit_id"]).glob("epoch_*.pt"))


def test_every_epoch_leaves_a_resumable_file_and_only_the_last_k_plus_the_best_are_kept(settings):
    meta = train_family(settings, FAMILY, records(), embed=CachedEmbedder(FakeEncoder()))
    names = epoch_files(settings, meta)
    wanted = {f"epoch_{e:04d}.pt" for e in (4, 5, meta["best_epoch"])}       # last 2 (keep_checkpoints) + the best
    assert set(names) == wanted and "epoch_0005.pt" in names                 # the final epoch is always there


def test_an_epoch_file_holds_the_full_resumable_state(settings):
    meta = train_family(settings, FAMILY, records(), embed=CachedEmbedder(FakeEncoder()))
    state = epochs.load_latest(epochs.epoch_dir(settings, meta["unit_id"]))
    assert {"epoch", "model", "optimizer", "scheduler", "history", "best_val_acc", "rng"} <= set(state)
    assert state["epoch"] == 5 and [h["epoch"] for h in state["history"]] == list(range(6))


def test_resume_picks_the_newest_complete_file_and_skips_a_corrupt_one(settings):
    meta = train_family(settings, FAMILY, records(), embed=CachedEmbedder(FakeEncoder()))
    d = epochs.epoch_dir(settings, meta["unit_id"])
    (d / "epoch_0009.pt").write_bytes(b"torn")
    assert epochs.load_latest(d)["epoch"] == 5
    assert epochs.load_latest(d / "nowhere") is None


def test_a_crash_at_any_epoch_resumes_to_identical_weights_and_history(settings, tmp_path, monkeypatch):
    expected = uninterrupted(settings, tmp_path)
    clean = {**settings, "gnn": {**settings["gnn"], "artifacts_dir": str(tmp_path / "clean")}}
    expected_w = load_checkpoint(checkpoint_path(clean, FAMILY))[0].state_dict()
    for crash_at in range(1, 7):
        s = copy.deepcopy(settings)
        s["gnn"]["artifacts_dir"] = str(tmp_path / f"crash{crash_at}")
        real, calls = epochs.save_epoch, {"n": 0}

        def crashing(*args, _real=real, _calls=calls, _at=crash_at, **kwargs):
            _calls["n"] += 1
            if _calls["n"] == _at:
                raise RuntimeError("simulated crash")
            return _real(*args, **kwargs)

        monkeypatch.setattr(epochs, "save_epoch", crashing)
        with pytest.raises(RuntimeError, match="simulated crash"):
            train_family(s, FAMILY, records(), embed=CachedEmbedder(FakeEncoder()))
        monkeypatch.undo()
        resumed = train_family(s, FAMILY, records(), embed=CachedEmbedder(FakeEncoder()))
        assert resumed["history"] == expected["history"], crash_at
        w = load_checkpoint(checkpoint_path(s, FAMILY))[0].state_dict()
        assert all(torch.equal(w[k], expected_w[k]) for k in w), crash_at


def test_keep_checkpoints_must_be_set_and_positive(settings):
    settings["gnn_train"]["keep_checkpoints"] = 0
    with pytest.raises(ValueError, match=r"gnn_train\.keep_checkpoints"):
        train_family(settings, FAMILY, records(), embed=CachedEmbedder(FakeEncoder()))
    del settings["gnn_train"]["keep_checkpoints"]
    with pytest.raises(core.Unset, match=r"gnn_train\.keep_checkpoints"):
        train_family(settings, FAMILY, records(), embed=CachedEmbedder(FakeEncoder()))
