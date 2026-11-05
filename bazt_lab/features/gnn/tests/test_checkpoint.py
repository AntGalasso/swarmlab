"""Checkpoints: torch.save of the state_dict plus metadata, atomic, loadable with weights_only."""
import pickle

import pytest
import torch

from bazt_lab.features.gnn.adapters.pyg_model import build_model
from bazt_lab.features.gnn.private.checkpoint import load_checkpoint, save_checkpoint

ARCH = dict(in_channels=6, hidden_channels=8, out_channels=1, heads=2, edge_dim=[3, 6], num_layers=2, dropout=0.2,
            aggr_type="mean")
META = {"arch": ARCH, "family": "insider_pi__mmlu", "best_epoch": 3, "best_val_acc": 0.875, "history": [{"epoch": 0}]}


def inputs():
    g = torch.Generator().manual_seed(0)
    return torch.randn(4, 6, generator=g), torch.tensor([[0, 1, 2], [1, 2, 3]]), torch.randn(3, 3, 6, generator=g)


def test_round_trip_restores_weights_metadata_and_eval_mode(tmp_path):
    torch.manual_seed(1)
    model = build_model(ARCH).eval()
    save_checkpoint(tmp_path / "ck" / "m.pt", model, META)
    loaded, meta = load_checkpoint(tmp_path / "ck" / "m.pt")
    assert meta == META and not loaded.training
    torch.testing.assert_close(loaded(*inputs()), model(*inputs()))


def test_a_save_leaves_no_temporary_file(tmp_path):
    save_checkpoint(tmp_path / "m.pt", build_model(ARCH), META)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.pt"]


def test_a_failing_save_keeps_the_previous_checkpoint_and_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "m.pt"
    first = build_model(ARCH).eval()
    save_checkpoint(path, first, META)
    with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
        save_checkpoint(path, build_model(ARCH), {**META, "unpicklable": lambda: None})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.pt"]
    loaded, _ = load_checkpoint(path)
    torch.testing.assert_close(loaded(*inputs()), first(*inputs()))


def test_metadata_without_an_architecture_is_refused(tmp_path):
    with pytest.raises(ValueError, match="arch"):
        save_checkpoint(tmp_path / "m.pt", build_model(ARCH), {"family": "x"})


def test_a_missing_checkpoint_is_reported_with_its_path(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"nope.pt"):
        load_checkpoint(tmp_path / "nope.pt")


def test_the_file_loads_with_the_safe_loader(tmp_path):
    save_checkpoint(tmp_path / "m.pt", build_model(ARCH), META)
    blob = torch.load(tmp_path / "m.pt", map_location="cpu", weights_only=True)
    assert set(blob) == {"state_dict", "meta"}
