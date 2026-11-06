"""Device handling: core.device(settings) drives training and inference; checkpoints stay device independent."""
import copy

import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.gnn.private.arm import GNNArm
from bazt_lab.features.gnn.private.checkpoint import load_checkpoint
from bazt_lab.features.gnn.private.embedding import CachedEmbedder
from bazt_lab.features.gnn.private.train import checkpoint_path, train_family
from bazt_lab.features.gnn.tests.conftest import FakeEncoder, make_records
from bazt_lab.features.gnn.tests.test_arm import ctx, full_adj, obs, play

FAMILY = "insider_pi__mmlu"


def records():
    return make_records(24, n_agents=6, rounds=4, p=0.8, k=1)


def test_training_asks_core_for_the_device_and_a_missing_compute_setting_is_named(settings, monkeypatch):
    asked = []
    real = core.device
    monkeypatch.setattr(core, "device", lambda s: asked.append(1) or real(s))
    train_family(settings, FAMILY, records(), embed=CachedEmbedder(FakeEncoder()))
    assert asked
    bad = copy.deepcopy(settings)
    bad["gnn"]["artifacts_dir"] += "x"
    del bad["compute"]
    with pytest.raises(core.Unset, match=r"compute\.device"):
        train_family(bad, FAMILY, records(), embed=CachedEmbedder(FakeEncoder()))


def test_the_arm_asks_core_for_the_device(settings, monkeypatch):
    train_family(settings, FAMILY, records(), embed=CachedEmbedder(FakeEncoder()))
    asked = []
    real = core.device
    monkeypatch.setattr(core, "device", lambda s: asked.append(1) or real(s))
    GNNArm(embed=CachedEmbedder(FakeEncoder())).reset(ctx(settings, full_adj()))
    assert asked


def test_a_saved_checkpoint_holds_cpu_tensors(settings):
    train_family(settings, FAMILY, records(), embed=CachedEmbedder(FakeEncoder()))
    blob = torch.load(checkpoint_path(settings, FAMILY), map_location="cpu", weights_only=True)
    assert all(v.device.type == "cpu" for v in blob["state_dict"].values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_training_gives_a_cpu_loadable_checkpoint_and_the_arm_runs_on_the_gpu(settings):
    settings["compute"]["device"] = "cuda"
    train_family(settings, FAMILY, records(), embed=CachedEmbedder(FakeEncoder()))
    model, _ = load_checkpoint(checkpoint_path(settings, FAMILY))
    assert next(model.parameters()).device.type == "cpu"
    a = GNNArm(embed=CachedEmbedder(FakeEncoder()))
    adj = full_adj()
    a.reset(ctx(settings, adj))
    assert next(a._model.parameters()).device.type == "cuda"
    assert play(a, adj, [1])[-1].statistics["gnn_logit"].device.type == "cpu"
    assert obs
