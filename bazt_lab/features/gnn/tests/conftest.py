"""Shared test helpers: tiny settings, a deterministic fake embedder, synthetic conversation records.

Every number in `TEST_SETTINGS` is a test value (small, fast), not an experiment setting.
"""
from pathlib import Path

import pytest
import torch

from bazt_lab.core import public as core

DIM = 16          # test embedding width (the real one is 384)


def pytest_configure(config):
    # torch_geometric 2.8.1 triggers torch.jit.script's deprecation warning at import; not ours to fix
    config.addinivalue_line("filterwarnings", "ignore:.*torch.jit.script.*:FutureWarning")


def fake_vector(text: str, dim: int = DIM) -> torch.Tensor:
    """Deterministic unit vector per text; texts starting with ATTACK share a strong common direction."""
    v = torch.randn(dim, generator=core.stream("fake_embed", text), dtype=torch.float32)
    if text.startswith("ATTACK"):
        v = v + 4.0 * torch.ones(dim)
    return v / v.norm()


class FakeEncoder:
    """The raw batch encoder an adapter would provide; counts what it was asked to encode."""

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.calls = []                       # one list of texts per call

    def __call__(self, texts):
        self.calls.append(list(texts))
        return torch.stack([fake_vector(t, self.dim) for t in texts])


def make_records(n: int, n_agents: int = 6, rounds: int = 4, p: float = 0.6, k: int = 1, seed: int = 0,
                 attack: str = "insider_pi", task: str = "mmlu", backbone: str = "test-backbone"):
    """Conversation records in the documented log format; attackers' texts start with ATTACK."""
    records = []
    for c in range(n):
        g = core.stream("records", seed, c)
        adj = (core.uniforms(g, (n_agents, n_agents)) <= p).to(torch.int64)
        adj.fill_diagonal_(0)
        attackers = sorted(torch.randperm(n_agents, generator=g)[:k].tolist())
        turns = [[(f"ATTACK says wrong {c} {r} {i}" if i in attackers else f"honest reasoning {c} {r} {i}")
                  for i in range(n_agents)] for r in range(rounds)]
        records.append({"id": f"s{seed}-c{c}", "attack": attack, "task": task, "backbone": backbone,
                        "adj": adj.tolist(), "attackers": attackers, "turns": turns})
    return records


TEST_SETTINGS = {
    "compute": {"device": "cpu", "dtype": "float64", "compile": False, "deterministic": True},
    "llm": {"rounds": 4},
    "gnn": {
        "cut": 0.5,
        "carry_flags": True,
        "fallback_family": "pi_same_task",
        "embedding": {"model": "fake/minilm", "revision": "rev0", "batch_size": 64},
    },
    "gnn_train": {
        "families": [["insider_pi", "mmlu"]],
        "agents": 6, "rounds": 4, "edge_probabilities": [0.6, 1.0], "attackers": [1, 2],
        "conversations_per_setting": 6, "backbone": "test-backbone",
        "hidden_dim": 16, "num_heads": 2, "num_layers": 2, "dropout": 0.2, "aggr_type": "mean",
        "epochs": 6, "keep_checkpoints": 2, "lr": 0.01, "weight_decay": 0.0002, "batch_size": 8,
        "scheduler_t_max": 10, "scheduler_eta_min": 1e-5, "train_fraction": 0.75, "split": "shuffled",
        "train_node_index": "target", "seed": 7,
    },
}


@pytest.fixture
def settings(tmp_path):
    import copy
    s = copy.deepcopy(TEST_SETTINGS)
    s["gnn"]["artifacts_dir"] = str(tmp_path / "artifacts")
    s["gnn_train"]["logs_dir"] = str(tmp_path / "logs")
    s["profile"] = "test"
    return s


@pytest.fixture
def encoder():
    return FakeEncoder()


UPSTREAM_PI = Path(__file__).resolve().parents[4] / "vendor" / "G-safeguard" / "PI"
