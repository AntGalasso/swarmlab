"""Loads upstream G-Safeguard code from the pinned submodule, for parity tests only.

Nothing is copied into this tree: `vendor/G-safeguard/PI/{gat_with_attr_conv,model}.py` are imported by path, and two
functions of its scripts (`embeddings2graph`, `gen_model_training_set`) are compiled out of their files by name, since
the scripts themselves do their work on import. Upstream's model.py imports `einops.rearrange` without using it; a stub
stands in when einops is not installed.
"""
from __future__ import annotations

import ast
import importlib.util
import pickle
import sys
import types
from pathlib import Path

import numpy
import torch
from torch_geometric.utils import scatter as pyg_scatter
from tqdm import tqdm


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # Reading parity code must not dirty the pinned checkout with __pycache__.
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)  # noqa: S102
    return module


def load_upstream(pi_dir: Path) -> types.SimpleNamespace:
    """`.GATwithEdgeConv` and `.MyGAT` of upstream PI/, loaded under private module names."""
    pi_dir = Path(pi_dir)
    # The conv module keeps its private name in sys.modules for good: PyG resolves a layer's annotations through
    # sys.modules[cls.__module__] each time one is built. Only the alias model.py imports it by is temporary.
    conv = _load("_gsafeguard_pi_conv", pi_dir / "gat_with_attr_conv.py")
    stub_einops = importlib.util.find_spec("einops") is None
    sys.modules["gat_with_attr_conv"] = conv
    if stub_einops:
        sys.modules["einops"] = types.SimpleNamespace(rearrange=None)
    try:
        model = _load("_gsafeguard_pi_model", pi_dir / "model.py")
    finally:
        sys.modules.pop("gat_with_attr_conv", None)
        if stub_einops:
            sys.modules.pop("einops", None)
    return types.SimpleNamespace(GATwithEdgeConv=conv.GATwithEdgeConv, MyGAT=model.MyGAT)


def _function_of(path: Path, name: str, namespace: dict):
    """One top-level function of an upstream script, compiled in `namespace` (the scripts run work on import)."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102 - upstream script function, parity tests only
    return namespace[name]


def _scatter_mean(src, index, dim=0, dim_size=None):
    """torch_scatter.scatter_mean (not installed here) through PyG scatter."""
    return pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce="mean")


def load_embeddings2graph(pi_dir: Path):
    """Upstream inference-side `embeddings2graph(embeddings, adj_matrix) -> (x, edge_index, edge_attr)`."""
    return _function_of(Path(pi_dir) / "main_defense_for_different_topology.py", "embeddings2graph",
                        {"np": numpy, "torch": torch, "scatter_mean": _scatter_mean})


def load_dataset_builder(pi_dir: Path):
    """Upstream `gen_model_training_set(language_dataset, embedding_model, save_path)`."""
    return _function_of(Path(pi_dir) / "gen_training_dataset.py", "gen_model_training_set",
                        {"np": numpy, "pickle": pickle, "tqdm": tqdm})
