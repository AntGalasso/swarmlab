"""GNN training, the upstream `train.py` of G-Safeguard (pinned commit 890c99f) on our conversation logs.

Loop: Adam(lr, weight_decay) with CosineAnnealingLR(T_max, eta_min) stepped once per epoch, BCEWithLogitsLoss on one
logit per node, training batches of `batch_size` graphs in a seeded shuffled order, the best epoch on the validation
graphs kept (strictly better accuracy replaces it). Validation runs one graph at a time with the node feature of the
deployed defence (`"target"`), loss averaged over graphs and accuracy counted over nodes, as upstream's `test()`.

Differences from upstream, all deliberate: the first epoch is always kept (upstream starts its best at 0.0 and could
keep nothing); every random draw is keyed (initial weights, split, per-epoch order and dropout), so a run is
reproducible and a crashed run resumes to the same result; the model is saved with its metadata (`save_checkpoint`).
Upstream's TA/MA `train()` draws a random number of turns and then discards the slice it made; that has no effect and
is not ported.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from bazt_lab.core import public as core
from bazt_lab.features.gnn.adapters.pyg_model import build_model
from bazt_lab.features.gnn.private import epochs
from bazt_lab.features.gnn.private.checkpoint import load_checkpoint, save_checkpoint
from bazt_lab.features.gnn.private.embedding import shared_embedder
from bazt_lab.features.gnn.private.graph import (
    NODE_INDEX,
    Example,
    check_record,
    collate,
    conversation_to_example,
    family_key,
    node_features,
)
from bazt_lab.features.gnn.private.logs import check_logged, read_conversations

SPLITS = ("sequential", "shuffled")
KEYS = ("backbone", "agents", "rounds", "hidden_dim", "num_heads", "num_layers", "dropout", "aggr_type", "epochs", "lr",
        "weight_decay", "batch_size", "scheduler_t_max", "scheduler_eta_min", "train_fraction", "split",
        "train_node_index", "seed")
HYPERPARAMETERS = ("epochs", "lr", "weight_decay", "batch_size", "scheduler_t_max", "scheduler_eta_min",
                   "train_fraction", "split", "train_node_index", "seed")


class TrainingDataIncomplete(RuntimeError):
    """The family's conversations are not all logged yet; train again after the remaining units have run."""

    def __init__(self, family: str, have: int, need: int) -> None:
        super().__init__(f"family {family} has {have} of {need} conversations; run the remaining gnn_train units first")
        self.family, self.have, self.need = family, have, need


def checkpoint_path(settings: Mapping, family: str) -> Path:
    return Path(core.get(settings, "gnn.artifacts_dir")) / f"gnn_{family}.pt"


def _fingerprints(settings: Mapping, cfg: dict, records: Sequence[dict]) -> dict:
    """What a training depends on: the gnn_train settings (and embedding), and the ids of the conversations."""
    emb = {k: core.get(settings, f"gnn.embedding.{k}") for k in ("model", "revision")}
    emb["fake"] = bool(core.get(settings, "gnn.embedding.fake")) if core.is_set(settings, "gnn.embedding.fake") else False
    digest = lambda obj: hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]   # noqa: E731
    return {"settings": digest({"cfg": cfg, "embedding": emb}), "records": digest([r["id"] for r in records])}


def training_finished(settings: Mapping, family: str) -> bool:
    """True when `family` has a checkpoint that ran all `gnn_train.epochs` epochs to the end with the current
    gnn_train settings and, when conversations are logged for the family, on exactly those conversations."""
    path = checkpoint_path(settings, family)
    if not path.is_file():
        return False
    meta = load_checkpoint(path)[1]
    if not (meta.get("finished") and len(meta.get("history", ())) == core.get(settings, "gnn_train.epochs")):
        return False
    logged = read_conversations(settings, family) if core.is_set(settings, "gnn_train.logs_dir") else []
    now = _fingerprints(settings, _config(settings), logged)
    saved = meta.get("identity") or {}
    return saved.get("settings") == now["settings"] and (not logged or saved.get("records") == now["records"])


def _config(settings: Mapping) -> dict:
    cfg = {k: core.get(settings, f"gnn_train.{k}") for k in KEYS}
    if cfg["train_node_index"] not in NODE_INDEX:
        raise ValueError(f"gnn_train.train_node_index must be one of {NODE_INDEX}, got {cfg['train_node_index']!r}")
    if cfg["split"] not in SPLITS:
        raise ValueError(f"gnn_train.split must be one of {SPLITS}, got {cfg['split']!r}")
    for key in ("epochs", "batch_size"):
        if cfg[key] < 1:
            raise ValueError(f"gnn_train.{key} must be at least 1, got {cfg[key]}")
    return cfg


def split_examples(items: Sequence, split: str, train_fraction: float, family: str, seed: int) -> Tuple[list, list]:
    """(train, validation). "sequential": the first `round(n * train_fraction)` items (upstream's positional 80/20);
    "shuffled": the same sizes from a permutation keyed by (seed, family)."""
    if split not in SPLITS:
        raise ValueError(f"gnn_train.split must be one of {SPLITS}, got {split!r}")
    items, n = list(items), len(items)
    n_train = round(n * train_fraction)
    if n_train < 1:
        raise ValueError(f"gnn_train.train_fraction={train_fraction} leaves no training graph out of {n}")
    if n_train >= n:
        raise ValueError(f"gnn_train.train_fraction={train_fraction} leaves no validation graph out of {n}")
    if split == "shuffled":
        items = [items[i] for i in torch.randperm(n, generator=core.stream("gnn_train", seed, family, "split")).tolist()]
    return items[:n_train], items[n_train:]


def evaluate(model: torch.nn.Module, examples: Sequence[Example]) -> Tuple[float, float]:
    """(mean over graphs of the graph's mean BCE, share of nodes classified right) with dropout off."""
    model.eval()
    dev = next(model.parameters(), torch.empty(0)).device       # a parameter-free stub model runs on the CPU
    losses, correct, total = [], 0, 0
    with torch.no_grad():
        for ex in examples:
            x, ei, ea, y = (t.to(dev) for t in (node_features(ex.edge_attr, ex.edge_index, ex.y.numel(), "target"),
                                                ex.edge_index, ex.edge_attr, ex.y))
            out = model(x, ei, ea)
            losses.append(F.binary_cross_entropy_with_logits(out, y.float().unsqueeze(-1)).item())
            correct += ((torch.sigmoid(out) >= 0.5).squeeze(-1) == y.bool()).sum().item()
            total += ex.y.numel()
    return sum(losses) / len(losses), correct / total


def _train_epoch(model, optimizer, train: Sequence[Example], cfg: dict, family: str, epoch: int) -> Tuple[float, float]:
    model.train()
    dev = next(model.parameters()).device
    order = torch.randperm(len(train), generator=core.stream("gnn_train", cfg["seed"], family, "order", epoch)).tolist()
    losses, correct, total = [], 0, 0
    for start in range(0, len(order), cfg["batch_size"]):
        x, edge_index, edge_attr, y = (t.to(dev) for t in collate(
            [train[i] for i in order[start:start + cfg["batch_size"]]], cfg["train_node_index"]))
        optimizer.zero_grad()
        out = model(x, edge_index, edge_attr)
        loss = F.binary_cross_entropy_with_logits(out, y.float().unsqueeze(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        correct += ((torch.sigmoid(out) >= 0.5).squeeze(-1) == y.bool()).sum().item()
        total += y.numel()
    return sum(losses) / len(losses), correct / total


def _unit_id(family: str, cfg: dict, embedding: dict, records: Sequence[dict]) -> str:
    """Names one training: the family, every setting it depends on and the conversations it trains on."""
    blob = json.dumps({"family": family, "cfg": cfg, "embedding": embedding, "ids": [r["id"] for r in records]},
                      sort_keys=True)
    return f"gnn-{family}-{hashlib.sha256(blob.encode()).hexdigest()[:12]}"


def train_family(settings: Mapping, family: str, records: Sequence[dict],
                 embed: Optional[Callable[[Sequence[str]], torch.Tensor]] = None) -> dict:
    """Train the GNN of `family` on `records` (conversation logs, see `logs`) and return its checkpoint metadata.

    The best validated epoch is written to `checkpoint_path`. Progress is saved after every epoch (state in
    `<gnn.artifacts_dir>/train_state`, an epoch log beside it), so a crashed call resumes after its last finished
    epoch and ends where an uninterrupted one would. A finished training is returned, not repeated."""
    cfg = _config(settings)
    keep = core.get(settings, "gnn_train.keep_checkpoints")
    if keep < 1:
        raise ValueError(f"gnn_train.keep_checkpoints must be at least 1, got {keep}")
    if not records:
        raise ValueError(f"no conversations to train {family} on")
    for record in records:
        check_record(record)
    embedding = {"model": core.get(settings, "gnn.embedding.model"), "revision": core.get(settings, "gnn.embedding.revision")}
    path, uid = checkpoint_path(settings, family), _unit_id(family, cfg, embedding, records)
    store = core.CheckpointStore(Path(core.get(settings, "gnn.artifacts_dir")) / "train_state")
    epoch_path = epochs.epoch_dir(settings, uid)

    if store.is_done(uid):
        meta = load_checkpoint(path)[1] if path.is_file() else None
        if meta is None or meta.get("unit_id") != uid:
            raise RuntimeError(f"training {uid} is finished but {path} is not its checkpoint; delete {store.root}, "
                               f"{epoch_path} and {path} to train {family} again")
        return meta

    embed = embed or shared_embedder(settings)
    embed([text for record in records for turn in record["turns"] for text in turn])      # one batched pass, cached
    examples = [conversation_to_example(record, embed) for record in records]
    train, val = split_examples(examples, cfg["split"], cfg["train_fraction"], family, cfg["seed"])
    dim = examples[0].edge_attr.shape[-1]
    arch = {"in_channels": dim, "hidden_channels": cfg["hidden_dim"], "out_channels": 1, "heads": cfg["num_heads"],
            "edge_dim": [cfg["rounds"], dim], "num_layers": cfg["num_layers"], "dropout": cfg["dropout"],
            "aggr_type": cfg["aggr_type"]}
    with torch.random.fork_rng():
        torch.manual_seed(core.seed_of("gnn_train", cfg["seed"], family, "init"))
        model = build_model(arch).to(core.device(settings))
    optimizer = Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["scheduler_t_max"], eta_min=cfg["scheduler_eta_min"])
    meta = {"arch": arch, "family": family, "backbone": cfg["backbone"], "n_train": len(train), "n_val": len(val),
            "hyperparameters": {k: cfg[k] for k in HYPERPARAMETERS}, "embedding": embedding,
            "torch": str(torch.__version__), "unit_id": uid}

    history, best, first = [], -1.0, 0
    state = epochs.load_latest(epoch_path)
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, best, first = state["history"], state["best_val_acc"], state["epoch"] + 1
        logged = {row["epoch"] for row in store.resume_rows(uid)}
        store.append_rows(uid, [row for row in history if row["epoch"] not in logged])    # a crash between the two saves

    for epoch in range(first, cfg["epochs"]):
        with torch.random.fork_rng():
            torch.manual_seed(core.seed_of("gnn_train", cfg["seed"], family, "dropout", epoch))
            train_loss, train_acc = _train_epoch(model, optimizer, train, cfg, family, epoch)
        val_loss, val_acc = evaluate(model, val)
        row = {"epoch": epoch, "lr": scheduler.get_last_lr()[0], "train_loss": train_loss, "train_acc": train_acc,
               "val_loss": val_loss, "val_acc": val_acc, "best": val_acc > best}
        scheduler.step()
        if row["best"]:
            best = val_acc
            save_checkpoint(path, model, {**meta, "best_epoch": epoch, "best_val_acc": val_acc,
                                          "best_val_loss": val_loss, "history": [*history, row]})
        history.append(row)
        epochs.save_epoch(epoch_path, {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                       "scheduler": scheduler.state_dict(), "history": history, "best_val_acc": best},
                          keep, max(r["epoch"] for r in history if r["best"]))
        store.append_rows(uid, [row])

    best_model, saved = load_checkpoint(path)
    if saved["best_epoch"] != max(row["epoch"] for row in history if row["best"]):
        raise RuntimeError(f"{path} holds epoch {saved['best_epoch']} but the training history says otherwise; "
                           f"delete {store.root}, {epoch_path} and {path} to train {family} again")
    meta = {**saved, "history": history, "finished": True, "identity": _fingerprints(settings, cfg, records)}
    save_checkpoint(path, best_model, meta)
    store.mark_done(uid)
    return meta


def train_unit(unit: Mapping, settings: Mapping,
               embed: Optional[Callable[[Sequence[str]], torch.Tensor]] = None) -> dict:
    """Train the GNN of the family (`unit["attack"]`, `unit["task"]`) of a `gnn_train` unit on its logged conversations.

    A family has one GNN, trained on all its units' conversations, so this raises `TrainingDataIncomplete` until the
    last unit of the family has logged its conversations: call it after any unit, it works once the family is whole."""
    family = family_key(unit["attack"], unit["task"])
    families = sorted(family_key(*f) for f in core.get(settings, "gnn_train.families"))
    if family not in families:
        raise ValueError(f"{family} is not one of the gnn_train.families: {families}")
    return _train_logged(settings, family, embed)


def _train_logged(settings: Mapping, family: str, embed) -> dict:
    need = (core.get(settings, "gnn_train.conversations_per_setting") * len(core.get(settings, "gnn_train.edge_probabilities"))
            * len(core.get(settings, "gnn_train.attackers")))
    check_logged(settings, family)
    records = read_conversations(settings, family)
    if len(records) < need:
        raise TrainingDataIncomplete(family, len(records), need)
    if len(records) > need:
        raise ValueError(f"family {family} has {len(records)} conversations, the gnn_train settings ask for {need}")
    return train_family(settings, family, records, embed)


def train_pending(settings: Mapping, embed: Optional[Callable[[Sequence[str]], torch.Tensor]] = None) -> list:
    """Train, or resume, every gnn_train family whose conversations are all logged and whose training is not
    finished. Returns the metadata of the trainings it ran; a no-op (empty list) when nothing is pending."""
    need = (core.get(settings, "gnn_train.conversations_per_setting") * len(core.get(settings, "gnn_train.edge_probabilities"))
            * len(core.get(settings, "gnn_train.attackers")))
    families = sorted(family_key(*f) for f in core.get(settings, "gnn_train.families"))
    for f in families:
        check_logged(settings, f)                 # foreign logs refuse before anything trains
    pending = [f for f in families
               if len(read_conversations(settings, f)) >= need and not training_finished(settings, f)]
    return [_train_logged(settings, f, embed) for f in pending]
