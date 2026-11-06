"""Graph tensors of the G-Safeguard GNN, built the way upstream builds them.

Convention (upstream `gen_training_dataset.py` and `main_defense_for_different_topology.py`, pinned commit 890c99f):
  adj[i, j] truthy        agent i sends to agent j (the agents' own prompts read `adj[:, j]` as j's inbox)
  edge_index              the row-major nonzeros of adj: row 0 = sender i, row 1 = receiver j
  edge_attr [E, turns, D] on edge (i, j) the embeddings of agent **j's** messages, one per turn
  node feature x          the mean of the first turn's edge_attr over the edges ending at the node, which is the
                          node's own first message (zeros for a node nobody sends to)
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Sequence, Tuple

import torch
from torch import Tensor

NODE_INDEX = ("target", "source")


class Example(NamedTuple):
    """One labelled conversation graph."""
    edge_index: Tensor          # int64 [2, E]
    edge_attr: Tensor           # float32 [E, turns, D]
    y: Tensor                   # int64 [N]: 1 = attacker


def family_key(attack: str, task: str) -> str:
    return f"{attack}__{task}"


def graph_inputs(adj: Tensor, turn_embeddings: Tensor) -> Tuple[Tensor, Tensor]:
    """(edge_index, edge_attr) of a graph `adj` [N, N] and the embeddings [turns, N, D] of what each agent said."""
    if adj.dim() != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adjacency must be square [N, N], got {tuple(adj.shape)}")
    if turn_embeddings.dim() != 3:
        raise ValueError(f"turn embeddings must be [turns, agents, dim], got {tuple(turn_embeddings.shape)}")
    if turn_embeddings.shape[1] != adj.shape[0]:
        raise ValueError(f"the graph has {adj.shape[0]} agents, the embeddings {turn_embeddings.shape[1]}")
    edge_index = torch.stack(torch.nonzero(adj, as_tuple=True))
    return edge_index, turn_embeddings[:, edge_index[1]].transpose(0, 1).contiguous()


def node_features(edge_attr: Tensor, edge_index: Tensor, num_nodes: int, index: str) -> Tensor:
    """x [num_nodes, D]: mean of the first turn's edge_attr over the edges whose `index` end is the node.

    index "target" is what upstream's validation and deployed defence use (the node's own first message);
    "source" is what upstream's train() uses (the mean over the node's receivers' first messages).
    """
    if index not in NODE_INDEX:
        raise ValueError(f"node index must be one of {NODE_INDEX}, got {index!r}")
    first = edge_attr[:, 0, :]
    where = edge_index[1 if index == "target" else 0]
    x = first.new_zeros(num_nodes, first.shape[-1])
    return x.scatter_reduce_(0, where[:, None].expand_as(first), first, reduce="mean", include_self=False)


def collate(examples: Sequence[Example], index: str) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """The disjoint union of several graphs as (x, edge_index, edge_attr, y)."""
    offsets, total = [], 0
    for e in examples:
        offsets.append(total)
        total += e.y.numel()
    edge_index = torch.cat([e.edge_index + o for e, o in zip(examples, offsets, strict=True)], dim=1)
    edge_attr = torch.cat([e.edge_attr for e in examples])
    return (node_features(edge_attr, edge_index, total, index), edge_index, edge_attr,
            torch.cat([e.y for e in examples]))


def predict_logits(model: torch.nn.Module, adj: Tensor, turn_embeddings: Tensor) -> Tensor:
    """Attacker logits [N] of the deployed defence for a conversation so far (`turn_embeddings` [turns, N, D])."""
    if model.training:
        raise ValueError("model must be in eval mode (dropout off) for prediction")
    device = next(model.parameters()).device
    edge_index, edge_attr = graph_inputs(adj.cpu(), turn_embeddings.cpu())
    with torch.no_grad():
        x = node_features(edge_attr, edge_index, adj.shape[0], "target")
        return model(x.to(device), edge_index.to(device), edge_attr.to(device)).squeeze(-1).cpu()


def check_record(record: dict) -> Tuple[Tensor, list, list]:
    """(adj bool [N, N], attackers, turns) of a conversation record, or ValueError naming the record."""
    rid = record.get("id", "<no id>")

    def bad(why: str):
        return ValueError(f"record {rid}: {why}")

    try:
        raw = torch.tensor(record["adj"])
    except (TypeError, ValueError, RuntimeError) as e:
        raise bad(f"adj must be a rectangular numeric matrix ({e})") from e
    if raw.dim() != 2 or raw.shape[0] != raw.shape[1]:
        raise bad(f"adj must be square, got shape {tuple(raw.shape)}")
    if not bool(((raw == 0) | (raw == 1)).all()):             # checked before any cast: 0.5 must not become 0
        raise bad("adj entries must be 0 or 1")
    adj = raw.to(torch.int64)
    n = adj.shape[0]
    attackers = record["attackers"]
    if not isinstance(attackers, (list, tuple)) or any(type(a) is not int for a in attackers):
        raise bad(f"attackers must be a list of int agent indices, got {attackers!r}")
    attackers = list(attackers)
    if len(set(attackers)) != len(attackers) or any(not 0 <= a < n for a in attackers):
        raise bad(f"attackers must be distinct agent indices in [0, {n}), got {attackers}")
    turns = record["turns"]
    if len(turns) < 1:
        raise bad("needs at least one turn")
    for r, turn in enumerate(turns):
        if len(turn) != n:
            raise bad(f"turn {r} has {len(turn)} texts for {n} agents")
        if not all(isinstance(t, str) for t in turn):
            raise bad(f"turn {r} must hold str texts")
    return adj.bool(), attackers, turns


def conversation_to_example(record: dict, embed: Callable[[Sequence[str]], Tensor]) -> Example:
    """A conversation record (see CONTRACT.md) as a labelled graph; `embed` maps texts to float32 [n, D] rows."""
    adj, attackers, turns = check_record(record)
    n = adj.shape[0]
    emb = embed([text for turn in turns for text in turn]).view(len(turns), n, -1)
    edge_index, edge_attr = graph_inputs(adj, emb)
    y = torch.zeros(n, dtype=torch.int64)
    y[attackers] = 1
    return Example(edge_index, edge_attr, y)
