"""G-Safeguard's GNN (Wang et al., ACL 2025) ported to torch_geometric 2.8.1.

Ported from `PI/gat_with_attr_conv.py` and `PI/model.py` of github.com/wslong20/G-safeguard at the pinned commit
890c99f (no code is copied into the repository's tree beyond this port; the submodule stays under vendor/).
Parameter names and shapes equal upstream's, so an upstream `state_dict` loads here and ours loads upstream.

Deliberate omissions (never used by MyGAT, noted so nobody expects them): bipartite `in_channels` tuples and
torch_sparse `SparseTensor` adjacency; only a dense COO `edge_index` tensor is supported.
"""
from __future__ import annotations

from typing import Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax


class GATwithEdgeConv(MessagePassing):
    """GATConv plus a second edge projection: message = alpha * (x_j + W_e e_ij).

    GATConv only lets the edge attribute shift the attention score (`lin_edge`, `att_edge`). Here the edge attribute
    also enters the message through the extra `lin_edge_attr`, and `forward` returns `(out, W_e e)`, so a stacked
    layer receives the projected edge attribute. `edge_dim` is mandatory.
    """

    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0, add_self_loops: bool = False,
                 edge_dim: Optional[int] = None, fill_value: Union[float, Tensor, str] = "mean", bias: bool = True,
                 residual: bool = False, **kwargs):
        if edge_dim is None:
            raise ValueError("GATwithEdgeConv needs edge_dim: every layer projects edge_attr")
        if not isinstance(in_channels, int):
            raise TypeError("only int in_channels is supported (no bipartite graphs)")
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)
        self.in_channels, self.out_channels, self.heads, self.concat = in_channels, out_channels, heads, concat
        self.negative_slope, self.dropout, self.add_self_loops = negative_slope, dropout, add_self_loops
        self.edge_dim, self.fill_value, self.residual = edge_dim, fill_value, residual

        self.lin = Linear(in_channels, heads * out_channels, bias=False, weight_initializer="glorot")
        self.att_src = Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = Parameter(torch.empty(1, heads, out_channels))
        self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False, weight_initializer="glorot")
        self.lin_edge_attr = Linear(edge_dim, heads * out_channels, bias=False, weight_initializer="glorot")
        self.att_edge = Parameter(torch.empty(1, heads, out_channels))

        total_out_channels = out_channels * (heads if concat else 1)
        if residual:
            self.res = Linear(in_channels, total_out_channels, bias=False, weight_initializer="glorot")
        else:
            self.register_parameter("res", None)
        if bias:
            self.bias = Parameter(torch.empty(total_out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        self.lin_edge.reset_parameters()
        self.lin_edge_attr.reset_parameters()
        if self.res is not None:
            self.res.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None, size=None,
                return_attention_weights: Optional[bool] = None):
        """Returns `(out, W_e e)`; with `return_attention_weights` a bool, `(out, (edge_index, alpha))` as upstream."""
        if edge_attr is None:
            raise ValueError("GATwithEdgeConv.forward needs edge_attr")
        if x.dim() != 2:
            raise ValueError("static graphs are not supported: x must be [num_nodes, channels]")
        H, C = self.heads, self.out_channels

        res = self.res(x) if self.res is not None else None
        x_src = x_dst = self.lin(x).view(-1, H, C)
        alpha = ((x_src * self.att_src).sum(dim=-1), (x_dst * self.att_dst).sum(-1))

        if self.add_self_loops:
            num_nodes = x_src.size(0)
            num_nodes = min(size) if size is not None else num_nodes
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=self.fill_value,
                                                   num_nodes=num_nodes)

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr, size=size)

        # propagate_type: (x: OptPairTensor, alpha: Tensor, edge_attr: OptTensor)
        edge_attr = self.lin_edge_attr(edge_attr)
        out = self.propagate(edge_index, x=(x_src, x_dst), alpha=alpha, edge_attr=edge_attr, size=size)

        out = out.view(-1, H * C) if self.concat else out.mean(dim=1)
        if res is not None:
            out = out + res
        if self.bias is not None:
            out = out + self.bias
        if isinstance(return_attention_weights, bool):
            return out, (edge_index, alpha)
        return out, edge_attr

    def edge_update(self, alpha_j: Tensor, alpha_i: Optional[Tensor], edge_attr: Optional[Tensor], index: Tensor,
                    ptr: Optional[Tensor], dim_size: Optional[int]) -> Tensor:
        # edge-level attention: source score + target score + edge score, leaky relu, softmax over a target's edges
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            alpha = alpha + (edge_attr * self.att_edge).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, dim_size)
        return F.dropout(alpha, p=self.dropout, training=self.training)

    def message(self, x_j: Tensor, alpha: Tensor, edge_attr: Optional[Tensor]) -> Tensor:
        msg = x_j + edge_attr.view(-1, self.heads, self.out_channels)
        return msg * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, heads={self.heads})"


class DialogueEmbeddingProcess(nn.Module):
    """Reduces the per-edge dialogue [E, turns, dim] over turns: the last turn or the mean (upstream `aggr_type`)."""

    def __init__(self, aggr_type: str):
        super().__init__()
        if aggr_type not in ("last", "mean"):
            raise ValueError(f"aggr_type must be 'last' or 'mean', got {aggr_type!r}")
        self.aggr_type = aggr_type

    def forward(self, diag_emb: Tensor) -> Tensor:
        return diag_emb[:, -1, :] if self.aggr_type == "last" else diag_emb.mean(dim=1)


class MyGAT(nn.Module):
    """G-Safeguard's node classifier: dialogue reduction, `num_layers` GATwithEdgeConv layers (ReLU + dropout), Linear.

    `edge_dim = (max_turns, embedding_dim)` as upstream passes it; only the embedding width is used (the turn count
    may vary at run time). Layer 1 takes edge_attr of width `embedding_dim`; later layers take the projected
    edge_attr of width `heads * head_channels`. Differences from upstream, all refusals of inputs that crash it:
    `concat=False` and `hidden_channels % heads != 0` raise at construction; `add_time_emb` (stored, never used
    upstream) is not kept. No experiment value has a default here: the caller passes every one.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int, edge_dim: Tuple[int, int],
                 num_layers: int, dropout: float, aggr_type: str, concat: bool = True, residual: bool = False):
        super().__init__()
        if not concat:
            raise ValueError("MyGAT needs concat=True: layer 2 expects heads * head_channels inputs")
        if hidden_channels % heads:
            raise ValueError(f"hidden_channels={hidden_channels} must be divisible by heads={heads}")
        self.in_channels, self.out_channels, self.num_layers, self.dropout = in_channels, out_channels, num_layers, dropout
        self.edge_dim = tuple(edge_dim)
        self.heads = heads
        self.head_channels = hidden_channels // heads
        self.hidden_channels = self.head_channels * heads

        _, embedding_dim = self.edge_dim
        self.convs = nn.ModuleList()
        self.convs.append(GATwithEdgeConv(in_channels, self.head_channels, heads=heads, concat=concat,
                                          edge_dim=embedding_dim, residual=residual))
        for _ in range(num_layers - 1):
            self.convs.append(GATwithEdgeConv(self.hidden_channels, self.head_channels, heads=heads, concat=concat,
                                              edge_dim=self.hidden_channels, residual=residual))
        self.diag_emb_proc = DialogueEmbeddingProcess(aggr_type)
        self.out = nn.Linear(self.hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        edge_attr = self.diag_emb_proc(edge_attr)
        for conv in self.convs:
            x, edge_attr = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out(x)


ARCH_KEYS = ("in_channels", "hidden_channels", "out_channels", "heads", "edge_dim", "num_layers", "dropout",
             "aggr_type")


def build_model(arch: Mapping) -> MyGAT:
    """MyGAT from an `arch` dict (the one stored in a checkpoint's metadata)."""
    return MyGAT(**{k: arch[k] for k in ARCH_KEYS})
