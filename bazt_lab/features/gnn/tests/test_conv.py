"""GATwithEdgeConv: hand-computed tiny graph, an independent loop oracle, guards, and parity with the upstream class."""
import pytest
import torch

from bazt_lab.features.gnn.adapters.pyg_model import GATwithEdgeConv

EDGE_INDEX = torch.tensor([[0, 1, 2], [2, 2, 0]])           # 0->2, 1->2, 2->0 ; node 1 receives nothing
X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
E = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])      # one attribute per edge, in EDGE_INDEX order


def tiny_conv() -> GATwithEdgeConv:
    conv = GATwithEdgeConv(2, 2, heads=1, edge_dim=2, bias=False)
    with torch.no_grad():
        conv.lin.weight.copy_(torch.eye(2))
        conv.att_src.copy_(torch.tensor([[[1.0, 0.0]]]))
        conv.att_dst.copy_(torch.tensor([[[0.0, 1.0]]]))
        conv.lin_edge.weight.copy_(torch.eye(2))
        conv.att_edge.copy_(torch.tensor([[[1.0, 1.0]]]))
        conv.lin_edge_attr.weight.copy_(2 * torch.eye(2))
    return conv


def test_hand_computed_tiny_graph():
    # Node 2 hears edges 0->2 and 1->2. Raw scores: a_src(x_i) + a_dst(x_2) + a_edge(e) =
    #   0->2: 1 + 1 + 1 = 3     1->2: 0 + 1 + 1 = 2        (all positive, leaky relu is identity)
    # softmax([3, 2]) = [sigmoid(1), sigmoid(-1)] = [0.7310586, 0.2689414]
    # message = x_j + W_e e, with W_e = 2 I:  0->2: [1,0]+[2,0] = [3,0]   1->2: [0,1]+[0,2] = [0,3]
    # out_2 = 0.7310586 [3,0] + 0.2689414 [0,3];  node 0 hears only 2->0: weight 1, message [1,1]+[2,2] = [3,3];
    # node 1 hears nothing: 0.   The returned edge attribute is W_e e.
    out, edge_attr = tiny_conv()(X, EDGE_INDEX, edge_attr=E)
    s = torch.sigmoid(torch.tensor(1.0)).item()
    expected = torch.tensor([[3.0, 3.0], [0.0, 0.0], [3 * s, 3 * (1 - s)]])
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(edge_attr, 2 * E)


def reference(conv, x, edge_index, edge_attr):
    """Independent per-node loop of the same maths (no PyG)."""
    H, C = conv.heads, conv.out_channels
    xs = (x @ conv.lin.weight.T).view(-1, H, C)
    a_src, a_dst = (xs * conv.att_src).sum(-1), (xs * conv.att_dst).sum(-1)
    ee = (edge_attr @ conv.lin_edge.weight.T).view(-1, H, C)
    a_edge = (ee * conv.att_edge).sum(-1)
    me = (edge_attr @ conv.lin_edge_attr.weight.T).view(-1, H, C)
    src, dst = edge_index
    out = torch.zeros(x.shape[0], H, C)
    for j in range(x.shape[0]):
        idx = [e for e in range(src.numel()) if dst[e] == j]
        if not idx:
            continue
        raw = torch.stack([a_src[src[e]] + a_dst[j] + a_edge[e] for e in idx])
        w = torch.softmax(torch.nn.functional.leaky_relu(raw, conv.negative_slope), dim=0)
        for pos, e in enumerate(idx):
            out[j] += w[pos].unsqueeze(-1) * (xs[src[e]] + me[e])
    out = out.reshape(x.shape[0], H * C)
    return out + conv.bias if conv.bias is not None else out


@pytest.mark.parametrize("heads,concat", [(1, True), (3, True)])
def test_matches_loop_oracle_on_random_weights(heads, concat):
    g = torch.Generator().manual_seed(0)
    conv = GATwithEdgeConv(5, 4, heads=heads, concat=concat, edge_dim=7)
    with torch.no_grad():
        conv.bias.copy_(torch.randn(4 * heads, generator=g))
    x, ei, ea = torch.randn(6, 5, generator=g), torch.tensor([[0, 1, 2, 3, 4, 5, 2], [1, 1, 1, 4, 4, 0, 5]]), \
        torch.randn(7, 7, generator=g)
    out, edge_attr = conv(x, ei, edge_attr=ea)
    torch.testing.assert_close(out, reference(conv, x, ei, ea), atol=1e-5, rtol=1e-5)
    assert edge_attr.shape == (7, heads * 4)


def test_negative_raw_scores_go_through_leaky_relu():
    conv = tiny_conv()
    with torch.no_grad():
        conv.att_edge.copy_(torch.tensor([[[-5.0, -5.0]]]))       # raw scores become negative
    out, _ = conv(X, EDGE_INDEX, edge_attr=E)
    torch.testing.assert_close(out, reference(conv, X, EDGE_INDEX, E), atol=1e-6, rtol=1e-6)


def test_attention_weights_sum_to_one_per_target():
    conv = tiny_conv()
    _, (ei, alpha) = conv(X, EDGE_INDEX, edge_attr=E, return_attention_weights=True)
    assert torch.equal(ei, EDGE_INDEX)
    torch.testing.assert_close(alpha[:2, 0].sum(), torch.tensor(1.0))      # the two edges into node 2
    torch.testing.assert_close(alpha[2, 0], torch.tensor(1.0))             # the single edge into node 0


def test_edge_dim_is_mandatory():
    with pytest.raises(ValueError, match="edge_dim"):
        GATwithEdgeConv(2, 2, heads=1)


def test_forward_needs_edge_attr():
    with pytest.raises(ValueError, match="edge_attr"):
        tiny_conv()(X, EDGE_INDEX)


def test_no_edges_returns_bias_only():
    conv = GATwithEdgeConv(2, 3, heads=2, edge_dim=4)
    out, edge_attr = conv(torch.randn(5, 2), torch.empty(2, 0, dtype=torch.long), edge_attr=torch.empty(0, 4))
    torch.testing.assert_close(out, conv.bias.expand(5, 6))
    assert edge_attr.shape == (0, 6)


def test_parameter_names_are_the_upstream_names():
    names = set(dict(GATwithEdgeConv(4, 3, heads=2, edge_dim=5).named_parameters()))
    assert names == {"lin.weight", "att_src", "att_dst", "att_edge", "lin_edge.weight", "lin_edge_attr.weight",
                     "bias"}
