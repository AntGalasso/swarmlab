"""Graph tensors: upstream's edge convention, node features, collation, record validation, parity with upstream code."""
import numpy as np
import pytest
import torch

from bazt_lab.features.gnn.private.embedding import CachedEmbedder
from bazt_lab.features.gnn.private.graph import (
    Example,
    collate,
    conversation_to_example,
    family_key,
    graph_inputs,
    node_features,
    predict_logits,
)
from bazt_lab.features.gnn.tests.conftest import DIM, UPSTREAM_PI, FakeEncoder, fake_vector, make_records

ADJ = torch.tensor([[0, 1, 1], [0, 0, 1], [1, 0, 0]])          # 0->1, 0->2, 1->2, 2->0


def emb(r=2, n=3, dim=4, seed=0):
    return torch.randn(r, n, dim, generator=torch.Generator().manual_seed(seed))


def test_family_key_joins_attack_and_task():
    assert family_key("insider_pi", "mmlu") == "insider_pi__mmlu"


# -------------------------------------------------------------------------------------------- graph_inputs
def test_edges_are_the_row_major_nonzeros_and_carry_the_receivers_messages():
    e = emb()
    edge_index, edge_attr = graph_inputs(ADJ, e)
    assert edge_index.tolist() == [[0, 0, 1, 2], [1, 2, 2, 0]] and edge_index.dtype == torch.int64
    assert edge_attr.shape == (4, 2, 4)
    for k in range(4):                                         # edge k = (sender i, receiver j): attr = j's turns
        assert torch.equal(edge_attr[k], e[:, edge_index[1, k], :])


def test_adjacency_may_be_bool_and_an_empty_graph_has_no_edges():
    edge_index, edge_attr = graph_inputs(ADJ.bool(), emb())
    assert edge_index.shape == (2, 4)
    edge_index, edge_attr = graph_inputs(torch.zeros(3, 3, dtype=torch.bool), emb())
    assert edge_index.shape == (2, 0) and edge_attr.shape == (0, 2, 4)


def test_shape_mismatches_are_refused():
    with pytest.raises(ValueError, match="square"):
        graph_inputs(torch.ones(3, 2), emb())
    with pytest.raises(ValueError, match="3 agents"):
        graph_inputs(ADJ, emb(n=4))
    with pytest.raises(ValueError, match=r"\[turns, agents, dim\]"):
        graph_inputs(ADJ, emb()[0])


# ----------------------------------------------------------------------------------------- node_features
def loop_features(edge_attr, edge_index, n, which):
    """Mean of the first turn's edge attribute over the edges ending (target) or starting (source) at each node."""
    out = torch.zeros(n, edge_attr.shape[-1])
    for v in range(n):
        rows = [k for k in range(edge_index.shape[1]) if edge_index[1 if which == "target" else 0, k] == v]
        if rows:
            out[v] = torch.stack([edge_attr[k, 0] for k in rows]).mean(0)
    return out


@pytest.mark.parametrize("which", ["target", "source"])
def test_node_features_match_a_loop_oracle(which):
    g = torch.Generator().manual_seed(3)
    adj = (torch.rand(6, 6, generator=g) < 0.4).to(torch.int64)
    adj.fill_diagonal_(0)
    edge_index, edge_attr = graph_inputs(adj, torch.randn(3, 6, 5, generator=g))
    torch.testing.assert_close(node_features(edge_attr, edge_index, 6, which),
                               loop_features(edge_attr, edge_index, 6, which))


def test_target_features_are_the_nodes_own_first_turn_message():
    e = emb()
    edge_index, edge_attr = graph_inputs(ADJ, e)
    x = node_features(edge_attr, edge_index, 3, "target")
    torch.testing.assert_close(x, e[0])                        # every node here has an in-edge


def test_a_node_without_in_edges_gets_zeros_under_target():
    adj = torch.tensor([[0, 1], [0, 0]])                       # node 0 receives nothing
    edge_index, edge_attr = graph_inputs(adj, emb(n=2))
    assert torch.equal(node_features(edge_attr, edge_index, 2, "target")[0], torch.zeros(4))


def test_unknown_node_index_is_refused():
    edge_index, edge_attr = graph_inputs(ADJ, emb())
    with pytest.raises(ValueError, match=r"target.*source"):
        node_features(edge_attr, edge_index, 3, "both")


# ---------------------------------------------------------------------------------------------- collate
def example(seed, n=4):
    g = torch.Generator().manual_seed(seed)
    adj = (torch.rand(n, n, generator=g) < 0.5).to(torch.int64)
    adj.fill_diagonal_(0)
    edge_index, edge_attr = graph_inputs(adj, torch.randn(2, n, 3, generator=g))
    return Example(edge_index, edge_attr, torch.tensor([1] + [0] * (n - 1)))


def test_collate_is_the_disjoint_union_with_shifted_node_ids():
    a, b = example(1), example(2)
    x, edge_index, edge_attr, y = collate([a, b], "target")
    assert torch.equal(edge_index[:, :a.edge_index.shape[1]], a.edge_index)
    assert torch.equal(edge_index[:, a.edge_index.shape[1]:], b.edge_index + 4)
    assert torch.equal(y, torch.cat([a.y, b.y]))
    torch.testing.assert_close(x[:4], node_features(a.edge_attr, a.edge_index, 4, "target"))
    torch.testing.assert_close(x[4:], node_features(b.edge_attr, b.edge_index, 4, "target"))


# ---------------------------------------------------------------------------------- conversation_to_example
def test_a_conversation_becomes_labelled_edges_with_all_turns_embedded_in_order():
    rec = make_records(1, n_agents=5, rounds=3, p=1.0, k=2, seed=4)[0]
    enc = FakeEncoder()
    ex = conversation_to_example(rec, CachedEmbedder(enc))
    assert enc.calls[0][:5] == rec["turns"][0] and len(enc.calls[0]) == 15       # turn-major, 3 x 5 distinct texts
    assert ex.y.tolist() == [1 if i in rec["attackers"] else 0 for i in range(5)]
    assert ex.edge_index.shape == (2, 20) and ex.edge_attr.shape == (20, 3, DIM)
    k = 7                                                                          # any edge: receiver's 3 messages
    recv = ex.edge_index[1, k].item()
    for r in range(3):
        assert torch.equal(ex.edge_attr[k, r], fake_vector(rec["turns"][r][recv]))


@pytest.mark.parametrize("change,message", [
    (lambda r: r.update(adj=[[0, 1], [1, 0], [0, 0]]), "square"),
    (lambda r: r.update(adj=[[0, 2], [1, 0]]), "0 or 1"),
    (lambda r: r.update(attackers=[9]), "attacker"),
    (lambda r: r.update(attackers=[0, 0]), "attacker"),
    (lambda r: r.update(turns=[]), "turn"),
    (lambda r: r.update(turns=[["a"]]), "agents"),
    (lambda r: r.update(turns=[["a", 3, "c", "d", "e"]]), "str"),
])
def test_a_malformed_record_is_refused_with_its_id(change, message):
    rec = make_records(1, n_agents=5, rounds=2, seed=1)[0]
    change(rec)
    with pytest.raises(ValueError, match=rf"{rec['id']}.*{message}"):
        conversation_to_example(rec, CachedEmbedder(FakeEncoder()))


# ------------------------------------------------------------------------------------------ predict_logits
def tiny_model():
    from bazt_lab.features.gnn.adapters.pyg_model import MyGAT
    torch.manual_seed(0)
    return MyGAT(in_channels=4, hidden_channels=8, out_channels=1, heads=2, edge_dim=(3, 4), num_layers=2,
                 dropout=0.5, aggr_type="mean")


def test_predict_logits_runs_the_defence_pipeline_without_dropout_or_grad():
    model = tiny_model().eval()
    e = emb()
    out = predict_logits(model, ADJ, e)
    edge_index, edge_attr = graph_inputs(ADJ, e)
    expected = model(node_features(edge_attr, edge_index, 3, "target"), edge_index, edge_attr).squeeze(-1)
    assert out.shape == (3,) and not out.requires_grad
    torch.testing.assert_close(out, expected.detach())


def test_predict_logits_refuses_a_model_in_training_mode():
    with pytest.raises(ValueError, match="eval"):
        predict_logits(tiny_model().train(), ADJ, emb())


# --------------------------------------------------------------------------------- parity with upstream code
needs_upstream = pytest.mark.skipif(not (UPSTREAM_PI / "main_defense_for_different_topology.py").exists(),
                                    reason="vendor/G-safeguard not checked out")


@needs_upstream
@pytest.mark.parametrize("n,p,seed", [(6, 1.0, 0), (8, 0.4, 1), (5, 0.2, 2)])
def test_inference_graph_equals_upstreams_embeddings2graph(n, p, seed):
    from bazt_lab.features.gnn.adapters.upstream import load_embeddings2graph
    up = load_embeddings2graph(UPSTREAM_PI)
    g = torch.Generator().manual_seed(seed)
    adj = (torch.rand(n, n, generator=g) <= p).to(torch.int64)
    adj.fill_diagonal_(0)
    turns = [torch.randn(n, DIM, generator=g) for _ in range(3)]
    x_up, edge_index_up, edge_attr_up = up(turns_as_numpy(turns), adj.numpy())
    edge_index, edge_attr = graph_inputs(adj, torch.stack(turns))
    assert torch.equal(edge_index, edge_index_up)
    torch.testing.assert_close(edge_attr, edge_attr_up)
    torch.testing.assert_close(node_features(edge_attr, edge_index, n, "target"), x_up)


def turns_as_numpy(turns):
    return [t.numpy() for t in turns]


@needs_upstream
def test_training_example_equals_upstreams_dataset_builder(tmp_path):
    import pickle

    from bazt_lab.features.gnn.adapters.upstream import load_dataset_builder

    rec = make_records(1, n_agents=6, rounds=4, p=0.5, k=2, seed=9)[0]

    class UpstreamEmbedder:                                    # `.encode(str)` as SentenceTransformer, one text at a time
        def encode(self, text):
            return fake_vector(text).numpy()

    meta = {"adj_matrix": rec["adj"], "attacker_idxes": rec["attackers"],
            "system_prompts": [f"agent {i}" for i in range(6)], "question": "q", "correct_answer": "A",
            "wrong_answer": ["B"],
            "communication_data": [[(i, text) for i, text in enumerate(turn)] for turn in rec["turns"]]}
    path = tmp_path / "dataset.pkl"
    load_dataset_builder(UPSTREAM_PI)([meta], UpstreamEmbedder(), str(path))
    (up,) = pickle.loads(path.read_bytes())  # noqa: S301 - the file was written by this test

    ours = conversation_to_example(rec, CachedEmbedder(FakeEncoder()))
    assert np.array_equal(ours.edge_index.numpy(), up["edge_index"])
    assert np.array_equal(ours.y.numpy(), up["labels"])
    np.testing.assert_allclose(ours.edge_attr.numpy(), up["edge_attr"], atol=0, rtol=0)
