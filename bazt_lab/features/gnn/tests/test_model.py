"""MyGAT: structure, the dialogue-embedding reduction, and parity with the upstream classes (vendored submodule)."""
import pytest
import torch

from bazt_lab.features.gnn.adapters.pyg_model import GATwithEdgeConv, MyGAT, build_model
from bazt_lab.features.gnn.tests.conftest import UPSTREAM_PI

EDGE_INDEX = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]])


def graph(n=4, turns=3, dim=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, dim, generator=g), torch.randn(EDGE_INDEX.shape[1], turns, dim, generator=g)


def small_model(**kw):
    args = dict(in_channels=10, hidden_channels=12, out_channels=1, heads=3, edge_dim=(3, 10), num_layers=2,
                dropout=0.2, aggr_type="mean")
    args.update(kw)
    return MyGAT(**args)


def test_state_dict_keys_are_the_upstream_keys():
    keys = set(small_model().state_dict())
    layer = {"lin.weight", "att_src", "att_dst", "att_edge", "lin_edge.weight", "lin_edge_attr.weight", "bias"}
    assert keys == {f"convs.{i}.{k}" for i in (0, 1) for k in layer} | {"out.weight", "out.bias"}


def test_second_layer_takes_the_projected_edge_attribute():
    m = small_model()
    assert m.convs[0].lin_edge_attr.in_channels == 10          # layer 1 edge_dim = embedding width
    assert m.convs[1].lin_edge_attr.in_channels == 12          # layer 2 edge_dim = hidden width (heads * head dim)
    x, ea = graph()
    assert m(x, EDGE_INDEX, ea).shape == (4, 1)


@pytest.mark.parametrize("aggr", ["last", "mean"])
def test_dialogue_reduction_over_turns(aggr):
    m = small_model(aggr_type=aggr).eval()
    x, ea = graph()
    reduced = ea[:, -1, :] if aggr == "last" else ea.mean(dim=1)
    torch.testing.assert_close(m(x, EDGE_INDEX, ea), m(x, EDGE_INDEX, reduced.unsqueeze(1)))


def test_unknown_reduction_is_refused():
    x, ea = graph()
    with pytest.raises(ValueError, match="aggr"):
        small_model(aggr_type="max")(x, EDGE_INDEX, ea)


def test_variable_number_of_turns_is_accepted():
    m = small_model().eval()
    x, _ = graph()
    for turns in (1, 2, 4):
        assert m(x, EDGE_INDEX, graph(turns=turns)[1]).shape == (4, 1)


def test_hidden_not_divisible_by_heads_is_refused():
    with pytest.raises(ValueError, match="divisible"):
        small_model(hidden_channels=13)


def test_eval_is_deterministic_and_train_dropout_is_active():
    m = small_model(dropout=0.5)
    x, ea = graph()
    m.eval()
    torch.testing.assert_close(m(x, EDGE_INDEX, ea), m(x, EDGE_INDEX, ea))
    m.train()
    assert not torch.equal(m(x, EDGE_INDEX, ea), m(x, EDGE_INDEX, ea))


def test_build_model_from_arch_dict():
    arch = dict(in_channels=10, hidden_channels=12, out_channels=1, heads=3, edge_dim=[3, 10], num_layers=2,
                dropout=0.2, aggr_type="mean")
    m = build_model(arch)
    assert isinstance(m, MyGAT) and m.edge_dim == (3, 10)


needs_upstream = pytest.mark.skipif(not (UPSTREAM_PI / "model.py").exists(), reason="vendor/G-safeguard not checked out")


@needs_upstream
@pytest.mark.parametrize("self_loops", [False, True])
def test_conv_matches_the_upstream_class(self_loops):
    from bazt_lab.features.gnn.adapters.upstream import load_upstream
    up = load_upstream(UPSTREAM_PI)
    ours = GATwithEdgeConv(10, 4, heads=3, edge_dim=10, add_self_loops=self_loops)
    theirs = up.GATwithEdgeConv(10, 4, heads=3, edge_dim=10, add_self_loops=self_loops)
    theirs.load_state_dict(ours.state_dict())
    x, ea = graph()
    ea = ea[:, 0, :]
    (o1, e1), (o2, e2) = ours(x, EDGE_INDEX, edge_attr=ea), theirs(x, EDGE_INDEX, edge_attr=ea)
    torch.testing.assert_close(o1, o2)
    torch.testing.assert_close(e1, e2)


@needs_upstream
@pytest.mark.parametrize("aggr", ["last", "mean"])
def test_mygat_matches_the_upstream_class_both_ways(aggr):
    from bazt_lab.features.gnn.adapters.upstream import load_upstream
    up = load_upstream(UPSTREAM_PI)
    args = dict(in_channels=10, hidden_channels=12, out_channels=1, heads=3, edge_dim=(3, 10), num_layers=2,
                dropout=0.2, aggr_type=aggr)
    ours, theirs = MyGAT(**args).eval(), up.MyGAT(**args).eval()
    assert set(ours.state_dict()) == set(theirs.state_dict())
    x, ea = graph(turns=3)
    theirs.load_state_dict(ours.state_dict())
    torch.testing.assert_close(ours(x, EDGE_INDEX, ea), theirs(x, EDGE_INDEX, ea))
    ref = up.MyGAT(**args).eval()                                  # and an upstream-initialised state loads here
    ours.load_state_dict(ref.state_dict())
    torch.testing.assert_close(ours(x, EDGE_INDEX, ea), ref(x, EDGE_INDEX, ea))


def test_zero_layers_are_refused():
    with pytest.raises(ValueError, match="num_layers"):
        small_model(num_layers=0)
