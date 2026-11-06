"""GNN_standin: detect with the cheap monitor above the calibrated threshold, then cut the agent's outgoing edges."""
import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.references.private.gnn_standin import GNNStandin
from bazt_lab.features.references.tests.synth import FakePlatform, make_ctx, star

THR = {"gnn_standin": 0.5}


def monitor_const(values, B=1):
    m = torch.tensor([values] * B, dtype=torch.float64)
    return lambda t: m


def test_it_is_labelled_a_stand_in_and_reports_the_monitor_as_its_statistic():
    assert GNNStandin.name == "GNN_standin" and GNNStandin.stand_in is True
    ctx = make_ctx(star(6), 2)                                          # no threshold: calibration, collects only
    fp = FakePlatform(ctx, monitor=monitor_const([0.1, 0.9, 0.2, 0.3, 0.4, 0.95]))
    fp.run(GNNStandin())
    for r in fp.log:
        assert r.actions.deny is None or not r.actions.deny.any()
        s = r.actions.statistics["gnn_standin"]
        assert s.dtype == torch.float64 and torch.equal(s, r.obs.monitor)


def test_flagged_agents_have_all_their_outgoing_deliveries_denied_and_nobody_else():
    ctx = make_ctx(star(6), 3, thresholds=THR)
    fp = FakePlatform(ctx, monitor=monitor_const([0.1, 0.9, 0.2, 0.3, 0.4, 0.95]))
    fp.run(GNNStandin())
    d = fp.log[0].actions.deny[0]
    assert d.dtype == torch.bool and d.shape == (6, 6)
    assert torch.equal(d[1], ctx.adj[0, 1]) and torch.equal(d[5], ctx.adj[0, 5])     # leaves 1 and 5 -> hub
    assert d[[0, 2, 3, 4]].sum() == 0
    assert not fp.blocked.any()                                         # D99: no persistent cut


def test_a_flagged_hub_is_denied_towards_every_leaf():
    ctx = make_ctx(star(6), 2, thresholds=THR)
    fp = FakePlatform(ctx, monitor=monitor_const([0.99, 0.0, 0.0, 0.0, 0.0, 0.0]))
    fp.run(GNNStandin())
    assert fp.log[0].actions.deny[0, 0].sum() == 5


def test_every_round_is_evaluated_afresh_and_charged_per_denied_delivery():
    ctx = make_ctx(star(6), 4, thresholds=THR)
    mon = lambda t: torch.tensor([[0.1, 0.9 if t in (1, 2) else 0.1, 0.2, 0.3, 0.4, 0.1]], dtype=torch.float64)  # noqa: E731
    fp = FakePlatform(ctx, monitor=mon)
    fp.run(GNNStandin())
    assert [r.charged.item() for r in fp.log] == [1.0, 1.0, 0.0, 0.0]   # flagged at t = 1, 2 only: a cut is not persistent
    assert [bool(r.actions.deny[0, 1].any()) for r in fp.log] == [True, True, False, False]


def test_threshold_is_strict_and_missing_threshold_means_never_act():
    ctx = make_ctx(star(4), 2, thresholds={"gnn_standin": 0.9})
    fp = FakePlatform(ctx, monitor=monitor_const([0.1, 0.9, 0.95, 0.0]))
    fp.run(GNNStandin())
    d = fp.log[0].actions.deny[0]
    assert d[2].any() and not d[1].any()


def test_denials_stay_inside_the_intervention_budget_highest_score_first():
    # hub denial = 5 deliveries, leaf = 1. Budget 1 pays the highest-scoring leaf only.
    ctx = make_ctx(star(6), 3, thresholds=THR, intervention_cap=1)
    fp = FakePlatform(ctx, monitor=monitor_const([0.99, 0.6, 0.9, 0.0, 0.0, 0.0]))
    fp.run(GNNStandin())                                                # the platform asserts no violation
    d = fp.log[0].actions.deny[0]
    assert d[2].any() and not d[0].any() and not d[1].any()
    assert fp.log[0].charged.item() == 1


def test_it_needs_the_cheap_monitor():
    ctx = make_ctx(star(4), 2, thresholds=THR)
    fp = FakePlatform(ctx)                                              # obs.monitor is None (LLM side)
    with pytest.raises(ValueError, match="monitor"):
        fp.run(GNNStandin())


def test_state_roundtrip_resumes_exactly(tmp_path):
    def world():
        return make_ctx(star(6), 5, B=2, thresholds=THR, intervention_cap=30)

    mon = lambda t: torch.tensor([[0.1, 0.9, 0.2, 0.3, 0.4, 0.1], [0.1, 0.1, 0.1, 0.1, 0.6, 0.1 + 0.2 * t]],   # noqa: E731
                                  dtype=torch.float64)
    ref = FakePlatform(world(), monitor=mon)
    ref.run(GNNStandin())
    part = FakePlatform(world(), monitor=mon)
    d1 = GNNStandin()
    d1.reset(part.ctx)
    for t in (1, 2):
        part.step(d1, t)
    store = core.CheckpointStore(tmp_path)
    store.save_state("u", d1.state_dict())
    d2 = GNNStandin()
    d2.reset(part.ctx)
    d2.load_state_dict(store.load_state("u"))
    for t in (3, 4, 5):
        part.step(d2, t)
    for a, b in zip(ref.log, part.log, strict=True):
        assert torch.equal(a.actions.deny, b.actions.deny)
