"""Keyed tie-breaking jitter on the reported statistics: ties split at a threshold (a randomized Neyman-Pearson test),
the order of distinct values never changes, and the draws depend on the episode's own key only."""
import itertools
import math

import torch

from bazt_lab.features.bazt_slot.private import jitter as jt
from bazt_lab.features.bazt_slot.tests.testkit import full_adj, make_ctx

F64 = torch.float64
NAN = math.nan


def ctx_of(B=3, N=4):
    return make_ctx(full_adj(B, N))


def test_equal_values_become_distinct_and_stay_within_the_jitter_scale():
    ctx = ctx_of(1, 4)
    out = jt.jitter(torch.full((1, 4, 4), 3.0, dtype=F64), ctx, "judge", 1)
    assert out.unique().numel() == 16 and (out - 3.0).abs().max().item() < 1e-6


def test_the_order_of_distinct_values_never_changes_whatever_their_gaps():
    ctx = ctx_of(1, 3)
    for gap in (1.0, 1e-3, 1e-9, 1e-13):
        base = torch.tensor([[[0.0, 1.0, 1.0], [1.0 + gap, 1.0 + gap, 2.0], [2.0, 2.0 + gap, 0.0]]], dtype=F64)
        out = jt.jitter(base, ctx, "rl_minus_ts", 2)
        lo = base.flatten().unique()
        for a, b in itertools.pairwise(lo):
            assert out[base == a].max() < out[base == b].min(), gap


def test_nan_stays_nan_and_infinities_are_untouched():
    ctx = ctx_of(1, 2)
    base = torch.tensor([[[NAN, 1.0], [math.inf, NAN]]], dtype=F64)
    out = jt.jitter(base, ctx, "judge", 1)
    assert out.isnan().tolist() == base.isnan().tolist() and out[0, 1, 0].item() == math.inf


def test_the_draw_of_an_episode_does_not_depend_on_the_batch():
    ctx = ctx_of(3, 4)
    stat = torch.full((3, 4, 4), 2.0, dtype=F64)
    full = jt.jitter(stat, ctx, "judge", 5)
    for b in range(3):
        one = core_one(ctx, b)
        assert torch.equal(jt.jitter(stat[b:b + 1], one, "judge", 5)[0], full[b])


def core_one(ctx, b):
    from bazt_lab.core import public as core
    return core.EpisodeContext(**{**ctx.__dict__, "B": 1, "adj": ctx.adj[b:b + 1], "sink": ctx.sink[b:b + 1],
                                  "audit_cap": ctx.audit_cap[b:b + 1], "intervention_cap": ctx.intervention_cap[b:b + 1],
                                  "unit_ids": (ctx.unit_ids[b],), "arm_stream": lambda ns, _b, b=b: ctx.arm_stream(ns, b)})


def test_the_draw_is_keyed_by_round_and_statistic_and_is_reproducible():
    ctx = ctx_of(1, 4)
    stat = torch.full((1, 4, 4), 2.0, dtype=F64)
    a = jt.jitter(stat, ctx, "judge", 1)
    assert torch.equal(a, jt.jitter(stat, ctx, "judge", 1))
    assert not torch.equal(a, jt.jitter(stat, ctx, "judge", 2))
    assert not torch.equal(a, jt.jitter(stat, ctx, "rl_minus_ts", 1))


def test_a_per_sender_statistic_gets_one_jitter_per_sender():
    ctx = ctx_of(1, 4)
    out = jt.jitter(torch.full((1, 4, 4), 2.0, dtype=F64), ctx, "bazt_risk", 1, per_sender=True)
    assert (out == out[:, :, :1]).all() and out[0, :, 0].unique().numel() == 4
