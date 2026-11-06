"""C's reported statistic carries a keyed tie-breaking jitter (a randomized test at the threshold).

RL - TS is a sum of integer-like weights, so many deliveries tie exactly and one threshold cannot hit alpha inside a
step. The jitter is smaller than half the gap between distinct values, so it never reorders them.
"""
import math

import pytest
import torch

from bazt_lab.features.conventional.private.arm_c import ArmC, _unit
from bazt_lab.features.conventional.tests.test_arm_c_oracle import GoOracle
from bazt_lab.features.conventional.tests.testkit import BINS, Fake, full_adj, make_ctx, make_settings

STAT = "rl_minus_ts"
SCORES = [-6.0, -3.0, 0.0]


def _run(B=1, N=3, adj=None, thresholds=None, ids=None, bins=BINS):
    adj = full_adj(B, N) if adj is None else adj
    ctx = make_ctx(adj, "C", thresholds=thresholds, settings=make_settings(rate_limit=None, bins=bins, window=3), ids=ids)
    arm = ArmC()
    arm.reset(ctx)
    return ctx, arm, Fake(ctx)


def test_the_unit_is_the_gcd_of_every_weight():
    assert _unit([[1.0, 4.0], [2.0, 12.0]]) == 1.0
    assert _unit([[1.0, 4.5], [2.0, 12.0]]) == 0.5
    assert _unit([[1.0, 0.0]]) == 1.0


def test_jitter_never_reorders_distinct_values_and_stays_below_half_the_gap():
    ctx, arm, fake = _run(N=6, thresholds={})
    stat = arm.decide(fake.obs(1, scores=[-8.0, -3.0, 0.0, 0.0, -3.0, -8.0])).statistics[STAT]
    exact = {-8.0: -26.0, -3.0: -22.0, 0.0: -14.0}
    for i, s in enumerate([-8.0, -3.0, 0.0, 0.0, -3.0, -8.0]):
        row = stat[0, i][~stat[0, i].isnan()]
        assert ((row - (exact[s] if s != -8.0 else -26.0)).abs() < 0.5).all()
    assert stat[0, 0][~stat[0, 0].isnan()].unique().numel() > 1            # ties inside a sender are split


def test_jitter_is_deterministic_and_independent_of_the_batch():
    B, N = 3, 5
    ids = [f"u{b}" for b in range(B)]
    ctx, arm, fake = _run(B=B, N=N, thresholds={}, ids=ids)
    batch = [arm.decide(fake.obs(t, scores=[-6.0, -3.0, 0.0, 0.0, -3.0])).statistics[STAT] for t in (1, 2)]
    for b in range(B):
        ctx1, arm1, fake1 = _run(B=1, N=N, thresholds={}, ids=[ids[b]])
        for t in (1, 2):
            alone = arm1.decide(fake1.obs(t, scores=[-6.0, -3.0, 0.0, 0.0, -3.0])).statistics[STAT]
            assert torch.equal(alone[0].nan_to_num(nan=-1e9), batch[t - 1][b].nan_to_num(nan=-1e9))
    again = _run(B=B, N=N, thresholds={}, ids=ids)
    assert torch.equal(again[1].decide(again[2].obs(1, scores=[-6.0, -3.0, 0.0, 0.0, -3.0])).statistics[STAT]
                       .nan_to_num(nan=-1e9), batch[0].nan_to_num(nan=-1e9))


def test_the_compared_value_is_the_reported_value_and_ties_split_at_the_threshold():
    N = 12
    ctx, arm, fake = _run(B=8, N=N, adj=full_adj(8, N), thresholds={STAT: -26.0}, ids=[f"t{b}" for b in range(8)])
    a = arm.decide(fake.obs(1, scores=[-8.0] * N))       # every statistic ties at -26 before the jitter
    stat = a.statistics[STAT]
    live = ~stat.isnan()
    assert torch.equal(a.deny, live & (stat > -26.0))
    rate = a.deny.sum().item() / live.sum().item()
    assert 0.3 < rate < 0.7                              # a tie at the threshold is split about evenly


@pytest.mark.parametrize("window", [0, 2])
def test_rounded_statistic_equals_the_scalar_oracle(window):
    B, N, T = 3, 6, 6
    g = torch.Generator().manual_seed(77 + window)
    adj = (torch.rand(B, N, N, generator=g) < 0.75) & ~torch.eye(N, dtype=torch.bool)
    ctx = make_ctx(adj, "C", thresholds={}, settings=make_settings(rate_limit=None, bins=BINS, window=window), T=T)
    arm = ArmC()
    arm.reset(ctx)
    fake = Fake(ctx)
    oracles = [GoOracle(N, window, BINS) for _ in range(B)]
    for t in range(1, T + 1):
        attempts = adj & (torch.rand(B, N, N, generator=g) < 0.8)
        scores = torch.tensor([-8.0, -5.0, -3.0, 0.0], dtype=torch.float64)[torch.randint(0, 4, (B, N), generator=g)]
        stat = arm.decide(fake.obs(t, attempts, scores=scores)).statistics[STAT]
        for b in range(B):
            stats, _ = oracles[b].round(adj[b].tolist(), attempts[b].tolist(), torch.ones(N, N).bool().tolist(),
                                        [True] * N, scores[b].tolist(), math.inf)
            got = {(i, j): round(stat[b, i, j].item()) for i in range(N) for j in range(N) if not stat[b, i, j].isnan()}
            assert got == {k: round(v) for k, v in stats.items()}


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_the_jitter_survives_the_compute_dtype_and_stays_far_below_the_gap(dtype):
    ctx = make_ctx(full_adj(4, 12), "C", dtype=dtype, thresholds={}, settings=make_settings(rate_limit=None, bins=BINS, window=3),
                   ids=[f"d{b}" for b in range(4)])
    arm = ArmC()
    arm.reset(ctx)
    stat = arm.decide(Fake(ctx).obs(1, scores=[-8.0] * 12)).statistics[STAT]
    live = stat[~stat.isnan()]
    assert stat.dtype == dtype
    assert live.unique().numel() > live.numel() // 2                    # the ties are really broken in this dtype
    assert (live + 26.0).abs().max() < 0.05                              # and stay tiny against the gap of 1
