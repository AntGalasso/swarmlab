"""Arm C: the network threat level table of the reference implementation, every level against its own weight.

The reference (additive_trust.go 130-139) weighs the number of flagged agents by the table below; level 0 has no entry
and weighs 0. C saturates at level 10. Restated here from those lines, independently of private/weights.py.
"""
import pytest

from bazt_lab.features.conventional.private.arm_c import ArmC
from bazt_lab.features.conventional.tests.testkit import BINS, Fake, full_adj, make_ctx, make_settings

STAT = "rl_minus_ts"
THREAT = {0: 0.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 7.0, 6: 9.0, 7: 12.0, 8: 15.0, 9: 20.0, 10: 25.0}   # 130-139
N = 12


@pytest.mark.parametrize("flagged", range(N + 1))
def test_each_number_of_flagged_agents_adds_its_reference_weight(flagged):
    ctx = make_ctx(full_adj(1, N), "C", thresholds={STAT: -20.0}, T=2,
                   settings=make_settings(rate_limit=None, bins=BINS, window=3))
    arm = ArmC()
    arm.reset(ctx)
    fake = Fake(ctx)
    scores = [0.0] * flagged + [-6.0] * (N - flagged)                   # risk weight 12 above -1, 0 below -5
    first = arm.decide(fake.obs(1, scores=scores))                       # TS 26: only the weight-12 senders exceed -20
    assert first.deny[0].any(-1).sum() == flagged
    stat = arm.decide(fake.obs(2, scores=scores)).statistics[STAT]       # TS 37 for everyone
    for i in range(N):
        row = stat[0, i][~stat[0, i].isnan()].unique()
        assert row.tolist() == [(12.0 if i < flagged else 0.0) + THREAT[min(flagged, 10)] - 37.0], i
