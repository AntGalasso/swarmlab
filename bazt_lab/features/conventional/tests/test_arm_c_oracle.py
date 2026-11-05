"""Arm C against a scalar transcription of the reference implementation.

`GoOracle` follows internal/app/algorithm/additive/additive_trust.go of zintpavowj/ZT_Score-based_Network_Level_AC_PDP_TA
(commit 46e099b) one delivery at a time. Histories are plain Python lists, searched with min()/max() exactly as
floatlib.MinMax does (empty history: weight 0), so it shares no code and no running-extremum trick with the batched
arm. Both sides use the mapping of the spec's attribute table documented in CONTRACT.md.
"""
import math

import pytest
import torch

from bazt_lab.features.conventional.private.arm_c import ArmC
from bazt_lab.features.conventional.tests.testkit import BINS, Fake, make_ctx, make_settings

STAT = "rl_minus_ts"
# Reference weights, restated from additive_trust.go (line numbers of that file).
AUTH = 8.0                                                    # 34  device_cert_auth
CHANNEL = 5.0 + 5.0 + 5.0                                     # 68, 71, 74  SHA256-RSA, AES, SHA256-RSA
SERVICE, RATE, PRESENCE, INPUT, HISTORY = 3.0, 5.0, 3.0, 3.0, 5.0     # 24, 22, 27, 21, 31
NETWORK_THREAT = {1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 7.0, 6: 9.0, 7: 12.0, 8: 15.0, 9: 20.0, 10: 25.0}   # 130-139


def min_max_weight(history, current, weight):
    """getUserUsualInputBehaviorWeight / getUserUsualAccessRateWeight / getUserTrustHistoryWeight (428-441)."""
    if not history:                                           # floatlib.MinMax returns an error for a nil slice
        return 0.0
    return weight if min(history) <= current <= max(history) else 0.0


def bin_weight(score, bins):
    for upper, weight in bins:
        if score <= upper:
            return weight
    return 0.0


class GoOracle:
    def __init__(self, N, window, bins):
        self.N, self.window, self.bins = N, window, bins
        self.service = [set() for _ in range(N)]              # userServiceUsageHistory
        self.rate = [[] for _ in range(N)]                    # userAccessRateHistory
        self.input = [[] for _ in range(N)]                   # userInputBehaviorHistory
        self.trust = [[] for _ in range(N)]                   # userTrustHistory
        self.absent = [0] * N
        self.flagged = set()

    def round(self, adj, attempts, authentic, active, scores, theta):
        """-> ({(i, j): statistic} of the evaluated deliveries, set of denied (i, j))."""
        N = self.N
        passes = {(i, j) for i in range(N) for j in range(N) if attempts[i][j] and adj[i][j] and authentic[i][j]}
        rates = [sum(1 for j in range(N) if attempts[i][j]) for i in range(N)]
        threat = NETWORK_THREAT.get(min(len(self.flagged), 10), 0.0)
        stats, ts_of = {}, {}
        for i, j in sorted(passes):
            partial = AUTH + CHANNEL
            partial += PRESENCE if self.absent[i] <= self.window else 0.0
            partial += SERVICE if j in self.service[i] else 0.0
            partial += min_max_weight(self.rate[i], rates[i], RATE)
            partial += min_max_weight(self.input[i], scores[i], INPUT)
            ts = partial + min_max_weight(self.trust[i], partial, HISTORY)
            stats[(i, j)] = (threat + bin_weight(scores[i], self.bins)) - ts
            ts_of[(i, j)] = ts
        denied = {k for k, v in stats.items() if v > theta}
        for i in {i for i, _ in passes}:                      # the histories learn from every evaluated delivery
            self.service[i] |= {j for a, j in passes if a == i}
            self.rate[i].append(rates[i])
            self.input[i].append(scores[i])
            self.trust[i] += [ts_of[(a, j)] for a, j in sorted(passes) if a == i]
        self.flagged |= {i for i, _ in denied}
        self.absent = [0 if active[i] else self.absent[i] + 1 for i in range(N)]
        return stats, denied


@pytest.mark.parametrize("window", [0, 2, 6])
def test_batched_arm_equals_the_scalar_transcription_on_random_episodes(window):
    B, N, T = 4, 6, 9
    g = torch.Generator().manual_seed(2024 + window)
    adj = (torch.rand(B, N, N, generator=g) < 0.75) & ~torch.eye(N, dtype=torch.bool)
    theta = -30.0
    ctx = make_ctx(adj, "C", thresholds={STAT: theta}, settings=make_settings(rate_limit=None, bins=BINS, window=window),
                   T=T)
    arm = ArmC()
    arm.reset(ctx)
    fake = Fake(ctx)
    oracles = [GoOracle(N, window, BINS) for _ in range(B)]
    seen_denies = seen_flags = 0
    for t in range(1, T + 1):
        extra = torch.rand(B, N, N, generator=g) < 0.05                       # now and then an off-allowlist attempt
        attempts = ((adj & (torch.rand(B, N, N, generator=g) < 0.8)) | extra) & ~torch.eye(N, dtype=torch.bool)
        authentic = torch.rand(B, N, N, generator=g) > 0.03                   # now and then altered in transit
        active = torch.rand(B, N, generator=g) > 0.15                         # agents drop out and come back
        attempts &= active[:, :, None]
        scores = torch.tensor([-8.0, -5.0, -3.0, -1.0, 0.0, 2.0], dtype=torch.float64)[
            torch.randint(0, 6, (B, N), generator=g)]
        actions = arm.decide(fake.obs(t, attempts, scores=scores, authentic=authentic, active=active))
        stat = actions.statistics[STAT]
        for b in range(B):
            stats, denied = oracles[b].round(adj[b].tolist(), attempts[b].tolist(), authentic[b].tolist(),
                                             active[b].tolist(), scores[b].tolist(), theta)
            got ={(i, j): stat[b, i, j].item() for i in range(N) for j in range(N) if not math.isnan(stat[b, i, j])}
            assert got == stats, (t, b)
            gate_drop = attempts[b] & ~(adj[b] & authentic[b])
            want_deny = gate_drop.clone()
            for i, j in denied:
                want_deny[i, j] = True
            assert torch.equal(actions.deny[b], want_deny), (t, b)
            seen_denies += len(denied)
            seen_flags = max(seen_flags, len(oracles[b].flagged))
    assert seen_denies > 0 and seen_flags > 0                                 # the scenario exercised the feedback loop
