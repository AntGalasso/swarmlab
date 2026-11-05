"""Arm C: contextual score-based zero trust (spec 2.3), the additive algorithm of Bradatsch et al. (TrustCom 2023,
arXiv 2402.08299, Eqs. 1 to 3) with the weights of the reference implementation (private/weights.py).

For every delivery i -> j that A lets through and whose sender's text was checked, TS is the sum of the weights of the
trust attributes met (Eq. 1), RL the sum of the weights of the risk attributes met (Eq. 2). The reported decision
statistic is RL - TS; the delivery is denied when it exceeds the calibrated threshold, which replaces the paper's fixed
risk offset (spec 2.3). Mapping of the spec's attribute table, in CONTRACT.md.
"""
from __future__ import annotations

import itertools
import math
from fractions import Fraction

import torch

from bazt_lab.core import public as core
from bazt_lab.features.conventional.private import weights as w
from bazt_lab.features.conventional.private.common import STAT_C, BudgetedArm, Tensor

BINS_KEY = "conventional.c.judge_score_bins"
WINDOW_KEY = "conventional.c.presence_window_rounds"
_NAN = math.nan


def _number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not math.isnan(x)


def _bins(value) -> tuple[Tensor, Tensor]:
    """[[upper, weight], ...] -> (upper bounds [K], weights [K + 1]). A score s weighs the weight of the first bin with
    s <= upper (bins are (previous upper, upper]); above the last bound it is outside every target set and weighs 0."""
    def refuse(why):
        return ValueError(f"{BINS_KEY}.<platform> {why}; expected [[upper, weight], ...] with strictly increasing upper bounds "
                          f"and finite weights >= 0, got {value!r}")

    if not isinstance(value, (list, tuple)) or not value:
        raise refuse("is not a non-empty list")
    if any(not isinstance(p, (list, tuple)) or len(p) != 2 or not all(map(_number, p)) for p in value):
        raise refuse("has an entry that is not a pair of numbers")
    upper, weight = (list(map(float, column)) for column in zip(*value, strict=False))
    if any(b <= a for a, b in itertools.pairwise(upper)):
        raise refuse("has upper bounds that do not strictly increase")
    if any(not math.isfinite(x) or x < 0 for x in weight):
        raise refuse("has a weight that is negative or not finite")
    return torch.tensor(upper, dtype=torch.float64), torch.tensor(weight + [0.0], dtype=torch.float64)


def _unit(bins) -> float:
    """The gcd of every weight that enters RL - TS (reference weights and the bin weights): any two distinct values of
    the statistic differ by at least this much."""
    fixed = [w.AUTHENTICATION, w.CHANNEL, w.SERVICE_USAGE, w.ACCESS_RATE, w.PRESENCE, w.INPUT_BEHAVIOUR,
             w.TRUST_HISTORY, *w.NETWORK_THREAT.tolist()]
    parts = [Fraction(x).limit_denominator(10**6) for x in fixed + [b[1] for b in bins]]
    den = math.lcm(*(p.denominator for p in parts))
    g = math.gcd(*(int(p * den) for p in parts))
    if g == 0:
        raise ValueError(f"{BINS_KEY}.<platform> and the reference weights give no positive weight")
    return g / den


def _window(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{WINDOW_KEY} must be an integer >= 0 (rounds), got {value!r}")
    return value


def _in_range(x: Tensor, lo: Tensor, hi: Tensor) -> Tensor:
    """min <= x <= max of a history kept as running extrema; an empty history (lo = inf, hi = -inf) contains nothing,
    which is the reference's `MinMax` error branch (weight 0)."""
    return (x >= lo) & (x <= hi)


def _extend(lo: Tensor, hi: Tensor, x: Tensor, where: Tensor) -> tuple[Tensor, Tensor]:
    return (torch.minimum(lo, torch.where(where, x, math.inf)), torch.maximum(hi, torch.where(where, x, -math.inf)))


def _gain(met: Tensor, weight: float) -> Tensor:
    return met.to(torch.float64) * weight


class ArmC(BudgetedArm):
    name = "C"
    jitter = True                    # tie-breaking jitter of the statistic (CONTRACT.md); the tests switch it off
    state_keys = ("recipients", "rate_lo", "rate_hi", "input_lo", "input_hi", "trust_lo", "trust_hi", "absent",
                  "flagged")

    def _allocate(self, ctx: core.EpisodeContext) -> None:
        self.window = _window(core.get(ctx.settings, WINDOW_KEY))
        if not ctx.platform:
            raise core.Unset(f"{BINS_KEY}.<platform> (ctx.platform is empty)")
        value = core.get(ctx.settings, f"{BINS_KEY}.{ctx.platform}")
        self.bin_upper, self.bin_weight = _bins(value)
        self.unit = _unit(value)
        B, N = ctx.B, ctx.N
        self.recipients = torch.zeros(B, N, N, dtype=torch.bool)            # j is among i's past recipients
        for history in ("rate", "input", "trust"):                           # running min / max of i's past values
            setattr(self, f"{history}_lo", torch.full((B, N), math.inf, dtype=torch.float64))
            setattr(self, f"{history}_hi", torch.full((B, N), -math.inf, dtype=torch.float64))
        self.absent = torch.zeros(B, N, dtype=torch.int64)                   # consecutive rounds i was not active
        self.flagged = torch.zeros(B, N, dtype=torch.bool)                   # agents C has denied so far

    def _policy(self, obs: core.Observation, evaluated: Tensor, audited: Tensor):
        scored = evaluated & ~audited.isnan()[:, :, None]                    # A passed and the text was checked
        sender = lambda x: x[:, :, None]                                     # noqa: E731  [B, N] -> [B, N, 1]
        rate = obs.attempts.sum(-1).to(torch.float64)                        # messages i sends this round

        partial = (w.CONSTANT_TRUST
                   + _gain(sender(self.absent <= self.window), w.PRESENCE)
                   + _gain(self.recipients, w.SERVICE_USAGE)
                   + _gain(sender(_in_range(rate, self.rate_lo, self.rate_hi)), w.ACCESS_RATE)
                   + _gain(sender(_in_range(audited, self.input_lo, self.input_hi)), w.INPUT_BEHAVIOUR))
        ts = partial + _gain(_in_range(partial, sender(self.trust_lo), sender(self.trust_hi)), w.TRUST_HISTORY)

        level = self.flagged.sum(-1).clamp(max=w.MAX_THREAT_LEVEL)
        judge = self.bin_weight[torch.bucketize(audited, self.bin_upper)]    # NaN lands beyond the last bin: 0
        rl = w.NETWORK_THREAT[level][:, None, None] + sender(judge)

        statistic = rl - ts
        if self.jitter:                                                      # strictly inside (-unit/2, unit/2)
            everyone = torch.arange(self.ctx.B)
            statistic = statistic + (self._draw("jitter", everyone) - 0.5) * (0.9 * self.unit)
        statistic = torch.where(scored, statistic, _NAN)
        want = scored & (statistic > float(self.ctx.thresholds.get(STAT_C, math.inf)))

        # every checked delivery enters the histories, denied or not: C never penalises an earlier flag (D36)
        sent = scored.any(-1)
        self.recipients = self.recipients | scored
        self.rate_lo, self.rate_hi = _extend(self.rate_lo, self.rate_hi, rate, sent)
        self.input_lo, self.input_hi = _extend(self.input_lo, self.input_hi, audited, sent)
        self.trust_lo = torch.minimum(self.trust_lo, torch.where(scored, ts, math.inf).amin(-1))
        self.trust_hi = torch.maximum(self.trust_hi, torch.where(scored, ts, -math.inf).amax(-1))
        self.flagged = self.flagged | want.any(-1)
        self.absent = torch.where(obs.active, torch.zeros_like(self.absent), self.absent + 1)
        return want, statistic, {STAT_C: statistic}
