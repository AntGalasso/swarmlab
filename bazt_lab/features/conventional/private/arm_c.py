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
from bazt_lab.features.conventional.private.common import STAT_C, BudgetedArm, Tensor, mask_and

BINS_KEY = "conventional.c.judge_score_bins"
WINDOW_KEY = "conventional.c.presence_window_rounds"
_NAN = math.nan
JITTER_ULPS = 64        # tie_jitter_scale is 8 ulp of the largest sum; x64 gives about 6000 distinct levels per tie


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


class ArmC(BudgetedArm):
    name = "C"
    JITTER = True                    # tie-breaking jitter of the statistic (CONTRACT.md); reset() copies it to `jitter`
    state_keys = ("recipients", "rate_lo", "rate_hi", "input_lo", "input_hi", "trust_lo", "trust_hi", "absent",
                  "flagged")

    def _allocate(self, ctx: core.EpisodeContext) -> None:
        self.jitter = bool(type(self).JITTER)                                # one value for the whole run
        self.window = _window(core.get(ctx.settings, WINDOW_KEY))
        if not ctx.platform:
            raise core.Unset(f"{BINS_KEY}.<platform> (ctx.platform is empty)")
        value = core.get(ctx.settings, f"{BINS_KEY}.{ctx.platform}")
        upper, weight = _bins(value)
        dev, dt = ctx.device, ctx.dtype
        self.bin_upper, self.bin_weight = upper.to(dev, dt), weight.to(dev, dt)
        self.threat = w.NETWORK_THREAT.to(dev, dt)
        self.unit = _unit(value)
        bound = w.CONSTANT_TRUST + w.PRESENCE + w.SERVICE_USAGE + w.ACCESS_RATE + w.INPUT_BEHAVIOUR + w.TRUST_HISTORY             + float(w.NETWORK_THREAT.max()) + float(weight.max())            # largest |RL - TS|
        self.jitter_scale = float(core.tie_jitter_scale(dt, bound)) * JITTER_ULPS
        if 2 * self.jitter_scale >= self.unit:
            raise ValueError(f"{BINS_KEY}.{ctx.platform} weights have a gcd of {self.unit}, too small for the {dt} "
                             f"tie-breaking jitter ({self.jitter_scale}): the statistic could not be told apart")
        B, N = ctx.B, ctx.N
        self.recipients = torch.zeros(B, N, N, dtype=torch.bool, device=dev)    # j is among i's past recipients
        for history in ("rate", "input", "trust"):                           # running min / max of i's past values
            setattr(self, f"{history}_lo", torch.full((B, N), math.inf, dtype=dt, device=dev))
            setattr(self, f"{history}_hi", torch.full((B, N), -math.inf, dtype=dt, device=dev))
        self.absent = torch.zeros(B, N, dtype=torch.int64, device=dev)       # consecutive rounds i was not active
        self.flagged = torch.zeros(B, N, dtype=torch.bool, device=dev)       # agents C has denied so far

    def _jitter(self, scored: Tensor) -> Tensor:
        """Tie-breaking jitter [B, N, N] (CONTRACT.md). Only evaluated deliveries need one, so the draw has one slot
        per such delivery: delivery number r of the episode in row-major (sender, receiver) order this round gets
        element r of the keyed stream. That depends on the episode and the round only, and costs K draws instead of N*N,
        K being the most evaluated deliveries in one episode (most deliveries of a sparse graph are not)."""
        flat = scored.flatten(1)
        rank = flat.cumsum(-1, dtype=torch.int64) - 1
        most = int(rank[:, -1].max()) + 1                                    # one host read per round
        if most <= 0:
            return torch.zeros_like(scored, dtype=self.ctx.dtype)
        u = core.keyed_uniform(self.keys, f"{self.name}|jitter", self.step, (most,), self.ctx.device, self.ctx.dtype)
        slot = u.gather(1, rank.clamp(min=0))
        return ((slot - 0.5) * (2 * self.jitter_scale)).view_as(scored)

    def _policy(self, obs: core.Observation, evaluated: Tensor, audited: Tensor):
        dt = self.ctx.dtype
        checked = (~audited.isnan())[:, :, None]
        scored = mask_and(evaluated, checked.expand_as(evaluated))          # A passed and the text was checked
        sender = lambda x: x[:, :, None]                                     # noqa: E731  [B, N] -> [B, N, 1]
        rate = obs.attempts.sum(-1, dtype=dt)                                # messages i sends this round

        # sender-level terms on [B, N]; only service usage and trust history are per delivery
        met = lambda mask, weight: mask.to(dt) * weight                      # noqa: E731  Eq. 1: the weight if met
        base = (w.CONSTANT_TRUST + met(self.absent <= self.window, w.PRESENCE)
                + met(_in_range(rate, self.rate_lo, self.rate_hi), w.ACCESS_RATE)
                + met(_in_range(audited, self.input_lo, self.input_hi), w.INPUT_BEHAVIOUR))
        partial = sender(base) + met(self.recipients, w.SERVICE_USAGE)
        ts = partial + met(_in_range(partial, sender(self.trust_lo), sender(self.trust_hi)), w.TRUST_HISTORY)

        level = self.flagged.sum(-1).clamp(max=w.MAX_THREAT_LEVEL)
        judge = self.bin_weight[torch.bucketize(audited, self.bin_upper)]    # NaN lands beyond the last bin: 0
        rl = (self.threat[level][:, None] + judge)[:, :, None]

        statistic = rl - ts
        if self.jitter:                                                      # strictly inside (-unit/2, unit/2)
            statistic = statistic + self._jitter(scored)
        statistic = torch.where(scored, statistic, _NAN)
        want = mask_and(scored, statistic > float(self.ctx.thresholds.get(STAT_C, math.inf)))

        # every checked delivery enters the histories, denied or not: C never penalises an earlier flag (D36)
        sent = scored.any(-1)
        self.recipients = self.recipients | scored
        self.rate_lo, self.rate_hi = _extend(self.rate_lo, self.rate_hi, rate, sent)
        self.input_lo, self.input_hi = _extend(self.input_lo, self.input_hi, audited, sent)
        self.trust_lo = torch.minimum(self.trust_lo, torch.where(scored, ts, math.inf).amin(-1))
        self.trust_hi = torch.maximum(self.trust_hi, torch.where(scored, ts, -math.inf).amax(-1))
        self.flagged = self.flagged | want.any(-1)
        live = ~self.exhausted[:, None]                                      # an exhausted episode's histories stay frozen
        updated = torch.where(obs.active, torch.zeros_like(self.absent), self.absent + 1)
        self.absent = torch.where(live, updated, self.absent)
        return want, statistic, {STAT_C: statistic}, want
