"""Keyed tie-breaking jitter for the reported decision statistics.

Several statistics are discrete (C's trust score is a sum of integer weights), so many deliveries tie exactly at a
calibrated threshold and the realised deny rate lands far below alpha. Adding a tiny keyed jitter splits the ties at the
threshold at random, as a randomized Neyman-Pearson test does, without changing the order of distinct values.

Magnitude. Per episode and call, `scale = min(SCALE, gap / 4)` where `gap` is the smallest gap between distinct finite
values of that episode's statistic; the jitter is uniform in (-scale / 2, scale / 2), so it is always strictly below half
the smallest gap (in fact an eighth of it). SCALE is a fixed constant so that, away from pathologically close values, the
same jitter scale is used in every round and the calibrated threshold stays comparable across rounds.

Keys. The draw of episode b is `ctx.arm_stream(f"jitter|<statistic>|<round>", b)`: one uniform per (sender, receiver)
(per sender for `per_sender=True`), so it depends on the episode's own key, never on the batch. Deterministic.
"""
from __future__ import annotations

import math

import torch

from bazt_lab.core import public as core

Tensor = torch.Tensor
SCALE = 2.0 ** -24


def _scale(stat: Tensor) -> Tensor:
    """float64 [B]: min(SCALE, smallest gap between distinct finite values / 4) per episode."""
    flat = stat.flatten(1)
    finite = torch.isfinite(flat)
    s = torch.where(finite, flat, torch.full_like(flat, math.inf)).sort(dim=1).values
    gap = s[:, 1:] - s[:, :-1]
    gap = torch.where(torch.isfinite(gap) & (gap > 0), gap, torch.full_like(gap, math.inf))
    smallest = gap.amin(1) if gap.shape[1] else torch.full((flat.shape[0],), math.inf, dtype=stat.dtype)
    return torch.clamp(smallest / 4.0, max=SCALE)


def unit(ctx: core.EpisodeContext, name: str, step: int, per_sender: bool = False) -> Tensor:
    """float64 [B, N, N] (or [B, N, 1]) uniforms in [0, 1): the keyed draws behind the jitter of round `step`."""
    shape = (ctx.N, 1) if per_sender else (ctx.N, ctx.N)
    return torch.stack([core.uniforms(ctx.arm_stream(f"jitter|{name}|{step}", b), shape) for b in range(ctx.B)])


def jitter(stat: Tensor, ctx: core.EpisodeContext, name: str, step: int, per_sender: bool = False) -> Tensor:
    """`stat` float64 [B, N, N] plus the keyed jitter; NaN and infinities are left as they are."""
    u = unit(ctx, name, step, per_sender)
    noise = (u - 0.5) * _scale(stat)[:, None, None]
    return torch.where(torch.isfinite(stat), stat + noise, stat)
