"""Keyed tie-breaking jitter for the reported decision statistics.

Several statistics are discrete (C's trust score is a sum of integer weights), so many deliveries tie exactly at a
calibrated threshold and the realised deny rate lands far below alpha. Adding a tiny keyed jitter splits the ties at the
threshold at random, as a randomized Neyman-Pearson test does, without changing the order of distinct values.

Magnitude. Per episode and call, `scale = min(top, gap / 4)` where `gap` is the smallest gap between distinct finite
values of that episode's statistic (for a sender-level statistic: of its per-sender values, a 12-fold smaller sort); the
jitter is uniform in (-scale / 2, scale / 2), so it is always strictly below half the smallest gap (an eighth of it).
`top = max(SCALE, core.tie_jitter_scale(dtype, M))` with M the largest |value| (at least 1): SCALE (2^-24) gives a fine grid
of jitter positions in float64, and the dtype term keeps the jitter above eight ulp when float32 rounds the sum. Away from
very close values the scale is the same in every round, so a calibrated threshold stays comparable across rounds.

Keys. One batched `core.keyed_uniform` call over all episodes, keyed by `core.episode_keys(ctx.unit_ids)`, the stream
`bazt|<arm>|jitter|<statistic>` and the round as the counter: one uniform per (sender, receiver) (per sender for
`per_sender=True`). A draw depends only on the episode's own key, the arm, the statistic, the round and the element, never on
the batch, the device or the order. Deterministic.
"""
from __future__ import annotations

import math

import torch

from bazt_lab.core import public as core

Tensor = torch.Tensor
SCALE = 2.0 ** -24


def top_scale(dtype: torch.dtype, magnitude) -> Tensor | float:
    """The largest jitter scale: SCALE, or eight ulp of `magnitude` (float or tensor) if that is larger."""
    top = core.tie_jitter_scale(dtype, torch.clamp(magnitude, min=1.0) if isinstance(magnitude, torch.Tensor) else max(magnitude, 1.0))
    return torch.clamp(top, min=SCALE) if isinstance(top, torch.Tensor) else max(top, SCALE)


def _gap(values: Tensor) -> Tensor:
    """[B]: smallest gap between distinct finite values per row (inf when there is none)."""
    finite = torch.isfinite(values)
    s = torch.where(finite, values, torch.full_like(values, math.inf)).sort(dim=1).values
    gap = s[:, 1:] - s[:, :-1]
    gap = torch.where(torch.isfinite(gap) & (gap > 0), gap, torch.full_like(gap, math.inf))
    return gap.amin(1) if gap.shape[1] else torch.full_like(values[:, 0], math.inf)


def unit(ctx: core.EpisodeContext, name: str, step: int, per_sender: bool = False, keys: Tensor | None = None,
         dtype: torch.dtype | None = None) -> Tensor:
    """[B, N, N] (or [B, N, 1]) uniforms in [0, 1): the keyed draws behind the jitter of round `step`."""
    keys = core.episode_keys(ctx.unit_ids) if keys is None else keys
    shape = (ctx.N, 1) if per_sender else (ctx.N, ctx.N)
    return core.keyed_uniform(keys, f"bazt|{ctx.arm}|jitter|{name}", step, shape, device=ctx.device,
                              dtype=ctx.dtype if dtype is None else dtype)


def at_own_scale(stat: Tensor, offset: Tensor) -> Tensor:
    """`stat` plus `offset` (a keyed u - 0.5, same shape) times top_scale(dtype, |value|) of each element: a jitter whose
    scale depends on the dtype and the element's own value only, so a value can be jittered the moment it is known. The
    upstream verdict needs that (it is decided audit by audit, before the round's other scores exist) and the reported
    `upstream_judge` uses the same function, so verdict and statistic are the same number. Tied values get the same
    scale; with no gap guard two distinct values closer than the scale (2^-24 in float64, eight ulp in float32) may swap.
    NaN and infinities are left as they are. Elementwise IEEE operations only: one element alone and the same element
    in a batch give the same bits."""
    finite = torch.isfinite(stat)
    scale = top_scale(stat.dtype, torch.where(finite, stat.abs(), torch.ones_like(stat)))
    return torch.where(finite, stat + offset * scale, stat)


def jitter(stat: Tensor, ctx: core.EpisodeContext, name: str, step: int, per_sender: bool = False,
           keys: Tensor | None = None) -> Tensor:
    """`stat` [B, N, N] plus the keyed jitter; NaN and infinities are left as they are."""
    u = unit(ctx, name, step, per_sender, keys, stat.dtype)
    finite = torch.isfinite(stat)
    magnitude = torch.where(finite, stat.abs(), torch.zeros_like(stat)).amax((1, 2))
    if per_sender:                      # one value per sender: the gap is taken among N values, not N x N
        flat = stat.nan_to_num(nan=-math.inf).amax(-1)
        gap = _gap(flat)
    else:
        gap = _gap(stat.flatten(1))
    scale = torch.minimum(top_scale(stat.dtype, magnitude), gap / 4.0)
    return torch.where(finite, stat + (u - 0.5) * scale[:, None, None], stat)
