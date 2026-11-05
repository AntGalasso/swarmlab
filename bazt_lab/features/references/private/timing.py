"""MatchedTiming: lets an arm act only at the rounds of a schedule (spec 2.7, D32).

The matched-timing control separates WHEN an arm acts from WHOM it picks: the wrapped arm runs every round, so its
own state (alarm runs, audit gaps, statistics for calibration) keeps evolving, but nothing it decides outside the
schedule reaches the platform. That covers its actions and its audits: at an unscheduled round the wrapper hands the
arm an observation whose `audit` performs and debits nothing (the arm sees NaN scores, exactly as for an audit it
could not afford), and it drops every action mask.

The schedule of an episode is the set of rounds, 1-based like `Observation.t`, at which BAZT acted in the same
episode. It reaches the wrapper as `ctx.timing_schedule`, a bool [B, T] mask (see CONTRACT.md, Requests), or through
the constructor. Without one the wrapper refuses to run instead of acting at every round, with one exception: a
threshold-driven arm that has no thresholds yet is a calibration run (it can only collect its statistics), and gets
an empty schedule.
"""
from __future__ import annotations

import dataclasses
from typing import Mapping, Optional, Sequence

import torch

from bazt_lab.core import public as core

_MASKS = {"deny": 2, "sanitize": 2, "delay": 2, "edge_quarantine": 2,
          "rate_limit": 1, "privilege_degradation": 1, "isolation": 1, "reinstatement": 1}   # field -> agent dims


def schedule_from_rounds(rounds: Sequence[Sequence[int]], T: int) -> torch.Tensor:
    """bool [B, T]: row b is True at the 1-based rounds listed for episode b."""
    mask = torch.zeros(len(rounds), T, dtype=torch.bool)
    for b, rs in enumerate(rounds):
        for r in rs:
            if not 1 <= r <= T:
                raise ValueError(f"schedule round {r} of episode {b} is outside 1..{T}")
            mask[b, r - 1] = True
    return mask


class MatchedTiming:
    def __init__(self, inner: core.Defender, schedule: Optional[torch.Tensor] = None) -> None:
        self.inner = inner
        self.name = f"{inner.name}_matched"
        self.needs_truth = bool(getattr(inner, "needs_truth", False))
        self.schedule = schedule

    def reset(self, ctx: core.EpisodeContext) -> None:
        schedule = getattr(ctx, "timing_schedule", None)
        if schedule is None:
            schedule = self.schedule
        if schedule is None and getattr(self.inner, "thresholded", False) and not ctx.thresholds:
            schedule = torch.zeros(ctx.B, ctx.T, dtype=torch.bool)   # calibration run: statistics only, no action
        if schedule is None:
            raise ValueError(f"{self.name} needs the episode's timing schedule (ctx.timing_schedule, bool [B, T]) "
                             f"to know the rounds at which it may act")
        if tuple(schedule.shape) != (ctx.B, ctx.T):
            raise ValueError(f"timing schedule has shape {tuple(schedule.shape)}, expected (B, T) = {(ctx.B, ctx.T)}")
        self.when = schedule.bool()
        self.inner.reset(ctx)

    def decide(self, obs: core.Observation) -> core.Actions:
        on = self.when[:, obs.t - 1]
        gated = dataclasses.replace(obs, audit=lambda senders: obs.audit(senders & on[:, None]))
        acts = self.inner.decide(gated)
        keep = {k: getattr(acts, k) & on.reshape(-1, *[1] * dims) for k, dims in _MASKS.items()
                if getattr(acts, k) is not None}
        return dataclasses.replace(acts, **keep)

    def state_dict(self) -> dict:
        return {f"inner.{k}": v for k, v in self.inner.state_dict().items()}

    def load_state_dict(self, state: Mapping) -> None:
        self.inner.load_state_dict({k.removeprefix("inner."): v for k, v in state.items()})
