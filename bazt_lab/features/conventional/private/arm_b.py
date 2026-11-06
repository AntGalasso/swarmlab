"""Arm B: deployed agent zero-trust pipeline (spec 2.2): A, then the LLM judge on every sent text, then a rate limit
per agent identity. Each hop is decided on its own: the only thing B carries across rounds is the budget latch."""
from __future__ import annotations

import math

import torch

from bazt_lab.core import public as core
from bazt_lab.features.conventional.private.common import STAT_JUDGE, BudgetedArm, Tensor, mask_and

RATE_LIMIT_KEY = "runtime.rate_limit_per_identity"


class ArmB(BudgetedArm):
    name = "B"

    def _allocate(self, ctx: core.EpisodeContext) -> None:
        limit = core.get(ctx.settings, RATE_LIMIT_KEY)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError(f"{RATE_LIMIT_KEY} must be an integer >= 1, got {limit!r}")
        self.rate_limit = limit

    def _policy(self, obs: core.Observation, evaluated: Tensor, audited: Tensor):
        # A missing threshold is +inf: the statistic is reported (calibration runs) and nothing is denied by it.
        threshold = float(self.ctx.thresholds.get(STAT_JUDGE, math.inf))
        score = audited[:, :, None].expand_as(evaluated)
        over = (audited > threshold)[:, :, None].expand_as(evaluated)    # NaN (unchecked sender) is never over
        flagged = mask_and(evaluated, over)
        checked = mask_and(evaluated, (~audited.isnan())[:, :, None].expand_as(evaluated))   # unchecked: only A's business
        passed = core.as_bool(core.as_uint8(checked) ^ core.as_uint8(flagged))            # flagged is a subset of checked
        want = core.as_bool(core.as_uint8(flagged) | core.as_uint8(self._rate_limited(passed)))
        statistic = score.masked_fill(~evaluated, math.nan)
        return want, statistic, {STAT_JUDGE: statistic}, flagged

    def _rate_limited(self, passed: Tensor) -> Tensor:
        """Deliveries beyond `rate_limit` receivers per sender and round. Which receivers get through is a keyed
        random draw: receiver order carries no meaning and an index rule would favour low indices."""
        limit = self.rate_limit
        rows = (passed.sum(-1) > limit).any(-1).nonzero().flatten()
        out = torch.zeros_like(passed)
        if len(rows):                                                   # rows: the episodes where the limit bites
            u = self._draw("rate_limit", rows).masked_fill(~passed[rows], 2.0)
            out[rows] = mask_and(passed[rows], u.argsort(-1).argsort(-1) >= limit)
        return out
