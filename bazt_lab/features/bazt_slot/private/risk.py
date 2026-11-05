"""Evidence to posterior. Each agent carries a log-odds of being compromised: free evidence now (the cheap monitor)
plus a decaying memory of paid verdicts. Float64 throughout, [B, N]."""
from __future__ import annotations

import math

import torch

Tensor = torch.Tensor
_EDGE = 1e-6


def logit(p: Tensor) -> Tensor:
    p = p.clamp(_EDGE, 1.0 - _EDGE)
    return torch.log(p) - torch.log1p(-p)


def sigmoid(x: Tensor) -> Tensor:
    return torch.sigmoid(x)


def free_logodds(monitor: Tensor | None, active: Tensor, prior: float) -> Tensor:
    """Log-odds from the free monitor reading of an active agent; the prior for an agent that sends nothing (isolated)
    or when the platform has no monitor."""
    base = torch.full(active.shape, math.log(prior / (1.0 - prior)), dtype=torch.float64)
    return base if monitor is None else torch.where(active, logit(monitor), base)


def remember(memory: Tensor, flagged: Tensor, audited: Tensor, decay: float, flag: float, clear: float) -> Tensor:
    """memory' = decay * memory + flag for a flagged sender, - clear for an audited sender the pipeline let through."""
    verdict = flagged.to(torch.float64) * flag - (audited & ~flagged).to(torch.float64) * clear
    return decay * memory + verdict


def eligible(free: Tensor, memory: Tensor, threshold: float) -> Tensor:
    """bool [B, N]: free evidence over the calibrated threshold (net of a clearing verdict), or a positive verdict.
    A missing threshold is +inf: only a verdict then makes an agent eligible. Eligibility only keeps the rate of
    actions on benign agents at the calibrated alpha; whether acting pays is the business of the pricing."""
    return ((free + memory) > threshold) | (memory > 0.0)
