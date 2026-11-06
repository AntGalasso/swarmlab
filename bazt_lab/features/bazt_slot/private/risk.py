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


def prior_logodds(prior: float) -> float:
    return math.log(prior / (1.0 - prior))


def free_logodds(monitor: Tensor | None, active: Tensor, prior: float, dtype: torch.dtype = torch.float64,
                 held: Tensor | None = None) -> Tensor:
    """Log-odds from the free monitor reading of an active agent. An agent that sends nothing (isolated) has no reading:
    silence is not evidence of being clean (D119), so it keeps `held`, the free evidence of its last active round (the
    prior for an agent that was never active). Without a monitor (the LLM side) every agent has the prior."""
    base = torch.full(active.shape, prior_logodds(prior), dtype=dtype if monitor is None else monitor.dtype,
                      device=active.device)
    if monitor is None:
        return base
    return torch.where(active, logit(monitor), base if held is None else held.to(base.dtype))


def cleared(audited: Tensor, flagged: Tensor) -> Tensor:
    """bool [B, N]: an audit of the sender this round found nothing the pipeline flags; the verdict `remember` writes as
    -clear_logodds into the memory. The only evidence that may end an isolation (D119)."""
    return audited & ~flagged


def remember(memory: Tensor, flagged: Tensor, audited: Tensor, decay: float, flag: float, clear: float) -> Tensor:
    """memory' = decay * memory + flag for a flagged sender, - clear for an audited sender the pipeline let through."""
    verdict = flagged.to(memory.dtype) * flag - (audited & ~flagged).to(memory.dtype) * clear
    return decay * memory + verdict


def eligible(free: Tensor, memory: Tensor, threshold: float, hold: float) -> Tensor:
    """bool [B, N]: free evidence over the calibrated threshold (net of a clearing verdict), or a verdict that still stands.
    A verdict stands while its decayed weight `memory` exceeds `hold` (the weight of a clearing verdict, bazt.clear_logodds):
    one flag lapses after a few rounds instead of keeping the sender eligible for ever, and an audit that clears it ends it at
    once. A missing threshold is +inf: only a standing verdict then makes an agent eligible. Eligibility only keeps the rate of
    actions on benign agents at the calibrated alpha; whether acting pays is the business of the pricing."""
    return ((free + memory) > threshold) | (memory > hold)
