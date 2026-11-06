"""Reference schedule of an attack: which rounds carry the payload (attack_params keys `start_frac` and `rate`)."""
from __future__ import annotations

import math

import torch

EPS = 1e-9      # float guard: 0.29 * 100 = 28.999999999999996 must still count as 29


def attack_rounds(T: int, start_frac: float = 0.0, rate: float = 1.0) -> torch.Tensor:
    """bool [T], index t - 1 for round t. The attack begins at round `1 + floor(start_frac * T)`; from there, the
    k-th opportunity (k = 1, 2, ...) is an attack round iff `ceil(k * rate) > ceil((k - 1) * rate)`. The first
    opportunity is always an attack round, and n opportunities carry `ceil(n * rate)` attacks, evenly spaced."""
    if not 0.0 <= start_frac <= 1.0:
        raise ValueError(f"start_frac must be in [0, 1], got {start_frac}")
    if not 0.0 < rate <= 1.0:
        raise ValueError(f"rate must be in (0, 1], got {rate}")
    start = 1 + math.floor(start_frac * T + EPS)
    ks = torch.arange(T, dtype=torch.float64) - (start - 1) + 1          # opportunity number of each round
    ceil_k = torch.ceil(ks * rate - EPS)
    ceil_prev = torch.ceil((ks - 1) * rate - EPS)
    return (ks >= 1) & (ceil_k > ceil_prev)
