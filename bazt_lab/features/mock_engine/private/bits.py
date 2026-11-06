"""Boolean mask algebra on uint8 views.

On CPU, `&`, `|` and `~` on bool tensors are 50-80x slower than the same operation on uint8 (a measured 650 us against
12 us for a [512, 12, 12] mask). A bool tensor is one byte holding 0 or 1, so `.view(torch.uint8)` is a zero-copy
reinterpretation, `&` and `|` keep the values in {0, 1}, and `x ^ 1` is the negation. Results are identical to the bool
operators; only the speed differs.
"""
from __future__ import annotations

import torch

Tensor = torch.Tensor


def u8(m: Tensor) -> Tensor:
    return m.view(torch.uint8)


def band(a: Tensor, b: Tensor, *more: Tensor) -> Tensor:
    out = u8(a) & u8(b)
    for m in more:
        out = out & u8(m)
    return out.view(torch.bool)


def bor(a: Tensor, b: Tensor, *more: Tensor) -> Tensor:
    out = u8(a) | u8(b)
    for m in more:
        out = out | u8(m)
    return out.view(torch.bool)


def bnot(m: Tensor) -> Tensor:
    return (u8(m) ^ 1).view(torch.bool)


def bandnot(a: Tensor, b: Tensor) -> Tensor:
    """a & ~b."""
    return (u8(a) & (u8(b) ^ 1)).view(torch.bool)
