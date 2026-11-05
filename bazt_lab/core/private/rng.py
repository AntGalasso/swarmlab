"""Reproducible random streams (torch.Generator), one per string key.

Common random numbers: draws that must be identical across arms of one episode come from keys WITHOUT the arm
(e.g. stream("adopt", topology, seed)); arm-specific draws add the arm to the key. Pre-draw arm-independent
uniforms for all rounds up front (`uniforms`) and compare `u < p`: drawing only the entries an arm needs would
desynchronise the streams across arms. Never use the global torch RNG. https://docs.pytorch.org/docs/2.14/notes/randomness.html
"""
from __future__ import annotations

import hashlib
from typing import Sequence

import torch


def seed_of(*key: object) -> int:
    """A 63-bit seed from a key; blake2b keeps distinct keys (e.g. seed 1 arm 12 vs seed 11 arm 2) apart."""
    text = "|".join(map(str, key)).encode()
    return int.from_bytes(hashlib.blake2b(text, digest_size=8).digest(), "little") >> 1


def stream(*key: object) -> torch.Generator:
    """A fresh CPU generator seeded from `key`."""
    return torch.Generator(device="cpu").manual_seed(seed_of(*key))


def uniforms(gen: torch.Generator, shape: Sequence[int]) -> torch.Tensor:
    """float64 uniforms in [0, 1), drawn in one call so the stream position does not depend on use."""
    return torch.rand(tuple(shape), generator=gen, dtype=torch.float64)


def batch_uniforms(keys: Sequence[tuple], shape: Sequence[int]) -> torch.Tensor:
    """Stack per-episode uniforms [B, *shape]; each episode's draws depend only on its own key, never on B."""
    return torch.stack([uniforms(stream(*k), shape) for k in keys])
