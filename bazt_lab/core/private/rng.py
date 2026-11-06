"""Reproducible random numbers, two families.

1. Keyed counter-based draws (the fast path, runs unchanged on CPU and CUDA): `keyed_uniform`, `keyed_normal`,
   `keyed_randint`, `keyed_permutation_scores`. A draw is a pure function of (episode key, stream name, counter,
   element index): it never depends on the batch size, the batch order, the device or the chunking.

   Construction. Per (episode, stream, counter), once per call and O(B): Philox4x32-10 (Salmon et al., SC11; checked
   against the Random123 known-answer vectors) maps the counter block (episode key lo, hi, stream hash lo, hi) under the
   key (counter lo, hi) to four 32-bit words; the first three are the 96-bit key (k0, k1, k2) of the stream. Per
   element, O(B x n): a four-stage xorshift-multiply hash of (element index, k0, k1, k2) gives two 32-bit words (hi, lo),
   and the draw is the 53-bit uniform (hi >> 5, lo >> 6) / 2**53. The hash is a keyed bijection of the index: stage
   j xors in key lane k_j, xorshifts and multiplies by an odd constant (the murmur3 / lowbias32 finalizer family), and
   the last stage has no key so every key lane is diffused. Its avalanche is ideal in the tests (every index and key bit
   flips every output bit with probability 1/2).

   Exactness. All lanes are int64 tensors that hold 32-bit words. A product x * C with x < 2**32 and C an odd 32-bit
   constant is taken as x * s with s = C or C - 2**32, the signed 32-bit form of C, so |x * s| < 2**63: no int64 overflow,
   no reliance on wrap-around; `& 0xFFFFFFFF` of the (possibly negative) product is exactly (x * C) mod 2**32. Shifts,
   xors, masks and integer-to-float64 conversion of values below 2**53 are exact, so the uniforms, randints and
   permutation scores are bit-identical on every device. Only the Box-Muller transform of `keyed_normal` uses float64
   log/sqrt/cos/sin, which agree across devices to within a few ulp.

   Collisions. The stream key has 96 bits (birthday bound ~1e-11 for a billion streams), the element index 32 bits
   (at most 2**32 draws per stream and call), the 53-bit uniform has the resolution of a double.

2. Per-episode `torch.Generator` objects (`stream`, `uniforms`, `batch_uniforms`): kept so lanes can migrate one by
   one. They cost Python time per episode and cannot run on a GPU.

Common random numbers: draws that must be identical across arms of one episode come from stream names WITHOUT the
arm (e.g. "adopt"); arm-specific draws put the arm in the stream name. Draw arm-independent uniforms for all
rounds up front and compare `u < p`: drawing only the entries an arm needs would desynchronise the arms. Never use
the global torch RNG. https://docs.pytorch.org/docs/2.14/notes/randomness.html
"""
from __future__ import annotations

import functools
import hashlib
import math
import operator
from typing import Iterator, Sequence

import torch

_MASK = 0xFFFFFFFF
_W0, _W1 = 0x9E3779B9, 0xBB67AE85                  # Philox key increments (golden ratio, sqrt(3) - 1)
# Philox multipliers are M0 = 0xD2511F53 and M1 = 0xCD9E8D57 (both above 2**31). x * M for a 32-bit lane x would
# overflow int64, so use M = 2**32 - N with N < 2**30: x * M = (x << 32) - x * N, and x * N < 2**62 never overflows.
# With r = x * (-N): hi = x + (r >> 32) (arithmetic shift = floor), lo = r & MASK. Four tensor ops per product.
_N0, _N1 = (1 << 32) - 0xD2511F53, (1 << 32) - 0xCD9E8D57
_DERIVE_KEYS = 3                                   # key lanes taken from the Philox output
_CHUNK_ELEMENTS = 1 << 18                          # CPU: draws per pass, ~2 MB int64 tensors (best of 2**15..2**20 measured)
_CUDA_CHUNK_ELEMENTS = 1 << 22                     # CUDA: ~35 small kernels per pass, so fewer and larger passes
_TWO_POW_53 = 2.0**-53
_FLOAT_DTYPES = (torch.float64, torch.float32)


def _signed32(c: int) -> int:
    """The signed 32-bit form of an odd 32-bit multiplier: congruent to c modulo 2**32, magnitude at most 2**31."""
    return c if c < 1 << 31 else c - (1 << 32)


# (xorshift, multiplier) of the four stages of the element hash; constants from murmur3 fmix32 and lowbias32 (Wellons).
_STAGES = ((16, _signed32(0x7FEB352D)), (15, _signed32(0x846CA68B)), (16, _signed32(0x9E3779B1)), (15, _signed32(0x85EBCA6B)))


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


# ------------------------------------------------------------------------------------------------ Philox4x32
def _mulhilo(x: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(hi, lo) 32-bit words of x * (2**32 - n) for a 32-bit lane x and a constant n < 2**30."""
    r = x * -n
    return x + (r >> 32), r & _MASK


def philox4x32(c0, c1, c2, c3, k0, k1, rounds: int = 10):
    """Philox4x32-`rounds` on int64 lanes in [0, 2**32). Counter lanes c0..c3 and key lanes k0, k1 are tensors that
    broadcast against each other (c0 and c2 must be tensors; the rest may be Python ints). Returns four int64
    tensors with the output words of the Random123 reference (known-answer tests are in the core test suite)."""
    for _ in range(rounds):
        h0, l0 = _mulhilo(c0, _N0)
        h1, l1 = _mulhilo(c2, _N1)
        c0, c1, c2, c3 = h1 ^ c1 ^ k0, l1, h0 ^ c3 ^ k1, l0
        k0 = (k0 + _W0) & _MASK
        k1 = (k1 + _W1) & _MASK
    return c0, c1, c2, c3


# ------------------------------------------------------------------------------------------------ element hash
def mix(index: torch.Tensor, k0, k1, k2) -> tuple[torch.Tensor, torch.Tensor]:
    """The element hash: 32-bit `index` and key lanes (int64 tensors in [0, 2**32), broadcasting) -> two 32-bit words."""
    (s1, m1), (s2, m2), (s3, m3), (s4, m4) = _STAGES
    x = index ^ k0
    a = ((x ^ (x >> s1)) * m1) & _MASK
    x = a ^ k1
    b = ((x ^ (x >> s2)) * m2) & _MASK
    x = a ^ b ^ k2
    a = ((x ^ (x >> s3)) * m3) & _MASK
    x = b ^ a
    b = ((x ^ (x >> s4)) * m4) & _MASK
    hi = a ^ b
    return hi ^ (hi >> 15), b ^ (b >> 15)


# ------------------------------------------------------------------------------------------------ public API
def episode_keys(ids: Sequence[str]) -> torch.Tensor:
    """int64 [B]: one 63-bit key per episode, `seed_of(id)`. The only Python loop of the keyed path; call it once
    per batch and keep the result on whatever device the draws run on."""
    return torch.tensor([seed_of(i) for i in ids], dtype=torch.int64)


@functools.lru_cache(maxsize=4096)
def _stream_hash(name: str) -> int:
    return seed_of(name)


def _check_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype not in _FLOAT_DTYPES:
        raise ValueError(f"dtype must be torch.float64 or torch.float32, got {dtype}")
    return dtype


def as_uniform_dtype(u: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """float64 uniforms in [0, 1) rounded to `dtype`, kept strictly below 1 (rounding 1 - 2**-53 gives 1.0 in float32)."""
    if dtype == u.dtype:
        return u
    return u.to(dtype).clamp_(max=1.0 - torch.finfo(dtype).eps / 2)


def _prepare(keys, stream_name, counter, shape, device):
    """Validate the arguments of every keyed draw and derive the three 32-bit key lanes [B, 1] of each stream."""
    if not isinstance(keys, torch.Tensor) or keys.dtype != torch.int64:
        raise TypeError("keys must be an int64 tensor [B] (build it with episode_keys)")
    if keys.ndim != 1:
        raise ValueError(f"keys must have shape [B], got {tuple(keys.shape)}")
    if not isinstance(stream_name, str):
        raise TypeError(f"stream must be a str, got {type(stream_name).__name__}")
    try:
        dims = tuple(operator.index(s) for s in shape)
    except TypeError:
        raise ValueError(f"shape must be a tuple of ints, got {shape!r}") from None
    if any(s < 0 for s in dims):
        raise ValueError(f"shape must be non-negative, got {shape!r}")
    dev = keys.device if device is None else torch.device(device)
    batch = keys.shape[0]
    keys = keys.to(dev)
    if isinstance(counter, torch.Tensor):
        if counter.dtype not in (torch.int64, torch.int32, torch.int16, torch.uint8, torch.int8):
            raise TypeError(f"counter tensor must be an integer tensor, got {counter.dtype}")
        count = counter.to(device=dev, dtype=torch.int64)
        if count.ndim == 0:
            count = count.expand(batch)
        elif count.shape != (batch,):
            raise ValueError(f"counter tensor must be a scalar or have shape [B]=[{batch}], got {tuple(counter.shape)}")
        c_lo, c_hi = count & _MASK, (count >> 32) & _MASK
    else:
        try:
            value = operator.index(counter)
        except TypeError:
            raise TypeError(f"counter must be an int or an int tensor, got {type(counter).__name__}") from None
        if not 0 <= value < 1 << 63:
            raise ValueError(f"counter must be in [0, 2**63), got {value}")
        c_lo, c_hi = value & _MASK, value >> 32
    sh = _stream_hash(stream_name)
    lo = keys & _MASK
    words = philox4x32(lo, (keys >> 32) & _MASK, torch.full_like(lo, sh & _MASK), torch.full_like(lo, sh >> 32), c_lo, c_hi)
    return dev, dims, tuple(w.reshape(batch, 1) for w in words[:_DERIVE_KEYS])


def _uniform_chunks(dev: torch.device, lanes: tuple, n: int, group: int = 1) -> Iterator[tuple]:
    """Yield (r0, r1, j0, j1, u) with u the float64 uniforms [r1 - r0, j1 - j0] of elements j0..j1 of episodes r0..r1.
    Work is done in chunks of about `_CHUNK_ELEMENTS` (CUDA: `_CUDA_CHUNK_ELEMENTS`) draws, so temporary memory is bounded
    whatever B and the shape are; column chunks hold a multiple of `group` elements."""
    if n > 1 << 32:
        raise ValueError(f"more than 2**32 draws per episode in one call ({n})")
    k0, k1, k2 = lanes
    batch = k0.shape[0]
    chunk = _CUDA_CHUNK_ELEMENTS if dev.type == "cuda" else _CHUNK_ELEMENTS
    rows = min(batch, chunk)
    cols = max(group, chunk // rows // group * group)
    for r0 in range(0, batch, rows):
        r1 = min(batch, r0 + rows)
        for j0 in range(0, n, cols):
            j1 = min(n, j0 + cols)
            index = torch.arange(j0, j1, dtype=torch.int64, device=dev).reshape(1, -1)
            hi, lo = mix(index, k0[r0:r1], k1[r0:r1], k2[r0:r1])
            yield r0, r1, j0, j1, ((hi >> 5) * 67108864 + (lo >> 6)).to(torch.float64) * _TWO_POW_53


def keyed_uniform(keys: torch.Tensor, stream: str, counter, shape: Sequence[int], device=None,
                  dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Uniforms in [0, 1), shape [B, *shape], from (keys[b], stream, counter, element index) only.

    `counter` is an int in [0, 2**63) or an integer tensor (scalar or [B], read as an unsigned 64-bit pattern),
    e.g. the round number. `device` defaults to the device of `keys`. `dtype` float32 returns the same numbers
    rounded (and kept below 1.0); the bits are always drawn as integers and float64 first."""
    _check_dtype(dtype)
    dev, dims, lanes = _prepare(keys, stream, counter, shape, device)
    batch, n = lanes[0].shape[0], math.prod(dims)
    out = torch.empty(batch, n, dtype=dtype, device=dev)
    for r0, r1, j0, j1, u in _uniform_chunks(dev, lanes, n) if batch and n else ():
        out[r0:r1, j0:j1] = as_uniform_dtype(u, dtype)
    return out.reshape(batch, *dims)


def keyed_normal(keys: torch.Tensor, stream: str, counter, shape: Sequence[int], device=None,
                 dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Standard normals [B, *shape] by Box-Muller in float64 on adjacent uniforms (elements 2k and 2k + 1 of the same
    stream give cos and sin of one pair), then rounded to `dtype`. Same keying rules as `keyed_uniform`."""
    _check_dtype(dtype)
    dev, dims, lanes = _prepare(keys, stream, counter, shape, device)
    batch, n = lanes[0].shape[0], math.prod(dims)
    padded = n + n % 2
    out = torch.empty(batch, padded // 2, 2, dtype=dtype, device=dev)
    for r0, r1, j0, j1, u in _uniform_chunks(dev, lanes, padded, group=2) if batch and n else ():
        radius = torch.sqrt(-2.0 * torch.log1p(-u[:, 0::2]))              # 1 - u is in (0, 1]: finite
        angle = (2.0 * math.pi) * u[:, 1::2]
        out[r0:r1, j0 // 2:j1 // 2, 0] = (radius * torch.cos(angle)).to(dtype)
        out[r0:r1, j0 // 2:j1 // 2, 1] = (radius * torch.sin(angle)).to(dtype)
    return out.view(batch, padded)[:, :n].reshape(batch, *dims)


def keyed_randint(keys: torch.Tensor, stream: str, counter, shape: Sequence[int], high: int, device=None) -> torch.Tensor:
    """int64 in [0, high), shape [B, *shape]: floor(keyed_uniform * high) in float64, `1 <= high <= 2**31`."""
    try:
        bound = operator.index(high)
    except TypeError:
        raise ValueError(f"high must be an int in [1, 2**31], got {high!r}") from None
    if not 1 <= bound <= 1 << 31:
        raise ValueError(f"high must be in [1, 2**31], got {bound}")
    u = keyed_uniform(keys, stream, counter, shape, device)
    return u.mul_(bound).floor_().to(torch.int64).clamp_(max=bound - 1)


def keyed_permutation_scores(keys: torch.Tensor, stream: str, counter, shape: Sequence[int], device=None) -> torch.Tensor:
    """float64 uniform scores [B, *shape] for random subsets and tie breaks: `torch.argsort(scores, dim=-1,
    stable=True)` is a keyed random permutation of the last axis (the first k indices a keyed random k-subset).
    Always float64: the order must not depend on compute.dtype. Equal to `keyed_uniform` of the same arguments."""
    return keyed_uniform(keys, stream, counter, shape, device)
