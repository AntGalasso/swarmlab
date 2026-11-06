"""Embedding of message texts: one batched encoder call per request, a cache keyed by the hash of the text.

The encoder is any callable `encode(texts) -> float32 [n, dim]` (the sentence-transformers adapter in production, a
fake in tests). A text's vector is computed once and never changes afterwards, whatever batch asks for it later; two
requests that batch the same text differently can differ at float32 rounding level (padding), no more.
"""
from __future__ import annotations

import hashlib
from typing import Callable, Dict, Sequence

import torch

from bazt_lab.core import public as core


def text_key(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=16).digest()


class CachedEmbedder:
    """`embedder(texts) -> float32 [len(texts), dim]`, rows in input order, duplicates and cached texts not re-encoded."""

    def __init__(self, encode: Callable[[Sequence[str]], torch.Tensor]) -> None:
        self._encode = encode
        # ponytail: unbounded dict, about 1.7 kB per distinct text; bound it (LRU) if a run embeds millions of texts.
        self._cache: Dict[bytes, torch.Tensor] = {}

    def __call__(self, texts: Sequence[str]) -> torch.Tensor:
        texts = list(texts)
        if not texts:
            raise ValueError("embedding needs at least one text")
        if not all(isinstance(t, str) for t in texts):
            raise TypeError("texts must all be str")
        keys = [text_key(t) for t in texts]
        misses: Dict[bytes, str] = {}
        for key, text in zip(keys, texts, strict=True):
            if key not in self._cache:
                misses.setdefault(key, text)
        if misses:
            out = self._encode(list(misses.values()))
            if out.shape[0] != len(misses):
                raise ValueError(f"{len(misses)} texts sent to the encoder, {out.shape[0]} row(s) came back")
            for key, row in zip(misses, out.detach().to(torch.float32), strict=True):
                self._cache[key] = row.clone()
        return torch.stack([self._cache[k] for k in keys])


_SHARED: Dict[tuple, CachedEmbedder] = {}


FAKE_DIM = 384          # the width of all-MiniLM-L6-v2


def fake_encode(texts: Sequence[str]) -> torch.Tensor:
    """Deterministic unit vectors keyed by the text: an offline stand-in for the real encoder (tests and e2e only)."""
    out = torch.stack([torch.randn(FAKE_DIM, generator=core.stream("gnn_fake_embed", t), dtype=torch.float32)
                       for t in texts])
    return out / out.norm(dim=1, keepdim=True)


def shared_embedder(settings) -> CachedEmbedder:
    """The process-wide embedder of the settings' sentence-transformers model, loaded on first use. With the
    test-only key `gnn.embedding.fake` true it is `fake_encode` instead, so offline runs download nothing."""
    key = tuple(core.get(settings, f"gnn.embedding.{k}") for k in ("model", "revision", "batch_size"))
    if core.is_set(settings, "gnn.embedding.fake") and core.get(settings, "gnn.embedding.fake"):
        key = ("fake",) + key
        _SHARED.setdefault(key, CachedEmbedder(fake_encode))
    if key not in _SHARED:
        from bazt_lab.features.gnn.adapters.sbert import load_encoder  # heavy import, only when really used
        _SHARED[key] = CachedEmbedder(load_encoder(*key))
    return _SHARED[key]
