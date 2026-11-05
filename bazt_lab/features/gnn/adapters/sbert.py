"""sentence-transformers adapter: the MiniLM encoder G-Safeguard embeds agent messages with.

Upstream calls `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").encode(text)` with the default
`normalize_embeddings=False`. The model's own pipeline ends in a `Normalize` module (modules.json at the pinned
revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41), so its vectors are unit length either way; asking the library
to normalise again would change nothing, and we pass False to stay literally equal to upstream.
"""
from __future__ import annotations

from functools import cache
from typing import Callable, Sequence

import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


@cache
def load_encoder(model: str, revision: str, batch_size: int) -> Callable[[Sequence[str]], torch.Tensor]:
    """`encode(texts) -> float32 [n, dim]` over a model loaded once per (model, revision, batch size) on CPU."""
    load_dotenv()                      # a public model needs no token; HF_TOKEN from .env is honoured if present
    st = SentenceTransformer(model, revision=revision, device="cpu")

    def encode(texts: Sequence[str]) -> torch.Tensor:
        with torch.inference_mode():
            out = st.encode(list(texts), batch_size=batch_size, convert_to_tensor=True,
                            normalize_embeddings=False, show_progress_bar=False)
        return out.to(device="cpu", dtype=torch.float32).clone()      # clone: a normal tensor, usable under autograd

    return encode
