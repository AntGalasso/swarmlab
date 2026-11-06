"""Communication graphs of an LLM seed run. Every draw is keyed (core.stream), so a graph depends on its key and
on N only. Edges are directed deliveries adj[i, j] = 1 for i -> j; all topologies but random_directed are symmetric.

random_directed, hub_star, barabasi_albert, watts_strogatz are the grid (spec 4.1); chain, tree and wheel (G-Safeguard's
"star") are supported and take no parameter.
"""
from __future__ import annotations

from typing import Mapping, Tuple

import numpy as np
import torch

from bazt_lab.core import public as core

TOPOLOGIES = ("random_directed", "hub_star", "barabasi_albert", "watts_strogatz", "chain", "tree", "wheel")
WS_ATTEMPTS = 20          # candidates for a rewiring target, as in swarmlab


def _sym(a: np.ndarray) -> np.ndarray:
    return ((a + a.T) > 0).astype(np.int64)


def _random_directed(n: int, p: float, gen: torch.Generator) -> np.ndarray:
    adj = (core.uniforms(gen, (n, n)).numpy() < p).astype(np.int64)
    np.fill_diagonal(adj, 0)
    return adj


def _barabasi_albert(n: int, m: int, gen: torch.Generator) -> np.ndarray:
    """Agents 0..m form a star around 0; each later agent attaches to m distinct earlier ones with probability
    proportional to degree (exponential race: the m smallest of -ln(u) / degree)."""
    if not 1 <= m < n:
        raise ValueError(f"runtime.ba_m={m} must satisfy 1 <= ba_m < N={n}")
    adj = np.zeros((n, n), dtype=np.int64)
    adj[0, 1:m + 1] = adj[1:m + 1, 0] = 1
    u = core.uniforms(gen, (n, n)).numpy()
    for v in range(m + 1, n):
        deg = adj[:v, :v].sum(1).astype(np.float64)
        race = -np.log(np.maximum(u[v, :v], 1e-300)) / deg
        for t in np.argsort(race, kind="stable")[:m]:
            adj[v, t] = adj[t, v] = 1
    return adj


def _watts_strogatz(n: int, k: int, p: float, gen: torch.Generator) -> np.ndarray:
    """Ring lattice with k // 2 neighbours on each side; each lattice edge is rewired with probability p to the first
    of 20 uniform candidates that is neither the agent nor already its neighbour (else it stays)."""
    half = k // 2
    if k < 2 or k % 2 or half >= n:
        raise ValueError(f"runtime.ws_k={k} must be even and satisfy 2 <= ws_k and ws_k // 2 < N={n}")
    u = core.uniforms(gen, (n * half, 1 + WS_ATTEMPTS)).numpy()
    adj = np.zeros((n, n), dtype=np.int64)
    for i in range(n):                                   # the full lattice first, so a rewiring never meets a later edge
        for d in range(1, half + 1):
            adj[i, (i + d) % n] = adj[(i + d) % n, i] = 1
    for e in range(n * half):                            # each lattice edge moves (never merges): the count is kept
        i = e // half
        j = (i + e % half + 1) % n
        if u[e, 0] < p:
            for c in (u[e, 1:] * n).astype(np.int64).clip(max=n - 1):
                if c != i and not adj[i, c]:
                    adj[i, j] = adj[j, i] = 0
                    adj[i, c] = adj[c, i] = 1
                    break
    return adj


def build_adjacency(topology: str, n: int, settings: Mapping, key: Tuple) -> np.ndarray:
    """int64 [n, n]. `key` seeds the stream of the random topologies. Reads runtime.* of the topology only."""
    if topology not in TOPOLOGIES:
        raise ValueError(f"unknown topology {topology!r}; the harness knows {TOPOLOGIES}")
    if n < 2:
        raise ValueError(f"a graph needs at least 2 agents, got N={n}")
    gen = core.stream(*key)
    idx = np.arange(n)
    if topology == "random_directed":
        return _random_directed(n, float(core.get(settings, "runtime.random_edge_probability")), gen)
    if topology == "barabasi_albert":
        return _barabasi_albert(n, int(core.get(settings, "runtime.ba_m")), gen)
    if topology == "watts_strogatz":
        return _watts_strogatz(n, int(core.get(settings, "runtime.ws_k")),
                               float(core.get(settings, "runtime.ws_rewire_p")), gen)
    adj = np.zeros((n, n), dtype=np.int64)
    if topology == "hub_star":
        adj[0, 1:] = adj[1:, 0] = 1
    elif topology == "chain":
        adj = (np.abs(idx[:, None] - idx[None, :]) == 1).astype(np.int64)
    elif topology == "tree":
        for child in range(1, n):
            adj[(child - 1) // 2, child] = 1
    else:                                              # wheel: hub 0 plus a ring through the leaves
        adj[0, 1:] = 1
        for i in range(1, n):
            adj[i, 1 + i % (n - 1)] = 1
    adj = _sym(adj)
    np.fill_diagonal(adj, 0)
    return adj
