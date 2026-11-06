"""Communication graphs for the mock, vectorised over a batch of episodes (B), plus the horizon rule.

Topologies (spec 4.2): star_hub, star_leaf (same graph, different attack entry, see world.py), chain, small_world
(Watts-Strogatz, replicating swarmlab's construction step by step) and barabasi_albert. Every graph is undirected
(a symmetric bool [B, N, N] adjacency with an empty diagonal): an edge is a pair of directed deliveries.

Random graphs consume a pre-drawn uniform tensor `u` ([B, *draw_shape]) so that each episode depends only on its own
keyed stream, never on the batch.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

TOPOLOGIES = ("star_hub", "star_leaf", "chain", "small_world", "barabasi_albert")
WS_ATTEMPTS = 20          # swarmlab's retries to find a rewiring target not already a neighbour


def draw_shape(topology: str, N: int, ws_k: Optional[int], ba_m: Optional[int]) -> Optional[Tuple[int, ...]]:
    """Shape of the uniforms one episode's graph needs (None for the deterministic topologies); validates N."""
    if topology not in TOPOLOGIES:
        raise ValueError(f"unknown topology {topology!r}; the mock knows {TOPOLOGIES}")
    if N < 2:
        raise ValueError(f"a swarm needs at least 2 agents, got N={N}")
    if topology == "small_world":
        if not 2 <= ws_k or ws_k // 2 >= N:
            raise ValueError(f"ws_k={ws_k} must satisfy 2 <= ws_k and ws_k // 2 < N={N}")
        return (N * (ws_k // 2), 1 + WS_ATTEMPTS)
    if topology == "barabasi_albert":
        if ba_m < 1 or N <= ba_m:
            raise ValueError(f"ba_m={ba_m} must satisfy 1 <= ba_m < N={N}")
        return (N, N)
    return None


def build_graphs(topology: str, N: int, u: Optional[torch.Tensor], B: int, ws_k: Optional[int] = None,
                 ws_p: Optional[float] = None, ba_m: Optional[int] = None) -> torch.Tensor:
    draw_shape(topology, N, ws_k, ba_m)
    idx = torch.arange(N)
    if topology in ("star_hub", "star_leaf"):
        one = torch.zeros(N, N, dtype=torch.bool)
        one[0, 1:] = one[1:, 0] = True
    elif topology == "chain":
        one = (idx[:, None] - idx[None, :]).abs() == 1
    elif topology == "small_world":
        return _watts_strogatz(u, N, ws_k, ws_p)
    else:
        return _barabasi_albert(u, N, ba_m)
    return one.expand(B, N, N).clone()


def _watts_strogatz(u: torch.Tensor, N: int, k: int, p: float) -> torch.Tensor:
    """swarmlab._build_graph: ring lattice edges (i, i+d), d = 1..k//2, in i-major order; each edge is rewired with
    probability p to the first of 20 uniform candidates that is not i and not already a neighbour of i (else it stays)."""
    B, half = u.shape[0], k // 2
    adj = torch.zeros(B, N, N, dtype=torch.bool)
    rows = torch.arange(B)
    for e in range(N * half):
        i, j = e // half, (e // half + e % half + 1) % N
        cand = (u[:, e, 1:] * N).long().clamp_(max=N - 1)                  # [B, 20] targets, uniform on 0..N-1
        ok = ~adj[:, i].gather(1, cand) & (cand != i)
        j2 = cand.gather(1, ok.long().argmax(1, keepdim=True))[:, 0]       # first valid attempt
        target = torch.where((u[:, e, 0] < p) & ok.any(1), j2, torch.full_like(j2, j))
        adj[rows, i, target] = adj[rows, target, i] = True
    return adj


def _barabasi_albert(u: torch.Tensor, N: int, m: int) -> torch.Tensor:
    """Initial star on agents 0..m (centre 0); each later agent v attaches to m distinct earlier agents drawn with
    probability proportional to their degree (exponential race: the m smallest of -ln(u)/degree)."""
    B = u.shape[0]
    adj = torch.zeros(B, N, N, dtype=torch.bool)
    adj[:, 0, 1:m + 1] = adj[:, 1:m + 1, 0] = True
    deg = adj.sum(2).double()
    rows = torch.arange(B)[:, None]
    for v in range(m + 1, N):
        race = -torch.log(u[:, v, :v].clamp_min(1e-300)) / deg[:, :v]
        tgt = race.topk(m, dim=1, largest=False).indices                    # [B, m], distinct
        adj[rows, v, tgt] = adj[rows, tgt, v] = True
        deg[rows, tgt] += 1.0
        deg[:, v] = m
    return adj


def _bfs(adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(diameter, connected) per graph: grow every agent's reach one hop per iteration until nothing changes."""
    B, N, _ = adj.shape
    A = adj.to(torch.float32)                                              # 0/1 products are exact in float32
    reach = torch.eye(N, dtype=torch.bool).expand(B, N, N)
    diam = torch.zeros(B, dtype=torch.int64)
    for d in range(1, N):
        nxt = reach | (torch.bmm(reach.to(torch.float32), A) > 0)
        grew = (nxt != reach).flatten(1).any(1)
        if not grew.any():
            break
        diam[grew] = d
        reach = nxt
    return diam, reach.flatten(1).all(1)


def diameters(adj: torch.Tensor) -> torch.Tensor:
    """Effective diameter, int64 [B]: the longest finite shortest path (METRICS_SPEC 14.3); unreachable pairs ignored."""
    return _bfs(adj)[0]


def reach_all(adj: torch.Tensor) -> torch.Tensor:
    """bool [B]: every agent can reach every other."""
    return _bfs(adj)[1]


def horizon_rounds(diameter: torch.Tensor, horizon_constant: float) -> torch.Tensor:
    """T = ceil(horizon_constant x effective diameter), at least 1 round. The product is rounded to 9 decimals before
    the ceiling so that 0.1 x 30 is 3 rounds, not 4 (binary floating point)."""
    t = torch.round(diameter.double() * horizon_constant * 1e9) / 1e9
    return torch.ceil(t).clamp_min(1).to(torch.int64)


def is_deterministic(topology: str) -> bool:
    return topology in ("star_hub", "star_leaf", "chain")


__all__ = ["TOPOLOGIES", "build_graphs", "diameters", "draw_shape", "horizon_rounds", "is_deterministic", "reach_all"]
