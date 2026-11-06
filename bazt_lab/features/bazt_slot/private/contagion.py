"""Topology-aware contagion over the remaining horizon, batched over episodes (dense torch, N <= ~200).

Conventions. A[b, i, j] = 1 when i sends to j on a live delivery. Infection travels along A, so the
out-neighbourhood of i is what i can infect. Everything is float64; B is the leading episode dimension and
episodes never interact.

  katz        truncated Katz walk reach by matvec recursion (Katz 1953; the finite-horizon form of
              "time-critical influence minimisation", Wang et al., PACMMOD 2025). Gives the reach of a node and,
              from the same recursion, the value of each of its out-edges (link blocking, Kimura et al., TKDD 2009).
  relative_decay
              the Katz decay of each episode's graph, bazt.katz_decay / rho(A) (D115). The k-th term of the Katz sum is
              decay^(k-1) A^k s, powers of the directed A, so the series converges exactly when decay x rho(A) < 1, rho
              the spectral radius of that directed A (rho(A) = rho(A^T); on the symmetric graphs of the grid directed and
              undirected coincide). A fraction in (0, 1) of 1 / rho is therefore valid on every topology.
  leading_pair / eigen_drop
              NetShield (Tong et al., TKDE 2016) and NetMelt (Tong et al., CIKM 2012): the relative drop of the
              spectral radius when a node (all its edges) or an edge is removed, to first order, from the leading
              right/left eigenvectors by batched power iteration. For a symmetric A this is NetShield's
              Sv(i) = 2 lambda u_i^2; the directed form is 2 lambda v_i u_i / (v.u).
"""
from __future__ import annotations

import torch

Tensor = torch.Tensor
_EPS = 1e-12
_RESIDUAL = 0.5          # largest ||A u - lam u|| / (lam ||u||) of a pair still taken as an eigenpair


def katz(A: Tensor, s: Tensor, decay: float | Tensor, order: int) -> tuple[Tensor, Tensor, Tensor]:
    """Truncated Katz reach by the recursion r_0 = s, r_m = s + decay * A r_{m-1}.

    s[b, j] is the susceptibility weight of node j (what is lost if j is infected). `decay` is one number or one per
    episode ([B], see `relative_decay`), applied in the dtype of s. Returns, all [B, N]:
      H       = A r_{order-1}   = sum_{k=1..order} decay^(k-1) A^k s    reach of each node over `order` hops
      R       = r_{order-1}                                             continuation value behind each node, so
                                                                        the edge i->j is worth R_j and H_i = sum_j A_ij R_j
      H_prev  = A r_{order-2}   = the same reach over `order - 1` hops  (what a one-round delay removes: H - H_prev)
    order <= 0 means no hop is left: everything is zero. The matrices are never powered: `order - 1` bmm calls.
    """
    if order <= 0:
        z = torch.zeros_like(s)
        return z, z.clone(), z.clone()
    d = torch.as_tensor(decay, dtype=s.dtype, device=s.device)
    d = d[:, None] if d.dim() == 1 else d
    r, r_prev = s, None
    for _ in range(order - 1):
        r_prev = r
        r = s + d * torch.bmm(A, r[..., None])[..., 0]
    H = torch.bmm(A, r[..., None])[..., 0]
    H_prev = torch.zeros_like(s) if r_prev is None else torch.bmm(A, r_prev[..., None])[..., 0]
    return H, r, H_prev


def spectral_radius(adj: Tensor, iterations: int) -> Tensor:
    """float64 [B]: the spectral radius of each episode's directed graph adj[b, i, j] (i -> j), the leading eigenvalue
    of `leading_pair` (for a nonnegative matrix the Perron root is the spectral radius). 0 where the power iteration
    finds no eigenpair: a graph without a cycle (nilpotent, rho = 0), or one that has not converged in `iterations`."""
    lam, _, _ = leading_pair(adj.to(torch.float64), iterations)
    return lam


def relative_decay(adj: Tensor, fraction: float, iterations: int) -> Tensor:
    """float64 [B]: the Katz decay of each episode, fraction / rho(adj) (D115), fraction = bazt.katz_decay in (0, 1).

    `adj` is the base graph. Every matrix the Katz sum runs on is entrywise at most adj (live deliveries times a factor
    in [0, 1], or adj without quarantined edges), so by Perron-Frobenius monotonicity its radius is at most rho(adj) and
    decay x its radius <= fraction < 1: the sum converges on all of them, and the decay does not move when an action
    cuts an edge. Where no eigenpair is found, the out-degree bound max_i sum_j adj_ij >= rho stands in for rho; the
    divisor is never below 1 (a 0/1 graph with a cycle has rho >= 1), so an empty graph gives the fraction, never NaN.
    Computed once per episode in float64 with a fixed iteration count, so it depends on the batch only as far as CPU bmm
    does (bitwise at the small N of the tests, last-ulp differences from about N = 30 with MKL)."""
    A = adj.to(torch.float64)
    rho = spectral_radius(A, iterations)
    scale = torch.where(rho > 0, rho, A.sum(-1).amax(-1)).clamp_min(1.0)
    return fraction / scale


def _unit(x: Tensor) -> Tensor:
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(_EPS)


def leading_pair(A: Tensor, iterations: int, u0: Tensor | None = None, v0: Tensor | None = None):
    """Leading eigenvalue with its right (A u = lam u) and left (A^T v = lam v) eigenvectors, [B], [B, N], [B, N].

    Power iteration on A + I. The shift keeps the dominant eigenvalue strictly largest in modulus even on a
    bipartite graph (a path has +-lambda), where plain power iteration oscillates. `u0`, `v0` warm-start the
    iteration (the graph changes little between rounds). Fixed iteration count, so the result is deterministic and
    independent of batch size.

    A graph with no cycle (nilpotent A) has no eigenpair: the iteration on A + I then converges only algebraically
    and a Rayleigh quotient reports a small positive lam. The residual ||A u - lam u|| tells the two cases apart (about
    lam ||u|| on a DAG, about 0 once converged), so a pair that fails it is reported as lam = 0.
    """
    B, N, _ = A.shape
    start = torch.full((B, N), N ** -0.5, dtype=A.dtype, device=A.device)
    u = start if u0 is None else u0
    v = start.clone() if v0 is None else v0
    At = A.transpose(1, 2)
    for _ in range(iterations):
        u = _unit(u + torch.bmm(A, u[..., None])[..., 0])
        v = _unit(v + torch.bmm(At, v[..., None])[..., 0])
    vu = (v * u).sum(-1)
    ok = vu > 1e-9
    Au = torch.bmm(A, u[..., None])[..., 0]
    lam = ((v * Au).sum(-1) / torch.where(ok, vu, torch.ones_like(vu))).clamp_min(0.0)
    residual = torch.linalg.vector_norm(Au - lam[:, None] * u, dim=-1)
    lam = torch.where(ok & (residual <= _RESIDUAL * lam), lam, torch.zeros_like(lam))
    return torch.minimum(lam, A.sum(-1).amax(-1)), u, v


def drops_from_pair(A: Tensor, lam: Tensor, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    """Relative eigen-drops from a leading pair: (node_rel [B, N], edge_rel [B, N, N]), both in [0, 1].

    node_rel[i] = 2 v_i u_i / (v.u)             isolating node i (all its in- and out-edges)
    edge_rel[i, j] = v_i u_j / (lam v.u) * A_ij blocking the edge i -> j
    """
    vu = (v * u).sum(-1)
    ok = (lam > _EPS) & (vu > 1e-9)
    safe_vu = torch.where(ok, vu, torch.ones_like(vu))
    safe_lam = torch.where(ok, lam, torch.ones_like(lam))
    node = (2.0 * v * u / safe_vu[:, None]).clamp(0.0, 1.0)
    edge = (v[:, :, None] * u[:, None, :] / (safe_lam * safe_vu)[:, None, None] * A).clamp(0.0, 1.0)
    keep = ok.to(A.dtype)
    return node * keep[:, None], edge * keep[:, None, None]


def eigen_drop(A: Tensor, iterations: int, u0: Tensor | None = None, v0: Tensor | None = None):
    """(lam [B], node_rel [B, N], edge_rel [B, N, N]) in one call."""
    lam, u, v = leading_pair(A, iterations, u0, v0)
    node, edge = drops_from_pair(A, lam, u, v)
    return lam, node, edge
