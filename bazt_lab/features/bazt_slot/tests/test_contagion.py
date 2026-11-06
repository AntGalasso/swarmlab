"""Contagion maths: truncated Katz by matvec recursion, NetShield / NetMelt eigen-drop by batched power iteration.

Every quantity is checked against an explicit numpy sum or against torch.linalg.eigvals on tiny graphs.
"""
import numpy as np
import pytest
import torch

from bazt_lab.features.bazt_slot.private import contagion as ct
from bazt_lab.features.bazt_slot.tests.testkit import barabasi_albert, small_world, spectral_radius_np, star_hub

F64 = torch.float64

# directed, with a cycle (0 -> 1 -> 2 -> 0) and a tail, so walks differ from first-reach paths
A5 = np.array([[0, 1, 0, 1, 0],
               [0, 0, 1, 0, 0],
               [1, 0, 0, 0, 1],
               [0, 0, 0, 0, 1],
               [0, 1, 0, 0, 0]], dtype=float)
S5 = np.array([0.9, 0.5, 1.0, 0.2, 0.7])


def explicit_katz(A, s, decay, order):
    """sum_{k=1}^{order} decay^(k-1) A^k s"""
    out = np.zeros_like(s)
    for k in range(1, order + 1):
        out = out + decay ** (k - 1) * np.linalg.matrix_power(A, k) @ s
    return out


def t3(a):
    return torch.as_tensor(a, dtype=F64)[None]


@pytest.mark.parametrize("order", [1, 2, 3, 5])
def test_katz_matches_explicit_sum(order):
    H, _, _ = ct.katz(t3(A5), t3(S5), 0.6, order)
    np.testing.assert_allclose(H[0].numpy(), explicit_katz(A5, S5, 0.6, order), rtol=1e-12, atol=1e-12)


def test_katz_previous_order_is_one_hop_shorter():
    _, _, H_prev = ct.katz(t3(A5), t3(S5), 0.6, 4)
    np.testing.assert_allclose(H_prev[0].numpy(), explicit_katz(A5, S5, 0.6, 3), rtol=1e-12, atol=1e-12)
    _, _, H_prev1 = ct.katz(t3(A5), t3(S5), 0.6, 1)
    assert torch.equal(H_prev1, torch.zeros_like(H_prev1))


def test_katz_edge_reach_decomposes_the_node_reach():
    """H_i = sum_j A_ij R_j: the reach of a node is the sum of the values of its out-edges."""
    H, R, _ = ct.katz(t3(A5), t3(S5), 0.6, 4)
    np.testing.assert_allclose(H[0].numpy(), A5 @ R[0].numpy(), rtol=1e-12)


def test_katz_order_zero_is_zero():
    H, R, Hp = ct.katz(t3(A5), t3(S5), 0.6, 0)
    assert not H.any() and not R.any() and not Hp.any()


def test_katz_batch_does_not_mix_episodes():
    gen = torch.Generator().manual_seed(3)
    A = (torch.rand(4, 6, 6, generator=gen, dtype=F64) < 0.4).to(F64)
    A = A * (1 - torch.eye(6, dtype=F64))
    s = torch.rand(4, 6, generator=gen, dtype=F64)
    H, R, Hp = ct.katz(A, s, 0.7, 3)
    for b in range(4):
        h1, r1, p1 = ct.katz(A[b:b + 1], s[b:b + 1], 0.7, 3)
        assert torch.equal(H[b], h1[0]) and torch.equal(R[b], r1[0]) and torch.equal(Hp[b], p1[0])


def max_real_eig(A):
    return torch.linalg.eigvals(A).real.amax(-1)


def random_strongly_connected(n, seed):
    gen = torch.Generator().manual_seed(seed)
    A = (torch.rand(n, n, generator=gen, dtype=F64) < 0.35).to(F64)
    A.fill_diagonal_(0)
    for i in range(n):                                 # a Hamiltonian cycle makes it strongly connected
        A[i, (i + 1) % n] = 1.0
    return A


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_leading_eigenvalue_matches_eigvals(seed):
    A = random_strongly_connected(8, seed)[None]
    lam, u, v = ct.leading_pair(A, iterations=2000)
    assert lam[0].item() == pytest.approx(max_real_eig(A[0]).item(), rel=1e-9)
    torch.testing.assert_close(A[0] @ u[0], lam[0] * u[0], atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(v[0] @ A[0], lam[0] * v[0], atol=1e-8, rtol=1e-8)


def test_leading_pair_converges_on_a_bipartite_path():
    """A path graph has eigenvalues +-lambda; plain power iteration oscillates, the shifted iteration must not."""
    n = 6
    A = torch.zeros(1, n, n, dtype=F64)
    for i in range(n - 1):
        A[0, i, i + 1] = A[0, i + 1, i] = 1.0
    lam, _, _ = ct.leading_pair(A, iterations=2000)
    assert lam[0].item() == pytest.approx(2 * np.cos(np.pi / (n + 1)), rel=1e-9)


def removal_drop(A, i):
    """Exact relative eigen-drop of isolating node i: (lambda(A) - lambda(A without i)) / lambda(A)."""
    B = A.clone()
    B[i, :] = 0
    B[:, i] = 0
    return ((max_real_eig(A) - max_real_eig(B)) / max_real_eig(A)).item()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_node_drop_is_the_first_order_eigen_drop(seed):
    """d lambda / d eps for removing a fraction eps of node i's edges = 2 lambda v_i u_i / (v.u)."""
    A = random_strongly_connected(7, seed)
    _, node_rel, _ = ct.eigen_drop(A[None], iterations=3000)
    eps = 1e-6
    for i in range(7):
        B = A.clone()
        B[i, :] *= 1 - eps
        B[:, i] *= 1 - eps
        fd = ((max_real_eig(A) - max_real_eig(B)) / (eps * max_real_eig(A))).item()
        assert min(fd, 1.0) == pytest.approx(node_rel[0, i].item(), rel=1e-4, abs=1e-6)


def test_edge_drop_is_the_first_order_link_blocking_value():
    A = random_strongly_connected(6, 5)
    _, _, edge_rel = ct.eigen_drop(A[None], iterations=3000)
    eps = 1e-6
    for i, j in [(0, 1), (2, 3), (5, 0)]:
        B = A.clone()
        B[i, j] *= 1 - eps
        fd = ((max_real_eig(A) - max_real_eig(B)) / (eps * max_real_eig(A))).item()
        assert fd == pytest.approx(edge_rel[0, i, j].item(), rel=1e-4, abs=1e-7)
    assert edge_rel[0][A == 0].abs().max().item() == 0.0       # only existing edges have a value


def test_symmetric_star_reproduces_netshield():
    """Undirected star with 5 leaves: u_hub^2 = 1/2, u_leaf^2 = 1/10, so Sv(i)/lambda = 2 u_i^2."""
    n = 6
    A = torch.zeros(1, n, n, dtype=F64)
    A[0, 0, 1:] = 1.0
    A[0, 1:, 0] = 1.0
    lam, node_rel, _ = ct.eigen_drop(A, iterations=3000)
    assert lam[0].item() == pytest.approx(5 ** 0.5, rel=1e-9)
    assert node_rel[0, 0].item() == pytest.approx(1.0, rel=1e-6)
    assert node_rel[0, 1:].tolist() == pytest.approx([0.2] * 5, rel=1e-6)


def test_node_drop_ranks_like_the_exact_drop():
    A = random_strongly_connected(8, 11)
    _, node_rel, _ = ct.eigen_drop(A[None], iterations=3000)
    exact = torch.tensor([removal_drop(A, i) for i in range(8)], dtype=F64)
    assert node_rel[0].argmax().item() == exact.argmax().item()


def test_graph_without_edges_has_no_drop():
    lam, node_rel, edge_rel = ct.eigen_drop(torch.zeros(2, 4, 4, dtype=F64), iterations=20)
    assert not lam.any() and not node_rel.any() and not edge_rel.any()


def test_warm_start_keeps_the_converged_pair():
    A = random_strongly_connected(7, 4)[None]
    lam, u, v = ct.leading_pair(A, iterations=3000)
    lam2, u2, v2 = ct.leading_pair(A, iterations=1, u0=u, v0=v)
    assert lam2[0].item() == pytest.approx(lam[0].item(), rel=1e-12)
    torch.testing.assert_close(u2, u, atol=1e-10, rtol=0)


def test_eigen_batch_does_not_mix_episodes():
    A = torch.stack([random_strongly_connected(6, s) for s in range(3)])
    lam, node, edge = ct.eigen_drop(A, iterations=50)
    for b in range(3):
        l1, n1, e1 = ct.eigen_drop(A[b:b + 1], iterations=50)
        assert torch.equal(lam[b], l1[0]) and torch.equal(node[b], n1[0]) and torch.equal(edge[b], e1[0])


# ------------------------------------------------------------------ graphs without a sustained cascade (acyclic)
def acyclic_chain(n=4):
    A = torch.zeros(1, n, n, dtype=F64)
    for i in range(n - 1):
        A[0, i, i + 1] = 1.0
    return A


@pytest.mark.parametrize("iterations", [5, 30, 200])
def test_acyclic_graph_has_no_spectral_radius_and_no_drop(iterations):
    """A DAG is nilpotent: lambda = 0 and no node or edge carries a NetShield value. Power iteration on A + I
    converges only algebraically on its Jordan blocks, so a Rayleigh quotient alone reports a small positive lambda."""
    lam, node, edge = ct.eigen_drop(acyclic_chain(), iterations=iterations)
    assert lam.item() == 0.0 and not node.any() and not edge.any()


def test_out_star_has_no_spectral_radius():
    A = torch.zeros(1, 5, 5, dtype=F64)
    A[0, 0, 1:] = 1.0
    lam, node, edge = ct.eigen_drop(A, iterations=30)
    assert lam.item() == 0.0 and not node.any() and not edge.any()


def test_acyclic_head_into_a_cycle_values_only_the_cycle():
    """0 -> 1 -> 2 <-> 3: lambda = 1 is carried by the 2-cycle; the acyclic head has no value."""
    A = torch.zeros(1, 4, 4, dtype=F64)
    A[0, 0, 1] = A[0, 1, 2] = A[0, 2, 3] = A[0, 3, 2] = 1.0
    lam, node, edge = ct.eigen_drop(A, iterations=60)
    assert lam[0].item() == pytest.approx(1.0, rel=1e-6)
    assert node[0, :2].abs().max().item() < 1e-6
    assert node[0, 2:].tolist() == pytest.approx([1.0, 1.0], rel=1e-6)


def test_batch_mixing_cyclic_and_acyclic_episodes_keeps_them_apart():
    cyc = random_strongly_connected(4, 2)[None]
    A = torch.cat([acyclic_chain(), cyc])
    lam, node, edge = ct.eigen_drop(A, iterations=200)
    assert lam[0].item() == 0.0 and not node[0].any()
    assert lam[1].item() == pytest.approx(max_real_eig(cyc[0]).item(), rel=1e-6) and node[1].any()


# ------------------------------------------------- D115: the Katz decay is bazt.katz_decay / spectral radius of the graph
GRID = {"star_hub": star_hub(12), "small_world": small_world(30), "barabasi_albert": barabasi_albert(100)}


def boolean(a):
    return torch.as_tensor(np.asarray(a) > 0)


@pytest.mark.parametrize("name", sorted(GRID))
def test_the_spectral_radius_of_a_grid_graph_matches_numpy(name):
    rho = ct.spectral_radius(boolean(GRID[name])[None], iterations=300)
    assert rho.dtype == F64 and rho[0].item() == pytest.approx(spectral_radius_np(GRID[name]), rel=1e-9)


@pytest.mark.parametrize("name", sorted(GRID))
def test_the_katz_decay_is_the_fraction_of_one_over_the_spectral_radius(name):
    d = ct.relative_decay(boolean(GRID[name])[None], 0.9, iterations=300)
    assert d.dtype == F64 and d[0].item() == pytest.approx(0.9 / spectral_radius_np(GRID[name]), rel=1e-9)


@pytest.mark.parametrize("name", sorted(GRID))
@pytest.mark.parametrize("order", [1, 2, 4, 6])
def test_katz_with_the_relative_decay_matches_the_explicit_sum(name, order):
    A = GRID[name]
    s = np.random.default_rng(order).uniform(0.0, 1.0, len(A))
    d = ct.relative_decay(boolean(A)[None], 0.6, iterations=300)
    H, _, _ = ct.katz(t3(A), t3(s), d, order)
    np.testing.assert_allclose(H[0].numpy(), explicit_katz(A, s, 0.6 / spectral_radius_np(A), order), rtol=1e-9)


@pytest.mark.parametrize("name", sorted(GRID))
def test_katz_with_the_relative_decay_converges_at_09(name):
    """sum_k d^(k-1) A^k s tends to A (I - d A)^-1 s when d rho < 1. At katz_decay = 0.9 the truncated reach grows with
    the horizon towards that limit and never past it (with 0.9 as an absolute decay it would grow like (0.9 rho)^K)."""
    A = GRID[name]
    s = np.random.default_rng(1).uniform(0.2, 1.0, len(A))
    d_np = 0.9 / spectral_radius_np(A)
    limit = A @ np.linalg.solve(np.eye(len(A)) - d_np * A, s)
    d = ct.relative_decay(boolean(A)[None], 0.9, iterations=300)
    short, mid, long_ = (ct.katz(t3(A), t3(s), d, K)[0][0].numpy() for K in (35, 100, 400))
    assert (short <= mid).all() and (mid <= limit * (1 + 1e-9)).all()
    np.testing.assert_allclose(long_, limit, rtol=1e-9)


def test_a_batch_of_different_graphs_takes_each_episodes_own_radius():
    # N = 12 (the star_hub size of the grid): CPU (MKL) bmm is bitwise batch-independent there; from about N = 30 it
    # differs in the last ulp between batch sizes, for every bmm of the lane (see CONTRACT.md, Compute)
    graphs = [star_hub(12), small_world(12, seed=2), barabasi_albert(12, m=2, seed=3)]
    A = torch.as_tensor(np.stack(graphs))
    s = torch.as_tensor(np.random.default_rng(4).uniform(0.0, 1.0, (3, 12)))
    d = ct.relative_decay(A > 0, 0.9, iterations=300)
    H, R, Hp = ct.katz(A, s, d, 6)
    for b, g in enumerate(graphs):
        assert d[b].item() == pytest.approx(0.9 / spectral_radius_np(g), rel=1e-9)
        np.testing.assert_allclose(H[b].numpy(), explicit_katz(g, s[b].numpy(), 0.9 / spectral_radius_np(g), 6),
                                   rtol=1e-9)
        d1 = ct.relative_decay(A[b:b + 1] > 0, 0.9, iterations=300)
        h1, r1, p1 = ct.katz(A[b:b + 1], s[b:b + 1], d1, 6)
        assert torch.equal(d[b], d1[0]) and torch.equal(H[b], h1[0]) and torch.equal(R[b], r1[0])
        assert torch.equal(Hp[b], p1[0])


def test_a_graph_without_a_cycle_takes_the_out_degree_bound_and_the_decay_stays_finite():
    """No eigenpair (lambda = 0, a DAG or an empty graph): the out-degree bound max_i sum_j A_ij >= rho stands in for
    rho, never below 1, so the decay is finite and decay x rho <= katz_decay still holds."""
    empty = torch.zeros(1, 5, 5, dtype=torch.bool)
    out_star = torch.zeros(1, 5, 5, dtype=torch.bool)
    out_star[0, 0, 1:] = True
    chain = acyclic_chain(5) > 0
    adj = torch.cat([empty, out_star, chain])
    d = ct.relative_decay(adj, 0.6, iterations=30)
    assert d.tolist() == pytest.approx([0.6, 0.6 / 4, 0.6], rel=1e-12)
    H, R, Hp = ct.katz(adj.to(F64), torch.ones(3, 5, dtype=F64), d, 5)
    assert H.isfinite().all() and R.isfinite().all() and Hp.isfinite().all() and not H[0].any()
    np.testing.assert_allclose(H[2].numpy(), explicit_katz(acyclic_chain(5)[0].numpy(), np.ones(5), 0.6, 5), rtol=1e-12)


def test_float32_katz_with_the_relative_decay_stays_float32_and_close_to_float64():
    A, s = GRID["barabasi_albert"], np.random.default_rng(5).uniform(0.0, 1.0, 100)
    d = ct.relative_decay(boolean(A)[None], 0.9, iterations=300)
    H64, _, _ = ct.katz(t3(A), t3(s), d, 35)
    H32, R32, _ = ct.katz(t3(A).float(), t3(s).float(), d, 35)
    assert H32.dtype == R32.dtype == torch.float32
    np.testing.assert_allclose(H32[0].double().numpy(), H64[0].numpy(), rtol=1e-4)
