"""bazt/contagion.py -- Active-graph reachability reusing `swarmlab.reach_H`.

`H_i` over active non-isolated nodes and open directed edges within
`min(h_t, d_max)` hops (`1.0` if `use_topology` is False).

`H_edge(i, j) = (1.0 - p'_j) * (1.0 + H_j_without_i)` on open directed edge
`i -> j`.

Horizon rule: at the final step (`h_t = T - t == 0`), future contagion reach is
`0.0`, so `BudgetedOptimizer` spends `0` budget at step `T`.
"""

from __future__ import annotations

from typing import List, Set

import swarmlab as sl


def node_reach(
    adj: List[List[int]],
    i: int,
    h_t: int,
    active: Set[int],
    isolated: Set[int],
    p_prime: List[float],
    sl_cfg,  # swarmlab.Config (dmax, gamma)
    use_topology: bool,
) -> float:
    """Horizon-bounded, path-decayed reachability from `i` over active,
    non-isolated nodes. `1.0` when topology is disabled; `0.0` at h_t == 0."""
    if h_t <= 0:
        return 0.0
    if not use_topology:
        return 1.0
    eff_active = {n for n in active if n not in isolated}
    return sl.reach_H(adj, i, min(h_t, sl_cfg.dmax), sl_cfg, eff_active, p_prime, use_p=True)


def edge_reach(
    adj: List[List[int]],
    i: int,
    j: int,
    h_t: int,
    active: Set[int],
    isolated: Set[int],
    p_prime: List[float],
    sl_cfg,  # swarmlab.Config (dmax, gamma)
    use_topology: bool,
) -> float:
    """`H_edge(i, j) = (1.0 - p'_j) * (1.0 + H_j_without_i)` on open edge `i -> j`.

    `H_j_without_i` = reachability from `j` over active, non-isolated nodes with
    `i` removed (marginal contagion through the directed edge `i -> j`).
    """
    if h_t <= 0:
        return 0.0
    if not use_topology:
        # topology disabled: every open edge carries unit marginal reach
        return (1.0 - p_prime[j]) * 1.0
    eff_active = {n for n in active if n not in isolated and n != i}
    H_j_without_i = sl.reach_H(adj, j, min(h_t, sl_cfg.dmax), sl_cfg, eff_active, p_prime, use_p=True)
    return (1.0 - p_prime[j]) * (1.0 + H_j_without_i)
