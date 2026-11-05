"""Deterministic role assignment: ingress / relay / worker, plus tool-holding (privileged) nodes.

Roles are a deployment fact (which agents face the outside world, which hold tools). They are drawn from
their own CRN namespace and never from the simulator's hidden state, so a gateway cannot learn where the
attack really started from its own role map.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Sequence, Tuple

import swarmlab as sl
from bazt.config import BaztConfig

INGRESS, RELAY, WORKER = "ingress", "relay", "worker"


@dataclass(frozen=True)
class AgentIdentity:
    node_id: int
    role: str
    privileged: bool
    allowed_out_edges: Tuple[int, ...]


def articulation_points(adj: Sequence[Sequence[int]]) -> FrozenSet[int]:
    """Nodes whose removal disconnects the graph (iterative Tarjan, O(N + E))."""
    n = len(adj)
    disc, low, parent = [-1] * n, [0] * n, [-1] * n
    cut, timer = set(), 0
    for root in range(n):
        if disc[root] != -1:
            continue
        disc[root] = low[root] = timer
        timer += 1
        stack, children = [(root, iter(adj[root]))], 0
        while stack:
            u, it = stack[-1]
            advanced = False
            for v in it:
                if disc[v] == -1:
                    parent[v] = u
                    disc[v] = low[v] = timer
                    timer += 1
                    if u == root:
                        children += 1
                    stack.append((v, iter(adj[v])))
                    advanced = True
                    break
                if v != parent[u]:
                    low[u] = min(low[u], disc[v])
            if advanced:
                continue
            stack.pop()
            if stack:
                p = stack[-1][0]
                low[p] = min(low[p], low[u])
                if p != root and low[u] >= disc[p]:
                    cut.add(p)
        if children > 1:
            cut.add(root)
    return frozenset(cut)


def assign_roles(topology: str, seed: int, adj: Sequence[Sequence[int]], cfg: BaztConfig,
                 seed_agents: frozenset = frozenset()) -> Dict[int, AgentIdentity]:
    n = len(adj)
    key = f"{topology}:{n}"
    if seed_agents:                                    # known entry points (fake-web corpus readers)
        ingress = frozenset(int(i) for i in seed_agents)
    else:                                              # designated external-facing candidates
        k = max(1, min(cfg.n_ingress, n))
        ingress = frozenset(int(i) for i in sl.stream_rng("bazt-roles", key, seed, "ingress").permutation(n)[:k])
    cut = articulation_points(adj)
    n_relay = max(1, round(0.25 * n))
    ranked = sorted((i for i in range(n) if i not in ingress), key=lambda i: (i not in cut, -len(adj[i]), i))
    relay = frozenset(ranked[:n_relay])
    k_priv = max(0, min(cfg.n_privileged, n))
    privileged = frozenset(int(i) for i in sl.stream_rng("bazt-roles", key, seed, "privileged").permutation(n)[:k_priv])
    out: Dict[int, AgentIdentity] = {}
    for i in range(n):
        role = INGRESS if i in ingress else RELAY if i in relay else WORKER
        nbrs: List[int] = list(adj[i])
        if role == INGRESS:                            # untrusted-facing nodes may only talk to relays
            via_relay = [j for j in nbrs if j in relay]
            nbrs = via_relay or nbrs
        out[i] = AgentIdentity(i, role, i in privileged, tuple(nbrs))
    return out
