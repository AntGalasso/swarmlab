"""bazt/roles.py -- Deterministic agent identity & role assignment.

Roles are assigned from an isolated CRN stream (`swarmlab.stream_rng("bazt-roles",
f"{topology}:{N}", seed)`) so role assignment never shifts a single draw of the
existing simulator and is reproducible per seed.

* `ingress`   : entry agents (from `hooks.seed_agents`, or designated entry
                candidates via the `bazt-roles` CRN namespace -- never leaking
                the true `world.p0`).
* `relay`     : top degree / bridge nodes.
* `worker`    : the remaining nodes.
* `privileged`: `n_privileged` tool-holding nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set

import swarmlab as sl


@dataclass(frozen=True)
class AgentIdentity:
    node_id: int
    role: str            # "ingress" | "relay" | "worker"
    privileged: bool
    allowed_out_edges: tuple  # tuple of neighbour ids the node may send to


def assign_roles(
    N: int,
    topology: str,
    seed: int,
    adj: List[List[int]],
    ingress_nodes: Optional[Set[int]] = None,
    n_privileged: int = 3,
) -> List[AgentIdentity]:
    """Deterministically assign every node a role + privilege flag.

    `ingress_nodes` (from `hooks.seed_agents`, default empty) are the t=1 entry
    agents.  When `ingress_nodes` is empty the entry candidates are drawn from the
    `bazt-roles` CRN namespace -- never from `world.p0`, so patient zero stays
    hidden from the role layer.
    """
    rng = sl.stream_rng("bazt-roles", f"{topology}:{N}", seed)
    deg = [len(a) for a in adj]
    ingress = set(ingress_nodes or [])
    # Never reveal the true patient zero through role assignment: ingress candidates,
    # when not supplied, are sampled from the CRN, not from p0.
    candidates = [i for i in range(N) if i not in ingress]
    rng.shuffle(candidates)
    n_ingress = max(1, min(len(ingress), N)) if ingress else max(1, N // 6)
    chosen_ingress = set(ingress)
    for i in candidates:
        if len(chosen_ingress) >= n_ingress:
            break
        chosen_ingress.add(i)
    # relay = top-degree nodes not already ingress
    relay_rank = sorted(range(N), key=lambda i: (-deg[i], i))
    relay = set()
    for i in relay_rank:
        if len(relay) >= max(1, N // 4):
            break
        if i not in chosen_ingress:
            relay.add(i)
    privileged = set()
    for i in relay_rank:
        if len(privileged) >= n_privileged:
            break
        privileged.add(i)
    identities: List[AgentIdentity] = []
    for i in range(N):
        if i in chosen_ingress:
            role = "ingress"
        elif i in relay:
            role = "relay"
        else:
            role = "worker"
        identities.append(
            AgentIdentity(
                node_id=i,
                role=role,
                privileged=(i in privileged),
                allowed_out_edges=tuple(sorted(adj[i])),
            )
        )
    return identities
