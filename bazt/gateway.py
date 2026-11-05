"""bazt/gateway.py -- `EnforcementGateway`: the BAZT enforcement decision point.

Plugged into `swarmlab.run_episode` via `EpisodeHooks.gateway_factory` (P3).
Owns node state (`normal` / `restricted` / `quarantined`), directed
`blocked_edges`, the `LineageTracker`, `BudgetLedger`, `AuditLog`, roles and
the `BudgetedOptimizer`.  All budget enforcement happens here; `swarmlab.py`
mirrors `isolate` -> `quar` and `sanitize` -> `san_now` bookkeeping.

STRICT GROUND-TRUTH FIREWALL: `StepView` carries no `infected_now`, `ever` or
`world.p0`.  Ground truth reaches the gateway only via
`inspect_oracle(i, t, truth_callable)`, which spends inspection budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import swarmlab as sl
from bazt.actions import ActionKind, Candidate, score_candidates
from bazt.audit import AuditLog
from bazt.contagion import edge_reach, node_reach
from bazt.config import BaztConfig, cost_of
from bazt.evidence import InspectionState, fused_p
from bazt.ledger import BudgetLedger
from bazt.metrics import compute_all
from bazt.provenance import LineageTracker
from bazt.roles import AgentIdentity
from bazt.optimizer import BudgetedOptimizer, Plan


@dataclass
class GatewayPlan:
    isolate: List[int] = field(default_factory=list)
    sanitize: List[int] = field(default_factory=list)
    restrict: List[int] = field(default_factory=list)
    block_edges: List[Tuple[int, int]] = field(default_factory=list)
    inspect: List[int] = field(default_factory=list)


class EnforcementGateway:
    def __init__(
        self,
        swarmlab_cfg,
        bazt_cfg: BaztConfig,
        world_topology: str,
        world_seed: int,
        adj: List[List[int]],
        roles: List[AgentIdentity],
        ingress_nodes: Set[int],
        entry_seed_nodes: Set[int],
    ) -> None:
        self.cfg = swarmlab_cfg
        self.bcfg = bazt_cfg
        self.topology = world_topology
        self.seed = world_seed
        self.adj = adj
        self.N = swarmlab_cfg.N
        self.T = swarmlab_cfg.T
        self.deg = [len(a) for a in adj]
        self.maxdeg = max(self.deg) if self.deg else 1
        self.roles = {r.node_id: r for r in roles}
        self.ingress_nodes = set(ingress_nodes)
        self.entry_seed_nodes = set(entry_seed_nodes)

        self.ledger = BudgetLedger(bazt_cfg)
        self.lineage = LineageTracker(bazt_cfg, self.N, ingress_nodes=set(ingress_nodes))
        self.evidence = InspectionState(bazt_cfg, self.N, world_topology, world_seed, self.T)
        self.audit = AuditLog()
        self.optimizer = BudgetedOptimizer()

        self.node_state: Dict[int, str] = {i: "normal" for i in range(self.N)}
        self.restrict_until: Dict[int, int] = {}
        self.quarantine_until: Dict[int, int] = {}
        self.blocked_edges: Set[Tuple[int, int]] = set()
        self.quarantined: Set[int] = set()
        self.isolated: Set[int] = set()
        self.stripped_messages: Set[int] = set()
        self.delivered_count = 0
        self._sanitized_this_step: Set[int] = set()
        self._restricted_this_step: Set[int] = set()
        self._p_prime: List[float] = [0.0] * self.N
        self._H: List[float] = [0.0] * self.N
        self._budget_violations = 0

    # ------------------------------------------------------------------ edge API
    def edge_open(self, j: int, i: int, t: int) -> bool:
        """Whether directed edge j -> i delivers at step t."""
        if (j, i) in self.blocked_edges:
            return False
        if j in self.quarantined or i in self.quarantined:
            return False
        return True

    def restricted(self, t: int) -> Set[int]:
        """Nodes currently restricted (soft channel constraint) at step t."""
        return {i for i, st in self.node_state.items() if st == "restricted"}

    # ------------------------------------------------------------------ ground truth (budget-gated)
    def inspect_oracle(self, i: int, t: int, truth_callable: Callable[[int], bool]) -> Optional[float]:
        """Spend one inspection budget on node i.

        Returns conclusive `p'` (0.98 emitting / 0.02 clean) or None if the
        `q_inspect` draw is inconclusive.  This is the *only* channel through
        which ground truth reaches the gateway.
        """
        if not self.ledger.can_afford(ActionKind.INSPECT):
            return None
        try:
            verdict = self.evidence.inspect(i, t, bool(truth_callable(i)))
        except Exception:
            verdict = None
        self.ledger.charge(ActionKind.INSPECT)
        if verdict is not None:
            if verdict == 0.02:  # clean -> clear taint
                self.lineage.inspect_clean(i, t + self.bcfg.inspect_ttl - 1)
        return verdict

    # ------------------------------------------------------------------ main decision
    def decide(self, sv, truth_callable: Callable[[int], bool]) -> GatewayPlan:
        t = sv.t
        active = list(sv.active)
        aset = set(active)
        p_monitor = list(sv.p_monitor)
        roll = list(sv.roll_scores)
        h_t = self.T - t

        # release expired restrictions (reassessment when p' < tau_release)
        self._release_restrictions(t, roll, p_monitor)
        # release expired quarantines
        self._release_quarantines(t)

        # lineage step advance + t=1 ingress seeding
        self.lineage.begin_step(t)
        if t == 1:
            is_seed_flags = [False] * self.N
            for i in self.entry_seed_nodes:
                if 0 <= i < self.N:
                    is_seed_flags[i] = True
            self.lineage.seed_ingress(t, is_seed_flags)

        # fused p' for every active node
        p_prime = [0.0] * self.N
        for i in range(self.N):
            insp = self.evidence.active(i, t)
            if insp is not None:
                p_prime[i] = insp
            else:
                p_prime[i] = fused_p(roll[i], self.lineage.value(i), self.bcfg,
                                     a=self.cfg.a, b=self.cfg.b)
        self._p_prime = p_prime

        # reachability
        H = [0.0] * self.N
        for i in active:
            H[i] = node_reach(self.adj, i, h_t, aset, self.isolated, p_prime,
                              self.cfg, self.bcfg.use_topology)
        self._H = H
        H_edge: Dict[Tuple[int, int], float] = {}
        for i in active:
            for j in self.adj[i]:
                if j in aset and self.edge_open(i, j, t):
                    H_edge[(i, j)] = edge_reach(self.adj, i, j, h_t, aset, self.isolated,
                                                p_prime, self.cfg, self.bcfg.use_topology)

        privileged = {r.node_id for r in self.roles.values() if r.privileged}
        candidates = score_candidates(
            p_prime, H, H_edge, self.deg, self.maxdeg, self.T, t, h_t,
            active, self.adj, self.bcfg, privileged,
        )

        def inspect_oracle(i: int) -> Optional[float]:
            return self.inspect_oracle(i, t, truth_callable)

        plan = self.optimizer.plan(candidates, self.ledger, inspect_oracle, p_prime)

        # apply side effects
        for (i, j) in plan.block_edges:
            self.blocked_edges.add((i, j))
        for i in plan.restrict:
            self.node_state[i] = "restricted"
            self.restrict_until[i] = t + self.bcfg.ttl_restrict - 1
        for i in plan.isolate:
            self.quarantined.add(i)
            self.isolated.add(i)
        if self.bcfg.quarantine_ttl is not None:
            for i in plan.isolate:
                self.quarantine_until[i] = t + self.bcfg.quarantine_ttl - 1

        self._sanitized_this_step = set(plan.sanitize)
        self._restricted_this_step = set(plan.restrict)

        # audit
        pool = self.ledger.capacity if self.ledger.capacity != float("inf") else float(self.ledger.spent)
        for i in plan.inspect:
            self._audit(t, "INSPECT", i, cost_of("INSPECT", self.bcfg), p_prime[i], "inspected")
        for i in plan.sanitize:
            self._audit(t, "SANITIZE", i, cost_of("SANITIZE", self.bcfg), p_prime[i], "sanitized")
        for i in plan.restrict:
            self._audit(t, "RESTRICT", i, cost_of("RESTRICT", self.bcfg), p_prime[i], "restricted")
        for (i, j) in plan.block_edges:
            self._audit(t, "BLOCK_EDGE", i, cost_of("BLOCK_EDGE", self.bcfg), p_prime[i], f"blocked {i}->{j}")
        for i in plan.isolate:
            self._audit(t, "ISOLATE", i, cost_of("ISOLATE", self.bcfg), p_prime[i], "isolated")

        return GatewayPlan(
            isolate=list(plan.isolate),
            sanitize=list(plan.sanitize),
            restrict=list(plan.restrict),
            block_edges=list(plan.block_edges),
            inspect=list(plan.inspect),
        )

    # ------------------------------------------------------------------ delivery
    def on_deliver(self, t: int, delivered_edges: List[Tuple[int, int, bool]]) -> None:
        """Lineage taint propagation + message-throughput accounting after delivery."""
        # mark which nodes were stripped this step (from the sanitized flags on edges)
        stripped_nodes = {dst for (src, dst, san) in delivered_edges if san}
        for i in stripped_nodes:
            self.lineage.mark_stripped(t, i, True)
        # also mark sanitized/restricted nodes that had no outgoing delivered edge
        for i in self._sanitized_this_step | self._restricted_this_step:
            if i not in stripped_nodes:
                self.lineage.mark_stripped(t, i, False)
        self.lineage.on_delivery(t, delivered_edges)
        self.delivered_count += len(delivered_edges)
        self.stripped_messages = {src for (src, _dst, san) in delivered_edges if san}
        self._sanitized_this_step = set()
        self._restricted_this_step = set()

    # ------------------------------------------------------------------ metrics
    def extra_metrics(self, infected_series: List[int], world) -> dict:
        """§15 metrics. Ground-truth evaluation (harm_exposure_auc, ...)
        computed post-episode at t == T, never inside decide()."""
        healthy = {i for i in range(self.N) if i not in self.entry_seed_nodes}
        spend = self.ledger.spent
        # prevented compromises ~ nodes kept out of ever-infected vs entry seeds
        ever_count = infected_series[-1] if infected_series else 0
        prevented = max(0, len(self.entry_seed_nodes) - 0)  # placeholder baseline
        rows = self.audit.rows()
        result = compute_all(
            rows=rows,
            infected_series=infected_series,
            N=self.N,
            T=self.T,
            delivered=self.delivered_count,
            healthy=healthy,
            spend=spend,
            prevented=prevented,
        )
        result.update({
            "bazt_budget_spent": spend,
            "bazt_budget_capacity": (self.ledger.capacity if self.ledger.capacity != float("inf") else -1.0),
            "bazt_budget_violations": float(self._budget_violations),
        })
        return result

    # ------------------------------------------------------------------ internals
    def _release_restrictions(self, t: int, roll: List[float], p_monitor: List[float]) -> None:
        for i in list(self.restrict_until):
            if self.node_state[i] != "restricted":
                continue
            if t > self.restrict_until[i]:
                self.node_state[i] = "normal"
                self.restrict_until.pop(i, None)
                continue
            # reassessment release when p' < tau_release
            pp = self._p_prime[i] if i < len(self._p_prime) else p_monitor[i]
            if pp < self.bcfg.tau_release:
                self.node_state[i] = "normal"
                self.restrict_until.pop(i, None)

    def _release_quarantines(self, t: int) -> None:
        if self.bcfg.quarantine_ttl is None:
            return
        for i in list(self.quarantine_until):
            if t > self.quarantine_until[i]:
                self.quarantined.discard(i)
                self.isolated.discard(i)
                self.node_state[i] = "normal"
                self.quarantine_until.pop(i, None)

    def _audit(
        self, t, action, target, cost, p_prime, reason: str,
    ) -> None:
        remaining = self.ledger.capacity - self.ledger.spent if self.ledger.capacity != float("inf") else float("inf")
        spent = self.ledger.spent
        H = self._H[target] if target < len(self._H) else 0.0
        taint = self.lineage.value(target)
        net = p_prime * H  # benefit proxy
        density = net / cost if cost else 0.0
        self.audit.record(
            t=t, action=action, target=target, cost=cost,
            pool=self.ledger.capacity if self.ledger.capacity != float("inf") else float("inf"),
            spent=spent, remaining=remaining,
            p_prime=float(p_prime), s_mon=0.0, taint=taint, H=H,
            net=net, density=density,
            theta=self.ledger.theta(), reason=reason,
        )
