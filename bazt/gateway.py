"""EnforcementGateway: the object `swarmlab.run_episode` talks to (P3-P7 hooks).

Information barrier: the gateway sees a `StepView` (monitor output) and lineage built from delivery events.
Ground truth arrives only through `inspect_oracle`, i.e. only when an inspection is paid for and conclusive.
"""
from __future__ import annotations

from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import swarmlab as sl
from bazt.actions import ActionKind
from bazt.audit import AuditLog
from bazt.config import BaztConfig
from bazt.evidence import InspectionState, fused_probability
from bazt.ledger import BudgetLedger
from bazt.metrics import EpisodeLog, episode_metrics
from bazt.optimizer import BudgetedOptimizer, Plan, Situation
from bazt.provenance import LineageTracker
from bazt.roles import INGRESS, assign_roles


def _none_if_inf(x: float) -> Optional[float]:
    return None if x == float("inf") else x


class EnforcementGateway:
    def __init__(self, policy: str, cfg: BaztConfig, sl_cfg: sl.Config, topology: str, seed: int,
                 base_adj: Sequence[Sequence[int]], recv: Sequence[Sequence[int]],
                 seed_agents: FrozenSet[int] = frozenset()):
        self.policy, self.cfg, self.sl_cfg = policy, cfg, sl_cfg
        self.n, self.T = sl_cfg.N, sl_cfg.T
        self.recv = recv
        self.identities = assign_roles(topology, seed, base_adj, cfg, seed_agents)
        self.privileged = frozenset(i for i, a in self.identities.items() if a.privileged)
        ingress = frozenset(i for i, a in self.identities.items() if a.role == INGRESS)
        self.lineage = LineageTracker(self.n, ingress, recv, cfg, sl_cfg.p_strip)
        # dual_pct caps are fractions of the episode's edge deliveries, so the ledger needs the denominator.
        self.e_dir = sum(len(r) for r in recv)
        self.ledger = BudgetLedger(cfg, total_deliveries=(self.T - 1) * self.e_dir)
        # Identity tier (arm_A): the authenticated directed allowlist is the BASE graph. A receive edge that the
        # base graph does not contain (an injected / rewritten incoming edge) fails authentication at zero cost.
        self._allow: Set[Tuple[int, int]] = {(u, v) for v in range(self.n) for u in recv[v] if v in base_adj[u]}
        self._identity_blocks: Set[Tuple[int, int, int]] = set()
        self.fallback_active = False                 # True from the step a dual_pct pool ran dry
        self.fallback_step = None
        self.budget_exhausted = []                   # [{"pool", "step", "cap"}]
        self.inspection = InspectionState(cfg.inspect_ttl)
        self.optimizer = BudgetedOptimizer(cfg, sl_cfg)
        self.audit = AuditLog()
        # CRN: one pre-drawn matrix, so WHICH nodes are inspected never shifts anyone's draw.
        self._u_inspect = sl.stream_rng("bazt-inspect", f"{topology}:{self.n}", seed).random((self.T, self.n))
        self._out_adj: List[List[int]] = [[] for _ in range(self.n)]       # open directed send lists
        for v in range(self.n):
            for u in recv[v]:
                self._out_adj[u].append(v)
        self._blocked: Set[Tuple[int, int]] = set()                        # (sender, receiver)
        self._restricted: Dict[int, int] = {}                              # node -> last restricted step
        self._quarantined: Dict[int, Optional[int]] = {}                   # node -> release step (None = never)
        self._log = EpisodeLog(self.n, self.T, sum(len(r) for r in recv))
        self._block_events: List[Tuple[int, int, int]] = []      # (t, sender, receiver) denied deliveries
        self.telemetry: dict = {}                                # experiment-only metrics (see telemetry_metrics)

    # ------------------------------------------------------------------------- run_episode hooks
    def edge_open(self, sender: int, receiver: int, t: int = 0) -> bool:
        """A delivery needs (a) an authenticated directed edge and (b) no active block.

        The identity check is free and never inspects content, so it is exactly arm_A's rule. When the base
        graph IS the receive graph (every legacy BAZT run) it is a no-op, which keeps those runs bit-identical.
        """
        if (sender, receiver) not in self._allow:
            self._identity_blocks.add((int(t), sender, receiver))
            return False
        return (sender, receiver) not in self._blocked

    def pop_released(self, t: int) -> List[int]:
        due = sorted(i for i, until in self._quarantined.items() if until is not None and until <= t)
        for i in due:
            del self._quarantined[i]
            self.audit.append(t=t, action="RELEASE", target=i, cost=0, reason="quarantine_ttl")
        return due

    def restricted(self, t: int) -> FrozenSet[int]:
        return frozenset(i for i, last in self._restricted.items() if last >= t)

    def inspect_oracle(self, node: int, t: int, truth: Callable[[int], bool]) -> Optional[bool]:
        """The ONLY door to ground truth: conclusive with probability q_inspect (pre-drawn), else None."""
        if self._u_inspect[t - 1, node] < self.cfg.q_inspect:
            return bool(truth(node))
        return None

    def decide(self, view: sl.StepView, truth: Callable[[int], bool]) -> Plan:
        t, aset = view.t, set(view.active)
        if self.fallback_active:
            # Budget exhausted: only the arm_A identity/allowlist rules survive for the rest of the episode.
            return Plan()
        self._restricted = {i: last for i, last in self._restricted.items() if last >= t}
        tau = self.lineage.advance(t, lambda j, i: (j, i) not in self._blocked, aset)
        for i in view.active:
            if self.inspection.clean_active(i, t):
                self.lineage.clear(i)
        p_prime, s_mon = [0.0] * self.n, [0.0] * self.n
        probe = self.cfg.probe_score_hook
        for i in view.active:
            roll = view.roll_scores[i]
            if probe is not None:
                r = probe(i, t, view)
                roll = roll if r is None else float(r)
            s_mon[i] = roll
            override = self.inspection.override(i, t)
            p_prime[i] = override if override is not None else fused_probability(self.sl_cfg, self.cfg, roll, tau[i])
        for i in [k for k in self._restricted if k in aset and p_prime[k] < self.cfg.tau_release]:
            del self._restricted[i]                                       # reassessment: evidence faded
            self.audit.append(t=t, action="RELEASE", target=i, cost=0, p_prime=p_prime[i], s_mon=s_mon[i],
                              taint=tau[i], reason="p_prime<tau_release")
        sit = Situation(t=t, T=self.T, active=view.active, p_prime=p_prime, deg=view.deg, maxdeg=view.maxdeg,
                        privileged=self.privileged, out_adj=self._out_adj,
                        restricted=frozenset(self._restricted),
                        verdict_active=self.inspection.active_nodes(t),
                        inspect=lambda i: self._inspect(i, t, truth))
        plan = self.optimizer.plan(sit, self.ledger)
        self._apply(t, plan, s_mon, tau)
        for ev in self.ledger.mark_exhausted(t):
            self.budget_exhausted.append(ev)
            self.fallback_active = True
            self.fallback_step = t
        return plan

    def action_steps(self) -> List[int]:
        """Steps at which this gateway spent a delivery-changing action (matched-timing reference)."""
        return sorted({t for t, _k, _n in self._log.actions})

    def on_deliver(self, t: int, outgoing: Dict[int, bool]) -> None:
        """`outgoing`: every sender whose message left the gateway at step t -> was it successfully stripped."""
        self.lineage.record_outgoing(outgoing)
        self._log.delivered[t] = [(j, i, stripped) for j, stripped in outgoing.items()
                                  for i in self._out_adj[j] if i not in self._quarantined]

    def extra_metrics(self, infected_series: Sequence[Sequence[bool]], world=None) -> dict:  # noqa: ARG002
        """EXACTLY `BAZT_METRICS` (the section-15 contract). Experiment telemetry travels separately."""
        m = episode_metrics(self._log, infected_series, self.ledger, self.audit)
        self.telemetry = self.telemetry_metrics(infected_series)
        return m

    def telemetry_metrics(self, infected_series: Sequence[Sequence[bool]]) -> dict:
        """Dual-pct / calibration telemetry for the sprint orchestrator (not part of BAZT_METRICS)."""
        T, ever = self.T, [False] * self.n
        ever_at: List[List[bool]] = []
        for row in infected_series:
            ever = [e or bool(x) for e, x in zip(ever, row)]
            ever_at.append(ever)
        benign_den = sum(1 for t, s, _r in self._block_events if t >= 1 and not ever_at[t - 1][s])
        benign_del = benign_den + sum(1 for t in range(1, T) for s, _r, _st in self._log.delivered.get(t, ())
                                     if not ever_at[t - 1][s])
        return {"benign_deliveries": benign_del, "benign_denied_deliveries": benign_den,
                "benign_delivery_deny_rate": (benign_den / benign_del) if benign_del else 0.0,
                "budget_exhausted": int(bool(self.budget_exhausted)),
                "budget_exhausted_first_step": self.fallback_step if self.fallback_step is not None else -1,
                "budget_exhausted_pools": ",".join(sorted({e["pool"] for e in self.budget_exhausted})),
                "audit_spend": self.ledger.audit_spend(),
                "intervention_spend": self.ledger.intervention_spend(),
                "identity_blocks": len(self._identity_blocks),
                "fallback_to_arm_a": int(self.fallback_active)}

    # ------------------------------------------------------------------------------ internals
    def _inspect(self, node: int, t: int, truth: Callable[[int], bool]) -> Optional[float]:
        verdict = self.inspect_oracle(node, t, truth)
        if verdict is None:
            return None
        self._log.inspections_conclusive += 1
        if not verdict:
            self.lineage.clear(node)
        return self.inspection.record(node, t, verdict)

    def _apply(self, t: int, plan: Plan, s_mon: List[float], tau: List[float]) -> None:
        for d in plan.decisions:
            c = d.cand
            self.audit.append(
                t=t, action=c.kind.name, target=c.node if c.peer < 0 else f"{c.node}->{c.peer}",
                cost=d.charge.cost, pool=d.charge.pool, spent=d.charge.spent,
                remaining=_none_if_inf(d.charge.remaining), p_prime=c.p_prime, s_mon=s_mon[c.node],
                taint=tau[c.node], H=c.H, net=c.net, density=c.density, theta=d.theta, reason=d.reason)
            if c.kind != ActionKind.INSPECT:
                self._log.actions.append((t, c.kind, c.node))
        ttl = self.cfg.quarantine_ttl
        for i in plan.isolate:
            self._quarantined[i] = None if ttl is None else t + ttl + 1   # inactive for ttl full steps after t
            self._restricted.pop(i, None)
        for i in plan.restrict:
            self._restricted[i] = t + self.cfg.ttl_restrict - 1
        for i, j in plan.block:
            self._blocked.add((i, j))
            self._block_events.append((int(t), int(i), int(j)))
            self._out_adj[i].remove(j)
        self._log.degraded[t] = set(self._restricted) | set(plan.sanitize) | set(self._quarantined)
