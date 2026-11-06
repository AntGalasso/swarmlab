"""Conventional Zero-Trust-Architecture ladder (arm_A / arm_B / arm_C) + reference baselines.

Every defender in this module speaks the SAME gateway protocol ``swarmlab.run_episode`` already uses for
``bazt.EnforcementGateway``:

    edge_open(sender, receiver, t) -> bool          delivery gate (identity / allowlist / blocks)
    pop_released(t) -> [nodes]                      TTL releases
    restricted(t) -> frozenset                      nodes whose message is stripped this step
    decide(StepView, truth) -> bazt.optimizer.Plan  budgeted decision (truth only via paid inspection)
    on_deliver(t, {sender: stripped})               delivery telemetry
    extra_metrics(infected_series, world) -> dict    post-episode metrics (ground truth allowed here)
    .audit, .ledger, .budget_exhausted               the same accounting objects as BAZT

SENDER-MEMORY PRESERVATION (shared invariant, tested)
    When a delivery on ``i -> j`` is denied, sanitized or delayed, sender ``i``'s own internal state and
    memory are UNCHANGED. Structurally, every action lives on the receiver-facing gate: ``run_episode``
    rebuilds ``out_prev`` from the agent's own raw output and the gateway can only blank or rewrite what a
    receiver is allowed to consume. No network in this module ever writes into an agent, into ``est``, into
    an inbox or into ``AgentCtx``. ``tests/test_experiments_suite.py`` pins the observable consequence: with
    every outgoing edge of node ``h`` blocked, ``h``'s estimate trajectory is bit-identical to the run with
    no blocks.

Arms
    arm_A   Identity-Tier ZTA (BeyondCorp / NIST SP 800-207A). Authenticates the sender identity and the
            directed edge allowlist ``(i, j) in E`` at ZERO budget cost (never capped). Stops unauthenticated
            ``aitm`` channel rewrites. Never inspects message content, never blocks a legitimate edge.
    arm_B   Stateless Agent Gateway: arm_A + 1 audit unit per unique broadcast to query the monitor/judge,
            denies delivery (1 intervention unit per blocked edge) above the clean-calibrated ``tau_alpha``,
            and enforces ``rate_limit_per_identity``. No state is carried across hops.
    arm_C   Bradatsch contextual score-based ZTA: arm_A + additive Trust Score ``TS`` (auth, historical
            recipient usage, access rate within range, active presence, message behaviour, and an unbounded
            ``[min, max] ``trust_history``) compared against a Risk Level ``RL`` (flagged-agent count +
            binned monitor score + calibrated offset). Grants iff ``TS > RL``.

Reference baselines (same boundary, same ledger)
    gsafeguard_standin  detect-and-cut: on detection, cut EVERY outgoing edge of the node.
    bra_audit_greedy    audit priority = rounds_since_last_audit * descendant_count.
    random_dual         random audits / blocks inside the same dual budget.
    oracle_dual         non-deployable upper bound: knows the truth, isolates infected nodes.
    mt_<legacy>         matched-timing wrapper: the legacy rule, executed only on the reference arm's steps.

All mock numbers are pipeline checks, not findings.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

import swarmlab as sl
from bazt.actions import ActionKind
from bazt.audit import AuditLog
from bazt.config import BaztConfig
from bazt.ledger import BudgetLedger
from bazt.metrics import BAZT_METRICS, episode_metrics
from bazt.optimizer import Plan

ZTA_ARMS = ("arm_A", "arm_B", "arm_C")
BASELINE_ARMS = ("gsafeguard_standin", "bra_audit_greedy", "random_dual", "oracle_dual")
MATCHED_TIMING_ARMS = ("mt_score_only", "mt_risk_aware", "mt_risk_aware_tiered")
LEGACY_PARENT = {"mt_score_only": "score_only", "mt_risk_aware": "risk_aware",
                 "mt_risk_aware_tiered": "risk_aware_tiered"}
ALL_CONVENTIONAL_ARMS = ZTA_ARMS + BASELINE_ARMS + MATCHED_TIMING_ARMS


# ===================================================================================== configuration
@dataclass(frozen=True)
class ZtaConfig:
    """Everything the conventional ladder needs beyond ``BaztConfig`` (budgets, costs, caps)."""
    tau_alpha: float = 0.5                  # clean-calibrated monitor/judge threshold (alpha = 0.02)
    alpha_clean_fpr: float = 0.02           # the false-positive rate tau_alpha was calibrated to
    rl_offset: float = 0.0                  # calibrated offset added to the Risk Level
    rl_bins: int = 10                       # monitor-score binning for RL
    trust_history_window: int = 12          # how many past TS values define [min, max]
    # arm_C additive weights (TS is a weighted mean of six [0, 1] terms)
    w_auth: float = 0.30
    w_hist: float = 0.15
    w_rate: float = 0.10
    w_presence: float = 0.10
    w_behaviour: float = 0.20
    w_trust: float = 0.15
    # risk level: flagged-agent count + binned score + offset
    rl_flagged_weight: float = 0.5
    rl_score_weight: float = 0.5
    # per-identity enforcement rate cap (arm_B)
    rate_limit_per_identity: int = 3
    # gsafeguard / bra knobs
    audit_rounds: int = 1
    # Optional external judge (out-of-family LLM judge in the Tier-1 harness). f(node, t, view) -> score|None.
    probe_score_hook: Optional[Callable] = None

    def __post_init__(self):
        if not 0.0 < self.tau_alpha < 1.0:
            raise ValueError("tau_alpha must be in (0, 1)")
        if not 0.0 <= self.alpha_clean_fpr < 1.0:
            raise ValueError("alpha_clean_fpr must be in [0, 1)")
        if self.rate_limit_per_identity < 1:
            raise ValueError("rate_limit_per_identity must be >= 1")
        total = (self.w_auth + self.w_hist + self.w_rate + self.w_presence + self.w_behaviour + self.w_trust)
        if total <= 0:
            raise ValueError("at least one Trust-Score weight must be positive")


# ========================================================================================== base class
class ConventionalGateway:
    """Shared identity tier + dual ledger + audit trail. Subclasses only implement ``_plan``."""

    arm = "arm_A"

    def __init__(self, arm: str, zcfg: ZtaConfig, bcfg: BaztConfig, sl_cfg: sl.Config, topology: str,
                 seed: int, base_adj: Sequence[Sequence[int]], recv: Sequence[Sequence[int]],
                 seed_agents: FrozenSet[int] = frozenset(), matched_steps: Optional[Sequence[int]] = None):
        self.arm, self.zcfg, self.bcfg, self.sl_cfg = arm, zcfg, bcfg, sl_cfg
        self.n, self.T = sl_cfg.N, sl_cfg.T
        self.recv = [list(r) for r in recv]
        self.base_adj = [list(a) for a in base_adj]
        # arm_A identity tier: authenticated directed allowlist = the BASE graph.
        self._allow: Set[Tuple[int, int]] = {(u, v) for v in range(self.n) for u in self.recv[v]
                                            if v in self.base_adj[u]}
        self._out: List[List[int]] = [[] for _ in range(self.n)]
        for v in range(self.n):
            for u in self.recv[v]:
                self._out[u].append(v)
        self._blocked: Dict[Tuple[int, int], int] = {}          # (s, r) -> last blocked step
        self._restricted: Dict[int, int] = {}
        self._quarantined: Dict[int, int] = {}
        self._identity_blocks: Set[Tuple[int, int, int]] = set()
        self._matched_steps = frozenset(int(s) for s in matched_steps) if matched_steps is not None else None
        e_dir = sum(len(r) for r in self.recv)
        self.e_dir = e_dir
        self.total_deliveries = (self.T - 1) * e_dir
        self.ledger = BudgetLedger(bcfg, total_deliveries=self.total_deliveries)
        self.audit = AuditLog()
        self.fallback_active = False
        self.fallback_step: Optional[int] = None
        self.budget_exhausted: List[dict] = []
        self._log_delivered: Dict[int, List[Tuple[int, int, bool]]] = {}
        self._block_events: List[Tuple[int, int, int]] = []      # (t, sender, receiver) denied deliveries
        self._actions: List[Tuple[int, ActionKind, int]] = []
        self._audits_used: Dict[int, int] = {}                  # identity -> interventions used
        self._audit_queries: Dict[int, int] = {}
        self._edge_history: Dict[Tuple[int, int], int] = {}      # (s, r) -> delivered steps
        self._step_count = 0
        self._first_action_step: Optional[int] = None
        # arm_C state: unbounded [min, max] trust history and last score
        self._ts_hist: List[List[float]] = [[] for _ in range(self.n)]
        self._last_score = [0.0] * self.n
        self._alarm_run = [0] * self.n
        self._u_rand = sl.stream_rng("zta-random", f"{topology}:{self.n}", seed).random((self.T, self.n))

    # ------------------------------------------------------------------------------ the arm_A gate
    def edge_open(self, sender: int, receiver: int, t: int = 0) -> bool:
        if (sender, receiver) not in self._allow:
            self._identity_blocks.add((int(t), sender, receiver))
            return False
        until = self._blocked.get((sender, receiver))
        if until is not None and (until is None or t <= until):
            return False
        return True

    def pop_released(self, t: int) -> List[int]:
        due = sorted(i for i, until in self._quarantined.items() if until > 0 and until <= t)
        for i in due:
            del self._quarantined[i]
        return due

    def restricted(self, t: int) -> FrozenSet[int]:
        return frozenset(i for i, last in self._restricted.items() if last >= t)

    def on_deliver(self, t: int, outgoing: Dict[int, bool]) -> None:
        self._step_count += 1
        self._log_delivered[t] = [(s, r, stripped) for s, stripped in outgoing.items()
                                  for r in self._out[s] if r not in self._quarantined]
        for s in outgoing:
            for r in self._out[s]:
                self._edge_history[(s, r)] = self._edge_history.get((s, r), 0) + 1

    def _matched(self, t: int) -> bool:
        return self._matched_steps is None or t in self._matched_steps

    # ------------------------------------------------------------------------------------ decisions
    def decide(self, view: sl.StepView, truth: Callable[[int], bool]) -> Plan:
        t = view.t
        for i in [k for k, last in self._restricted.items() if last < t]:
            del self._restricted[i]
        if self.arm == "arm_A":
            return Plan()                                  # identity tier only: zero budget, no content access
        if self.fallback_active or not self._matched(t):
            return Plan()
        plan = self._plan(view, truth)
        self._apply(t, plan)
        for ev in self.ledger.mark_exhausted(t):
            self.budget_exhausted.append(ev)
            self.fallback_active = True
            self.fallback_step = t
        return plan

    def _plan(self, view: sl.StepView, truth: Callable[[int], bool]) -> Plan:   # pragma: no cover - abstract
        raise NotImplementedError

    # -------------------------------------------------------------------------------- accounting
    def _charge(self, kind: ActionKind, t: int, node: int, reason: str, **extra) -> bool:
        ch = self.ledger.try_charge(kind)
        if ch is None:
            return False
        self.audit.append(t=t, action=kind.name, target=node, cost=ch.cost, pool=ch.pool,
                          spent=ch.spent, remaining=(None if math.isinf(ch.remaining) else ch.remaining),
                          reason=reason, **{k: v for k, v in extra.items() if k in
                                            {"p_prime", "s_mon", "taint", "H", "net", "density", "theta"}})
        if kind != ActionKind.INSPECT:
            self._actions.append((t, kind, node))
            if self._first_action_step is None:
                self._first_action_step = t
        return True

    def _apply(self, t: int, plan: Plan) -> None:
        for i in plan.isolate:
            self._quarantined[i] = t + self.bcfg.quarantine_ttl if self.bcfg.quarantine_ttl else 0
            self._restricted.pop(i, None)
        for i in plan.restrict:
            self._restricted[i] = t + self.bcfg.ttl_restrict - 1
        for i, j in plan.block:
            self._blocked[(i, j)] = t + self.bcfg.ttl_restrict - 1

    def _block_edges(self, t: int, sender: int, receivers: Iterable[int], reason: str,
                     p: float = float("nan")) -> List[Tuple[int, int]]:
        """Deny delivery on ``sender -> receiver`` and charge 1 intervention unit per blocked edge."""
        out = []
        for r in list(receivers):
            if not self.edge_open(sender, r, t):
                continue
            if self._charge(ActionKind.BLOCK_EDGE, t, sender, reason, p_prime=p):
                out.append((sender, r))
                self._block_events.append((int(t), int(sender), int(r)))
                # `_blocked` is the single source of truth for delivery and it EXPIRES at the end of the step,
                # so the send graph itself is never mutated (that is what keeps the sender side untouched).
                self._blocked[(sender, r)] = t
        return out

    def _audit(self, t: int, node: int, view: sl.StepView, score: float) -> bool:
        """Charge 1 audit unit to query the monitor/judge for this node's broadcast."""
        if not self._charge(ActionKind.INSPECT, t, node, "audit", p_prime=sl.sigmoid(self.sl_cfg.a * score -
                                                                                     self.sl_cfg.b),
                            s_mon=score):
            return False
        self._audit_queries[node] = self._audit_queries.get(node, 0) + 1
        return True

    # ------------------------------------------------------------------------------------ metrics
    def extra_metrics(self, infected_series: Sequence[Sequence[bool]], world=None) -> dict:  # noqa: ARG002
        T, N = self.T, self.n
        capacity = max((T - 1) * self.e_dir, 0)
        harm = delivered = 0
        for t in range(1, T):
            for src, _dst, stripped in self._log_delivered.get(t, ()):
                delivered += 1
                if infected_series[t - 1][src] and not stripped:
                    harm += 1
        ever = [False] * N
        ever_at: List[List[bool]] = []
        for row in infected_series:
            ever = [e or bool(x) for e, x in zip(ever, row)]
            ever_at.append(ever)
        false_actions = sum(1 for t, _k, node in self._actions if not ever_at[t - 1][node])
        benign_den = sum(1 for t, s, _r in self._block_events if t >= 1 and not ever_at[t - 1][s])
        benign_del = benign_den + sum(1 for t in range(1, T) for s, _r, _st in self._log_delivered.get(t, ())
                                     if not ever_at[t - 1][s])
        recovery = float("nan")
        if self._actions:
            t_first = min(t for t, _k, _n in self._actions)
            recovery = float(T - t_first + 1)
            for t in range(t_first, T + 1):
                if not any(infected_series[t - 1]):
                    recovery = float(t - t_first)
                    break
        sk = self.ledger.spent_by_kind
        m = {
            "enforcement_spend": self.ledger.spent,
            "spend_inspect": sk[ActionKind.INSPECT], "spend_sanitize": sk[ActionKind.SANITIZE],
            "spend_restrict": sk[ActionKind.RESTRICT], "spend_block": sk[ActionKind.BLOCK_EDGE],
            "spend_isolate": sk[ActionKind.ISOLATE],
            "harm_exposure_auc": harm / capacity if capacity else 0.0,
            "message_throughput": delivered / capacity if capacity else 0.0,
            "degraded_node_step_frac": sum(len(self.restricted(t)) for t in range(1, T + 1)) / (N * T)
            if N * T else 0.0,
            "false_action_rate": false_actions / len(self._actions) if self._actions else 0.0,
            "recovery_time": recovery,
            "cost_per_prevented_compromise": float("nan"),
            "budget_violations": self.audit.overspent_rows(),
            "inspections_conclusive": self.ledger.spent_by_kind[ActionKind.INSPECT],
            "budget_exhausted": int(bool(self.budget_exhausted)),
            "budget_exhausted_first_step": self.fallback_step if self.fallback_step is not None else -1,
            "budget_exhausted_pools": ",".join(sorted({e["pool"] for e in self.budget_exhausted})),
            "audit_spend": self.ledger.audit_spend(),
            "intervention_spend": self.ledger.intervention_spend(),
            "identity_blocks": len(self._identity_blocks),
            "fallback_to_arm_a": int(self.fallback_active),
            # clean-calibration telemetry: what fraction of BENIGN deliveries this arm denied
            "benign_deliveries": benign_del,
            "benign_denied_deliveries": benign_den,
            "benign_delivery_deny_rate": (benign_den / benign_del) if benign_del else 0.0,
        }
        for key in BAZT_METRICS:                     # every arm exposes the same metric vocabulary
            m.setdefault(key, float("nan"))
        return m


# ======================================================================================== the ladder
class _MonitorMixin:
    """Shared monitor/score arithmetic: identical to the simulator's own calibration."""

    def _scores(self, view: sl.StepView) -> List[float]:
        probe = self.zcfg.probe_score_hook
        out = []
        for i in range(self.n):
            r = probe(i, view.t, view) if probe is not None else None
            out.append(float(view.roll_scores[i]) if r is None else float(r))
        return out

    def action_steps(self) -> List[int]:
        """Steps at which this defender actually spent a delivery-changing action (matched-timing reference)."""
        return sorted({t for t, _k, _n in self._actions})


class IdentityTierGateway(ConventionalGateway, _MonitorMixin):
    """arm_A -- authenticate identity + edge allowlist, at zero budget cost. Never inspects content."""

    arm = "arm_A"

    def _plan(self, view, truth):  # noqa: ARG002
        return Plan()

    def decide(self, view, truth):  # noqa: ARG002
        return Plan()


class StatelessGateway(ConventionalGateway, _MonitorMixin):
    """arm_B -- arm_A + 1 audit per unique broadcast, block above tau_alpha, per-identity rate limit."""

    arm = "arm_B"

    def _plan(self, view: sl.StepView, truth: Callable[[int], bool]) -> Plan:  # noqa: ARG002
        plan = Plan()
        t = view.t
        scores = self._scores(view)
        for i in view.active:
            msg = view.observed_texts[i]
            if not msg:
                continue
            if not self._audit(t, i, view, scores[i]):
                continue
            self._last_score[i] = scores[i]
            if scores[i] <= self.zcfg.tau_alpha:
                continue
            used = self._audits_used.get(i, 0)
            if used >= self.zcfg.rate_limit_per_identity:
                continue                                     # per-identity enforcement rate cap
            targets = [r for r in self._out[i] if (i, r) not in self._blocked]
            blocked = self._block_edges(t, i, targets, "tau_alpha", p=sl.sigmoid(
                self.sl_cfg.a * scores[i] - self.sl_cfg.b))
            self._audits_used[i] = used + len(blocked)
            plan.block.extend(blocked)
        return plan


class ContextualGateway(ConventionalGateway, _MonitorMixin):
    """arm_C -- Bradatsch style: additive Trust Score vs Risk Level, grant iff ``TS > RL``."""

    arm = "arm_C"

    # ------------------------------------------------------------------ components of TS and RL
    def _ts(self, i: int, r: int, t: int, score: float, active: Set[int]) -> Tuple[float, Dict[str, float]]:
        z = self.zcfg
        auth = 1.0 if (i, r) in self._allow else 0.0
        # historical recipient usage: how often this pair actually delivered, normalised by steps seen
        hist = min(1.0, self._edge_history.get((i, r), 0) / max(1, t - 1))
        # access rate within range: deviation of this identity's audit volume from the per-step norm
        rate = 1.0 - min(1.0, abs(self._audit_queries.get(i, 0) - (t - 1)) / max(1, t))
        presence = 1.0 if i in active else 0.0
        # message behaviour: a high monitor score is bad behaviour
        behaviour = 1.0 - min(1.0, max(0.0, score) / max(1e-9, 3.0 * self.sl_cfg.mu_inf))
        hist_values = self._ts_hist[i][-z.trust_history_window:]
        if len(hist_values) >= 2:
            lo, hi = min(hist_values), max(hist_values)          # UNBOUNDED [min, max]
            prev = hist_values[-1]
            trust = (prev - lo) / (hi - lo) if hi > lo else 1.0
        else:
            trust = 1.0
        parts = {"auth": auth, "hist": hist, "rate": rate, "presence": presence,
                 "behaviour": behaviour, "trust": trust}
        ts = (z.w_auth * auth + z.w_hist * hist + z.w_rate * rate + z.w_presence * presence
              + z.w_behaviour * behaviour + z.w_trust * trust)
        return ts, parts

    def _rl(self, i: int, score: float, active: Set[int]) -> Tuple[float, Dict[str, float]]:
        z = self.zcfg
        flagged = sum(1 for j in active if self._last_score[j] > z.tau_alpha)
        flagged_norm = flagged / max(1, len(active))
        binned = math.floor(min(1.0, max(0.0, score) / max(1e-9, 3.0 * self.sl_cfg.mu_inf))
                            * z.rl_bins) / max(1, z.rl_bins)
        rl = z.rl_flagged_weight * flagged_norm + z.rl_score_weight * binned + z.rl_offset
        return min(1.0, rl), {"flagged": flagged_norm, "binned": binned, "offset": z.rl_offset}

    def _plan(self, view: sl.StepView, truth: Callable[[int], bool]) -> Plan:  # noqa: ARG002
        plan = Plan()
        t, active = view.t, set(view.active)
        scores = self._scores(view)
        for i in view.active:
            msg = view.observed_texts[i]
            if not msg:
                continue
            if not self._audit(t, i, view, scores[i]):
                continue
            self._last_score[i] = scores[i]
            targets = [r for r in self._out[i] if (i, r) not in self._blocked]
            denied = []
            for r in targets:
                ts, _parts = self._ts(i, r, t, scores[i], active)
                rl, _rl_parts = self._rl(i, scores[i], active)
                self._ts_hist[i].append(ts)
                if ts > rl:
                    continue                                  # grant
                denied.append(r)
            if denied:
                blocked = self._block_edges(t, i, denied, "TS<=RL", p=sl.sigmoid(
                    self.sl_cfg.a * scores[i] - self.sl_cfg.b))
                plan.block.extend(blocked)
        return plan


# ===================================================================================== baselines
class GsafeguardStandinGateway(ConventionalGateway, _MonitorMixin):
    """Detect-and-cut: on detection, cut EVERY outgoing edge of the node (G-Safeguard stand-in)."""

    arm = "gsafeguard_standin"

    def _plan(self, view: sl.StepView, truth: Callable[[int], bool]) -> Plan:  # noqa: ARG002
        plan = Plan()
        t = view.t
        scores = self._scores(view)
        for i in view.active:
            if not view.observed_texts[i]:
                continue
            if not self._audit(t, i, view, scores[i]):
                continue
            self._last_score[i] = scores[i]
            if scores[i] <= self.zcfg.tau_alpha:
                continue
            targets = [r for r in self._out[i]]
            blocked = self._block_edges(t, i, targets, "gsafeguard_cut_all", p=sl.sigmoid(
                self.sl_cfg.a * scores[i] - self.sl_cfg.b))
            plan.block.extend(blocked)
        return plan


class BraAuditGreedyGateway(ConventionalGateway, _MonitorMixin):
    """BRA audit greedy: priority = rounds_since_last_audit * descendant_count."""

    arm = "bra_audit_greedy"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._last_audit = {i: 0 for i in range(self.n)}
        self._desc = {i: float(self._descendants(i)) for i in range(self.n)}

    def _descendants(self, i: int) -> int:
        seen, frontier = {i}, [i]
        while frontier:
            nxt = []
            for u in frontier:
                for v in self.base_adj[u]:
                    if v not in seen:
                        seen.add(v)
                        nxt.append(v)
            frontier = nxt
        return len(seen) - 1

    def _plan(self, view: sl.StepView, truth: Callable[[int], bool]) -> Plan:  # noqa: ARG002
        plan = Plan()
        t = view.t
        scores = self._scores(view)
        order = sorted(view.active, key=lambda i: (-((t - self._last_audit[i]) * self._desc[i]), i))
        for round_ in range(max(1, self.zcfg.audit_rounds)):
            for i in order:
                if not view.observed_texts[i]:
                    continue
                if round_ > 0 and i != order[0]:
                    continue                                   # one deep audit per step, then triage
                if not self._audit(t, i, view, scores[i]):
                    continue
                self._last_audit[i] = t
                self._last_score[i] = scores[i]
                if scores[i] > self.zcfg.tau_alpha:
                    blocked = self._block_edges(t, i, list(self._out[i]), "bra_flagged", p=sl.sigmoid(
                        self.sl_cfg.a * scores[i] - self.sl_cfg.b))
                    plan.block.extend(blocked)
        return plan


class RandomDualGateway(ConventionalGateway, _MonitorMixin):
    """Random audits and random blocks inside the SAME dual budget (schedule control)."""

    arm = "random_dual"

    def _plan(self, view: sl.StepView, truth: Callable[[int], bool]) -> Plan:  # noqa: ARG002
        plan = Plan()
        t = view.t
        scores = self._scores(view)
        for i in view.active:
            if not view.observed_texts[i]:
                continue
            if self._u_rand[t - 1, i] > 0.5:
                continue
            if not self._audit(t, i, view, scores[i]):
                continue
            self._last_score[i] = scores[i]
            targets = [r for r in self._out[i] if self._u_rand[t - 1, r] < 0.5]
            if targets and scores[i] > self.zcfg.tau_alpha:
                plan.block.extend(self._block_edges(t, i, targets, "random", p=scores[i]))
        return plan


class OracleDualGateway(ConventionalGateway, _MonitorMixin):
    """Non-deployable upper bound: knows ground truth and isolates infected nodes under the same caps."""

    arm = "oracle_dual"

    def _plan(self, view: sl.StepView, truth: Callable[[int], bool]) -> Plan:
        plan = Plan()
        t = view.t
        for i in view.active:
            if not truth(i):
                continue
            if self._charge(ActionKind.ISOLATE, t, i, "oracle"):
                plan.isolate.append(i)
                self._quarantined[i] = 0
        return plan


class MatchedTimingGateway(ConventionalGateway, _MonitorMixin):
    """Legacy rule, executed ONLY on the reference arm's action steps (matched-timing control).

    The decision rule is the legacy one (``score_only`` / ``risk_aware`` / ``risk_aware_tiered``), but it may
    only spend on ``matched_steps`` -- the steps at which the reference defender actually acted. This isolates
    ``when`` a defender spends from ``how`` it decides.
    """

    def __init__(self, arm: str, *a, **kw):
        super().__init__(arm, *a, **kw)
        self.parent = LEGACY_PARENT[arm]

    def _plan(self, view: sl.StepView, truth: Callable[[int], bool]) -> Plan:  # noqa: ARG002
        plan = Plan()
        t = view.t
        scores = self._scores(view)
        active = list(view.active)
        aset = set(active)
        zero = [0.0] * self.n
        deg = [len(a) for a in self.base_adj]
        maxdeg = max(deg) if deg else 1
        p = [sl.sigmoid(self.sl_cfg.a * scores[i] - self.sl_cfg.b) for i in range(self.n)]
        for i in active:
            self._alarm_run[i] = self._alarm_run[i] + 1 if p[i] >= self.sl_cfg.tau_p else 0
        if self.parent in ("score_only", "risk_aware", "risk_aware_tiered") and not any(
                p[i] >= self.sl_cfg.tau_p for i in active):
            return plan
        if self.parent == "score_only":
            hits = sorted((i for i in active if p[i] >= self.sl_cfg.tau_p
                           and self._alarm_run[i] >= self.sl_cfg.confirm_steps), key=lambda i: -p[i])
            for i in hits:
                if self._charge(ActionKind.ISOLATE, t, i, "mt_score_only", p_prime=p[i]):
                    plan.isolate.append(i)
                    self._quarantined[i] = 0
            return plan
        scores_r = []
        for i in active:
            H = sl.reach_H(self.base_adj, i, self.sl_cfg.T - t, self.sl_cfg, aset, zero, use_p=False)
            C = sl.centrality(i, t, deg, maxdeg, self.sl_cfg.T)
            scores_r.append((p[i] * H - self.sl_cfg.lam * (1 - p[i]) * C, i))
        tau = self.sl_cfg.tau_quar if self.parent == "risk_aware_tiered" else self.sl_cfg.tau_R
        for r, i in sorted(scores_r, reverse=True):
            if r < tau:
                break
            if self.parent == "risk_aware_tiered":
                if self._charge(ActionKind.SANITIZE, t, i, "mt_tier1", p_prime=p[i]):
                    plan.sanitize.append(i)
            if self._charge(ActionKind.ISOLATE, t, i, "mt_tier2", p_prime=p[i]):
                plan.isolate.append(i)
                self._quarantined[i] = 0
        return plan


# ========================================================================================== factory
_CLASSES = {
    "arm_A": IdentityTierGateway, "arm_B": StatelessGateway, "arm_C": ContextualGateway,
    "gsafeguard_standin": GsafeguardStandinGateway, "bra_audit_greedy": BraAuditGreedyGateway,
    "random_dual": RandomDualGateway, "oracle_dual": OracleDualGateway,
    "mt_score_only": MatchedTimingGateway, "mt_risk_aware": MatchedTimingGateway,
    "mt_risk_aware_tiered": MatchedTimingGateway,
}


def make_conventional_gateway(arm: str, zcfg: ZtaConfig, bcfg: BaztConfig, sl_cfg: sl.Config, topology: str,
                              seed: int, base_adj: Sequence[Sequence[int]], recv: Sequence[Sequence[int]],
                              seed_agents: FrozenSet[int] = frozenset(),
                              matched_steps: Optional[Sequence[int]] = None) -> ConventionalGateway:
    if arm not in _CLASSES:
        raise ValueError(f"unknown conventional arm {arm!r}; choose from {sorted(_CLASSES)}")
    return _CLASSES[arm](arm, zcfg, bcfg, sl_cfg, topology, seed, base_adj, recv, seed_agents,
                         matched_steps=matched_steps)


def conventional_bazt_config(zcfg: ZtaConfig, audit_pct: Optional[float], intervention_pct: Optional[float],
                             total_deliveries: int = 0) -> BaztConfig:
    """The SHARED dual-percentage budget every compared defender runs under (arm_B, arm_C, bazt, baselines)."""
    return BaztConfig(budget_mode="dual_pct", audit_pct=audit_pct, intervention_pct=intervention_pct,
                      cost_inspect=1, cost_sanitize=1, cost_restrict=2, cost_block=1, cost_isolate=3,
                      strip_degrades_weight=True)


def sender_memory_fingerprint(results: dict) -> Tuple[float, ...]:
    """Observable fingerprint of a run, used by the sender-memory-preservation test."""
    return (float(results["auc_ever_infected"]), float(results["task_mae"]),
            float(results["quarantines_used"]), float(results["total_deviated_messages"]))
