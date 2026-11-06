"""Contracts shared by every feature. Changing anything here is a cross-feature change: owner = core.

Shapes: B episodes in a batch, N agents, T rounds. Agents send at every round; a delivery is one message on one
directed edge (sender i -> receiver j), indexed [b, i, j]. Every mask is bool, every index is int64.

Device and precision (settings [compute]): tensors live on `ctx.device` and float tensors that carry probabilities,
scores, statistics and model values use `ctx.dtype` (float64 or float32); a feature creates them with
`device=ctx.device, dtype=ctx.dtype`. Draws come from `core.keyed_uniform` / `keyed_normal`, which return the SAME
numbers on every device and dtype (up to rounding). Accounting is exact and NEVER follows compute.dtype: audit and
intervention caps, remaining budgets, spend per round and action costs are float64 (integer-valued unit counts,
exact up to 2**53) or int64, so budget sums, caps and exhaustion tests give the same answer in a float32 run.
Statistics that are compared across dtypes (ties, thresholds) get a rounding-safe jitter from
`core.tie_jitter_scale(ctx.dtype, magnitude)`.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Protocol, runtime_checkable

import torch

Tensor = torch.Tensor


class ActionKind(str, enum.Enum):
    """The eight actions of the spec (§2.5). Persistent actions last until `REINSTATE`."""
    INSPECT = "inspect"                      # spend one audit unit per checked sent text (audit budget)
    SANITIZE = "sanitize"                    # deliver with the payload stripped, this round only
    DELAY = "delay"                          # deliver one round later
    EDGE_QUARANTINE = "edge_quarantine"      # persistent: block one (i, j) edge
    RATE_LIMIT = "rate_limit"                # persistent: sender reaches at most `rate_limit_per_identity` receivers
    PRIVILEGE_DEGRADATION = "privilege_degradation"   # persistent: every outgoing delivery is sanitized
    ISOLATION = "isolation"                  # persistent: agent neither sends nor receives
    REINSTATEMENT = "reinstatement"          # lift every persistent action on an agent
    DENY = "deny"                            # drop the delivery this round (A, B, C block; BAZT may use it too)


@dataclass(frozen=True)
class EpisodeContext:
    """Fixed for a batch of episodes. Built by the platform before round 1."""
    B: int
    N: int
    T: int
    adj: Tensor                              # bool [B, N, N]: base graph = A's allowlist, adj[b, i, j] is i -> j
    sink: Tensor                             # int64 [B]: protected sink agent
    audit_cap: Tensor                        # float64 [B]: audit units for the episode, inf if uncapped
    intervention_cap: Tensor                 # float64 [B]: intervention units for the episode, inf if uncapped
    costs: Mapping[str, float]               # ActionKind value -> cost per affected delivery
    thresholds: Mapping[str, float]          # thresholds this arm acts on: empty in an open-loop calibration pass (it collects
                                             # statistics and never acts), the candidate thresholds of a closed-loop pass, the
                                             # calibrated ones in an evaluation run
    settings: Mapping                        # resolved settings (read with core.get)
    arm: str
    unit_ids: tuple                          # length B, for checkpointing and logging
    arm_stream: Callable[[str, int], torch.Generator]   # (namespace, b) -> arm-specific generator
    platform: str = ""                       # "mock" | "llm": lets an arm pick platform-specific settings
    timing_schedule: Optional[Tensor] = None   # bool [B, T]: rounds a matched-timing control may act (0-based);
                                             # None for every other arm
    cell: Optional[Mapping] = None           # {"attack", "task"} of the batch's units (LLM side; the GNN arm picks
                                             # its trained family from it); None where not supplied
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))   # compute.device (core.device(settings)): where this batch's tensors live
    dtype: torch.dtype = torch.float64       # compute.dtype (core.dtype(settings)): float tensors other than accounting


@dataclass
class Observation:
    """What a defender may see at round t. Ground truth (infection, attackers) is NEVER here."""
    t: int                                   # 1..T, counted over the whole episode (LLM: the seed run, D54)
    T: int
    active: Tensor                           # bool [B, N]: not isolated
    attempts: Tensor                         # bool [B, N, N]: deliveries the agents try this round
    monitor: Optional[Tensor]                # ctx.dtype [B, N] cheap monitor probability (free); None on the LLM side
    remaining_audit: Tensor                  # float64 [B]
    remaining_intervention: Tensor           # float64 [B]
    audit: Callable[[Tensor], Tensor]        # bool [B, N] senders -> ctx.dtype [B, N] judge/auditor score of
                                             # their text this round, NaN where not requested or unaffordable;
                                             # debits one audit unit per sender it grants (a request beyond the affordable
                                             # number is granted as a keyed random subset, on both platforms)
    authentic: Optional[Tensor] = None       # bool [B, N, N]: message authentication (mTLS / signature) passed;
                                             # False where content was altered in transit (AiTM)
    oracle: Optional[Callable[[], Tensor]] = None   # bool [B, N] infected now; supplied ONLY to arms whose
                                             # ArmSpec.needs_truth is True (oracle allocation)
    round_in_conversation: Optional[int] = None   # LLM side: 1-based round inside the current conversation
                                             # (t and T always count the whole episode = seed run, D54); None on the mock
    texts: Optional[List[List[str]]] = None   # [B][N] text each agent wrote this round (LLM side only;
                                             # delivered or not); None on the mock side
    audit_isolated: Optional[Callable[[Tensor], Tensor]] = None
                                             # D123: bool [B, N] isolated agents -> ctx.dtype [B, N] auditor/judge score
                                             # of the isolated agent (mock: its carried state, D111; LLM: the text it
                                             # still wrote this round), NaN where not requested, not isolated, without
                                             # a text or unaffordable. Same unit cost, same audit budget and same keyed
                                             # random subset on over-request as `audit`, but its own call count, so
                                             # `audit` itself never grants an isolated agent and its numbers never move
                                             # for an arm that does not call this. None where a platform has no such channel.


@dataclass
class Actions:
    """A defender's decision for round t. None = no such action. The platform applies them in this order:
    reinstatement, isolation, edge_quarantine, privilege_degradation, rate_limit, deny, delay, sanitize;
    charges intervention units per affected delivery (agent-level actions: the deliveries they remove over the remaining
    rounds, e.g. isolation: live out-degree x remaining rounds); an action on a delivery an earlier action already removed
    costs nothing. Rejects all actions of an episode whose cost exceeds its remaining budget (counted as a budget
    violation)."""
    deny: Optional[Tensor] = None            # bool [B, N, N]
    sanitize: Optional[Tensor] = None        # bool [B, N, N]
    delay: Optional[Tensor] = None           # bool [B, N, N]
    edge_quarantine: Optional[Tensor] = None   # bool [B, N, N]
    rate_limit: Optional[Tensor] = None      # bool [B, N]
    privilege_degradation: Optional[Tensor] = None   # bool [B, N]
    isolation: Optional[Tensor] = None       # bool [B, N]
    reinstatement: Optional[Tensor] = None   # bool [B, N]
    threshold_restricted: Optional[Tensor] = None   # bool [B, N, N]: deliveries restricted BECAUSE a calibrated
                                             # statistic exceeded its threshold (D108: alpha counts only these;
                                             # rate limits and A's rules are not); None = every restriction counts
    events: List[dict] = field(default_factory=list)   # e.g. {"event": "budget_exhausted", "b": 0, "budget": "audit"}
    statistics: Dict[str, Tensor] = field(default_factory=dict)
    # Decision statistics for alpha calibration: name -> ctx.dtype [B, N, N] (or [B, N]), higher = more suspicious;
    # the arm acts when statistic > ctx.thresholds[name]. A threshold missing from ctx.thresholds means +inf
    # (never act on it), which is how calibration runs collect the statistic without acting.


@runtime_checkable
class Defender(Protocol):
    """An arm. One instance per batch; keeps its own state across rounds."""
    name: str

    def reset(self, ctx: EpisodeContext) -> None: ...

    def decide(self, obs: Observation) -> Actions: ...

    def state_dict(self) -> dict: ...                  # tensors and plain values only (progressive checkpoints)

    def load_state_dict(self, state: Mapping) -> None: ...


@dataclass
class EpisodeTrace:
    """Ground-truth record of a batch, produced by a platform, consumed by metrics. Rounds are 0-based here.

    Time convention, identical on both platforms (index r = round r+1 of the episode):
      infected[r]  the agent's OWN output of round r carries the payload;
      active[r]    the agent is active DURING round r, i.e. before the defender's decision taken after round r;
      attempts/delivered/restricted/payload_delivered[r] and meta['tampered'][r]  the deliveries that carry the
                   outputs of round r (sender i's round-r text or value), whenever they arrive (a delay keeps the
                   index of the round the output was produced in);
      correct[r]   the agent's answer of round r.
    A delivery (r, i, j) is malicious iff infected[r, i] or it was tampered; a decision taken after round r shows
    in active[r+1] and in meta['flagged']."""
    unit_ids: tuple                          # length B
    infected: Tensor                         # bool [B, T, N]: the agent's own output carries the payload
    attackers: Tensor                        # bool [B, N]: insider attackers / AiTM victim's adversary = none
    active: Tensor                           # bool [B, T, N]
    attempts: Tensor                         # bool [B, T, N, N]
    delivered: Tensor                        # bool [B, T, N, N]: delivered unmodified
    restricted: Tensor                       # bool [B, T, N, N]: denied, sanitized, delayed or quarantined
    payload_delivered: Tensor                # bool [B, T, N, N]
    correct: Tensor                          # bool [B, T, N]: answer correct (mock: |est - mean| <= eps_correct)
    goal_reached: Tensor                     # bool [B]: the attack family's specific goal (targeted ASR)
    sink: Tensor                             # int64 [B]
    spend_audit: Tensor                      # float64 [B, T]
    spend_intervention: Tensor               # float64 [B, T]
    exhausted_step: Tensor                   # int64 [B]: first round a budget hit zero, 0 if never
    violations: Tensor                       # int64 [B]
    action_counts: Tensor                    # int64 [B, len(ActionKind)] in ActionKind order
    attack_free: bool                        # True for calibration / false-positive runs
    meta: dict = field(default_factory=dict)   # tokens, latency, torch version, threads, dtype, seeds


@runtime_checkable
class Tracker(Protocol):
    """Experiment tracking (Weights & Biases adapter in features/tracking)."""

    def start(self, block: str, config: Mapping, resume_id: Optional[str]) -> str: ...

    def log(self, metrics: Mapping, step: int) -> None: ...

    def log_rows(self, table: str, rows: List[Mapping]) -> None: ...

    def artifact(self, path: str, name: str, kind: str) -> None: ...

    def finish(self) -> None: ...


class NullTracker:
    """Tracker that records nothing; used in tests and when tracking is off."""

    def start(self, block, config, resume_id):
        return resume_id or "null"

    def log(self, metrics, step):
        pass

    def log_rows(self, table, rows):
        pass

    def artifact(self, path, name, kind):
        pass

    def finish(self):
        pass


@dataclass(frozen=True)
class ArmSpec:
    """How the runner finds an arm. Each defender feature exports a tuple ARMS of these in its public.py."""
    name: str                                # unit["arm"], e.g. "A", "B", "C", "BAZT", "GNN_standin"
    factory: Callable[[], Defender]
    platforms: tuple = ("mock", "llm")
    needs_truth: bool = False                # only oracle allocation
    calibrated: bool = False                 # reports Actions.statistics; thresholds set at alpha (spec §2.8)


@runtime_checkable
class Platform(Protocol):
    """Runs units. Mock: a batch of episodes vectorised over B. LLM: one seed run (its conversations in order)."""
    name: str                                # "mock" | "llm"
    adaptive_conditions: tuple               # the adaptive conditions (unit["condition"]) it runs; a unit carrying any
                                             # other is refused by run_batch, and the runner refuses its block before
                                             # anything runs. Mock: the adaptive feature's CONDITIONS. LLM: ("gray_box",), the replay of the mock's best response (D110).

    def run_batch(self, units: List[dict], arm: ArmSpec, settings: Mapping, store, tracker: Tracker) -> List[dict]:
        """Run the units (all with the same arm, budget cell, topology, attack and N), append rows to `store`
        progressively, mark each unit done, return its rows. Units already done in `store` are skipped."""
        ...

    def calibrate(self, units: List[dict], arm: ArmSpec, settings: Mapping, store) -> Dict[str, float]:
        """Attack-free runs of the uncapped arm; returns {statistic name: threshold} at deny rate alpha."""
        ...


@runtime_checkable
class Judge(Protocol):
    """Content check of B / C / BAZT. Mock: auditor preset (binormal ROC). LLM: PromptArmor judge, logprob('yes')."""

    def score(self, texts: List[str]) -> Tensor: ...   # float64 [len(texts)], higher = more likely injected
