"""Arm registry and per-episode assembly (policy string + swarm config + hooks + agent) for every arm.

Arms
  legacy      no_defense, score_only, centrality_only, risk_aware, risk_aware_tiered (+ offline_oracle bound)
  static_zta  always-on role-based edge policy + schema-only channel (env_ladder.HardConstraintSpec), no adaptivity
  BAZT        bazt and its six single-mechanism ablations (`swarmlab.BAZT_POLICIES`)

Every arm of one run shares the same world, decoys, hooks base, edge-weight rule and rationale-carrying agent,
so differences between arms come from the defence only.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional

import env_ladder as el
import swarmlab as sl
from bazt.channel import RationaleAgent, make_edge_weight_modifier, sensor_classes
from bazt.config import BaztConfig
from bazt.gateway import EnforcementGateway
from bazt.roles import assign_roles

LEGACY_ARMS = ("no_defense", "score_only", "centrality_only", "risk_aware", "risk_aware_tiered")
REFERENCE_ARMS = ("offline_oracle",)                  # non-deployable upper bound, used by delta_rel
STATIC_ARM = "static_zta"
BAZT_ARMS = tuple(sl.BAZT_POLICIES)
DEFAULT_ARMS = LEGACY_ARMS + REFERENCE_ARMS + (STATIC_ARM,) + BAZT_ARMS
# Sprint experiment registry (additive; DEFAULT_ARMS is untouched so `bazt.runner` behaves exactly as before).
from bazt.conventional_zta import (ALL_CONVENTIONAL_ARMS, BASELINE_ARMS, MATCHED_TIMING_ARMS,  # noqa: E402
                                   ZTA_ARMS)
EXPERIMENT_ARMS = tuple(BAZT_ARMS) + ALL_CONVENTIONAL_ARMS

_BAZT_OVERRIDES: Dict[str, dict] = {
    "bazt": {}, "bazt_no_prov": {"use_provenance": False}, "bazt_no_topo": {"use_topology": False},
    "bazt_no_pacing": {"use_pacing": False}, "bazt_no_util": {"use_utility": False},
    "bazt_binary": {"binary_actions": True}, "bazt_unlimited": {"unlimited": True},
}
assert set(_BAZT_OVERRIDES) == set(BAZT_ARMS)


def bazt_config_for(arm: str, base: BaztConfig) -> BaztConfig:
    return replace(base, **_BAZT_OVERRIDES[arm])


def equalized_swarm_cfg(arm: str, swarm_cfg: sl.Config, bcfg: BaztConfig) -> sl.Config:
    """Give a legacy arm the same TOTAL cost budget as the BAZT arms.

    Hard-only baselines: K = floor(B / cost_isolate). risk_aware_tiered keeps its tier-1 sanitize budget
    (capped by what B can pay for) and spends the remainder on hard quarantines (or K_hard / B_soft when the
    run uses tiered budgets).
    """
    if arm in BAZT_ARMS or arm == STATIC_ARM:
        return swarm_cfg
    total = int(bcfg.total_capacity)
    if arm == "risk_aware_tiered":
        if bcfg.budget_mode == "tiered":
            return replace(swarm_cfg, K=bcfg.K_hard, B_san=bcfg.B_soft)
        b_san = min(swarm_cfg.B_san, total // bcfg.cost_sanitize)
        return replace(swarm_cfg, B_san=b_san, K=max(0, (total - b_san * bcfg.cost_sanitize) // bcfg.cost_isolate))
    return replace(swarm_cfg, K=total // bcfg.cost_isolate)


@dataclass
class RunContext:
    bcfg: BaztConfig
    agent_factory: Callable[[sl.Config, sl.World], object]
    equalize: bool = False
    entry_factory: Optional[Callable[[int, int], sl.EpisodeHooks]] = None   # env_ladder fake-web entry
    adapter: Optional[el.CoTPayloadAdapter] = None
    constraint: Optional[el.HardConstraintSpec] = None
    sink: Optional[List[EnforcementGateway]] = None                          # collects gateways for audit export
    # --- sprint experiments (all defaulted: a run that does not set them is unchanged) -----------
    zcfg: Optional[object] = None                    # bazt.conventional_zta.ZtaConfig
    matched_steps: Optional[dict] = None             # arm -> frozenset(steps) for the matched-timing controls
    scenario_hooks: Optional[Callable[[int, int], sl.EpisodeHooks]] = None   # attack-suite hooks


@dataclass
class EpisodeSetup:
    policy: str
    cfg: sl.Config
    hooks: sl.EpisodeHooks
    agent: object


def make_gateway_factory(ctx: RunContext) -> Callable:
    """Matches `EpisodeHooks.gateway_factory`. `decoys` is part of the signature but never read: a gateway
    must not know which benign nodes are decoys.

    BAZT arms get `EnforcementGateway`; the sprint's conventional ladder and baselines get the matching
    `bazt.conventional_zta` class. Both speak the identical protocol and share the identical dual ledger.
    """
    def factory(policy, world, cfg, base_adj, decoys, hooks):  # noqa: ARG001
        recv = hooks.recv_from if hooks.recv_from is not None else base_adj
        if policy in ALL_CONVENTIONAL_ARMS:
            from bazt.conventional_zta import make_conventional_gateway
            steps = (ctx.matched_steps or {}).get(policy)
            gw = make_conventional_gateway(policy, ctx.zcfg, ctx.bcfg, cfg, world.topology, world.seed,
                                           base_adj, recv, frozenset(hooks.seed_agents), matched_steps=steps)
        else:
            gw = EnforcementGateway(policy, bazt_config_for(policy, ctx.bcfg), cfg, world.topology, world.seed,
                                    base_adj, recv, frozenset(hooks.seed_agents))
        if ctx.sink is not None:
            ctx.sink.append(gw)
        return gw
    return factory


def _compose(first: Optional[Callable], second: Callable) -> Callable:
    return second if first is None else (lambda text, i, t: second(first(text, i, t), i, t))


def _base_hooks(ctx: RunContext, world: sl.World, n: int) -> sl.EpisodeHooks:
    hooks = (ctx.scenario_hooks(world.seed, n) if ctx.scenario_hooks is not None
             else ctx.entry_factory(world.seed, n) if ctx.entry_factory else sl.EpisodeHooks())
    kw: dict = {}
    if ctx.adapter is not None:       # ground truth = the adapter's payload; the monitor still reads the tag
        kw.update(infection_marker=ctx.adapter.detect_ground_truth_payload,
                  monitor_marker=lambda s: sl.CANARY in s)
    if ctx.constraint is not None:
        mask = el.routing_mask(world.adj, ctx.constraint.movement_direction_mode)
        spec = ctx.constraint
        kw.update(recv_from=mask.recv_from, blocked_sends=mask.blocked_sends, total_sends=mask.total_sends,
                  channel_filter=lambda text, i, t: el.apply_hard_constraints(text, i, t, spec))
    return replace(hooks, **kw)


def _static_zta(hooks: sl.EpisodeHooks, world: sl.World, ctx: RunContext) -> sl.EpisodeHooks:
    """Always-on least-privilege: role-based send lists plus the schema-only channel. No budget, no adaptivity."""
    ids = assign_roles(world.topology, world.seed, world.adj, ctx.bcfg, hooks.seed_agents)
    base = hooks.recv_from if hooks.recv_from is not None else world.adj
    recv = [[u for u in base[v] if v in ids[u].allowed_out_edges] for v in range(len(world.adj))]
    schema = el.HardConstraintSpec(action_channel_mode="structured_kv_only")
    return replace(hooks, recv_from=recv,
                   channel_filter=_compose(hooks.channel_filter,
                                           lambda text, i, t: el.apply_hard_constraints(text, i, t, schema)))


def build_episode(arm: str, world: sl.World, swarm_cfg: sl.Config, ctx: RunContext) -> EpisodeSetup:
    if arm not in DEFAULT_ARMS + ALL_CONVENTIONAL_ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {DEFAULT_ARMS + ALL_CONVENTIONAL_ARMS}")
    cfg = equalized_swarm_cfg(arm, swarm_cfg, ctx.bcfg) if ctx.equalize else swarm_cfg
    hooks = _base_hooks(ctx, world, cfg.N)
    if arm == STATIC_ARM:
        hooks = _static_zta(hooks, world, ctx)
    weigh = ctx.bcfg.strip_degrades_weight
    hooks = replace(hooks, gateway_factory=make_gateway_factory(ctx),
                    edge_weight_modifier=make_edge_weight_modifier(ctx.bcfg) if weigh else None)
    agent = ctx.agent_factory(cfg, world)
    if weigh:
        agent = RationaleAgent(agent, sensor_classes(world.topology, cfg.N, world.seed, ctx.bcfg.noisy_frac))
    return EpisodeSetup("no_defense" if arm == STATIC_ARM else arm, cfg, hooks, agent)
