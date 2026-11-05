"""bazt/runner.py -- BAZT CLI.

    python -m bazt.runner --mode mock --split tune --seeds 12

Writes strictly to `results/bazt/` (never `results/per_seed_main.csv` or
`results/per_seed_heldout.csv`).

`--mode llm` is wired through `swarmlab._enforce_kickoff_guard` +
`llm_client.make_llm_policy` (openrouter / adaption): before kickoff it raises
`SystemExit` unconditionally; after kickoff (`--mode llm --allow-live-llm`) it
works without code edits.  Today (before kickoff) every run is `--mode mock`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from typing import List, Optional

import swarmlab as sl
from swarmlab import (
    Config, EpisodeHooks, run_episode, write_outputs, BAZT_POLICIES,
    _enforce_kickoff_guard, _seeds, make_world, pick_decoys,
)
from bazt.arms import (
    ALL_ARMS, BAZT_ARMS, COST_MATCHED_LEGACY, STATIC_ZTA,
    bazt_config, equalized_K, is_bazt_arm, static_zta_spec,
)
from bazt.config import BaztConfig
from bazt.gateway import EnforcementGateway
from bazt.roles import assign_roles

import env_ladder as el


# ------------------------------------------------------------------ hooks factory
def make_hooks(policy, seed, cfg, entry_source, corpus, entry_readers, B):
    """Policy-specific `EpisodeHooks`.  Default-off: legacy arms get `None`."""
    seed_agents: frozenset = frozenset()
    suppress_patient_zero = False
    recv_from = None
    channel_filter = None
    gateway_factory = None
    edge_weight_modifier = None

    if entry_source == "fake_web_jsonl":
        factory = el.make_entry_hooks_factory(corpus=corpus, n_readers=entry_readers)
        eh = factory(seed, cfg.N)
        seed_agents = eh.seed_agents
        suppress_patient_zero = bool(eh.suppress_patient_zero)

    if policy == STATIC_ZTA:
        spec = static_zta_spec()
        # recv_from/channel_filter are filled by the caller (needs world.adj).
        recv_from = "__STATIC_ZTA__"
        channel_filter = el.make_channel_filter(el.ladder_spec(1), spec)

    if policy in BAZT_POLICIES:
        bazt_cfg = bazt_config(policy, B=B)
        gw_cell = {"gw": None}

        def gateway_factory(p2, world2, cfg2, base_adj2, decoys2, hooks2):
            if p2 not in BAZT_POLICIES:
                return None
            gw = _build_gateway(p2, world2, cfg2, base_adj2, decoys2, hooks2, seed, B)
            gw_cell["gw"] = gw
            return gw

        def edge_weight_modifier(j, i, t, msg):
            gw = gw_cell["gw"]
            if gw is None or not gw.bcfg.strip_degrades_weight:
                return 1.0
            return gw.bcfg.stripped_edge_weight if j in gw.stripped_messages else 1.0

    return EpisodeHooks(
        seed_agents=seed_agents,
        suppress_patient_zero=suppress_patient_zero,
        recv_from=recv_from,
        channel_filter=channel_filter,
        gateway_factory=gateway_factory,
        edge_weight_modifier=edge_weight_modifier,
    )


def _build_gateway(policy, world, cfg, base_adj, decoys, hooks, seed, B):
    bazt_cfg = bazt_config(policy, B=B)
    ingress_nodes: set = set(hooks.seed_agents) if (hooks and hooks.seed_agents) else set()
    roles = assign_roles(cfg.N, world.topology, seed, base_adj, ingress_nodes, bazt_cfg.n_privileged)
    if ingress_nodes:
        entry_seed_nodes = set(ingress_nodes)
    else:
        entry_seed_nodes = {r.node_id for r in roles if r.role == "ingress"}
    return EnforcementGateway(cfg, bazt_cfg, world.topology, seed, base_adj,
                              roles, ingress_nodes, entry_seed_nodes)


# ------------------------------------------------------------------ runner
def run(args) -> int:
    if args.mode == "llm":
        _enforce_kickoff_guard(args.allow_live_llm)

    topos: List[str] = list(args.topologies)
    regimes: List[str] = list(args.regimes)
    policies: List[str] = list(args.policies) if args.policies else list(BAZT_ARMS)
    seeds: List[int] = _seeds(args)
    if not seeds:
        seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    B = int(args.B)
    equalize_budget = bool(args.equalize_budget)

    backend = None
    if args.mode == "llm":
        from llm_client import SQLiteLLMCache, make_llm_policy
        backend = make_llm_policy(args.model, SQLiteLLMCache(args.cache))

    def agent_factory(c):
        if backend is not None:
            return sl.LLMAgentPolicy(backend, payload=args.payload)
        return sl.MockAgentPolicy(c)

    rows: List[dict] = []
    episodes: List[dict] = []
    t0 = time.time()
    done = 0
    total = len(topos) * len(regimes) * len(seeds) * len(policies)

    for topo in topos:
        for seed in seeds:
            world = make_world(topo, seed, Config())
            base_adj = world.adj
            # static_zta movement mask (shared across seeds for this topology)
            static_mask = None
            if STATIC_ZTA in policies:
                static_mask = el.routing_mask(base_adj, static_zta_spec().movement_direction_mode)
            for regime in regimes:
                decoys = pick_decoys(world, 3 if regime == "decoy" else 0)
                for pol in policies:
                    hooks = None
                    if pol == STATIC_ZTA:
                        hooks = EpisodeHooks(
                            seed_agents=frozenset(), suppress_patient_zero=False,
                            recv_from=static_mask.recv_from,
                            channel_filter=static_zta_spec() and el.make_channel_filter(
                                el.ladder_spec(1), static_zta_spec()),
                        )
                    elif pol in BAZT_POLICIES or pol in COST_MATCHED_LEGACY:
                        hooks = make_hooks(pol, seed, Config(), args.entry_source,
                                           args.corpus, args.entry_readers, B)

                    c = Config()
                    if pol in COST_MATCHED_LEGACY and equalize_budget:
                        ref = bazt_config(pol, B=B) if pol in BAZT_POLICIES else BaztConfig(B=B)
                        c = replace(Config(), K=equalized_K(ref))

                    r = run_episode(world, pol, c, agent_factory(c), decoys, hooks=hooks)
                    curve = r.pop("ever_curve", None)
                    cd = r.pop("cum_decoy_quarantines", None)
                    ct = r.pop("cum_true_quarantines", None)
                    meta = dict(topology=topo, regime=regime, policy=pol, seed=seed, B=B)
                    rows.append({**meta, **r})
                    episodes.append({**meta, **r,
                                     "ever_curve": curve, "cum_decoy_quarantines": cd,
                                     "cum_true_quarantines": ct})
                    done += 1
                    if done % 25 == 0:
                        print(f"  {done}/{total} episodes ({time.time()-t0:.0f}s)", file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    write_outputs(rows, episodes, args.out, tag=tag, split=args.split)
    print(f"wrote {args.out}/per_seed_{args.split}{tag}.csv  ({done} episodes)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="BAZT defence runner (mock before kickoff)")
    ap.add_argument("--mode", choices=["mock", "llm"], default="mock")
    ap.add_argument("--allow-live-llm", action="store_true", default=False)
    ap.add_argument("--split", choices=["tune", "main", "heldout"], default="tune")
    ap.add_argument("--budget-sweep", action="store_true",
                    help="sweep B in (3, 6, 12, 24), one output file per B")
    ap.add_argument("--entry-source", choices=["patient_zero", "fake_web_jsonl"],
                    default="patient_zero")
    ap.add_argument("--corpus", default="data/fake_web_corpus.jsonl")
    ap.add_argument("--entry-readers", type=int, default=2)
    ap.add_argument("--evasion-adapter", choices=["none", "naive", "compressed", "live_cot"],
                    default="none")
    ap.add_argument("--model", default="qwen/qwen-2.5-7b-instruct")
    ap.add_argument("--cache", default="llm_cache.sqlite")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--topologies", nargs="+", default=["small_world", "star_hub", "chain"])
    ap.add_argument("--regimes", nargs="+", default=["decoy"])
    ap.add_argument("--policies", nargs="+", default=None)
    ap.add_argument("--B", type=int, default=12, help="BAZT pooled budget")
    ap.add_argument("--equalize-budget", action="store_true",
                    help="map pooled B -> K = B // cost_isolate for hard-only baselines")
    ap.add_argument("--out", default="results/bazt")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    if args.budget_sweep:
        rc = 0
        for B in (3, 6, 12, 24):
            args2 = argparse.Namespace(**vars(args))
            args2.B = B
            args2.tag = f"{args.tag}_B{B}" if args.tag else f"B{B}"
            rc |= run(args2)
        return rc
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
