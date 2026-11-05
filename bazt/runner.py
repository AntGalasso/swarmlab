#!/usr/bin/env python3
"""bazt.runner -- run the legacy arms, static_zta and the BAZT arms on identical worlds (strict CRN).

    python -m bazt.runner --mode mock --split tune --seeds 12        # smoke check, tune seeds only
    python -m bazt.runner --mode mock --split tune --budget-sweep    # cost-matched budget sweep

COMPLIANCE
  * Everything is written under results/bazt/. Legacy artifacts (results/per_seed_main.csv, ...) are never touched.
  * `--mode llm` goes through swarmlab's own kickoff guard, unchanged: before SWARMLAB_KICKOFF_UTC it exits
    unconditionally; afterwards it needs `--allow-live-llm`. The provider (openrouter | adaption) is chosen by
    $SWARMLAB_PROVIDER inside llm_client, so no code edit is needed on sprint day.
  * `--split main|heldout` needs `--confirm-frozen-config`: BaztConfig must be frozen on tune seeds first.
  * Mock numbers are pipeline checks, never findings.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import json
import os
import sys
import time
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

import numpy as np

import env_ladder as el
import swarmlab as sl
from bazt.arms import BAZT_ARMS, DEFAULT_ARMS, RunContext, build_episode
from bazt.config import BUDGET_MODES, BaztConfig
from bazt.metrics import BAZT_METRICS, add_paired_metrics, legacy_spend_columns

DEFAULT_OUT = os.path.join("results", "bazt")
META_COLS = ["topology", "regime", "policy", "seed", "N", "budget_mode", "B"]
CONSTRAINT_COLS = ["constraint_blocked_sends", "message_drop_rate"]
COLUMNS = META_COLS + list(sl.METRICS) + sl.EXTRA_COLS + BAZT_METRICS + CONSTRAINT_COLS   # sl.METRICS is NOT modified
EVASION_ADAPTERS = ("none", "naive", "compressed", "live_cot")
SWEEP_B = (4, 8, 12, 16, 24)


# ================================================================================ compliance / LLM
def _enforce_kickoff_guard(allow_live_llm: bool, mode: str = "llm") -> None:
    """swarmlab's own guard, unchanged: SystemExit before kickoff (always) or without --allow-live-llm."""
    sl._kickoff_guard(SimpleNamespace(mode=mode, allow_live_llm=allow_live_llm))


def _make_backend(args):
    """Seam for tests. Provider and key come from the environment (llm_client), never from this file."""
    from llm_client import SQLiteLLMCache, make_llm_policy
    return make_llm_policy(args.model, SQLiteLLMCache(args.cache))


class AdapterLLMAgentPolicy(sl.LLMAgentPolicy):
    """LLM agent whose prompts pass through a CoTPayloadAdapter (Kunal's live hook, post-kickoff only)."""

    def __init__(self, backend, payload: str, adapter: el.CoTPayloadAdapter, constraint: el.HardConstraintSpec):
        super().__init__(backend, payload)
        self.adapter, self.constraint = adapter, constraint

    def step(self, ctxs):
        reqs = []
        for c in ctxs:
            system, user, salt = self._prompts(c)
            infected = bool(c.is_seed or any(self.adapter.detect_ground_truth_payload(m)
                                             for m in ([c.own_prev] if c.own_prev else []) + list(c.inbox)))
            system, user = self.adapter.transform_agent_step_prompt(system, user, infected, self.constraint)
            reqs.append((system, user, salt))
        return self.backend.generate(reqs)


def make_adapter(name: str) -> Optional[el.CoTPayloadAdapter]:
    if name == "none":
        return None
    if name in el.PAYLOAD_FORMS:
        return el.TemplateCoTPayloadAdapter(payload_form=name)
    from live_cot_adapter import LiveCoTAdapter            # lazy: only the live_cot path needs it
    return LiveCoTAdapter()


def make_agent_factory(args, adapter, constraint, backend=None):
    if args.mode == "llm":
        if adapter is not None:
            return lambda cfg, world: AdapterLLMAgentPolicy(backend, args.payload, adapter, constraint)
        return lambda cfg, world: sl.LLMAgentPolicy(backend, payload=args.payload)
    if adapter is not None:
        return lambda cfg, world: el.ConstrainedMockAgentPolicy(adapter, constraint, cfg)
    return lambda cfg, world: sl.MockAgentPolicy(cfg)


# ======================================================================================== the grid
def resolve_seeds(split: Optional[str], n: Optional[int], seed_start: int) -> List[int]:
    if split:
        lo, hi = sl.SPLITS[split]
        full = list(range(lo, hi + 1))
        return full if n is None else full[:n]
    return list(range(seed_start, seed_start + (n or 30)))


def run_bazt_grid(topologies: Sequence[str], regimes: Sequence[str], arms: Sequence[str], seeds: Sequence[int],
                  swarm_cfg: sl.Config, ctx: RunContext, progress: bool = True,
                  collect_audit: bool = False) -> Tuple[List[dict], List[dict]]:
    rows: List[dict] = []
    audit_rows: List[dict] = []
    if collect_audit:
        ctx.sink = []
    total = len(topologies) * len(seeds) * len(regimes) * len(arms)
    done, t0 = 0, time.time()
    for topo, seed in itertools.product(topologies, seeds):
        world = sl.make_world(topo, seed, swarm_cfg)
        for regime in regimes:
            decoys = sl.pick_decoys(world, swarm_cfg.decoys_in_decoy_regime if regime == "decoy" else 0)
            for arm in arms:
                setup = build_episode(arm, world, swarm_cfg, ctx)
                r = sl.run_episode(world, setup.policy, setup.cfg, setup.agent, decoys, hooks=setup.hooks)
                for k in ("ever_curve", "cum_decoy_quarantines", "cum_true_quarantines"):
                    r.pop(k)
                row = {"topology": topo, "regime": regime, "policy": arm, "seed": seed, "N": swarm_cfg.N,
                       "budget_mode": ctx.bcfg.budget_mode, "B": ctx.bcfg.B, **r}
                if arm not in BAZT_ARMS:
                    legacy_spend_columns(row, ctx.bcfg.cost_isolate, ctx.bcfg.cost_sanitize)
                rows.append(row)
                if ctx.sink:
                    gw = ctx.sink.pop()
                    audit_rows += [{"topology": topo, "regime": regime, "arm": arm, "seed": seed, **a}
                                   for a in gw.audit.rows()]
                done += 1
        if progress:
            print(f"  {done}/{total} episodes  ({time.time() - t0:.0f}s)", file=sys.stderr)
    return rows, audit_rows


def finalize(rows: List[dict], extra_keys: Tuple[str, ...] = ()) -> None:
    keys = ("topology", "regime", "seed") + extra_keys
    sl.add_deltas(rows, keys=keys)
    add_paired_metrics(rows, keys=keys)


def run_budget_sweep(topologies, regimes, arms, seeds, swarm_cfg, base_ctx: RunContext, values=SWEEP_B,
                     progress=True) -> List[dict]:
    """One grid per pooled budget B, legacy arms cost-matched (equalize forced on)."""
    rows: List[dict] = []
    for b in values:
        ctx = dataclasses.replace(base_ctx, bcfg=dataclasses.replace(base_ctx.bcfg, B=int(b), budget_mode="pooled"),
                                  equalize=True)
        rows += run_bazt_grid(topologies, regimes, arms, seeds, swarm_cfg, ctx, progress=progress)[0]
    finalize(rows, ("B",))
    return rows


# ==================================================================================== output / print
def write_rows(rows: Sequence[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)


def _mean(rows: Sequence[dict], key: str) -> float:
    xs = [r[key] for r in rows if key in r and r[key] == r[key]]
    return float(np.mean(xs)) if xs else float("nan")


SUMMARY = [("AUC_ever", "auc_ever_infected", 9, 3), ("MAE", "task_mae", 7, 2), ("spend", "enforcement_spend", 7, 2),
           ("harm", "harm_exposure_auc", 7, 3), ("thruput", "message_throughput", 9, 3),
           ("falseAct", "false_action_rate", 9, 3), ("$/prev", "cost_per_prevented_compromise", 8, 2),
           ("viol", "budget_violations", 6, 0)]


def format_summary(rows: Sequence[dict], regime: str = "decoy") -> str:
    sub = [r for r in rows if r["regime"] == regime]
    pol_order = [a for a in DEFAULT_ARMS if any(r["policy"] == a for r in sub)]
    lines = [f"\n== mean over topologies x seeds | regime={regime} | n_rows={len(sub)}",
             f"{'arm':<18}" + "".join(f"{h:>{w}}" for h, _, w, _ in SUMMARY)]
    for pol in pol_order:
        rs = [r for r in sub if r["policy"] == pol]
        lines.append(f"{pol:<18}" + "".join(f"{_mean(rs, k):>{w}.{d}f}" for _, k, w, d in SUMMARY))
    return "\n".join(lines)


def run_meta(args, bcfg: BaztConfig, swarm_cfg: sl.Config, seeds: Sequence[int], arms: Sequence[str]) -> dict:
    cfg_dict = {k: v for k, v in dataclasses.asdict(bcfg).items() if k != "probe_score_hook"}
    return {"module": "bazt.runner", "mode": args.mode, "split": args.split, "seeds": [seeds[0], seeds[-1]],
            "n_seeds": len(seeds), "arms": list(arms), "topologies": args.topologies, "regimes": args.regimes,
            "entry_source": args.entry_source, "evasion_adapter": args.evasion_adapter,
            "equalize_budget": bool(args.equalize_budget), "bazt_config": cfg_dict,
            "swarm_config": {"N": swarm_cfg.N, "T": swarm_cfg.T, "K": swarm_cfg.K, "B_san": swarm_cfg.B_san},
            "note": "mock numbers are pipeline checks, not findings" if args.mode == "mock" else "live run"}


# ============================================================================================= CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="BAZT runner (writes only under results/bazt/)")
    ap.add_argument("--mode", choices=["mock", "llm"], default="mock")
    ap.add_argument("--allow-live-llm", action="store_true")
    ap.add_argument("--split", choices=list(sl.SPLITS), default="tune")
    ap.add_argument("--seeds", type=int, default=None, help="use only the first N seeds of the split")
    ap.add_argument("--seed-start", type=int, default=1, help="first seed when no --split is meaningful")
    ap.add_argument("--confirm-frozen-config", action="store_true",
                    help="required for main/heldout: certifies BaztConfig was frozen on tune seeds")
    ap.add_argument("--topologies", nargs="+", default=sl.TOPOLOGIES)
    ap.add_argument("--regimes", nargs="+", default=["clean", "decoy"])
    ap.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    ap.add_argument("--entry-source", choices=["patient_zero", "fake_web_jsonl"], default="patient_zero")
    ap.add_argument("--corpus", default=el.DEFAULT_CORPUS)
    ap.add_argument("--entry-readers", type=int, default=2)
    ap.add_argument("--evasion-adapter", choices=EVASION_ADAPTERS, default="none")
    ap.add_argument("--channel-mode", choices=el.CHANNEL_MODES, default="free_note")
    ap.add_argument("--movement-mode", choices=el.MOVEMENT_MODES, default="unrestricted")
    ap.add_argument("--budget-mode", choices=BUDGET_MODES, default="pooled")
    ap.add_argument("--B", type=int, default=12)
    ap.add_argument("--K-hard", type=int, default=3)
    ap.add_argument("--B-soft", type=int, default=6)
    ap.add_argument("--equalize-budget", action="store_true",
                    help="give legacy arms the same total cost budget (hard-only: K = B // cost_isolate)")
    ap.add_argument("--budget-sweep", action="store_true", help="sweep pooled B (legacy arms cost-matched)")
    ap.add_argument("--sweep-B", nargs="+", type=int, default=list(SWEEP_B))
    ap.add_argument("--no-strip-cost", action="store_true", help="disable the filtering utility cost")
    ap.add_argument("--model", default="qwen/qwen-2.5-7b-instruct")
    ap.add_argument("--payload", choices=sl.PAYLOADS, default="canary_tag")
    ap.add_argument("--cache", default="llm_cache.sqlite")
    ap.add_argument("--write-audit", action="store_true", help="also write the per-decision audit CSV")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--out-file", default=None)
    ap.add_argument("--quiet", action="store_true")
    return ap


def _out_name(args) -> str:
    if args.out_file:
        return args.out_file
    if args.budget_sweep:
        return f"budget_sweep_{args.split}.csv"
    return "smoke_tune_check.csv" if args.split == "tune" else f"per_seed_{args.split}.csv"


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "llm":                                   # first thing, before any world is built
        _enforce_kickoff_guard(args.allow_live_llm)
    if args.split in ("main", "heldout") and not args.confirm_frozen_config:
        sys.exit(f"Refusing: --split {args.split} needs --confirm-frozen-config (tune BaztConfig on tune seeds only).")
    name = _out_name(args)
    if os.path.basename(name) in ("per_seed_main.csv", "per_seed_heldout.csv") and \
            os.path.basename(os.path.normpath(args.out)) != "bazt":
        sys.exit("Refusing: BAZT never writes the legacy per_seed_main/heldout files outside a 'bazt' directory.")
    unknown = [a for a in args.arms if a not in DEFAULT_ARMS]
    if unknown:
        sys.exit(f"unknown arms: {unknown}")

    seeds = resolve_seeds(args.split, args.seeds, args.seed_start)
    bcfg = BaztConfig(budget_mode=args.budget_mode, B=args.B, K_hard=args.K_hard, B_soft=args.B_soft,
                      strip_degrades_weight=not args.no_strip_cost)
    swarm_cfg = sl.Config()
    adapter = make_adapter(args.evasion_adapter)
    constraint = None
    if adapter is not None or args.channel_mode != "free_note" or args.movement_mode != "unrestricted":
        constraint = el.HardConstraintSpec(movement_direction_mode=args.movement_mode,
                                           action_channel_mode=args.channel_mode)
    entry_factory = None
    if args.entry_source == "fake_web_jsonl":
        entry_factory = el.make_entry_hooks_factory(corpus=args.corpus, n_readers=args.entry_readers)
    backend = _make_backend(args) if args.mode == "llm" else None
    agent_factory = make_agent_factory(args, adapter, constraint or el.HardConstraintSpec(), backend)
    ctx = RunContext(bcfg=bcfg, agent_factory=agent_factory, equalize=args.equalize_budget,
                     entry_factory=entry_factory, adapter=adapter, constraint=constraint)

    print(f"mode={args.mode} split={args.split} seeds={seeds[0]}..{seeds[-1]} arms={len(args.arms)} "
          f"topologies={list(args.topologies)} regimes={args.regimes} budget={bcfg.budget_mode}/B={bcfg.B}",
          file=sys.stderr)
    try:
        audit: List[dict] = []
        if args.budget_sweep:
            rows = run_budget_sweep(args.topologies, args.regimes, args.arms, seeds, swarm_cfg, ctx,
                                    values=args.sweep_B, progress=not args.quiet)
        else:
            rows, audit = run_bazt_grid(args.topologies, args.regimes, args.arms, seeds, swarm_cfg, ctx,
                                        progress=not args.quiet, collect_audit=args.write_audit)
            finalize(rows)
    finally:
        if backend is not None:
            print("llm cache:", backend.cache.stats(), file=sys.stderr)
            backend.close()
            backend.cache.close()

    path = os.path.join(args.out, name)
    write_rows(rows, path)
    with open(path[:-4] + ".meta.json", "w") as f:
        json.dump(run_meta(args, bcfg, swarm_cfg, seeds, args.arms), f, indent=2)
    if audit:
        apath = path[:-4] + ".audit.csv"
        with open(apath, "w", newline="") as f:
            from bazt.audit import AUDIT_FIELDS
            wr = csv.DictWriter(f, fieldnames=["topology", "regime", "arm", "seed"] + AUDIT_FIELDS)
            wr.writeheader()
            wr.writerows(audit)
        print("wrote", apath)
    print("wrote", path)
    if "decoy" in args.regimes:
        print(format_summary(rows, "decoy"))
    if "clean" in args.regimes:
        print(format_summary(rows, "clean"))
    if args.mode == "mock":
        print("\nNOTE: mock agents -> pipeline check ONLY. No number above is a finding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
