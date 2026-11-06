"""Clean-only alpha = 0.02 threshold calibration for the conventional ladder and BAZT.

Pre-registered rule (``configs/sprint_preregistered.json``): every UNCAPPED comparison arm must deny at most
``alpha_clean_fpr = 0.02`` of BENIGN deliveries. Nothing here ever sees an attack: calibration runs ONLY
attack-free episodes, so no attack-specific information can leak into a threshold.

Calibrated knobs (one monotone aggressiveness dial per arm, largest value that still satisfies alpha wins):

    arm_B              tau_alpha       monitor/judge score above which a broadcast is blocked
    arm_C              rl_offset       additive offset on the Risk Level (TS > RL grants, so a LOW offset
                                       is aggressive and a HIGH offset is permissive)
    bazt               theta0          density threshold of the budgeted optimizer (+ p_floor)
    gsafeguard_standin tau_alpha       same monitor dial as arm_B

Determinism: mock runs use tune seeds 9001..9030 with CRN streams; LLM runs use 3 seeds x 8 conversations per
cell. The output JSON records a SHA-256 of the config it was calibrated under, so a re-run with a changed
config cannot silently reuse stale thresholds.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import swarmlab as sl
from bazt.config import BaztConfig
from bazt.conventional_zta import ZtaConfig, conventional_bazt_config

DEFAULT_OUT = os.path.join("results", "experiments", "frozen_thresholds.json")
CALIBRATION_SEEDS = (9001, 9030)          # tune split: never graded
CALIBRATION_TOPOLOGIES = ("chain", "small_world")


def config_sha256(config: dict) -> str:
    """Stable hash of a pre-registered config (order-insensitive, floats repr-stable)."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------- measuring
def benign_deny_rate(rows: Sequence[dict]) -> float:
    """Fraction of BENIGN deliveries an arm denied, averaged over episodes (0 when nothing was delivered)."""
    vals = []
    for r in rows:
        den = float(r.get("benign_deliveries", 0.0) or 0.0)
        if den <= 0:
            continue
        vals.append(float(r.get("benign_denied_deliveries", 0.0)) / den)
    return float(np.mean(vals)) if vals else 0.0


def _run_clean(arm: str, cfg: sl.Config, bcfg: BaztConfig, zcfg: ZtaConfig, seeds: Sequence[int],
               topologies: Sequence[str], agent_factory: Callable) -> List[dict]:
    """Attack-free episodes only. `decoys` are absent: calibration must not depend on benign noise either."""
    from bazt.arms import RunContext, build_episode
    clean_cfg = replace(cfg, decoys_in_decoy_regime=0)
    rows = []
    for topo in topologies:
        for seed in seeds:
            world = sl.make_world(topo, seed, clean_cfg)
            ctx = RunContext(bcfg=bcfg, agent_factory=agent_factory, zcfg=zcfg)
            setup = build_episode(arm, world, clean_cfg, ctx)
            # ATTACK-FREE: no patient zero, no decoys. Only benign deliveries can be denied here.
            hooks = replace(setup.hooks, suppress_patient_zero=True, recv_from=None)
            r = sl.run_episode(world, setup.policy, setup.cfg, setup.agent, frozenset(), hooks=hooks)
            r.update(topology=topo, seed=seed)
            rows.append(r)
    return rows


def _deny_counts(rows: List[dict]) -> List[dict]:
    """`benign_deliveries` / `benign_denied_deliveries` are produced by the gateways' extra_metrics."""
    return rows


# ------------------------------------------------------------------------------------- calibration
def _bisect(candidates: Sequence[float], measure: Callable[[float], float], alpha: float,
            aggressive_high: bool) -> Tuple[float, float]:
    """Largest-magnitude candidate whose benign deny rate stays <= alpha.

    `aggressive_high=True` means a LARGER value is more aggressive (tau_alpha, theta0);
    `aggressive_high=False` means a SMALLER value is more aggressive (rl_offset).
    """
    best, best_rate = candidates[0], float("inf")
    for v in candidates:
        rate = measure(v)
        ok = rate <= alpha + 1e-12
        if ok and (best_rate == float("inf") or (v > best if aggressive_high else v < best)):
            best, best_rate = v, rate
    if best_rate == float("inf"):                     # nothing satisfied alpha: take the MOST PERMISSIVE end
        best = max(candidates)
        best_rate = measure(best)
    return float(best), float(best_rate)


def calibrate_thresholds(config: dict, platform: str = "mock", split: str = "tune",
                         out_path: Optional[str] = None, verbose: bool = True,
                         seeds: Optional[Sequence[int]] = None,
                         topologies: Optional[Sequence[str]] = None) -> dict:
    """Calibrate the uncapped arms so each denies <= alpha_clean_fpr of benign deliveries.

    Returns the frozen-threshold document (also written to `out_path`). Only clean episodes are ever built.
    """
    if platform not in ("mock", "llm"):
        raise ValueError(f"platform must be 'mock' or 'llm', got {platform!r}")
    alpha = float(config["alpha_clean_fpr"])
    ni_ins = int(config["n_insiders"])
    rate_limit = int(config["rate_limit_per_identity"])
    horizon_constant = float(config["horizon_constant"])

    if split != "tune":
        raise SystemExit("calibration must run on the tune split; main/heldout are evaluated once and never tuned")
    lo, hi = config.get("splits", {}).get("tune", list(CALIBRATION_SEEDS))
    seeds = list(seeds) if seeds is not None else list(range(int(lo), int(hi) + 1))
    if platform == "llm" and seeds is None:            # 3 seeds x 8 conversations per cell
        seeds = seeds[:3]

    cfg = sl.Config(N=12, T=max(4, int(round(horizon_constant * 2))))
    topologies = list(topologies) if topologies is not None else list(CALIBRATION_TOPOLOGIES)
    base_z = ZtaConfig(alpha_clean_fpr=alpha, rate_limit_per_identity=rate_limit)
    agent_factory = (lambda c, _w: sl.MockAgentPolicy(c))

    doc: Dict[str, object] = {
        "platform": platform, "split": split, "alpha_clean_fpr": alpha,
        "calibration_seeds": [seeds[0], seeds[-1]], "n_seeds": len(seeds),
        "topologies": topologies, "n_insiders": ni_ins, "horizon_constant": horizon_constant,
        "method": "clean-only, largest-permissible aggressiveness per arm",
        "config_sha256": config_sha256(config),
        "thresholds": {}, "measured_benign_deny_rate": {},
    }

    # ---- arm_B / gsafeguard: the monitor-score dial (larger tau = less aggressive) -------------------
    tau_grid = [round(v, 4) for v in np.linspace(0.05, 0.99, 20)]

    def tau_rate(arm: str, tau: float) -> float:
        z = replace(base_z, tau_alpha=float(tau))
        b = conventional_bazt_config(z, None, None)
        rows = _run_clean(arm, cfg, b, z, seeds, topologies, agent_factory)
        return benign_deny_rate(_deny_counts(rows))

    for arm in ("arm_B", "gsafeguard_standin"):
        tau, rate = _bisect(tau_grid, lambda v, a=arm: tau_rate(a, v), alpha, aggressive_high=True)
        doc["thresholds"][arm] = {"tau_alpha": tau}
        doc["measured_benign_deny_rate"][arm] = rate
        if verbose:
            print(f"  {arm:<20} tau_alpha={tau:.3f}  benign deny rate={rate:.4f} (alpha={alpha})")

    # ---- arm_C: the risk-level offset dial (larger offset = less aggressive) ------------------------
    off_grid = [round(v, 4) for v in np.linspace(-0.6, 0.8, 15)]

    def off_rate(off: float) -> float:
        z = replace(base_z, rl_offset=float(off), tau_alpha=float(doc["thresholds"]["arm_B"]["tau_alpha"]))
        b = conventional_bazt_config(z, None, None)
        return benign_deny_rate(_deny_counts(_run_clean("arm_C", cfg, b, z, seeds, topologies, agent_factory)))

    off, c_rate = _bisect(off_grid, off_rate, alpha, aggressive_high=False)
    doc["thresholds"]["arm_C"] = {"rl_offset": off,
                                  "tau_alpha": doc["thresholds"]["arm_B"]["tau_alpha"]}
    doc["measured_benign_deny_rate"]["arm_C"] = c_rate
    if verbose:
        print(f"  {'arm_C':<20} rl_offset={off:.3f}  benign deny rate={c_rate:.4f} (alpha={alpha})")

    # ---- bazt: the pacing density dial (theta0) ----------------------------------------------------
    theta_grid = [round(v, 4) for v in np.linspace(0.15, 12.0, 16)]

    def theta_rate(theta: float) -> float:
        b = BaztConfig(budget_mode="pooled", B=10 ** 6, theta0=float(theta))
        z = replace(base_z, tau_alpha=float(doc["thresholds"]["arm_B"]["tau_alpha"]))
        return benign_deny_rate(_deny_counts(_run_clean("bazt", cfg, b, z, seeds, topologies, agent_factory)))

    theta, b_rate = _bisect(theta_grid, theta_rate, alpha, aggressive_high=False)
    doc["thresholds"]["bazt"] = {"theta0": theta}
    doc["measured_benign_deny_rate"]["bazt"] = b_rate
    if verbose:
        print(f"  {'bazt':<20} theta0={theta:.3f}  benign deny rate={b_rate:.4f} (alpha={alpha})")

    # ---- shared dual-budget envelope (the caps every capped arm is evaluated under) ----------------
    doc["dual_budget_grid"] = config.get("dual_budget_grid", [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, None])

    out = out_path or DEFAULT_OUT
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
    if verbose:
        print("wrote", out)
    return doc


def load_thresholds(path: str = DEFAULT_OUT) -> dict:
    with open(path) as f:
        return json.load(f)
