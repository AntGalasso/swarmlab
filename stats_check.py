#!/usr/bin/env python3
"""stats_check.py -- PRE-EXISTING SCAFFOLDING. Pre-registered tests for H1-H4.

Reads results/per_seed.csv written by swarmlab.py. Paired by seed. All tests: two-sided Wilcoxon
signed-rank, Holm correction within each family, 1,000-resample paired-seed bootstrap 95% CI of the
mean paired difference. rank-biserial keeps its sign (positive: first-named condition is LOWER, i.e. better,
for damage metrics; the sign convention is printed next to every table).

  python stats_check.py --file results/per_seed.csv
  python stats_check.py --file results/per_seed.csv --h4 results/model_a.csv results/model_b.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats

NUM = ["auc_ever_infected", "auc_active_unquarantined", "final_ever_infected_rate", "total_deviated_messages",
       "task_mae", "quarantines_used", "collateral_healthy_quarantined", "decoy_quarantines_wasted",
       "time_to_containment", "delta_abs", "delta_rel"]
ALPHA = 0.05


def load(path: str) -> List[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in NUM + ["seed"]:
                if k in r and r[k] != "":
                    r[k] = float(r[k])
            rows.append(r)
    return rows


def vec(rows, topo, regime, policy, metric) -> Dict[int, float]:
    return {int(r["seed"]): r[metric] for r in rows
            if r["topology"] == topo and r["regime"] == regime and r["policy"] == policy and metric in r and r[metric] == r[metric]}


def paired(a: Dict[int, float], b: Dict[int, float]) -> Tuple[np.ndarray, np.ndarray]:
    ks = sorted(set(a) & set(b))
    return np.array([a[k] for k in ks]), np.array([b[k] for k in ks])


def wilcoxon_rrb(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, int]:
    """Two-sided Wilcoxon on a-b. rrb>0 means a is LOWER than b more often than higher. Returns (p, rrb, n_nonzero)."""
    d = b - a  # positive when a is lower
    nz = d[d != 0]
    if len(nz) < 1:
        return 1.0, 0.0, 0
    ranks = stats.rankdata(np.abs(nz))
    wp, wm = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    rrb = float((wp - wm) / (wp + wm))
    try:
        p = float(stats.wilcoxon(nz, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        p = 1.0
    return p, rrb, len(nz)


def boot_ci(d: np.ndarray, B: int = 1000, seed: int = 42) -> Tuple[float, float]:
    if len(d) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, adj, run = len(items), {}, 0.0
    for k, (name, p) in enumerate(items):
        run = max(run, min(1.0, (m - k) * p))
        adj[name] = run
    return adj


def compare(rows, topo, regime, pol_a, pol_b, metric) -> dict:
    a, b = paired(vec(rows, topo, regime, pol_a, metric), vec(rows, topo, regime, pol_b, metric))
    p, rrb, nnz = wilcoxon_rrb(a, b)
    lo, hi = boot_ci(a - b)
    return {"a": pol_a, "b": pol_b, "metric": metric, "n": int(len(a)), "mean_diff_a_minus_b": float((a - b).mean()) if len(a) else float("nan"),
            "ci95": [lo, hi], "p": p, "rrb": rrb, "n_nonzero": nnz}


# --------------------------------------------------------------------------------------- H1
def h1(rows, topos, regime="decoy"):
    """Fixed windows differ in infection AUC; supported if Holm p<0.05 for >=1 pair on >=2 topologies."""
    res, hits = {}, 0
    for t in topos:
        fam = {f"{x}_vs_{y}": compare(rows, t, regime, x, y, "auc_ever_infected")
               for x, y in (("fixed_early", "fixed_mid"), ("fixed_mid", "fixed_late"), ("fixed_early", "fixed_late"))}
        adj = holm({k: v["p"] for k, v in fam.items()})
        for k in fam:
            fam[k]["p_holm"] = adj[k]
        res[t] = fam
        hits += any(v["p_holm"] < ALPHA for v in fam.values())
    return {"hypothesis": "H1 timing", "topologies_with_significant_pair": hits, "supported": hits >= 2, "detail": res}


# --------------------------------------------------------------------------------------- H2
def h2(rows, topos):
    """Decoys degrade score_only and fixed_early (AUC increase clean->decoy), paired by seed."""
    res, fam = {}, {}
    for t in topos:
        for pol in ("score_only", "fixed_early"):
            a, b = paired(vec(rows, t, "decoy", pol, "auc_ever_infected"), vec(rows, t, "clean", pol, "auc_ever_infected"))
            d = a - b  # positive = worse under decoys
            nz = d[d != 0]
            p = float(stats.wilcoxon(nz).pvalue) if len(nz) else 1.0
            lo, hi = boot_ci(d)
            fam[f"{t}:{pol}"] = {"n": int(len(d)), "mean_increase": float(d.mean()) if len(d) else float("nan"), "ci95": [lo, hi], "p": p}
    adj = holm({k: v["p"] for k, v in fam.items()})
    for k in fam:
        fam[k]["p_holm"] = adj[k]
        fam[k]["degraded"] = bool(fam[k]["mean_increase"] > 0 and adj[k] < ALPHA)
    sup = sum(v["degraded"] for v in fam.values())
    return {"hypothesis": "H2 budget exhaustion", "n_significant_degradations": sup, "n_tests": len(fam),
            "supported": sup >= max(1, len(fam) // 2), "detail": fam}


# --------------------------------------------------------------------------------------- H3
def h3(rows, topos, regime="decoy", rivals=("score_only", "centrality_only", "random_capped", "fixed_early", "fixed_mid", "fixed_late")):
    """risk_aware lower AUC at equal-or-lower MAE than every rival. Holm within (topology, metric)."""
    out = {}
    for t in topos:
        per = {}
        for metric in ("auc_ever_infected", "task_mae"):
            fam = {r: compare(rows, t, regime, "risk_aware", r, metric) for r in rivals}
            adj = holm({k: v["p"] for k, v in fam.items()})
            for k in fam:
                fam[k]["p_holm"] = adj[k]
            per[metric] = fam
        verdict = {}
        for r in rivals:
            auc, mae = per["auc_ever_infected"][r], per["task_mae"][r]
            better_auc = auc["mean_diff_a_minus_b"] < 0 and auc["p_holm"] < ALPHA
            mae_ok = not (mae["mean_diff_a_minus_b"] > 0 and mae["p_holm"] < ALPHA)
            verdict[r] = {"lower_auc": bool(better_auc), "mae_not_worse": bool(mae_ok), "pareto_win": bool(better_auc and mae_ok)}
        out[t] = {"tests": per, "verdict": verdict}
    wins = sum(v["pareto_win"] for t in out.values() for v in t["verdict"].values())
    total = sum(len(t["verdict"]) for t in out.values())
    return {"hypothesis": "H3 risk-aware Pareto gain", "pareto_wins": wins, "comparisons": total,
            "supported": wins == total, "detail": out}


# --------------------------------------------------------------------------------------- H4
def h4(file_a: str, file_b: str, regime="decoy", metric="auc_ever_infected", threshold=0.8):
    """Spearman rho between policy rankings (mean metric, pooled over topologies) of two model arms."""
    def means(path):
        rows, acc = load(path), defaultdict(list)
        for r in rows:
            if r["regime"] == regime and metric in r and r[metric] == r[metric]:
                acc[r["policy"]].append(r[metric])
        return {k: float(np.mean(v)) for k, v in acc.items()}
    ma, mb = means(file_a), means(file_b)
    pols = sorted(set(ma) & set(mb))
    rho, p = stats.spearmanr([ma[k] for k in pols], [mb[k] for k in pols])
    inv = [(x, y) for i, x in enumerate(pols) for y in pols[i + 1:]
           if (ma[x] - ma[y]) * (mb[x] - mb[y]) < 0]
    return {"hypothesis": "H4 cheap stand-in fidelity", "rho": float(rho), "p": float(p), "n_policies": len(pols),
            "supported": bool(rho >= threshold), "rank_inversions": inv}


# ----------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="results/per_seed.csv")
    ap.add_argument("--out", default="results/stats_report.json")
    ap.add_argument("--h4", nargs=2, default=None, metavar=("MODEL_A_CSV", "MODEL_B_CSV"))
    args = ap.parse_args(argv)
    rows = load(args.file)
    topos = sorted({r["topology"] for r in rows})
    rep = {"h1": h1(rows, topos), "h2": h2(rows, topos) if {"clean", "decoy"} <= {r["regime"] for r in rows} else None,
           "h3": h3(rows, topos)}
    if args.h4:
        rep["h4"] = h4(*args.h4)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(rep, open(args.out, "w"), indent=2)

    print("Sign convention: mean_diff = a - b on a DAMAGE metric; negative = a better. rrb>0 = a lower more often.")
    print(f"\nH1 (window timing, decoy regime): significant pair on {rep['h1']['topologies_with_significant_pair']} topologies -> supported={rep['h1']['supported']}")
    for t, fam in rep["h1"]["detail"].items():
        for k, v in fam.items():
            print(f"  {t:<12}{k:<18} diff={v['mean_diff_a_minus_b']:+.3f} CI[{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] p={v['p']:.3g} holm={v['p_holm']:.3g} rrb={v['rrb']:+.2f}")
    if rep["h2"]:
        print(f"\nH2 (decoy degradation): {rep['h2']['n_significant_degradations']}/{rep['h2']['n_tests']} significant -> supported={rep['h2']['supported']}")
        for k, v in rep["h2"]["detail"].items():
            print(f"  {k:<26} +AUC={v['mean_increase']:+.3f} CI[{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] holm={v['p_holm']:.3g}")
    print(f"\nH3 (risk_aware vs rivals, decoy): Pareto wins {rep['h3']['pareto_wins']}/{rep['h3']['comparisons']} -> supported={rep['h3']['supported']}")
    for t, d in rep["h3"]["detail"].items():
        for r, v in d["verdict"].items():
            auc = d["tests"]["auc_ever_infected"][r]
            mae = d["tests"]["task_mae"][r]
            print(f"  {t:<12}vs {r:<16} dAUC={auc['mean_diff_a_minus_b']:+.3f} (holm {auc['p_holm']:.3g})  dMAE={mae['mean_diff_a_minus_b']:+.2f} (holm {mae['p_holm']:.3g})  {'WIN' if v['pareto_win'] else '-'}")
    if "h4" in rep:
        print(f"\nH4: rho={rep['h4']['rho']:.2f} (>=0.8 -> {rep['h4']['supported']}); rank inversions: {rep['h4']['rank_inversions']}")
    print("\nwrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
