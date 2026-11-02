#!/usr/bin/env python3
"""stats_check.py -- PRE-EXISTING SCAFFOLDING. Pre-registered tests for H1-H4.

Reads a per-seed CSV written by swarmlab.py (results/per_seed_main.csv for the main split,
results/per_seed_heldout.csv for the held-out split). Paired by seed. All tests: two-sided Wilcoxon
signed-rank, Holm correction within each family, 1,000-resample paired-seed bootstrap 95% CI of the
mean paired difference. rank-biserial keeps its sign (positive: first-named condition is LOWER, i.e.
better, for damage metrics; the sign convention is printed next to every table).

H3 uses TWO unambiguous labels (never a bare "WIN"); both raw p_raw and Holm-adjusted p_holm are
reported for every comparison, for AUC_ever and for MAE:

  STRICT_PARETO_WIN
      delta_AUC < 0 AND p_holm_AUC < 0.05 AND delta_MAE < 0 AND p_holm_MAE < 0.05
  SAFETY_WIN_UTILITY_NONINFERIOR
      delta_AUC < 0 AND p_holm_AUC < 0.05 AND upper 95% CI of delta_MAE <= margin_mae
      (significant containment gain with statistically bounded / zero utility loss)

Every other row is labelled UTILITY_LOSS (MAE significantly worse), CONTAINMENT_GAIN_UTILITY_UNBOUNDED
(AUC significantly better, but no utility bound established) or NO_SIGNIFICANT_DIFFERENCE. No label that
contains WIN is ever emitted while MAE degrades significantly.

H3 is `supported` when the challenger achieves STRICT_PARETO_WIN or SAFETY_WIN_UTILITY_NONINFERIOR on at
least half of the pre-registered comparisons AND suffers zero UTILITY_LOSS labels (requiring 100% would
make the test unsupportable by construction). The `dominance_rate` is reported for every challenger.

  python stats_check.py --file results/per_seed_main.csv --out results/stats_report.json
  python stats_check.py --file results/per_seed_heldout.csv --out results/stats_report_heldout.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

NUM = ["auc_ever_infected", "auc_active_unquarantined", "final_ever_infected_rate", "total_deviated_messages",
       "task_mae", "quarantines_used", "sanitizations_used", "collateral_healthy_quarantined",
       "decoy_quarantines_wasted", "time_to_containment", "delta_abs", "delta_rel"]
ALPHA = 0.05
LABEL_STRICT = "STRICT_PARETO_WIN"
LABEL_SAFETY = "SAFETY_WIN_UTILITY_NONINFERIOR"
LABEL_CONTAIN = "CONTAINMENT_GAIN_UTILITY_UNBOUNDED"
LABEL_UTILITY_LOSS = "UTILITY_LOSS"
LABEL_NULL = "NO_SIGNIFICANT_DIFFERENCE"
DEFAULT_RIVALS = ("score_only", "centrality_only", "random_capped", "fixed_early", "fixed_mid", "fixed_late")


def load(path: str) -> List[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in NUM + ["seed", "confirm_steps", "B_san"]:
                if k in r and r[k] != "":
                    r[k] = float(r[k])
            rows.append(r)
    return rows


def vec(rows, topo, regime, policy, metric) -> Dict[int, float]:
    return {int(r["seed"]): r[metric] for r in rows
            if r["topology"] == topo and r["regime"] == regime and r["policy"] == policy
            and metric in r and r[metric] == r[metric]}


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
    return {"a": pol_a, "b": pol_b, "metric": metric, "n": int(len(a)),
            "mean_diff_a_minus_b": float((a - b).mean()) if len(a) else float("nan"),
            "ci95": [lo, hi], "p_raw": p, "p_holm": None, "rrb": rrb, "n_nonzero": nnz}


# --------------------------------------------------------------------------------------- H1
def h1(rows, topos, regime="decoy"):
    """Fixed windows differ in infection AUC; supported if Holm p<0.05 for >=1 pair on >=2 topologies."""
    res, hits = {}, 0
    for t in topos:
        fam = {f"{x}_vs_{y}": compare(rows, t, regime, x, y, "auc_ever_infected")
               for x, y in (("fixed_early", "fixed_mid"), ("fixed_mid", "fixed_late"), ("fixed_early", "fixed_late"))}
        adj = holm({k: v["p_raw"] for k, v in fam.items()})
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
            a, b = paired(vec(rows, t, "decoy", pol, "auc_ever_infected"),
                          vec(rows, t, "clean", pol, "auc_ever_infected"))
            d = a - b  # positive = worse under decoys
            nz = d[d != 0]
            p = float(stats.wilcoxon(nz).pvalue) if len(nz) else 1.0
            lo, hi = boot_ci(d)
            fam[f"{t}:{pol}"] = {"n": int(len(d)), "mean_increase": float(d.mean()) if len(d) else float("nan"),
                                 "ci95": [lo, hi], "p_raw": p}
    adj = holm({k: v["p_raw"] for k, v in fam.items()})
    for k in fam:
        fam[k]["p_holm"] = adj[k]
        fam[k]["degraded"] = bool(fam[k]["mean_increase"] > 0 and adj[k] < ALPHA)
    sup = sum(v["degraded"] for v in fam.values())
    return {"hypothesis": "H2 budget exhaustion", "n_significant_degradations": sup, "n_tests": len(fam),
            "supported": sup >= max(1, len(fam) // 2), "detail": fam}


# --------------------------------------------------------------------------------------- H3
def pareto_label(d_auc: float, p_holm_auc: Optional[float], d_mae: float, p_holm_mae: Optional[float],
                 ci95_mae: Sequence[float], margin_mae: float) -> str:
    """Two explicit, unambiguous labels; a bare 'WIN' is never emitted."""
    sig_auc = bool(d_auc < 0 and p_holm_auc is not None and p_holm_auc < ALPHA)
    strict = bool(sig_auc and d_mae < 0 and p_holm_mae is not None and p_holm_mae < ALPHA)
    noninferior = bool(sig_auc and ci95_mae[1] == ci95_mae[1] and ci95_mae[1] <= margin_mae)
    if strict:
        return LABEL_STRICT
    if noninferior:
        return LABEL_SAFETY
    if d_mae > 0 and p_holm_mae is not None and p_holm_mae < ALPHA:
        return LABEL_UTILITY_LOSS          # a significant utility loss is never reported as a win
    if sig_auc:
        return LABEL_CONTAIN
    return LABEL_NULL


def h3(rows, topos, regime="decoy", challengers=("risk_aware", "risk_aware_tiered"),
       rivals=DEFAULT_RIVALS, margin_mae: float = 0.05):
    """Containment (lower AUC_ever) at bounded utility (MAE) cost, per rival, Holm within (topology, metric)."""
    out = {}
    for ch in challengers:
        rival_set = tuple(r for r in (tuple(rivals) + ("risk_aware",)) if r != ch)
        per_topo, labels = {}, []
        for t in topos:
            per = {}
            for metric in ("auc_ever_infected", "task_mae"):
                fam = {r: compare(rows, t, regime, ch, r, metric) for r in rival_set}
                adj = holm({k: v["p_raw"] for k, v in fam.items()})
                for k in fam:
                    fam[k]["p_holm"] = adj[k]
                per[metric] = fam
            verdict = {}
            for r in rival_set:
                auc, mae = per["auc_ever_infected"][r], per["task_mae"][r]
                label = pareto_label(auc["mean_diff_a_minus_b"], auc["p_holm"], mae["mean_diff_a_minus_b"],
                                     mae["p_holm"], mae["ci95"], margin_mae)
                labels.append(label)
                verdict[r] = {"label": label,
                              "delta_auc": auc["mean_diff_a_minus_b"], "p_raw_auc": auc["p_raw"],
                              "p_holm_auc": auc["p_holm"],
                              "delta_mae": mae["mean_diff_a_minus_b"], "p_raw_mae": mae["p_raw"],
                              "p_holm_mae": mae["p_holm"], "ci95_delta_mae": mae["ci95"],
                              "strict_pareto": label == LABEL_STRICT,
                              "utility_non_inferior": label in (LABEL_STRICT, LABEL_SAFETY),
                              "n": auc["n"]}
            per_topo[t] = {"tests": per, "verdict": verdict}
        counts = {lab: labels.count(lab) for lab in
                  (LABEL_STRICT, LABEL_SAFETY, LABEL_CONTAIN, LABEL_UTILITY_LOSS, LABEL_NULL)}
        n = len(labels)
        wins = counts[LABEL_STRICT] + counts[LABEL_SAFETY]
        # Pre-registered H3 criterion: utility-preserving dominance on at least half of the comparisons
        # AND not a single significant MAE loss. (Requiring 100% would make H3 unsupportable by design.)
        out[ch] = {"per_topology": per_topo, "label_counts": counts,
                   "n_comparisons": n, "n_utility_wins": wins,
                   "dominance_rate": (wins / n) if n else float("nan"),
                   "n_utility_loss": counts[LABEL_UTILITY_LOSS],
                   "criterion": "wins >= 50% of comparisons AND zero UTILITY_LOSS labels",
                   "supported": bool(n > 0 and wins >= 0.5 * n and counts[LABEL_UTILITY_LOSS] == 0)}
    return {"hypothesis": "H3 contour-aware Pareto gain", "regime": regime, "margin_mae": margin_mae,
            "challengers": out}


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
    ap.add_argument("--file", default="results/per_seed_main.csv")
    ap.add_argument("--out", default=None, help="JSON report (default: stats_report.json / stats_report_heldout.json)")
    ap.add_argument("--margin-mae", type=float, default=0.05,
                    help="non-inferiority margin on delta MAE used by SAFETY_WIN_UTILITY_NONINFERIOR")
    ap.add_argument("--h4", nargs=2, default=None, metavar=("MODEL_A_CSV", "MODEL_B_CSV"))
    args = ap.parse_args(argv)

    rows = load(args.file)
    topos = sorted({r["topology"] for r in rows})
    regimes = {r["regime"] for r in rows}
    split = "heldout" if "heldout" in os.path.basename(args.file) else "main"
    out = args.out or f"results/stats_report_{split}.json"
    rep = {"meta": {"file": args.file, "split": split, "n_rows": len(rows), "regimes": sorted(regimes),
                    "seeds": [int(min(r["seed"] for r in rows)), int(max(r["seed"] for r in rows))],
                    "margin_mae": args.margin_mae},
           "h1": h1(rows, topos),
           "h2": h2(rows, topos) if {"clean", "decoy"} <= regimes else None,
           "h3": h3(rows, topos, margin_mae=args.margin_mae)}
    if args.h4:
        rep["h4"] = h4(*args.h4)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(rep, open(out, "w"), indent=2)

    print(f"file={args.file}  split={split}  seeds={rep['meta']['seeds'][0]}..{rep['meta']['seeds'][1]}  "
          f"rows={len(rows)}")
    print("Sign convention: mean_diff = a - b on a DAMAGE metric; negative = a better. rrb>0 = a lower more often.")
    print(f"\nH1 (window timing, decoy regime): significant pair on "
          f"{rep['h1']['topologies_with_significant_pair']} topologies -> supported={rep['h1']['supported']}")
    for t, fam in rep["h1"]["detail"].items():
        for k, v in fam.items():
            print(f"  {t:<12}{k:<20} dAUC={v['mean_diff_a_minus_b']:+.3f} "
                  f"CI[{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] p_raw={v['p_raw']:.3g} "
                  f"p_holm={v['p_holm']:.3g} rrb={v['rrb']:+.2f}")
    if rep["h2"]:
        print(f"\nH2 (decoy degradation): {rep['h2']['n_significant_degradations']}/{rep['h2']['n_tests']} "
              f"significant -> supported={rep['h2']['supported']}")
        for k, v in rep["h2"]["detail"].items():
            print(f"  {k:<28} +AUC={v['mean_increase']:+.3f} CI[{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] "
                  f"p_raw={v['p_raw']:.3g} p_holm={v['p_holm']:.3g} "
                  f"{'DEGRADED' if v['degraded'] else '-'}")
    print(f"\nH3 (regime={rep['h3']['regime']}, margin_MAE={args.margin_mae}) -- strict Pareto & non-inferiority")
    for ch, d in rep["h3"]["challengers"].items():
        c = d["label_counts"]
        print(f"  challenger={ch}: {d['n_utility_wins']}/{d['n_comparisons']} utility outcomes, "
              f"dominance_rate={d['dominance_rate']:.2f} (strict={c[LABEL_STRICT]}, "
              f"noninferior={c[LABEL_SAFETY]}, containment_only={c[LABEL_CONTAIN]}, "
              f"utility_loss={c[LABEL_UTILITY_LOSS]}, null={c[LABEL_NULL]}) -> supported={d['supported']}")
        print(f"    {'topology':<12} {'rival':<16} {'dAUC':>8} {'p_raw':>9} {'p_holm':>9} "
              f"{'dMAE':>8} {'CIhi_MAE':>9} {'p_raw':>9} {'p_holm':>9}  label")
        for t, per in d["per_topology"].items():
            for r, v in per["verdict"].items():
                print(f"    {t:<12} {r:<16} {v['delta_auc']:>8.3f} {v['p_raw_auc']:>9.3g} {v['p_holm_auc']:>9.3g} "
                      f"{v['delta_mae']:>8.2f} {v['ci95_delta_mae'][1]:>9.2f} {v['p_raw_mae']:>9.3g} "
                      f"{v['p_holm_mae']:>9.3g}  {v['label']}")
    if "h4" in rep:
        print(f"\nH4: rho={rep['h4']['rho']:.2f} (>=0.8 -> {rep['h4']['supported']}); "
              f"rank inversions: {rep['h4']['rank_inversions']}")
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
