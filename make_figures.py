#!/usr/bin/env python3
"""make_figures.py -- publication vector figures from the CSVs written by swarmlab.py.

  fig1_pareto_safety_utility.pdf   AUC_ever vs MAE frontier across policies (decoy regime)
  fig2_budget_exhaustion_dynamics.pdf  cumulative quarantines on decoys vs true positives, t = 1..12
  fig3_scaling_curve.pdf           AUC_ever and MAE across N = {12, 24, 36} (small_world)
  fig4_ladder_breaking_point.pdf   env_ladder Stage 2: MAE and pass rate vs difficulty level
  fig5_constraint_evasion.pdf      env_ladder Stage 3: payload delivery vs monitor detection

  python make_figures.py                                   # main split
  python make_figures.py --file results/per_seed_main.csv --out results/figures

Mock-mode numbers are pipeline checks, not findings. Pass `--file results/per_seed_<model>_main.csv`
for a live-LLM arm.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C_HIGHLIGHT = {"score_only": "#d62728", "fixed_early": "#ff7f0e", "centrality_only": "#8c564b",
               "risk_aware": "#1f77b4", "risk_aware_tiered": "#2ca02c", "no_defense": "#7f7f7f",
               "offline_oracle": "#9467bd", "fixed_mid": "#bcbd22", "fixed_late": "#17becf",
               "random_capped": "#e377c2"}
FIG2_POLICIES = ["fixed_early", "score_only", "risk_aware", "risk_aware_tiered"]


def load(path: str) -> List[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k, v in list(r.items()):
                if k in ("seed", "N", "K", "m") :
                    r[k] = int(float(v)) if v not in ("", None) else r[k]
                elif k not in ("topology", "regime", "policy", "quarantine_mode", "budget_rule"):
                    try:
                        r[k] = float(v)
                    except (TypeError, ValueError):
                        pass
            rows.append(r)
    return rows


def mean_sem(xs) -> tuple:
    xs = [x for x in xs if x == x]
    if not xs:
        return float("nan"), float("nan")
    return float(np.mean(xs)), float(np.std(xs, ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 else 0.0


# --------------------------------------------------------------------------------- fig 1
def fig1(rows, path, regime="decoy"):
    sub = [r for r in rows if r["regime"] == regime]
    pols = sorted({r["policy"] for r in sub})
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    for pol in pols:
        rs = [r for r in sub if r["policy"] == pol]
        col = C_HIGHLIGHT.get(pol, "#333333")
        # faint per-topology points keep the cross-topology average honest
        for topo in sorted({r["topology"] for r in rs}):
            ts = [r for r in rs if r["topology"] == topo]
            ax.plot(np.mean([r["auc_ever_infected"] for r in ts]), np.mean([r["task_mae"] for r in ts]),
                    marker="o", ms=3.5, color=col, alpha=0.28, zorder=1)
        auc, auc_e = mean_sem([r["auc_ever_infected"] for r in rs])
        mae, mae_e = mean_sem([r["task_mae"] for r in rs])
        hl = pol in ("score_only", "fixed_early", "risk_aware", "risk_aware_tiered")
        ax.errorbar(auc, mae, xerr=auc_e, yerr=mae_e, fmt="o", color=col, ms=10 if hl else 7,
                    capsize=3, lw=1.3, zorder=3, markeredgecolor="white", markeredgewidth=0.6,
                    label=pol)
    ax.set_xlabel(r"containment damage  $AUC_{ever}$  (lower is better)")
    ax.set_ylabel(r"consensus error  $MAE$  (lower is better)")
    ax.set_title(f"Safety–utility frontier, regime = {regime}", fontsize=11)
    ax.text(0.99, 0.02, "solid marker + SEM bars = mean over topologies × seeds;\n"
                        "faint = per-topology mean",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="#555555")
    ax.grid(alpha=0.25)
    ax.margins(x=0.10, y=0.10)
    leg = ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2, handletextpad=0.3,
                    columnspacing=1.1, borderaxespad=0.6, title="control arm", title_fontsize=8)
    for txt in leg.get_texts():
        if txt.get_text() in ("score_only", "fixed_early", "risk_aware", "risk_aware_tiered"):
            txt.set_fontweight("bold")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------- fig 2
def fig2(episodes_path, path, regime="decoy"):
    curves = defaultdict(lambda: {"decoy": [], "true": []})
    with open(episodes_path) as f:
        for line in f:
            e = json.loads(line)
            if e["regime"] != regime or e["policy"] not in FIG2_POLICIES:
                continue
            curves[e["policy"]]["decoy"].append(e["cum_decoy_quarantines"])
            curves[e["policy"]]["true"].append(e["cum_true_quarantines"])
    if not curves:
        return None
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    for pol in FIG2_POLICIES:
        c = curves.get(pol)
        if not c:
            continue
        col = C_HIGHLIGHT[pol]
        d = np.mean(np.array(c["decoy"]), axis=0)
        t = np.mean(np.array(c["true"]), axis=0)
        x = np.arange(1, len(d) + 1)
        ax.plot(x, d, "--", color=col, lw=2, marker="o", ms=4, label=f"{pol}: on decoys (wasted)")
        ax.plot(x, t, "-", color=col, lw=2, marker="s", ms=4, label=f"{pol}: on true positives")
    ax.set_ylabel("cumulative quarantines spent")
    ax.set_title("Budget exhaustion: decoy alerts vs true positives (decoy regime)", fontsize=11)
    ax.set_ylim(bottom=-0.08)
    ax.grid(alpha=0.25)
    ax.set_xlabel("step $t$   (dashed = budget wasted on benign decoys; solid = spent on true positives)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------- fig 3
def fig3(rows, path):
    sub = rows if rows else []
    if not sub:
        return None
    ns = sorted({r["N"] for r in sub})
    pols = sorted({r["policy"] for r in sub})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, metric, label in ((axes[0], "auc_ever_infected", r"$AUC_{ever}$"),
                              (axes[1], "task_mae", r"$MAE$")):
        for pol in pols:
            rules = (("prop", "--"),) if pol == "no_defense" else (("fixed", "-"), ("prop", "--"))
            for rule, ls in rules:
                rs = [r for r in sub if r["policy"] == pol and r["budget_rule"] == rule]
                if not rs:
                    continue
                xs, ys, es = [], [], []
                for n in ns:
                    ys_n, e_n = mean_sem([r[metric] for r in rs if r["N"] == n])
                    xs.append(n)
                    ys.append(ys_n)
                    es.append(e_n)
                hl = pol in C_HIGHLIGHT
                lab = pol if pol == "no_defense" else f"{pol} ({'K=3' if rule == 'fixed' else 'K=0.25N'})"
                ax.errorbar(xs, ys, yerr=es, fmt="o" if hl else "s", ls=ls, color=C_HIGHLIGHT.get(pol, "#333"),
                            ms=6 if hl else 4, capsize=3, lw=1.6, alpha=1.0 if hl else 0.7, label=lab)
        ax.set_xlabel("swarm size $N$ (small_world)")
        ax.set_ylabel(label)
        ax.set_xticks(ns)
        ax.grid(alpha=0.25)
        ax.set_title(f"{label} vs swarm size", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Targeted scaling sweep, small_world (solid = fixed scarce K=3, dashed = proportional K=0.25N)",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------------------- fig 4 / fig 5
def fig4(rows, path):
    """Stage-2 ladder: consensus error and pass rate against the frozen success bar, per level."""
    if not rows:
        return None
    rs = sorted(rows, key=lambda r: int(r["level"]))
    x = [int(r["level"]) for r in rs]
    tau = [float(r["target_mae_threshold"]) for r in rs]
    mae = [float(r["mean_mae"]) for r in rs]
    mae_e = [float(r["sem_mae"]) for r in rs]
    pr = [float(r["pass_rate"]) for r in rs]
    pr_e = [float(r["pass_rate_se"]) for r in rs]
    breaking = next((int(r["level"]) for r in rs if int(float(r["breaking"])) == 1), None)
    critical = next((int(r["level"]) for r in rs if int(float(r["critical_operating"])) == 1), None)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))
    ax = axes[0]
    ax.errorbar(x, mae, yerr=mae_e, fmt="o-", color=C_HIGHLIGHT["risk_aware"], ms=6, capsize=3,
                lw=1.8, label=r"consensus $MAE$ (mean $\pm$ SEM)")
    ax.plot(x, tau, "--", color="#7f7f7f", lw=1.6, label=r"success bar $\tau_l$")
    ax.set_ylabel(r"consensus error $MAE$")
    ax.set_title("Difficulty ladder: task error vs the frozen success bar", fontsize=11)
    ax = axes[1]
    ax.errorbar(x, pr, yerr=pr_e, fmt="s-", color=C_HIGHLIGHT["risk_aware_tiered"], ms=6, capsize=3,
                lw=1.8, label="pass rate (fraction of seeds with MAE <= tau)")
    ax.axhline(0.5, ls=":", color="#d62728", lw=1.4, label="breaking threshold (0.50)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("pass rate")
    ax.set_title("Capability breaking point", fontsize=11)
    for a in axes:
        a.set_xlabel("sandbox difficulty level")
        a.set_xticks(x)
        a.grid(alpha=0.25)
        if breaking is not None:
            a.axvline(breaking, color="#d62728", alpha=0.35, lw=1.2, ls="--")
        if critical is not None:
            a.axvline(critical, color="#2ca02c", alpha=0.30, lw=1.2, ls=":")
        a.legend(fontsize=7.5, frameon=False, loc="best")
    note = (f"dashed red = breaking_level {breaking}; dotted green = critical_operating_level {critical}\n"
            "mock agents: pipeline check, not a finding")
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=7.5, color="#555555")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fig5(rows, path):
    """Stage-3: does the payload survive the channel filter, and does the monitor still see it?

    Restricted to one defender arm and one movement mask when present, so the bars compare channel
    filters only; payload delivery (left) vs monitor detection (right) is the evasion gap.
    """
    if not rows:
        return None
    pol = "risk_aware_tiered" if any(r["policy"] == "risk_aware_tiered" for r in rows) else rows[0]["policy"]
    sub = [r for r in rows if r["policy"] == pol]
    if any(r["movement_direction_mode"] == "unrestricted" for r in sub):
        sub = [r for r in sub if r["movement_direction_mode"] == "unrestricted"]
    lvl = min(int(float(r["level"])) for r in sub)
    sub = [r for r in sub if int(float(r["level"])) == lvl]
    channels = sorted({r["action_channel_mode"] for r in sub})
    forms = sorted({r["payload_form"] for r in sub})
    colors = {"naive": "#d62728", "compressed": "#2ca02c"}
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))
    width = 0.8 / max(1, len(forms))
    for ax, metric, title in ((axes[0], "mean_payload_delivery_rate", "payload survives the filter"),
                              (axes[1], "mean_monitor_detection_rate", "monitor still sees it")):
        for k, form in enumerate(forms):
            vals, errs, xs = [], [], []
            for j, ch in enumerate(channels):
                cell = [float(r[metric]) for r in sub if r["payload_form"] == form
                        and r["action_channel_mode"] == ch]
                v, e = mean_sem(cell)
                vals.append(v)
                errs.append(e)
                xs.append(j + (k - (len(forms) - 1) / 2) * width)
            ax.bar(xs, vals, width=width, color=colors.get(form, "#333333"), alpha=0.85, label=form)
        ax.set_xticks(range(len(channels)))
        ax.set_xticklabels([c.replace("_", "\n", 1) for c in channels], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("fraction of infected node-steps")
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"Monitor evasion under hard channel constraints (level {lvl}, {pol})", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--file", default=None, help="per-seed CSV (default results/per_seed_main.csv)")
    ap.add_argument("--episodes", default=None, help="episodes JSONL (default results/episodes_main.jsonl)")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--regime", default="decoy")
    args = ap.parse_args(argv)

    per_seed = args.file or os.path.join(args.results, "per_seed_main.csv")
    episodes = args.episodes or os.path.join(args.results, "episodes_main.jsonl")
    scaling = os.path.join(args.results, "scaling_per_seed.csv")
    ladder = os.path.join(args.results, "ladder_breaking_point.csv")
    evasion = os.path.join(args.results, "ladder_constraint_summary.csv")
    os.makedirs(args.out, exist_ok=True)
    made = []
    if os.path.exists(per_seed):
        rows = load(per_seed)
        made.append(fig1(rows, os.path.join(args.out, "fig1_pareto_safety_utility.pdf"), args.regime))
    else:
        print("missing", per_seed, file=sys.stderr)
    if os.path.exists(episodes):
        p = fig2(episodes, os.path.join(args.out, "fig2_budget_exhaustion_dynamics.pdf"), args.regime)
        if p:
            made.append(p)
    else:
        print("missing", episodes, file=sys.stderr)
    if os.path.exists(scaling):
        p = fig3(load(scaling), os.path.join(args.out, "fig3_scaling_curve.pdf"))
        if p:
            made.append(p)
    else:
        print("missing", scaling, file=sys.stderr)
    if os.path.exists(ladder):
        p = fig4(load(ladder), os.path.join(args.out, "fig4_ladder_breaking_point.pdf"))
        if p:
            made.append(p)
    else:
        print("missing", ladder, file=sys.stderr)
    if os.path.exists(evasion):
        p = fig5(load(evasion), os.path.join(args.out, "fig5_constraint_evasion.pdf"))
        if p:
            made.append(p)
    else:
        print("missing", evasion, file=sys.stderr)
    for p in made:
        print("wrote", p)
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
