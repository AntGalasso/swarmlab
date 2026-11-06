"""The pre-registered hypotheses and secondary analyses of EXPERIMENT_PLAN.md section 7, one function each.

Every function takes a table loaded for one platform and returns a JSON-able dict:
  {"status": "not_configured"}                    no block is listed for it in analysis.blocks
  {"status": "no_data", "blocks_missing": [...]}  none of its blocks is on disk
  {"status": "ok", ...}
"""
from __future__ import annotations

import math

import numpy as np

from bazt_lab.features.analysis.adapters import stats_check_adapter as sc
from bazt_lab.features.analysis.private import compare, stats
from bazt_lab.features.analysis.private.config import Cfg
from bazt_lab.features.analysis.private.table import Table


def _report(results, skipped, **more) -> dict:
    return {"summary": {**compare.summarize(results), "complete": not skipped}, "comparisons": results,
            "skipped": skipped, **more}


def _per_challenger(t: Table, cfg: Cfg, platform: str, name: str, hypothesis: str, extra=None, **options) -> dict:
    """The family of every challenger against the baseline. `extra(raws, results, challenger)` adds fields."""
    blocks, status = compare.gate(t, cfg, hypothesis, platform)
    if status:
        return status
    out = {}
    for ch in cfg.challengers:
        raws, skipped = compare.arm_vs_arm(t, cfg, platform, blocks, ch, cfg.baseline, **options)
        results = compare.evaluate(raws, cfg, platform, (name, platform))
        out[ch] = _report(results, skipped, baseline=cfg.baseline, **(extra(raws, results, ch) if extra else {}))
    return {"status": "ok", "blocks": list(blocks), "challengers": out}


def h_budget(t: Table, cfg: Cfg, platform: str) -> dict:
    """H-budget: at each capped budget cell the challenger beats the baseline (capped C) with non-inferior accuracy."""
    return _per_challenger(t, cfg, platform, "H-budget", "budget", cells="capped")


# ------------------------------------------------------------------------------------------------ H-topology
def _advantage(p: compare.Pairs, cfg: Cfg, platform: str):
    """(baseline ASR - challenger ASR per pair, seed of the pair): positive = the challenger is safer."""
    x, y, cluster = compare.pair_arrays(p, "continuous", cfg, platform)
    keep = ~(np.isnan(x) | np.isnan(y))
    return (y - x)[keep], cluster[keep]


def holm_finite(pvals: dict) -> dict:
    """Holm over the p-values that exist; a NaN p-value (empty sample) is left out of the family, not ranked in it."""
    return sc.holm({k: p for k, p in pvals.items() if not math.isnan(p)})


def _growth(raws, cfg: Cfg, platform: str, challenger: str) -> dict:
    """The challenger's advantage at the largest agent count against the smallest, per heterogeneous cell, as two
    independent samples (graphs of different size share no episode). One Holm family."""
    cells = {}
    for raw in raws:
        if raw.context["topology"] in cfg.heterogeneous[platform]:
            cells.setdefault(tuple((f, v) for f, v in raw.context.items() if f != "agents"), []).append(raw)
    found = []
    for cell, group in cells.items():
        sizes = sorted({r.context["agents"] for r in group})
        if len(sizes) < 2:
            continue
        lo, hi = (next(r for r in group if r.context["agents"] == n) for n in (sizes[0], sizes[-1]))
        (a_hi, c_hi), (a_lo, c_lo) = _advantage(hi.safety, cfg, platform), _advantage(lo.safety, cfg, platform)
        key = ("H-topology", platform, challenger, "growth", *(str(v) for _, v in cell))
        found.append({"context": dict(cell), "agents": [sizes[0], sizes[-1]],
                      **stats.unpaired_test(a_hi, c_hi, a_lo, c_lo, key, cfg.resamples, cfg.level)})
    for c in found:
        c["p_holm"] = None                                   # stays None where the test has no p-value (empty sample)
    for i, p in holm_finite({i: c["p_raw"] for i, c in enumerate(found)}).items():
        found[i]["p_holm"] = p
    for c in found:
        significant = c["p_holm"] is not None and c["p_holm"] < cfg.alpha
        c["grows"], c["shrinks"] = bool(significant and c["delta"] > 0), bool(significant and c["delta"] < 0)
    return {"evaluated": bool(found), "contrasts": found, "n_grow": sum(c["grows"] for c in found),
            "n_shrink": sum(c["shrinks"] for c in found)}


def _topology_extra(cfg: Cfg, platform: str):
    def extra(raws, results, challenger):
        hetero_set, homo_set = cfg.heterogeneous[platform], cfg.homogeneous[platform]
        classes = {"heterogeneous": hetero_set, "homogeneous": homo_set}
        named = {n: compare.summarize([r for r in results if r["context"]["topology"] in members])
                 for n, members in classes.items()}
        named["other"] = compare.summarize([r for r in results
                                            if r["context"]["topology"] not in hetero_set + homo_set])
        growth = _growth(raws, cfg, platform, challenger)
        hetero, homo = named["heterogeneous"], named["homogeneous"]
        parts = {"appears_on_heterogeneous": hetero["supported"] if hetero["n_comparisons"] else None,
                 "absent_on_homogeneous": (not homo["supported"]) if homo["n_comparisons"] else None,
                 "grows_with_n": (growth["n_grow"] >= 0.5 * len(growth["contrasts"]) and growth["n_shrink"] == 0)
                 if growth["evaluated"] else None}
        known = [v for v in parts.values() if v is not None]
        return {"classes": named, "growth_with_n": growth, "parts": parts,
                "unevaluated": [k for k, v in parts.items() if v is None], "supported": bool(known and all(known))}
    return extra


def h_topology(t: Table, cfg: Cfg, platform: str) -> dict:
    """H-topology: the challenger's advantage appears on heterogeneous topologies, not on the homogeneous ones, and
    grows with N. Each part uses the rule of a hypothesis (at least half the class's comparisons win and none is a
    utility loss); a part the data cannot speak to is left out of `supported` and listed in `unevaluated`."""
    return _per_challenger(t, cfg, platform, "H-topology", "topology", extra=_topology_extra(cfg, platform))


# ------------------------------------------------------------------------------------------------ H-adaptive
def h_adaptive(t: Table, cfg: Cfg, platform: str) -> dict:
    """H-adaptive, one comparison per knowledge condition and cell: d = delta_challenger - factor * delta_baseline,
    where delta is the ASR increase under the adaptive attacker against the non-adaptive run of the same episodes."""
    ref = cfg.blocks["adaptive_reference"][platform]
    out = _per_challenger(t, cfg, platform, "H-adaptive", "adaptive", reference_blocks=ref)
    if out["status"] == "ok":
        out["reference_blocks"] = list(ref)
    return out


# ------------------------------------------------------------------------------------ secondary analyses
def ladder(t: Table, cfg: Cfg, platform: str) -> dict:
    """The ladder A > B > C: every rung against the one below it, uncapped, one Holm family. A cell is ordered when
    every step of the ladder is a significant safety gain."""
    blocks, status = compare.gate(t, cfg, "ladder", platform)
    if status:
        return status
    raws, skipped = [], []
    for challenger, baseline in zip(cfg.ladder[1:], cfg.ladder[:-1], strict=True):
        r, s = compare.arm_vs_arm(t, cfg, platform, blocks, challenger, baseline, cells="uncapped")
        raws, skipped = raws + r, skipped + s
    results = compare.evaluate(raws, cfg, platform, ("ladder", platform))
    steps, by_cell = len(cfg.ladder) - 1, {}
    for r in results:
        by_cell.setdefault(tuple(r["context"].items()), []).append(r)
    cells = [{"context": dict(k), "ordered": len(v) == steps and all(
        c["safety"]["delta"] < 0 and c["safety"]["p_holm"] < cfg.alpha for c in v)} for k, v in by_cell.items()]
    ordered = sum(c["ordered"] for c in cells)
    return {"status": "ok", "blocks": list(blocks), "order": list(cfg.ladder),
            **_report(results, skipped, cells=cells, n_cells=len(cells), ordered_cells=ordered,
                      holds=(ordered == len(cells)) if cells else None)}


def ablations(t: Table, cfg: Cfg, platform: str) -> dict:
    """Each ablation arm against the full arm (a win means the removed component helps), one Holm family."""
    blocks, status = compare.gate(t, cfg, "ablations", platform)
    if status:
        return status
    raws, skipped = [], {}
    for abl in cfg.ablations:
        r, skipped[abl] = compare.arm_vs_arm(t, cfg, platform, blocks, cfg.ablation_reference, abl)
        raws += r
    results = compare.evaluate(raws, cfg, platform, ("ablations", platform))
    per = {abl: _report([r for r in results if r["baseline"] == abl], skipped[abl], baseline=abl)
           for abl in cfg.ablations}
    return {"status": "ok", "blocks": list(blocks), "reference": cfg.ablation_reference,
            "summary": {**compare.summarize(results), "complete": not any(skipped.values())}, "ablations": per}
