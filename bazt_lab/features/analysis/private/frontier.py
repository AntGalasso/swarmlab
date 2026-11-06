"""H-efficiency (descriptive): the smallest budget at which a challenger reaches the security of the uncapped baseline.

A challenger cell *reaches* when the mean ASR difference to the uncapped baseline, over the same episodes, is at most
`analysis.efficiency_tolerance`. The budget of a cell is audit + intervention as fractions of all deliveries; a cell
with one budget uncapped has an unbounded budget. The interval comes from resampling whole seeds once and applying the
same draw to every cell of the frontier, so the smallest budget is read off each replicate frontier.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np

from bazt_lab.features.analysis.private import compare, stats
from bazt_lab.features.analysis.private.config import SAFETY, Cfg
from bazt_lab.features.analysis.private.table import UNCAPPED, Key, Table, budget_order, sort_key


def _site(site: Key) -> dict:
    return {k: v for k, v in compare.describe(site).items() if k not in ("audit", "intervention")}


def _total(audit, intervention) -> float:
    return budget_order(audit) + budget_order(intervention)


def _shown(total: float):
    return UNCAPPED if math.isinf(total) else total


def _cell_arrays(t: Table, cell: Key, ref: Key, fields: Sequence[str]):
    """d = ASR(cell) - ASR(reference) per aligned episode, its seed, and the split-field codes."""
    rows = t.aligned([cell, ref])
    asr = t.metrics[SAFETY]
    d = asr[rows[0]] - asr[rows[1]]
    keep = ~np.isnan(d)
    codes = np.stack([t.codes[f][rows[0]] for f in fields], axis=1) if fields else np.zeros((len(d), 0), dtype=np.int64)
    return d[keep], t.codes["seed"][rows[0]][keep], codes[keep], asr[rows[1]][keep]


def _unit_sums(d: np.ndarray, cluster: np.ndarray, universe: np.ndarray, mode: str):
    """(sums, counts) per cluster of `universe`; in seed mode a seed is one pair whose value is its mean."""
    sums, counts = stats.cluster_sums(d, cluster, universe)
    if mode == "seed":
        return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0), (counts > 0).astype(np.float64)
    return sums, counts


def _minimal(cells: List[dict]) -> List[dict]:
    """The reaching cells that no other reaching cell undercuts on both budgets."""
    reach = [c for c in cells if c["reaches"]]
    key = lambda c: (budget_order(c["audit"]), budget_order(c["intervention"]))  # noqa: E731
    return [c for c in reach if not any(key(o) != key(c) and all(a <= b for a, b in zip(key(o), key(c), strict=True))
                                        for o in reach)]


def _frontier(t: Table, cfg: Cfg, platform: str, name: str, site: Key, cells: List[Key], ref: Key, fields,
              combo: tuple, labels: tuple):
    """The frontier of one site and one split combination, or None if no cell has data for it."""
    arrays = []
    for cell in cells:
        d, seed, codes, base = _cell_arrays(t, cell, ref, fields)
        at = np.flatnonzero((codes == np.asarray(combo, dtype=np.int64)).all(axis=1)) if fields else np.arange(len(d))
        if len(at):
            arrays.append((cell, d[at], seed[at], base[at]))
    if not arrays:
        return None
    mode = cfg.pair_unit[platform]
    universe = np.unique(np.concatenate([a[2] for a in arrays]))
    idx = stats.resample_index(("H-efficiency", platform, name, *map(str, site), *map(str, labels)), cfg.resamples,
                               len(universe))
    sums = [_unit_sums(d, s, universe, mode) for _, d, s, _ in arrays]
    point = np.array([s.sum() / c.sum() for s, c in sums])
    rep = np.stack([np.divide(s[idx].sum(axis=1), c[idx].sum(axis=1), out=np.full(len(idx), np.nan),
                              where=c[idx].sum(axis=1) > 0) for s, c in sums], axis=1)       # [resamples, cells]
    reach_rep = rep <= cfg.efficiency_tolerance                                              # NaN compares False
    total = np.array([_total(c.audit, c.intervention) for c, *_ in arrays])
    cell_rows = [{"audit": c.audit, "intervention": c.intervention, "delta": float(point[k]),
                  "ci": list(stats.quantile_ci(rep[:, k][~np.isnan(rep[:, k])], cfg.level))
                  if (~np.isnan(rep[:, k])).any() else [math.nan, math.nan],
                  "reaches": bool(point[k] <= cfg.efficiency_tolerance), "p_reach": float(reach_rep[:, k].mean()),
                  "budget_total": _shown(total[k]), "n_pairs": len(arrays[k][1])}
                 for k, (c, *_) in enumerate(arrays)]
    reaching = [k for k, c in enumerate(cell_rows) if c["reaches"]]
    best = min(reaching, key=lambda k: (total[k], budget_order(arrays[k][0].audit), budget_order(arrays[k][0].intervention)),
               default=None)
    smallest = None if best is None else {"audit": cell_rows[best]["audit"], "intervention": cell_rows[best]["intervention"],
                                          "total": _shown(total[best])}
    reached = reach_rep.any(axis=1)
    per_rep = np.where(reach_rep, total, np.inf).min(axis=1)[reached]
    ci = None if not len(per_rep) else [_shown(float(v)) for v in np.quantile(per_rep, [(1 - cfg.level) / 2, (1 + cfg.level) / 2],
                                                                           method="inverted_cdf")]
    return {"context": {**_site(site), **dict(zip(fields, labels, strict=True))},
            "reference_asr": float(np.mean(max(arrays, key=lambda a: len(a[1]))[3])), "n_pairs": max(c["n_pairs"] for c in cell_rows),
            "n_clusters": len(universe), "cells": cell_rows, "smallest": smallest, "smallest_ci": ci,
            "p_reached": float(reached.mean()), "pareto_minimal": _minimal(cell_rows)}


def efficiency(t: Table, cfg: Cfg, platform: str) -> dict:
    blocks, status = compare.gate(t, cfg, "efficiency", platform)
    if status:
        return status
    fields, out = cfg.split_by[platform], {}
    for ch in cfg.challengers:
        sites: Dict[Key, List[Key]] = {}
        for key in t.index:
            if key.block in blocks and key.arm == ch and not key.attack_free:
                sites.setdefault(key._replace(audit=None, intervention=None, arm=None, attack_free=None), []).append(key)
        frontiers, skipped = [], []
        for site in sorted(sites, key=sort_key):
            ref = site._replace(arm=cfg.baseline, audit=UNCAPPED, intervention=UNCAPPED, attack_free=False)
            if not t.has(ref):
                skipped.append({"context": _site(site),
                                "reason": f"no uncapped {cfg.baseline} run to compare with"})
                continue
            cells = sorted(sites[site], key=lambda k: (budget_order(k.audit), budget_order(k.intervention)))
            combos = {}                                  # every cell contributes its split combinations
            for cell in cells:
                aligned = t.aligned([cell, ref])
                if len(aligned[0]):
                    combos.update((c, v) for c, v, _ in compare.split_rows(t, aligned[0], fields))
            combos = sorted(combos.items())
            for combo, labels in combos:
                f = _frontier(t, cfg, platform, ch, site, cells, ref, fields, combo, labels)
                if f is not None:
                    frontiers.append(f)
        out[ch] = {"baseline": cfg.baseline, "tolerance": cfg.efficiency_tolerance, "frontiers": frontiers,
                   "skipped": skipped}
    return {"status": "ok", "blocks": list(blocks), "challengers": out}
