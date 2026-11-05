"""The report: every analysis of EXPERIMENT_PLAN.md section 7 on both platforms, written as strict JSON.

    {"meta": ..., "hypotheses": {H-budget, H-efficiency, H-topology, H-adaptive}, "secondary": {ladder, ablations,
     rank_agreement}, "containment_per_cost": ..., "verdicts": ...}

Each analysis other than the rank agreement is {platform: result}; a result is {"status": "not_configured" |
"no_data" | "ok", ...} (see private/hypotheses.py). Nothing in a report is NaN or infinite: those become null.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping

import numpy as np

from bazt_lab.core import public as core
from bazt_lab.features.analysis.private import containment, frontier, hypotheses, rank, table
from bazt_lab.features.analysis.private.config import PLATFORMS, Cfg

REPORT_NAME = "analysis_report.json"
HYPOTHESES = (("H-budget", hypotheses.h_budget), ("H-efficiency", frontier.efficiency),
              ("H-topology", hypotheses.h_topology), ("H-adaptive", hypotheses.h_adaptive))
SECONDARY = (("ladder", hypotheses.ladder), ("ablations", hypotheses.ablations))
# What "supported" means for each confirmatory hypothesis (H-efficiency is descriptive and has no verdict).
SUPPORTED = {"H-budget": lambda c: c["summary"]["supported"], "H-adaptive": lambda c: c["summary"]["supported"],
             "H-topology": lambda c: c["supported"]}
PRE_REGISTERED = ("ni_margin_accuracy", "rank_agreement_threshold")


def jsonable(x):
    """Plain JSON values: numpy scalars and arrays unwrapped, NaN and infinity as null. An object that has no JSON
    form is an error, never a string."""
    if isinstance(x, Mapping):
        return {k if isinstance(k, str) else str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if x is None or isinstance(x, (bool, int, str)):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    raise TypeError(f"{type(x).__name__} has no JSON form")


def fingerprint(settings: Mapping) -> str:
    """sha256 of the analysis settings and the two pre-registered values: equal fingerprints, same analysis."""
    chosen = {"analysis": core.get(settings, "analysis"),
              **{f"runtime.{k}": core.get(settings, f"runtime.{k}") for k in PRE_REGISTERED}}
    return hashlib.sha256(json.dumps(chosen, sort_keys=True).encode()).hexdigest()


def check_manifests(root: Path, blocks, cfg: Cfg) -> dict:
    """The pre-registered values (fixed before any result, spec section 10) against what each block's manifest
    recorded when it ran. A difference is reported, never corrected."""
    expected = {"ni_margin_accuracy": cfg.ni_margin, "rank_agreement_threshold": cfg.rank_threshold}
    checked, mismatch = 0, []
    for block in sorted(blocks):
        for path in sorted((root / block).glob("manifest-*.json")):
            checked += 1
            where = {"block": block, "file": path.name}
            try:
                recorded = json.loads(path.read_text(encoding="utf-8"))["pre_registered"]
                differs = [(k, recorded.get(k)) for k in PRE_REGISTERED if recorded.get(k) != expected[k]]
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                mismatch.append({**where, "key": "manifest", "manifest": "unreadable", "settings": None})
                continue
            mismatch += [{**where, "key": k, "manifest": v, "settings": expected[k]} for k, v in differs]
    return {"checked": checked, "pre_registered_mismatch": mismatch}


def verdicts(found: Mapping) -> dict:
    """{hypothesis: {platform: {challenger: supported} | status}} for the confirmatory hypotheses."""
    return {name: {p: (found[name][p]["status"] if found[name][p]["status"] != "ok" else
                       {ch: bool(read(body)) for ch, body in found[name][p]["challengers"].items()})
                   for p in PLATFORMS}
            for name, read in SUPPORTED.items()}


def sink_attacker_excluded(t: table.Table) -> int:
    """Attacked rows whose sink is an insider: metrics gives them a NaN protected-sink harm, so every paired test drops
    them (their count per test is `n_dropped_nan`)."""
    attacked = t.codes["attack_free"] == 0
    return int(np.nansum(t.metrics["sink_is_attacker"][attacked] == 1.0)) if t.n else 0


def build(root: Path, tables: Mapping[str, table.Table], cfg: Cfg, settings: Mapping) -> dict:
    def per_platform(fn):
        return {p: fn(tables[p], cfg, p) for p in PLATFORMS}

    found = {name: per_platform(fn) for name, fn in HYPOTHESES}
    blocks = tuple(dict.fromkeys(b for p in PLATFORMS for b in cfg.all_blocks(p)))
    return {
        "meta": {"profile": cfg.profile, "results_dir": str(root),
                 "pre_registered": {"ni_margin_accuracy": cfg.ni_margin,
                                    "rank_agreement_threshold": cfg.rank_threshold},
                 "settings_sha256": fingerprint(settings), "analysis_settings": core.get(settings, "analysis"),
                 "manifests": check_manifests(root, blocks, cfg),
                 "platforms": {p: {"rows": t.n, "blocks": list(cfg.all_blocks(p)), "blocks_missing": t.missing_blocks,
                                   "torn_lines": t.torn_lines, "empty_metrics": t.empty_metrics,
                                   "sink_attacker_excluded": sink_attacker_excluded(t)}
                               for p, t in tables.items()}},
        "hypotheses": found,
        "secondary": {**{name: per_platform(fn) for name, fn in SECONDARY},
                      "rank_agreement": rank.rank_agreement(tables, cfg)},
        "containment_per_cost": per_platform(containment.containment),
        "verdicts": verdicts(found),
    }


def analyze(results_dir, settings: Mapping, out=None) -> dict:
    """Read the rows under `results_dir` (the folder that holds the block folders), run every analysis and write the
    report to `out` (default `<results_dir>/analysis_report.json`). Returns the report as written.

    Raises core.Unset naming the first setting that is not set, before anything is read; FileNotFoundError if
    `results_dir` is not a directory; ValueError for an invalid setting or malformed rows.
    """
    cfg = Cfg.from_settings(settings)
    root = Path(results_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"results directory {root} does not exist")
    tables = {p: table.load(root, cfg.all_blocks(p), cfg.metric_columns,
                            cfg.recovery_margin if p == "llm" else None) for p in PLATFORMS}
    report = jsonable(build(root, tables, cfg, settings))
    target = Path(out) if out else root / REPORT_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return report
