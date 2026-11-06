"""Command line: python -m bazt_lab.features.analysis [--profile P] [--set key=value] [--results-root DIR] [--out FILE]

Reads the committed rows files of <results-root>/<profile>/<block>/units/ and of its worker shards units/<shard>/ (a
<stem>.jsonl with its <stem>.done beside it), writes <results-root>/<profile>/analysis_report.json and prints the
verdicts. Exit 0 on success, 2 on a refusal (a setting that is not set, a missing results directory, malformed rows).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Mapping, Sequence

from bazt_lab.core import public as core
from bazt_lab.features.analysis.private import config, report
from bazt_lab.features.analysis.private.compare import CRITERION
from bazt_lab.features.analysis.private.config import PLATFORMS

DEFAULT_ROOT = core.SETTINGS_PATH.parents[1] / "results" / "experiments"


def _comparison_line(name: str, platform: str, challenger: str, body: Mapping, verdict: bool) -> str:
    s = body["summary"]
    counts = s["label_counts"]
    return (f"{name:<12} {platform:<5} {challenger:<14} {'SUPPORTED' if verdict else 'not supported':<13} "
            f"wins {s['n_wins']}/{s['n_comparisons']}  utility_loss {s['n_utility_loss']}  "
            f"strict {counts['STRICT_PARETO_WIN']}  "
            f"{'complete' if s['complete'] else 'INCOMPLETE (cells skipped)'}")


def render(r: Mapping) -> str:
    meta, lines = r["meta"], []
    lines.append(f"analysis of profile {meta['profile']} in {meta['results_dir']}")
    for p in PLATFORMS:
        info = meta["platforms"][p]
        missing = f", blocks without data: {', '.join(info['blocks_missing'])}" if info["blocks_missing"] else ""
        k = info["uncommitted_files"]
        uncommitted = f", {k} uncommitted rows file{'s' if k != 1 else ''} not read" if k else ""
        lines.append(f"  {p}: {info['rows']} rows{uncommitted}{missing}")
    lines.extend(f"WARNING {m['block']}/{m['file']}: pre-registered {m['key']} was {m['manifest']} when the block "
                 f"ran, the settings now say {m['settings']}" for m in meta["manifests"]["pre_registered_mismatch"])
    lines += ["", f"confirmatory hypotheses ({CRITERION})"]
    for name, per_platform in r["verdicts"].items():
        for p, v in per_platform.items():
            if isinstance(v, str):
                lines.append(f"{name:<12} {p:<5} {v}")
                continue
            for ch, ok in v.items():
                lines.append(_comparison_line(name, p, ch, r["hypotheses"][name][p]["challengers"][ch], ok))
    lines.append("")
    for p in PLATFORMS:
        e = r["hypotheses"]["H-efficiency"][p]
        if e["status"] != "ok":
            lines.append(f"H-efficiency {p:<5} {e['status']}")
            continue
        for ch, body in e["challengers"].items():
            reach = sum(f["smallest"] is not None for f in body["frontiers"])
            lines.append(f"H-efficiency {p:<5} {ch:<14} reaches the uncapped {body['baseline']} in "
                         f"{reach}/{len(body['frontiers'])} frontiers (descriptive)")
    for p in PLATFORMS:
        s = r["secondary"]["ladder"][p]
        lines.append(f"ladder       {p:<5} " + (s["status"] if s["status"] != "ok" else
                                                 f"holds={s['holds']} ({s['ordered_cells']}/{s['n_cells']} cells ordered)"))
    for p in PLATFORMS:
        s = r["secondary"]["ablations"][p]
        lines.append(f"ablations    {p:<5} " + (s["status"] if s["status"] != "ok" else
                                                 f"wins {s['summary']['n_wins']}/{s['summary']['n_comparisons']}"))
    ra = r["secondary"]["rank_agreement"]
    lines.append("rank agreement " + (f"not evaluable: {ra['reason']}" if ra["status"] != "ok" else
                                      f"{ra['statistic']}={ra['value']} threshold {ra['threshold']} "
                                      f"supported={ra['supported']}"))
    return "\n".join(lines)


def _refuse(message: str) -> int:
    print(f"Refusing analysis: {message}", file=sys.stderr)
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bazt_lab.features.analysis", description=__doc__.split("\n\n")[0])
    ap.add_argument("--settings", default=str(core.SETTINGS_PATH), help="settings file (default: experiments/profiles.toml)")
    ap.add_argument("--profile", default=None, help="profile (default: active_profile of the settings file)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override one setting (repeatable)")
    ap.add_argument("--results-root", default=str(DEFAULT_ROOT), help="folder that holds one folder per profile")
    ap.add_argument("--out", default=None, help="report file (default: <results-root>/<profile>/analysis_report.json)")
    args = ap.parse_args(argv)
    try:
        settings = core.load_settings(Path(args.settings), args.profile, args.set)
        missing: List[str] = config.missing_keys(settings)
        if missing:
            return _refuse("settings not set: " + ", ".join(missing))
        result = report.analyze(Path(args.results_root) / settings["profile"], settings, args.out)
    except (OSError, ValueError) as e:                       # includes FileNotFoundError and malformed rows
        return _refuse(str(e))
    print(render(result))
    return 0
