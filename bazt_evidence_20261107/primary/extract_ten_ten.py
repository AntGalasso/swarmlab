"""Reproduce descriptive prespecified sprint-cell means from authoritative CI artifacts.

No inference, model calls, source/result mutations or selection of favorable budget cells.
Usage: python extract_ten_ten.py [--artifacts-dir PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

RUN = 42381690293
FAMILIES = ("aitm", "infection", "insider_ma", "insider_pi", "insider_ta")
ARMS = ("C", "BAZT", "BAZT_upstream")
TOPOS = ("barabasi_albert", "chain", "small_world", "star_hub", "star_leaf")
DRAW_SEED, DRAWS = 20261107, 20000


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def extract(artifacts, out):
    out.mkdir(parents=True, exist_ok=True)
    rows, sources = {}, []
    for topo in TOPOS:
        path = artifacts / f"bazt-headline-evaluation-{topo}.zip"
        with ZipFile(path) as archive:
            meta = {name: archive.read(name).decode().strip() for name in
                    ("source_commit.txt", "settings_sha256.txt") if name in archive.namelist()}
            selected_files = 0
            for name in archive.namelist():
                if "/mock_frontier/units/" not in name or not name.endswith(".jsonl"):
                    continue
                with archive.open(name) as stream:
                    first = stream.readline()
                    if not first:
                        continue
                    row = json.loads(first)
                    if (row["audit"], row["intervention"]) != (.1, .1) or row["arm"] not in ARMS:
                        continue
                    selected_files += 1
                    for row in [row, *(json.loads(line) for line in stream if line.strip())]:
                        key = (row["topology"], row["arm"], row["attack"], row["seed"], row["sink"])
                        if key in rows:
                            raise RuntimeError(f"Duplicate authoritative row: {key}")
                        if row["topology"] != topo or row["agents"] != 12 or row["violations"] != 0:
                            raise RuntimeError("Wrong topology/N or budget violation in selected cell.")
                        rows[key] = row
            sources.append({"file": path.name, "sha256": sha256(path),
                            "selected_batch_files": selected_files, **meta})
    seeds, sinks = list(range(1, 31)), list(range(12))
    expected = {(t, a, f, s, i) for t in TOPOS for a in ARMS for f in (*FAMILIES, "none")
                for s in seeds for i in sinks}
    if set(rows) != expected:
        raise RuntimeError(f"Grid mismatch missing={len(expected - set(rows))}, extra={len(set(rows) - expected)}")
    arrays = {}
    for topo in TOPOS:
        for arm in ARMS:
            arrays[topo, arm] = {
                # Explicit nested averages: sinks, families, then independent seeds.
                "asr": np.array([np.mean([np.mean([rows[topo, arm, fam, seed, i]["asr_final"]
                                                   for i in sinks]) for fam in FAMILIES]) for seed in seeds]),
                "clean_accuracy": np.array([np.mean([rows[topo, arm, "none", seed, i]["accuracy_clean"]
                                                       for i in sinks]) for seed in seeds])}
    rng = np.random.default_rng(DRAW_SEED)
    bootstrap_indices = rng.integers(0, len(seeds), size=(DRAWS, len(seeds)))
    table = []
    for topo in TOPOS:
        row = {"topology": topo, "n_independent_seed_clusters": len(seeds)}
        for arm in ARMS:
            row[f"{arm}_asr"] = float(arrays[topo, arm]["asr"].mean())
            row[f"{arm}_clean_accuracy"] = float(arrays[topo, arm]["clean_accuracy"].mean())
        for metric in ("asr", "clean_accuracy"):
            differences = arrays[topo, "BAZT"][metric] - arrays[topo, "C"][metric]
            lo, hi = np.quantile(differences[bootstrap_indices].mean(1), [.025, .975])
            row[f"BAZT_minus_C_{metric}"] = float(differences.mean())
            row[f"BAZT_minus_C_{metric}_pointwise_95ci_low"] = float(lo)
            row[f"BAZT_minus_C_{metric}_pointwise_95ci_high"] = float(hi)
        table.append(row)
    result = {"workflow_run": RUN, "classification": "DESCRIPTIVE PRELIMINARY; full-grid adjusted statistics separate",
              "budget": {"audit_fraction": .1, "intervention_fraction": .1},
              "selection": "Prespecified sprint (10%,10%) cell, not chosen for observed performance",
              "population": {"N": 12, "topologies": list(TOPOS), "families": list(FAMILIES),
                             "seeds": seeds, "sinks": sinks, "selected_rows": len(rows)},
              "aggregation": "Mean 12 sinks inside each family/seed, equal mean of five families inside seed, then mean of 30 seeds; clean utility uses each seed's 12 attack-free sinks once.",
              "uncertainty": {"method": "paired seed bootstrap", "draws": DRAWS, "random_seed": DRAW_SEED,
                              "interval": "pointwise percentile 95%, no multiplicity adjustment",
                              "not": "No confirmatory Pareto/NI labels or full-family p-values are issued here."},
              "table": table,
              "per_seed": {topo: {arm: {m: v.tolist() for m, v in arrays[topo, arm].items()}
                                   for arm in ARMS} for topo in TOPOS},
              "provenance": {"source_artifacts": sources, "extractor_sha256": sha256(Path(__file__)),
                             "raw_rows_modified": False, "live_calls": 0}}
    write(out / "ten_ten_summary.json", result)
    with (out / "ten_ten_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    lines = ["# Preliminary interpretation of the prespecified 10/10 cell", "",
             "These numbers come from workflow 42381690293 and the frozen (10%,10%) sprint cell. They are descriptive while the full-grid analysis runs. All values below are percentages; differences and intervals are percentage points.", "",
             "| Topology | ASR C | ASR BAZT | BAZT−C ASR [pointwise 95% CI] | Clean accuracy C | Clean accuracy BAZT | BAZT−C clean accuracy [pointwise 95% CI] |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in table:
        f = lambda key: f"{100 * row[key]:.2f}"
        lines.append(f"| {row['topology']} | {f('C_asr')} | {f('BAZT_asr')} | {f('BAZT_minus_C_asr')} [{f('BAZT_minus_C_asr_pointwise_95ci_low')}, {f('BAZT_minus_C_asr_pointwise_95ci_high')}] | {f('C_clean_accuracy')} | {f('BAZT_clean_accuracy')} | {f('BAZT_minus_C_clean_accuracy')} [{f('BAZT_minus_C_clean_accuracy_pointwise_95ci_low')}, {f('BAZT_minus_C_clean_accuracy_pointwise_95ci_high')}] |")
    lines += ["", "Thirty seeds are the independent clusters. All 12 sinks and five attack families are averaged within each seed. Clean utility uses each seed's 12 clean sinks once. Bootstrap intervals use 20,000 paired seed resamples and are pointwise: they do not establish full-grid significance, non-inferiority or Pareto labels.", "",
              "The large ASR reductions on Barabasi–Albert, chain and small-world are encouraging for the BAZT lab stack. Both star topologies show small absolute ASR changes; their clean accuracy is already low because the diameter-scaled horizon leaves Metropolis consensus far from convergence. Report absolute ASR together with clean accuracy and the matched clean-reference degradation; do not label ordinary baseline consensus errors as attack successes caused by the attacker.", "",
              "The mock gives BAZT and BAZT_upstream a free monitor that C does not receive. These comparisons therefore establish behavior of the evaluated stacks, rather than isolating the allocator/topology mechanism. Insider PI/TA/MA also share closely related parametric dynamics here. A separate LLM task validation is required for broader semantic claims.", "",
              "BAZT_upstream is a separate adapter and must not inherit BAZT's conclusions. Its means are retained in the JSON/CSV; it loses substantial clean utility on several topologies.", "",
              "Do not change the already frozen LLM model/task/topology/budget scope after seeing these results. The full budget frontier, adjusted tests, utility non-inferiority and method limitations belong in the final report.", "",
              "Reproduce: `python extract_ten_ten.py --artifacts-dir ../artifacts_42381690293 --out .` (run from this directory with NumPy installed). Artifact/source/settings hashes and every per-seed value are in ten_ten_summary.json. No raw result rows are changed and no model calls are made."]
    (out / "INTERPRETATION.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"selected_rows": len(rows), "table": table}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(__file__).resolve().parents[1] / f"artifacts_{RUN}")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    extract(args.artifacts_dir.resolve(), args.out.resolve())
