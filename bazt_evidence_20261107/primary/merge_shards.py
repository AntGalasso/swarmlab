#!/usr/bin/env python3
"""Validate and merge disjoint mock topology artifacts for the existing seed-paired analysis.

Downloads are read only. Only committed row/done pairs are copied, under distinct CI shard names.
An incomplete grid, duplicate unit, altered model configuration or budget violation refuses the merge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source" if (HERE / "source/bazt_lab").exists() else HERE.parent
sys.path.insert(0, str(SOURCE))
from bazt_lab.core import public as core
from bazt_lab.features.runner.private import identity
from bazt_lab.features.runner.private.blocks import BLOCKS
from bazt_lab.features.runner.private.units import unit_id


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge(settings_path: Path, artifacts: Path, out_root: Path, block="mock_frontier"):
    s = core.load_settings(settings_path)
    profile = s["profile"]
    if out_root.resolve().is_relative_to(artifacts.resolve()):
        raise ValueError("Output must be outside the read-only artifact tree.")
    destination = out_root / profile
    if destination.exists():
        raise ValueError(f"Destination already exists: {destination}")
    sources = sorted(p for p in artifacts.rglob(block) if p.is_dir() and p.parent.name == profile)
    if not sources:
        raise ValueError(f"No artifacts contain {profile}/{block}")
    grid = BLOCKS[block].units(s, True)
    expected = {unit_id(u) for u in grid}
    seen = set()
    model = identity.block_view(s, block, "mock")
    allowed_changes = {"mock.topologies", "mock.epidemiology.topologies"}
    reports, calibration_hashes, source_commits = [], set(), set()
    out_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="merge-staging-", dir=out_root) as td:
        staging_profile = Path(td) / profile
        output = staging_profile / block
        output.mkdir(parents=True)
        for index, source in enumerate(sources):
            saved = json.loads((source / "settings.json").read_text())["settings"]
            for key in set(model) | set(saved):
                if key not in allowed_changes and model.get(key) != saved.get(key):
                    raise ValueError(f"Model setting differs in {source}: {key}")
            tops = saved.get("mock.epidemiology.topologies" if block == "mock_epidemiology" else "mock.topologies", [])
            planned_tops = model.get("mock.epidemiology.topologies" if block == "mock_epidemiology" else "mock.topologies", [])
            if not tops or not set(tops).issubset(planned_tops):
                raise ValueError(f"Unexpected topology scope in {source}: {tops}")
            files, orphans = core.committed_row_files(source)
            count = 0
            for rows_path in files:
                done = rows_path.with_suffix(".done")
                listed = json.loads(done.read_text()) if done.stat().st_size else [done.stem]
                actual = []
                with rows_path.open() as stream:
                    for line in stream:
                        row = json.loads(line)
                        uid = row["id"]
                        if uid in seen:
                            raise ValueError(f"Duplicate unit across committed rows: {uid}")
                        if uid not in expected:
                            raise ValueError(f"Unexpected unit outside planned grid: {uid}")
                        if row.get("platform") != "mock" or row.get("violations") != 0:
                            raise ValueError(f"Invalid platform or budget violation in {uid}")
                        # The upstream row contract leaves attack metrics undefined on clean rows.
                        # Attacked accuracy_clean can be deferred to its clean twin; analysis fills it.
                        required = ("accuracy_clean",) if row.get("attack") == "none" else ("asr_final", "accuracy_attack")
                        for metric in required:
                            value = row.get(metric)
                            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                                raise ValueError(f"Invalid {metric} in {uid}")
                        if row.get("attack") != "none":
                            clean = row.get("accuracy_clean")
                            if not isinstance(clean, (int, float)) or not math.isfinite(clean):
                                if row.get("twin_id") not in expected:
                                    raise ValueError(f"Missing clean twin for deferred utility in {uid}")
                        actual.append(uid)
                        seen.add(uid)
                if len(actual) != len(set(actual)) or set(actual) != set(listed):
                    raise ValueError(f"Rows do not match committed done list: {rows_path}")
                shard = f"ci-{index}-" + (rows_path.parent.name if rows_path.parent.name != "units" else "main")
                target = output / "units" / shard
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(rows_path, target / rows_path.name)
                shutil.copy2(done, target / done.name)
                count += len(actual)
            for manifest in source.glob("manifest-*.json"):
                shutil.copy2(manifest, output / (f"manifest-ci-{index}-" + manifest.name[9:]))
            shutil.copy2(source / "settings.json", output / f"source-settings-ci-{index}.json")
            calibration = source.parent / "mock_calibration"
            thresholds = calibration / "thresholds.json"
            calibration_hashes.add(digest(thresholds))
            if index == 0:
                shutil.copytree(calibration, staging_profile / "mock_calibration")
            commit = source.parents[1] / "source_commit.txt"
            if commit.exists():
                source_commits.add(commit.read_text().strip())
            reports.append({"source": str(source), "topologies": tops, "units": count,
                            "committed_files": len(files), "ignored_uncommitted_files": orphans})
        if len(calibration_hashes) != 1:
            raise ValueError("Shards used different calibration thresholds.")
        if len(source_commits) > 1:
            raise ValueError("Shards were produced by different source commits.")
        missing = expected - seen
        if missing:
            raise ValueError(f"Incomplete grid: missing {len(missing)} of {len(expected)} units")
        (output / "settings.json").write_text(json.dumps({"settings": model}, indent=2) + "\n")
        report = {"status": "COMPLETE_DISJOINT_GRID", "block": block, "profile": profile,
                  "expected_units": len(expected), "merged_units": len(seen), "shards": reports,
                  "thresholds_sha256": next(iter(calibration_hashes)),
                  "source_commits": sorted(source_commits), "settings_sha256": digest(settings_path),
                  "pair_unit_mock": s["analysis"]["pair_unit"]["mock"],
                  "utility_endpoint": s["analysis"]["utility_endpoint"]}
        (staging_profile / "merge_validation.json").write_text(json.dumps(report, indent=2) + "\n")
        staging_profile.rename(destination)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--settings", type=Path, required=True)
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--block", choices=("mock_frontier", "mock_epidemiology"), default="mock_frontier")
    a = p.parse_args()
    print(json.dumps(merge(a.settings, a.artifacts, a.out_root, a.block), indent=2))


if __name__ == "__main__":
    main()
