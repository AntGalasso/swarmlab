"""Stand-ins for bazt_lab.features.analysis.public.analyze, to test the end-to-end harness. Not a test module."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from bazt_lab.core import public as core

PLATFORMS = ("mock", "llm")


def _rows(path: Path) -> int:
    """Lines of a rows file that are a JSON object (a torn line is not a row)."""
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            count += isinstance(json.loads(line), dict)
        except json.JSONDecodeError:
            continue
    return count


def _committed(block_dir: Path) -> int:
    """The analysis' reader: core.committed_row_files, units/ and every worker shard, done marker required."""
    try:
        files, _ = core.committed_row_files(block_dir)
    except FileNotFoundError:
        return 0
    return sum(_rows(path) for path in files)


def _top_folder(block_dir: Path) -> int:
    """A regressed reader: units/<stem>.jsonl with its done marker, the worker shards units/w<k>/ never read."""
    units = block_dir / "units"
    return sum(_rows(p) for p in sorted(units.glob("*.jsonl")) if p.with_suffix(".done").exists())


def _report(results_dir: Path, settings, count: Callable[[Path], int]) -> dict:
    """{"meta": {"platforms": {p: {"rows", "blocks", "blocks_missing"}}}} like the real report, for the blocks that
    analysis.blocks names (none when it is not set)."""
    try:
        blocks = core.get(settings, "analysis.blocks")
    except core.Unset:
        blocks = {}
    platforms = {}
    for platform in PLATFORMS:
        names = list(dict.fromkeys(b for per_platform in blocks.values() for b in per_platform.get(platform, [])))
        rows = {name: count(Path(results_dir) / name) for name in names}
        platforms[platform] = {"rows": sum(rows.values()), "blocks": names,
                               "blocks_missing": [name for name, n in rows.items() if not n], "uncommitted_files": 0}
    return {"meta": {"profile": settings["profile"], "platforms": platforms}}


def analyze(results_dir: Path, settings) -> dict:
    """Counts the committed rows of the blocks analysis.blocks names, per platform, the way the real analysis reads them;
    returns {} when there are none (the harness must refuse)."""
    report = _report(results_dir, settings, _committed)
    return report if any(p["rows"] for p in report["meta"]["platforms"].values()) else {}


def top_folder_only(results_dir: Path, settings) -> dict:
    """The analysis with the shard reader regressed: it never reads units/w<k>/ and, like the real analysis, still
    returns a full report (rows 0, the block in blocks_missing) when it read nothing."""
    return _report(results_dir, settings, _top_folder)


def echo(results_dir: Path, settings) -> dict:
    """Returns what the harness passed in, so a test can see it (and an empty meta.platforms: it reads no block)."""
    return {"results_dir": str(results_dir), "profile": settings["profile"],
            "n_insiders": core.get(settings, "runtime.n_insiders"), "baseline": core.get(settings, "analysis.baseline"),
            "meta": {"platforms": {}}}


def needs_setting(results_dir: Path, settings) -> dict:
    """An analysis whose first setting is not set: the harness must name the key."""
    return {"alpha": core.get(settings, "analysis.never_set_anywhere")}
