"""Stand-ins for bazt_lab.features.analysis.public.analyze, to test the end-to-end harness. Not a test module."""
from __future__ import annotations

from pathlib import Path

from bazt_lab.core import public as core


def analyze(results_dir: Path, settings) -> dict:
    """Counts the rows under results_dir/<block>/units; returns {} when there are none (the harness must refuse)."""
    rows = sum(len(p.read_text(encoding="utf-8").splitlines()) for p in Path(results_dir).glob("*/units/*.jsonl"))
    return {"rows": rows, "profile": settings["profile"]} if rows else {}


def echo(results_dir: Path, settings) -> dict:
    """Returns what the harness passed in, so a test can see it."""
    return {"results_dir": str(results_dir), "profile": settings["profile"],
            "n_insiders": core.get(settings, "runtime.n_insiders"), "baseline": core.get(settings, "analysis.baseline")}


def needs_setting(results_dir: Path, settings) -> dict:
    """An analysis whose first setting is not set: the harness must name the key."""
    return {"alpha": core.get(settings, "analysis.never_set_anywhere")}
