"""run_search: the best responses of every cell of the mock_adaptive evaluation units, saved as they are found."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional

from bazt_lab.core import public as core
from bazt_lab.features.adaptive.private import search as se
from bazt_lab.features.adaptive.private import space as sp
from bazt_lab.features.adaptive.private import table as tb

TABLE_DIR = "best_responses"        # under the output directory: meta.json and one file per finished cell
PROGRESS_DIR = "progress"           # under the output directory: one jsonl per cell being searched
VERSION = 1
# Keys that cannot change what an episode does or which strategy wins: batch size (performance only), the seed splits
# (the search seeds are checked against them), which arms, cells, topologies and conditions a block covers (every cell
# is searched on its own), the calibration inputs (they enter each cell's own digest), the LLM side, tracking and the
# dry-run estimates. Every other setting is part of the fingerprint, so a changed model, auditor, cost or run-time
# value refuses a stale table instead of reusing it.
IGNORED = (
    "profile", "llm", "gnn_train", "gnn", "bazt_upstream", "bazt_slot", "llm_data", "analysis", "tracking",
    "estimates", "budgets.levels_percent", "budgets.alpha",
    "mock.batch_size", "mock.main_seeds", "mock.heldout_seeds", "mock.tune_seeds", "mock.topologies", "mock.families",
    "mock.frontier_agents", "mock.epidemiology_agents", "mock.uncapped_arms", "mock.capped_arms",
    "mock.allocation_arms", "mock.calibrated_arms", "mock.epidemiology", "mock.ablations", "mock.presets_robustness",
    "mock.costs_robustness", "mock.legacy", "mock.adaptive.arms", "mock.adaptive.cells", "mock.adaptive.conditions",
    "mock.adaptive.max_candidates", "runtime.llm_seeds", "runtime.llm_calibration_seeds", "runtime.llm_sink_positions",
    "runtime.ni_margin_accuracy", "runtime.rank_agreement_threshold", "runtime.recovery_margin",
)


def fingerprint(settings: Mapping) -> str:
    tree = copy.deepcopy(dict(settings))
    for dotted in IGNORED:
        *path, leaf = dotted.split(".")
        node = tree
        for part in path:
            node = node.get(part) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(leaf, None)
    text = json.dumps({"version": VERSION, "metric": se.METRIC, "settings": tree}, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _digest(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def load_best_responses(out_dir, settings: Mapping) -> tb.BestResponses:
    """The saved table, refused if `settings` differ from those it was searched with (see `fingerprint`)."""
    return tb.BestResponses.load(Path(out_dir) / TABLE_DIR, fingerprint(settings))


def run_search(units: Iterable[Mapping], arm_of: Callable[[str], core.ArmSpec], platform: core.Platform,
               settings: Mapping, out_dir, thresholds_of: Optional[Callable[[Mapping], Mapping]] = None,
               tracker: Optional[core.Tracker] = None) -> tb.BestResponses:
    """Search the best response of each cell of the evaluation `units` (mock_adaptive block units; attack-free units
    are skipped) and return the table. `arm_of(name)` gives the arm's ArmSpec, `thresholds_of(cell)` its calibrated
    thresholds (none by default). Refuses before writing anything; resumes after a crash at the last saved chunk."""
    space = sp.load_space(settings)
    batch_size = core.get(settings, "mock.batch_size")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise sp.Refusal(f"mock.batch_size must be a positive integer, got {batch_size!r}")
    jobs = se.plan_jobs(units, space)
    specs = {name: arm_of(name) for name in dict.fromkeys(j.arm for j in jobs)}
    thresholds_of = thresholds_of or (lambda cell: {})
    tracker = tracker or core.NullTracker()

    out = Path(out_dir)
    table_dir, fp = out / TABLE_DIR, fingerprint(settings)
    tb.ensure_meta(table_dir, fp, dataclasses.asdict(space))
    sizes = [sp.checked_count(j.family, j.agents, j.condition, space) * len(j.pairs) for j in jobs]
    total, finished = sum(sizes), 0
    for job, size in zip(jobs, sizes, strict=True):
        thresholds = dict(thresholds_of(job.cell))
        inputs = _digest(thresholds)
        if tb.has_cell(table_dir, job.cell, inputs) and tb.covers(table_dir, job.cell, job.groups):
            finished += size
            continue
        cands = sp.candidates(job.family, job.agents, job.condition, space)
        key = _digest(fp, job.cell, job.level, job.pairs, sorted(job.ids.items()), inputs)
        progress = out / PROGRESS_DIR / f"{key}.jsonl"

        def on_chunk(done: int, base: int = finished) -> None:
            tracker.log({"adaptive/episodes_done": base + done, "adaptive/episodes_total": total}, step=base + done)

        def run_batch(batch, arm_spec=specs[job.arm]):
            # the platform gets a tracker that records nothing: search episodes are not result units (mock_units table)
            return se.run_chunk(platform, arm_spec, settings, core.NullTracker(), batch)

        vals = se.evaluate(job, cands, run_batch, progress, batch_size, on_chunk, thresholds)
        fresh = se.choose(job, cands, vals, se.victims_of(job.family, job.agents, space.n_insiders))
        kept = [r for r in tb.read_records(table_dir, job.cell) if (r["seed"], r["sink"]) not in job.groups]
        tb.write_cell(table_dir, job.cell, kept + fresh, inputs)
        progress.unlink(missing_ok=True)
        finished += size
    return tb.BestResponses.load(table_dir, fp)
