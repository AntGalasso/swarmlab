"""Launching a block: refusals first (nothing is written until every one has passed), then the manifest and the
tracking run, then calibration cells or batches of units.

A mock block runs in batches of `mock.batch_size` units that share an arm, budget cell, topology, attack and N; an LLM
block runs one unit (a seed run) per batch. Platform.run_batch and Platform.calibrate append rows to the block's
CheckpointStore as they produce them, so a crashed launch resumes after the last saved row; the runner skips the
units the store already marks done and the calibration cells already saved in thresholds.json.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import itertools
import math
import platform as pyplatform
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy
import torch

from bazt_lab.core import public as core
from bazt_lab.features.runner.adapters import kickoff, workers
from bazt_lab.features.runner.private import files, identity, ledger, matched, needs, replay, thresholds
from bazt_lab.features.runner.private.blocks import BLOCKS, CALIBRATION_BLOCK, Block, unrunnable
from bazt_lab.features.runner.private.runtime import PRE_REGISTERED, missing_runtime
from bazt_lab.features.runner.private.units import cell_key, group_key, unit_id
from bazt_lab.features.runner.private.wiring import FeatureMissing, Wiring

ROOT = Path(__file__).resolve().parents[4]
RESULTS_ROOT = ROOT / "results" / "experiments"
FREEZE_MARKER = "ANALYSIS_FROZEN"      # created by hand in the results root when the analysis is frozen


@dataclass
class _Plan:
    grid: object                        # a UnitList or MockGrid
    specs: dict                         # arm name -> ArmSpec, for every arm of the grid
    platform: core.Platform
    batch_size: int
    thresholds_path: Path               # the calibration block's thresholds.json
    found: dict                         # evaluation blocks: {cell: thresholds}
    sha: str | None                     # sha256 of thresholds.json as read, for the manifest
    alpha: float | None                 # calibration blocks only
    adaptive: object | None             # the adaptive feature's public module, for mock_adaptive
    tracker: core.Tracker
    tracking: bool
    sync: object | None = None          # the sync feature's public module when sync.enabled is true
    view: dict | None = None            # the block's settings identity, written on its first launch
    out: Path | None = None             # the block directory
    manifest: dict | None = None        # the manifest being written (calibration diagnostics are added to it)
    manifest_path: Path | None = None
    workers: int = 1                    # compute.workers: processes that run the batches of a mock evaluation block
    gnn_skipped: dict | None = None     # {(arm, attack, task): units} of GNN cells that have no GNN (fallback none)
    replay: object | None = None        # llm_adaptive_check: the replayed best responses (private/replay.py)
    timing: object | None = None        # D116: the BAZT rounds the matched-timing controls replay (private/matched.py)
    matched_thresholds: str | None = None   # mock.legacy.matched_thresholds when a calibrated matched arm runs


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _tracking(s: Mapping, name: str, wiring: Wiring):
    """(tracker, enabled). The key is optional: unset means tracking is off, as in the NullTracker contract."""
    if not core.is_set(s, "tracking.enabled"):
        return core.NullTracker(), False
    enabled = core.get(s, "tracking.enabled")
    if not isinstance(enabled, bool):
        sys.exit(f"Refusing {name}: tracking.enabled must be true or false, got {enabled!r}.")
    if not enabled:
        return core.NullTracker(), False
    try:
        return wiring.tracker(s), True
    except RuntimeError as e:      # FeatureMissing, or the tracking feature's error naming the setting or variable
        sys.exit(f"Refusing {name}: {e}")


def _arm_specs(block: Block, grid, wiring: Wiring) -> dict:
    specs = {}
    for arm in sorted(grid.arms()):
        if matched.is_matched(arm) and block.kind == "evaluate" and not block.timing_source:
            sys.exit(f"Refusing {block.name}: arm {arm!r} is a matched-timing control: it acts at the rounds BAZT "
                     f"intervened in the same unit (D116), which only a block with a timing source hands over "
                     f"({', '.join(f'{b.name} from {b.timing_source}' for b in BLOCKS.values() if b.timing_source)}); "
                     f"{block.name} has none.")
        try:
            spec = wiring.arm(arm)
        except (FeatureMissing, LookupError) as e:
            sys.exit(f"Refusing {block.name}: {e}")
        if block.platform not in spec.platforms:
            sys.exit(f"Refusing {block.name}: arm {arm!r} does not run on the {block.platform} platform "
                     f"(ArmSpec.platforms = {spec.platforms}).")
        specs[arm] = spec
    return specs


def _gnn_feature(wiring: Wiring):
    if wiring.gnn is None:
        raise FeatureMissing("gnn", "the composition root wires no gnn feature")
    return wiring.gnn()


def _gnn_checkpoints(s: Mapping, name: str, grid, wiring: Wiring) -> dict:
    """Every GNN family the block deploys must be trained and finished before any arm of the block runs. Cells with no
    GNN (deploy_family is None, e.g. gnn.fallback_family = none) are skipped: {(arm, attack, task): units}."""
    gnn = _gnn_feature(wiring)
    families: dict = {}
    skipped: dict = {}
    for unit in grid:
        if unit["arm"] in needs.GNN_ARMS:
            try:
                family = gnn.deploy_family(unit["attack"], unit["task"], s)
            except ValueError as e:
                sys.exit(f"Refusing {name}: {e}")
            if family is None:
                key = (unit["arm"], unit["attack"], unit["task"])
                skipped[key] = skipped.get(key, 0) + 1
            else:
                families[family] = None
    for family in families:
        if not gnn.training_finished(s, family):
            sys.exit(f"Refusing {name}: the GNN checkpoint of family {family!r} is missing or not finished; run "
                     f"--block gnn_train first (a crashed training resumes).")
    return skipped


def _finish_gnn_training(s: Mapping, name: str, wiring: Wiring) -> None:
    """Even when every unit is done, a crash inside the training leaves it unfinished: resume it, then check."""
    gnn = _gnn_feature(wiring)
    gnn.train_pending(s)
    unfinished = [f"{a}__{t}" for a, t in map(tuple, core.get(s, "gnn_train.families"))
                  if not gnn.training_finished(s, f"{a}__{t}")]
    if unfinished:
        sys.exit(f"{name}: GNN training is not finished for {', '.join(unfinished)}; launch the block again "
                 f"(a crashed training resumes).")


def _oracle_thresholds(name: str, grid, found: Mapping) -> None:
    """D96: oracle_alloc audits and must act at random_alloc's calibrated audit_score; refuse when it is missing."""
    cells = sorted((c for c in grid.cells() if c[0] == needs.ORACLE), key=lambda c: tuple(map(str, c)))
    for cell in cells:
        source = (needs.ORACLE_SOURCE, *cell[1:])
        if "audit_score" not in found.get(source, {}):
            sys.exit(f"Refusing {name}: {needs.ORACLE} needs the calibrated audit_score of {needs.ORACLE_SOURCE} "
                     f"({thresholds.describe(source)}), which is missing; list {needs.ORACLE_SOURCE} in "
                     f"mock.calibrated_arms and run the calibration block first.")


def oracle_source_cell(unit: Mapping) -> tuple:
    return (needs.ORACLE_SOURCE, *cell_key(unit)[1:])


SETTINGS_FILE = "settings.json"


def _sync_feature(s: Mapping, name: str, wiring: Wiring):
    """The sync feature when sync.enabled is true (refused before anything is written when it is not built)."""
    enabled = core.get(s, "sync.enabled") if core.is_set(s, "sync.enabled") else False
    if not isinstance(enabled, bool):
        sys.exit(f"Refusing {name}: sync.enabled must be true or false, got {enabled!r}.")
    if not enabled:
        return None
    try:
        if wiring.sync is None:
            raise FeatureMissing("sync", "the composition root wires no sync feature")
        return wiring.sync()
    except FeatureMissing as e:
        sys.exit(f"Refusing {name}: {e}")


@contextlib.contextmanager
def _syncing(s: Mapping, block: Block, plan: _Plan, out: Path):
    """A low-priority upload of the block's results (and, for gnn_train, the GNN checkpoints) beside the block."""
    if plan.sync is None:
        yield
        return
    try:
        # the profile folder, so the repository mirrors <profile>/<block>/...; the GNN checkpoints as the gnn feature
        # reads the setting (relative to the working directory)
        paths = [out.parent]
        if block.name == "gnn_train":
            paths.append(Path(core.get(s, "gnn.artifacts_dir")))
            paths.append(Path(core.get(s, "gnn_train.logs_dir")))
        handle = plan.sync.start(s, paths)
    except core.Unset as e:
        sys.exit(f"Refusing {block.name}: setting not set in profiles.toml: {e.args[0]}")
    except RuntimeError as e:
        sys.exit(f"Refusing {block.name}: {e}")
    try:
        yield
    finally:
        handle.stop()
        summary = getattr(handle, "summary", None)
        if summary and summary.get("failures"):
            print(f"{block.name}: sync reported failures {summary['failures']} ({summary.get('commits')} commits)")


def _threshold_cell(cell: tuple, choice: str | None) -> tuple:
    """The calibration cell whose thresholds the units of `cell` take: under mock.legacy.matched_thresholds = 'base' a
    matched-timing control takes those of the legacy arm it wraps (same topology and N)."""
    arm = cell[0]
    return (matched.base_arm(arm), *cell[1:]) if choice == "base" and matched.is_matched(arm) else cell


def _probe(s: Mapping, block: Block, grid, specs: Mapping, platform: core.Platform, found: Mapping,
           annotate=None, choice: str | None = None) -> None:
    """One tiny real batch (one episode) per (arm, capped or not, topology, attack, condition) in a temporary store,
    before the block starts: a setting or a bug an arm meets only at run time stops the launch now, not hours in.
    `annotate(unit)` adds what the block's units are handed with (the replayed schedule of llm_adaptive_check)."""
    name = block.name
    if block.platform == "llm":      # one tiny conversation on the fake API: no live call, no file outside the temp store
        s = {**s, "llm": {**s["llm"], "api": "fake", "allow_live": False}}
    with tempfile.TemporaryDirectory(prefix="bazt-probe-") as tmp:
        store = core.CheckpointStore(Path(tmp) / "preflight-probe")
        for plain in _probe_units(block, grid):
            unit = annotate(plain) if annotate is not None else plain
            probe = {**unit, "id": unit_id(unit), "thresholds": (dict(found.get(_threshold_cell(cell_key(unit), choice), {}))
                                                                 if unit["arm"] != needs.ORACLE else
                                                                 _oracle_thresholds_of(found, unit))}
            try:
                platform.run_batch([probe], specs[unit["arm"]], s, store, core.NullTracker())
            except ValueError as e:
                sys.exit(f"Refusing {name}: a test episode of arm {unit['arm']!r} ({_describe(unit)}) was refused: "
                         f"{e.args[0] if e.args else e}")
            except core.Unset as e:
                sys.exit(f"Refusing {name}: setting not set in profiles.toml: {e.args[0]} (needed by arm "
                         f"{unit['arm']!r}, found by a test episode)")
            except Exception as e:
                sys.exit(f"Refusing {name}: a test episode of arm {unit['arm']!r} ({_describe(unit)}) failed: "
                         f"{type(e).__name__}: {e}")


def _probe_units(block: Block, grid):
    if block.platform == "mock":
        yield from grid.probes()
        return
    seen = set()
    for u in grid:
        key = (u["arm"], u["attack"], u.get("sanitize_mode"), u["audit"] == "uncapped", u["intervention"] == "uncapped")
        if key not in seen:
            seen.add(key)
            yield {**u, "conversations": 1}


def _describe(unit: Mapping) -> str:
    fields = ("topology", "attack", "audit", "intervention", "preset", "condition", "cost_condition")
    return ", ".join(f"{f}={unit[f]}" for f in fields if unit.get(f) is not None)


def _oracle_thresholds_of(found: Mapping, unit: Mapping) -> dict:
    return {"audit_score": found.get(oracle_source_cell(unit), {}).get("audit_score")}


def _preflight(s: Mapping, block: Block, allow_live_llm: bool, wiring: Wiring, root: Path) -> _Plan:
    name = block.name
    freeze = root / FREEZE_MARKER
    if block.requires_freeze and not freeze.exists():
        sys.exit(f"Refusing {name}: the analysis is not frozen ({freeze} does not exist).")
    missing = missing_runtime(s, block.platform)
    if missing:
        sys.exit(f"Refusing {name}: run-time variables not set in profiles.toml [runtime]: {', '.join(missing)}")
    reason = unrunnable(block, s, wiring)  # no flag or kickoff makes these units runnable: said first
    if reason:
        sys.exit(f"Refusing {name}: {reason}")
    if block.platform == "llm":
        kickoff.guard(allow_live_llm)
    try:
        grid = block.units(s, True)
        batch_size = core.get(s, "mock.batch_size") if block.platform == "mock" else 1
        n_workers = core.get(s, "compute.workers")
        if not isinstance(n_workers, int) or isinstance(n_workers, bool) or n_workers < 1:
            raise ValueError(f"compute.workers must be an integer >= 1, got {n_workers!r}")
        if n_workers > 1 and block.platform == "mock" and block.kind == "evaluate" and not wiring.rebuild:
            raise ValueError(f"compute.workers={n_workers} needs a wiring a worker process can rebuild "
                             f"(Wiring.rebuild is not set)")
        alpha = core.get(s, "budgets.alpha") if block.kind == "calibrate" else None
        tracker, tracking = _tracking(s, name, wiring)
    except ValueError as e:      # a grid the settings cannot build (an unknown preset, a bad range)
        sys.exit(f"Refusing {name}: {e.args[0] if e.args else e}")
    except core.Unset as e:
        sys.exit(f"Refusing {name}: setting not set in profiles.toml: {e.args[0]}")
    if not isinstance(batch_size, int) or batch_size < 1:
        sys.exit(f"Refusing {name}: mock.batch_size must be a positive integer, got {batch_size!r}.")
    specs = _arm_specs(block, grid, wiring)
    adaptive = None
    gnn_skipped: dict = {}
    try:
        needs.resolve(s, needs.required_keys(s, block.platform, grid.arms(), name))
        if block.platform == "mock" and not block.requires_freeze and needs.heldout_overlap(s, grid.seed_set()):
            sys.exit(f"Refusing {name}: its seeds include held-out seeds and the analysis is not frozen "
                     f"({freeze} does not exist).") if not freeze.exists() else None
        needs.check_seeds(s, block.platform, name)
        if block.kind == "calibrate" and core.get(s, "calibration.mode") not in needs.CALIBRATION_MODES:
            raise ValueError(f"calibration.mode must be one of {list(needs.CALIBRATION_MODES)}, "
                             f"got {core.get(s, 'calibration.mode')!r}")
        if block.name == "gnn_train":
            _gnn_feature(wiring)
        if set(needs.GNN_ARMS) & grid.arms() and block.kind in ("evaluate", "calibrate"):
            found_skipped = _gnn_checkpoints(s, name, grid, wiring)      # a calibration has no cell without a GNN
            gnn_skipped = found_skipped if block.kind == "evaluate" else {}
        if block.adaptive:
            if wiring.adaptive is None:
                raise FeatureMissing("adaptive", "the composition root wires no adaptive feature")
            adaptive = wiring.adaptive()
            adaptive.search_episodes(grid, s)          # refuses what run_search would refuse, before any write
        replayed = None
        if block.replays:                              # D110: the schedule the check replays, read before any write
            if wiring.adaptive is None:
                raise FeatureMissing("adaptive", "the composition root wires no adaptive feature")
            replayed = replay.load(s, grid, wiring.adaptive(), root, BLOCKS[block.replays].units(s, True))
            replay.check_pin(root / s["profile"] / name / replay.PIN, replayed)
        timing = None
        if block.timing_source and any(matched.is_matched(a) for a in grid.arms()):     # D116: BAZT's rounds, read first
            timing = matched.load(s, name, grid, block.timing_source, BLOCKS[block.timing_source].units(s, True), root)
            matched.check_pin(root / s["profile"] / name / matched.PIN, timing)
        choice = None
        if block.kind == "evaluate" and any(matched.is_matched(a) and specs[a].calibrated for a in grid.arms()):
            choice = matched.threshold_choice(s)
    except ValueError as e:      # a setting with an invalid value; the adaptive feature's Refusal is one, and an Unset too
        sys.exit(f"Refusing {name}: {e.args[0] if e.args else e}")
    except core.Unset as e:
        sys.exit(f"Refusing {name}: setting not set in profiles.toml: {e.args[0]}")
    except FeatureMissing as e:
        sys.exit(f"Refusing {name}: {e}")
    try:
        platform = wiring.platform(block.platform)
    except FeatureMissing as e:
        sys.exit(f"Refusing {name}: {e}")

    calibration = CALIBRATION_BLOCK[block.platform]
    path = root / s["profile"] / calibration / thresholds.FILE
    found, sha = {}, None
    if block.kind == "calibrate":
        for arm in sorted(grid.arms()):
            if not specs[arm].calibrated:
                sys.exit(f"Refusing {name}: arm {arm!r} is not calibrated (ArmSpec.calibrated is False) but this "
                         f"block's settings list it for calibration.")
            if matched.is_matched(arm) and core.get(s, "calibration.mode") == "closed_loop":
                sys.exit(f"Refusing {name}: arm {arm!r} is a matched-timing control; calibration.mode = closed_loop runs "
                         f"it acting, and the attack-free tune split has no BAZT rounds for it to act at (D116). "
                         f"Calibrate it open loop, or take the thresholds of {matched.base_arm(arm)!r} "
                         f"({matched.THRESHOLDS_KEY} = 'base') and drop it from {block.platform}.calibrated_arms.")
        saved_alpha = thresholds.saved_alpha(path)
        if saved_alpha is not None and saved_alpha != alpha:
            sys.exit(f"Refusing {name}: {path} was calibrated at alpha={saved_alpha}, this launch has alpha={alpha}. "
                     f"Delete it to recalibrate every cell at alpha={alpha}.")
        stale = identity.read(path)
        if stale is not None and stale != identity.threshold_view(s, block.platform):
            sys.exit(f"Refusing {name}: {path} holds cells calibrated under other settings: "
                     f"{identity.differences(stale, identity.threshold_view(s, block.platform))}. Delete it to "
                     f"recalibrate every cell, or use a fresh results root.")
    elif block.kind == "evaluate":
        for arm in sorted(grid.arms()):
            source_arm = _threshold_cell((arm,), choice)[0]
            if specs[arm].calibrated and source_arm not in core.get(s, f"{block.platform}.calibrated_arms"):
                sys.exit(f"Refusing {name}: arm {arm!r} reports a decision statistic (ArmSpec.calibrated) but "
                         f"{source_arm!r} is not listed in {block.platform}.calibrated_arms, so no threshold would exist "
                         f"for it. Add it to {block.platform}.calibrated_arms and run --block {calibration}.")
        stale = identity.read(path)
        if stale is not None and stale != identity.threshold_view(s, block.platform):
            sys.exit(f"Refusing {name}: the thresholds in {path} were calibrated under other settings: "
                     f"{identity.differences(stale, identity.threshold_view(s, block.platform))}. Recalibrate "
                     f"(--block {calibration}) in a fresh results root.")
        found, sha = thresholds.load(path), thresholds.sha256(path)
        _oracle_thresholds(name, grid, found)
        lacking = sorted((_threshold_cell(c, choice) for c in grid.cells()
                          if specs[c[0]].calibrated and _threshold_cell(c, choice) not in found),
                         key=lambda c: tuple(map(str, c)))
        if lacking:
            shown = "; ".join(thresholds.describe(c) for c in lacking[:3])
            sys.exit(f"Refusing {name}: no calibrated thresholds for {len(lacking)} cell(s) of calibrated arms, "
                     f"e.g. {shown}. Run --block {calibration} first (expects {path}).")
    out = root / s["profile"] / name
    view = identity.block_view(s, name, block.platform)
    saved = identity.read(out / SETTINGS_FILE)
    if saved is not None and saved != view:
        sys.exit(f"Refusing {name}: {out} holds rows produced under other settings: "
                 f"{identity.differences(saved, view)}. Rows of two experiments would mix; use another "
                 f"--results-root or --profile.")
    if block.platform == "mock" or block.kind in ("calibrate", "evaluate"):
        annotate = replayed.annotate if replayed is not None else timing.annotate if timing is not None else None
        _probe(s, block, grid, specs, platform, found, annotate, choice)
    return _Plan(grid, specs, platform, batch_size, path, found, sha, alpha, adaptive, tracker, tracking,
                 _sync_feature(s, name, wiring), view, out, gnn_skipped=gnn_skipped, workers=n_workers,
                 replay=replayed, timing=timing, matched_thresholds=choice)


def _manifest(s: Mapping, block: Block, plan: _Plan, started_at: str) -> dict:
    return {"block": block.name, "profile": s["profile"], "git_commit": _git_commit(), "started_at": started_at,
            "versions": {"python": pyplatform.python_version(), "torch": torch.__version__,
                         "numpy": numpy.__version__},
            "threads": torch.get_num_threads(),
            "pre_registered": {k: core.get(s, f"runtime.{k}") for k in PRE_REGISTERED},
            "units": len(plan.grid), "thresholds_sha256": plan.sha, "settings": dict(s)}


def _log(plan: _Plan, done: int, skipped: int, batches: int, size: int) -> None:
    plan.tracker.log({"runner/units_done": done, "runner/units_skipped": skipped, "runner/batches": batches,
                      "runner/batch_units": size}, step=done + skipped)


def _report_no_gnn(plan: _Plan, s: Mapping, name: str, root: Path) -> None:
    """Say, and record for `status`, which GNN cells are skipped because no GNN exists for them."""
    cells = [f"{arm}: attack={attack} task={task}" for (arm, attack, task) in sorted(plan.gnn_skipped or {})]
    for line in cells:
        print(f"{name}: skipped: no GNN for this cell (gnn.fallback_family = {core.get(s, 'gnn.fallback_family')}): "
              f"{line.split(': ', 1)[1]}")
    ledger.record_skipped(root, s["profile"], name, sum((plan.gnn_skipped or {}).values()), cells)


def _search(plan: _Plan, s: Mapping, out: Path, wiring: Wiring):
    """mock_adaptive: the best responses of every cell, found against the cell's calibrated thresholds."""
    def thresholds_of(cell):
        return _thresholds_of(plan, cell)
    try:
        return plan.adaptive.run_search(plan.grid, wiring.arm, plan.platform, s, out, thresholds_of, plan.tracker)
    except ValueError as e:                # the search refuses before it writes anything
        sys.exit(f"Refusing {out.name}: {e.args[0] if e.args else e}")


def _thresholds_of(plan: _Plan, unit: Mapping) -> dict:
    """The thresholds of the unit's cell; oracle_alloc takes random_alloc's audit_score (D96)."""
    if unit["arm"] == needs.ORACLE:
        return {"audit_score": plan.found[oracle_source_cell(unit)]["audit_score"]}
    return dict(plan.found.get(_threshold_cell(cell_key(unit), plan.matched_thresholds), {}))


def _pending(plan: _Plan, store: core.CheckpointStore, table, counts: dict):
    """The batches still to run, in grid order: done units dropped, each unit annotated with its id, the thresholds
    of its cell, attack_params, and (attacked mock units) the id of its attack-free twin."""
    no_gnn = set(plan.gnn_skipped or {})
    for _, group in itertools.groupby(plan.grid, key=group_key):
        pending = []
        for unit in group:
            if (unit["arm"], unit["attack"], unit.get("task")) in no_gnn:
                continue
            uid = unit_id(unit)
            if store.is_done(uid):
                counts["skipped"] += 1
                continue
            annotated = table.annotate(unit) if table is not None else unit
            twin = ({"twin_id": unit_id({**unit, "attack": "none"})}
                    if unit["platform"] == "mock" and unit["attack"] != "none" else {})
            pending.append({**annotated, "id": uid, "thresholds": _thresholds_of(plan, unit), **twin})
        for i in range(0, len(pending), plan.batch_size):
            yield pending[i:i + plan.batch_size]


def _evaluate(plan: _Plan, s: Mapping, store: core.CheckpointStore, table=None, wiring: Wiring | None = None,
              out: Path | None = None) -> tuple:
    """Run the pending batches, in this process or on compute.workers worker processes."""
    counts = {"skipped": 0}
    batches = _pending(plan, store, table, counts)
    if plan.workers > 1 and plan.platform.name == "mock" and plan.manifest is not None:
        done = workers.run(batches, plan.workers, wiring.rebuild, s, plan.platform.name, out, store,
                           lambda finished, dispatched, n: _log(plan, finished, counts["skipped"], n, 0))
        return done, counts["skipped"]
    done = nbatches = 0
    for batch in batches:
        plan.platform.run_batch(batch, plan.specs[batch[0]["arm"]], s, store, plan.tracker)
        undone = [b["id"] for b in batch if not store.is_done(b["id"])]
        if undone:      # unmarked units would run again, and append their rows twice, at the next launch
            raise RuntimeError(f"{plan.platform.name} platform returned from run_batch without marking "
                               f"{len(undone)} of {len(batch)} units done: {', '.join(undone[:3])}")
        done, nbatches = done + len(batch), nbatches + 1
        _log(plan, done, counts["skipped"], nbatches, len(batch))
    return done, counts["skipped"]


def _calibrate(plan: _Plan, block: Block, s: Mapping, store: core.CheckpointStore) -> tuple:
    """One Platform.calibrate call per cell, on that cell's units; each finished cell is saved at once."""
    saved = thresholds.load(plan.thresholds_path)
    cells: dict = {}
    for unit in plan.grid:
        cells.setdefault(cell_key(unit), []).append({**unit, "id": unit_id(unit), "thresholds": {}})
    done = skipped = batches = 0
    for cell, units in cells.items():
        if cell in saved:
            skipped += len(units)
            continue
        result = {str(k): float(v) for k, v in (plan.platform.calibrate(units, plan.specs[cell[0]], s, store)
                                               or {}).items()}
        if not result:
            sys.exit(f"{block.name}: calibrate returned no thresholds for {thresholds.describe(cell)}; "
                     f"nothing was saved for it.")
        bad = sorted(k for k, v in result.items() if not math.isfinite(v))
        if bad:       # nan never fires and inf never fires: the arm would silently do nothing in this cell
            sys.exit(f"{block.name}: calibrate returned a threshold that is not finite for {thresholds.describe(cell)}: "
                     f"{', '.join(bad)}; nothing was saved for it.")
        diagnostics = getattr(plan.platform, "last_calibration", None)
        thresholds.save_cell(plan.thresholds_path, plan.alpha, cell, result,
                             identity.threshold_view(s, block.platform), diagnostics)
        if diagnostics and plan.manifest is not None:
            plan.manifest.setdefault("calibration", {})[thresholds.describe(cell)] = dict(diagnostics)
            files.write_json(plan.manifest_path, plan.manifest)
        done, batches = done + len(units), batches + 1
        _log(plan, done, skipped, batches, len(units))
    return done, skipped


LOCK = ".launch.lock"


def run_block(s: Mapping, name: str, allow_live_llm: bool, *, wiring: Wiring,
              results_root: Path = RESULTS_ROOT) -> int:
    block = BLOCKS[name]
    root = Path(results_root)
    s = {**s, "llm": {**s.get("llm", {}), "allow_live": bool(allow_live_llm) and block.platform == "llm"}}
    plan = _preflight(s, block, allow_live_llm, wiring, root)

    out = root / s["profile"] / name
    store = core.CheckpointStore(out)
    busy = (f"Refusing {name}: another launch is running this block ({out / LOCK} is held). "
            f"Two launches would append the same rows twice.")
    with files.exclusive(out / LOCK, busy), _syncing(s, block, plan, out):
        started_at = dt.datetime.now(dt.UTC).isoformat(timespec="microseconds")
        manifest = _manifest(s, block, plan, started_at)
        manifest_path = out / f"manifest-{started_at.replace(':', '').replace('.', '')}.json"

        run_id_file = out / "tracker_run_id.txt"
        resume_id = (run_id_file.read_text(encoding="utf-8").strip()
                     if plan.tracking and run_id_file.exists() else None)
        config = {**s, "manifest": {k: v for k, v in manifest.items() if k != "settings"}}
        try:
            run_id = plan.tracker.start(name, config, resume_id)
        except core.Unset as e:
            sys.exit(f"Refusing {name}: setting not set in profiles.toml: {e.args[0]}")
        except RuntimeError as e:
            sys.exit(f"Refusing {name}: {e}")
        if not (out / SETTINGS_FILE).exists():          # pinned only once the tracker started: a dead launch pins nothing
            files.write_json(out / SETTINGS_FILE, {"settings": plan.view})
        if plan.replay is not None and not (out / replay.PIN).exists():      # the best responses the rows replay
            files.write_json(out / replay.PIN, plan.replay.pin())
        if plan.timing is not None and not (out / matched.PIN).exists():     # D116: the BAZT rounds the rows replay
            files.write_json(out / matched.PIN, plan.timing.pin())
        plan.manifest, plan.manifest_path = manifest, manifest_path
        files.write_json(manifest_path, manifest)
        try:
            if plan.tracking:
                files.write_text(run_id_file, run_id)
            plan.tracker.artifact(str(manifest_path), f"{name}-manifest", "manifest")
            if block.kind == "calibrate":
                done, skipped = _calibrate(plan, block, s, store)
            else:
                _report_no_gnn(plan, s, name, root)
                table = (_search(plan, s, out, wiring) if block.adaptive else
                         plan.replay if plan.replay is not None else plan.timing)
                done, skipped = _evaluate(plan, s, store, table, wiring, out)
                if name == "gnn_train":
                    _finish_gnn_training(s, name, wiring)
        except core.Unset as e:
            sys.exit(f"Refusing {name}: setting not set in profiles.toml: {e.args[0]}")
        finally:
            plan.tracker.finish()
    print(f"{name}: {done} units run, {skipped} already done, results in {out}")
    return 0
