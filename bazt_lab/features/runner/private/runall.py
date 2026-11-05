"""`run-all`: every block of the active profile in dependency order, stopping at the first refusal."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Mapping, Tuple

from bazt_lab.core import public as core
from bazt_lab.features.runner.private import ledger
from bazt_lab.features.runner.private.blocks import BLOCKS
from bazt_lab.features.runner.private.launch import FREEZE_MARKER, run_block
from bazt_lab.features.runner.private.runtime import PRE_REGISTERED, RUNTIME_BY_PLATFORM
from bazt_lab.features.runner.private.sizing import dry_run
from bazt_lab.features.runner.private.wiring import FeatureMissing, Wiring

KIND_ORDER = ("calibrate", "train", "evaluate")      # thresholds first, then the GNN, then the evaluations


def own_variables(platform: str) -> list:
    """The run-time variables only this platform reads (those it shares with the other one say nothing)."""
    others = {k for p, keys in RUNTIME_BY_PLATFORM.items() if p != platform for k in keys} | set(PRE_REGISTERED)
    return [k for k in RUNTIME_BY_PLATFORM[platform] if k not in others]


def configured(s: Mapping, platform: str) -> bool:
    """A platform is configured once any of its own run-time variables is set (the two pre-registered values are
    shared). A half-configured platform is not skipped: its launch refuses and names what is missing."""
    return any(core.is_set(s, f"runtime.{k}") for k in own_variables(platform))


def plan(s: Mapping, root: Path, allow_live_llm: bool) -> Tuple[List[str], List[Tuple[str, str]]]:
    """(blocks to run, [(block, reason it is skipped)]) in dependency order."""
    run, skipped = [], []
    ordered = sorted(BLOCKS.values(), key=lambda b: KIND_ORDER.index(b.kind))        # stable: registry order inside
    for block in ordered:
        if block.requires_freeze and not (Path(root) / FREEZE_MARKER).exists():
            skipped.append((block.name, "the analysis is not frozen"))
        elif block.platform == "llm" and not allow_live_llm:
            skipped.append((block.name, "LLM blocks need --allow-live-llm (after the kickoff)"))
        elif not configured(s, block.platform):
            first = own_variables(block.platform)[0]
            skipped.append((block.name, f"no {block.platform} run-time variable is set (runtime.{first})"))
        else:
            try:
                block.units(s, True)       # strict: a placeholder (an unset backbone) is a scope not set
            except core.Unset as e:
                skipped.append((block.name, f"scope not set ({e.args[0]})"))
                continue
            except SystemExit as e:
                skipped.append((block.name, f"scope invalid ({e.code})"))
                continue
            run.append(block.name)
    return run, skipped


def _analysis(s: Mapping, root: Path, wiring: Wiring, dry: bool) -> int:
    try:
        feature = wiring.analysis() if wiring.analysis else None
        if feature is None:
            raise FeatureMissing("analysis", "the composition root wires no analysis feature")
    except FeatureMissing as e:
        print(f"analysis: skipped: {e}")
        return 0
    missing = feature.missing_keys(s)
    if missing:
        print(f"analysis: skipped: analysis settings not set ({missing[0]})")
        return 0
    results = Path(root) / s["profile"]
    if dry:
        print(f"analysis: would read {results}")
        return 0
    try:
        feature.analyze(results, s)
    except (core.Unset, ValueError, FileNotFoundError) as e:
        print(f"Refusing analysis: {e.args[0] if e.args else e}", file=sys.stderr)
        return 1
    print(f"analysis: report written in {results}")
    return 0


def run_all(s: Mapping, root: Path, allow_live_llm: bool, dry: bool, wiring: Wiring) -> int:
    run, skipped = plan(s, root, allow_live_llm)
    for name, reason in skipped:
        print(f"{name}: skipped: {reason}")
    if dry:
        print(f"would run, in this order: {', '.join(run) or 'nothing'}")
        if run:
            dry_run(s, run, wiring)
        return _analysis(s, root, wiring, dry)
    for name in run:
        print(f"== {name}")
        try:
            run_block(s, name, allow_live_llm, wiring=wiring, results_root=root)
        except SystemExit as e:
            if isinstance(e.code, str):
                ledger.record(root, s["profile"], name, e.code)
                print(e.code, file=sys.stderr)
                return 1
            raise
        except Exception as e:
            ledger.record(root, s["profile"], name, f"{type(e).__name__}: {e}")
            raise
        ledger.record(root, s["profile"], name, None)
    return _analysis(s, root, wiring, dry)
