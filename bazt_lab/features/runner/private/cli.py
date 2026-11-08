"""The command line: --list, --dry-run, --run."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Mapping

from bazt_lab.core import public as core
from bazt_lab.features.runner.private import files, ledger, runall, status
from bazt_lab.features.runner.private.blocks import BLOCKS, Block
from bazt_lab.features.runner.private.launch import RESULTS_ROOT, run_block
from bazt_lab.features.runner.private.runtime import load_settings, missing_runtime
from bazt_lab.features.runner.private.sizing import dry_run
from bazt_lab.features.runner.private.wiring import FeatureMissing, Wiring

DESCRIPTION = ("BAZT experiment launcher (docs/EXPERIMENT_PLAN.md). Every setting comes from experiments/profiles.toml. "
               "Simplest use: `run-all` runs every block in order (resumable), `status` shows where you are and what is "
               "still unset. Finer control: --list, --dry-run, --run --block NAME.")


def _unbuilt(block: Block, arms, wiring: Wiring) -> List[str]:
    """The features a block needs that are not built, then any unknown arms."""
    features, unknown = set(), []
    for call, arg in [(wiring.platform, block.platform)] + [(wiring.arm, arm) for arm in sorted(arms)]:
        try:
            call(arg)
        except FeatureMissing as e:
            features.update(f.strip() for f in e.feature.split(","))
        except LookupError:
            unknown.append(arg)
    return sorted(features) + [f"arm {arm!r} unknown" for arm in unknown]


def list_blocks(s: Mapping, wiring: Wiring) -> None:
    print(f"profile: {s['profile']}\n")
    print(f"{'block':<26}{'platform':<10}{'units':>10}  {'features':<11}needs")
    for name, block in BLOCKS.items():
        needs, arms = [], set()
        try:
            grid = block.units(s, False)
            count, arms = f"{len(grid):,}", grid.arms()
        except core.Unset as e:
            count = "-"
            needs.append(f"set {e.args[0]}")
        runtime = missing_runtime(s, block.platform)
        if runtime:
            needs.append(f"{len(runtime)} run-time variables")
        unbuilt = _unbuilt(block, arms, wiring)
        if unbuilt:
            needs.append("not built: " + ", ".join(unbuilt))
        print(f"{name:<26}{block.platform:<10}{count:>10}  {'not built' if unbuilt else 'ok':<11}{'; '.join(needs)}")


COMMANDS = ("run-all", "status")
DEVICE = re.compile(r"(cpu|cuda(:\d+)?)")


def device(text: str) -> str:
    if not DEVICE.fullmatch(text):
        raise argparse.ArgumentTypeError(f"{text!r} is not cpu, cuda or cuda:N")
    return text


def add_compute_flags(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--device", type=device, default=None, metavar="{cpu,cuda,cuda:N}",
                    help="where the tensors run; same as --set compute.device=... (results do not depend on it)")
    ap.add_argument("--workers", type=int, default=None, metavar="N",
                    help="worker processes per block; same as --set compute.workers=N (rows do not depend on it)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the hot kernels; same as --set compute.compile=true")


def compute_overrides(args) -> list:
    return ([f"compute.device={json.dumps(args.device)}"] if args.device else []) + (
        ["compute.compile=true"] if args.compile else []) + (
        [f"compute.workers={args.workers}"] if args.workers is not None else [])


def _launch_each(s: Mapping, names, allow_live: bool, wiring: Wiring, root: Path) -> None:
    """--run: each block in turn; its error is remembered for `status`."""
    for name in names:
        try:
            run_block(s, name, allow_live, wiring=wiring, results_root=root)
        except files.BusyLaunch:
            raise
        except SystemExit as e:
            if isinstance(e.code, str):
                ledger.record(root, s["profile"], name, e.code)
            raise
        except Exception as e:
            ledger.record(root, s["profile"], name, f"{type(e).__name__}: {e}")
            raise
        ledger.record(root, s["profile"], name, None)


def _command(command: str, argv, wiring: Wiring) -> int:
    ap = argparse.ArgumentParser(prog=f"run_experiments.py {command}", description=DESCRIPTION)
    ap.add_argument("--settings", default=str(core.SETTINGS_PATH))
    ap.add_argument("--profile", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--results-root", default=str(RESULTS_ROOT))
    add_compute_flags(ap)
    if command == "run-all":
        ap.add_argument("--dry-run", action="store_true", help="show the order and the sizes, run nothing")
        ap.add_argument("--allow-live-llm", action="store_true", help="also run the LLM blocks (after kickoff only)")
    args = ap.parse_args(argv)
    s = load_settings(Path(args.settings), args.profile, [*args.set, *compute_overrides(args)])
    root = Path(args.results_root)
    if command == "status":
        print(status.render(s, root, Path(args.settings), wiring=wiring))
        return 0
    if args.dry_run:
        return runall.run_all(s, root, args.allow_live_llm, True, wiring)
    return runall.run_all(s, root, args.allow_live_llm, False, wiring)


def main(argv=None, *, wiring: Wiring) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in COMMANDS:
        return _command(argv[0], argv[1:], wiring)
    ap = argparse.ArgumentParser(description=DESCRIPTION)
    ap.add_argument("--settings", default=str(core.SETTINGS_PATH),
                    help="settings file (default: experiments/profiles.toml)")
    ap.add_argument("--profile", default=None, help="sprint or research (default: active_profile in the settings)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override one setting for this launch, e.g. --set llm.conversations_per_cell=80")
    ap.add_argument("--block", action="append", default=[], choices=list(BLOCKS), help="block(s) to act on")
    ap.add_argument("--results-root", default=str(RESULTS_ROOT),
                    help="where results/<profile>/<block> are written (default: results/experiments)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="list blocks with unit counts and what they still need")
    mode.add_argument("--dry-run", action="store_true", help="print sizes without running anything")
    mode.add_argument("--run", action="store_true", help="run the given block(s)")
    ap.add_argument("--allow-live-llm", action="store_true", help="required for LLM blocks (after kickoff only)")
    add_compute_flags(ap)
    args = ap.parse_args(argv)

    s = load_settings(Path(args.settings), args.profile, [*args.set, *compute_overrides(args)])
    if args.list:
        list_blocks(s, wiring)
        return 0
    names = args.block or list(BLOCKS)
    if args.dry_run:
        dry_run(s, names, wiring)
        return 0
    if not args.block:
        sys.exit("--run needs at least one --block.")
    _launch_each(s, names, args.allow_live_llm, wiring, Path(args.results_root))
    return 0
