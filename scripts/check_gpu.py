#!/usr/bin/env python3
"""The first thing to run on a GPU machine: what torch sees, and whether every device/precision gives the same answers.

    python scripts/check_gpu.py                # full slice, every combination this machine can run
    python scripts/check_gpu.py --quick        # smaller slice (a few minutes on CPU)
    python scripts/check_gpu.py --seeds 6 --agents 8 --devices cpu,cuda
    python scripts/check_gpu.py --require-cuda   # on the GPU machine: exit 2 at once if torch sees no CUDA device

It prints the torch, CUDA and GPU facts, then runs one small REAL slice of the mock frontier through the launcher's own
block runner (calibration, then the evaluation block): arms A, B, C and BAZT, topologies chain and small_world, capped and
uncapped. The slice runs once for each combination of device (cpu, cuda), dtype (float64, float32) and torch.compile
(off, on). The first combination (cpu, float64, compile off) is the reference: every run computes in float32 (D117,
compute.dtype in profiles.toml), and float64 is what float32 is checked against. Every other combination is compared with
it episode by episode:

  float64 combinations        every column identical (the engine promises that device and compile do not change results)
  float32 combinations        budget accounting (spend, violations, exhaustion step) identical for at least 99% of the
                              episodes; every other number within 0.005 (the documented float32 tolerance of
                              mock_engine/CONTRACT.md, accuracy and so ASR) for at least 99% of the episodes

It reports, for each combination, the episodes the slice plans, the rows it read back and episodes per second. A
combination that read fewer rows than it planned fails (two empty runs would compare as identical). Exit code 1 on any
mismatch, short read or failed run; 0 otherwise. Without CUDA the cuda combinations are reported as skipped and the CPU
part still runs: that is not a failure, unless --require-cuda is given (exit code 2 before anything runs; use it on the
GPU machine, where a CPU build of torch would otherwise pass).

The settings it passes are E2E TEST VALUES (scripts/e2e_common.py): they start a tiny run and say nothing about what the
experiments should use. Results go to a temporary directory.
"""
from __future__ import annotations

import argparse
import itertools
import math
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import e2e_common as e2e

ACCOUNTING = ("spend_audit_total", "spend_intervention_total", "spend_total", "violations", "exhausted_step")
IGNORED = (*e2e.VOLATILE_PREFIXES, "meta_")      # timings and the run's own provenance (dtype, threads, versions)
TOLERANCE = 0.005          # documented float32 tolerance (accuracy, and so ASR)
ALLOWED_SHARE = 0.01       # share of episodes that may leave the tolerance in a float32 combination
ARMS_UNCAPPED = ["A", "B", "C", "BAZT"]
ARMS_CAPPED = ["B", "C", "BAZT"]
TOPOLOGIES = ["chain", "small_world"]


@dataclass(frozen=True)
class Combo:
    device: str
    dtype: str
    compile: bool

    def __str__(self) -> str:
        return f"{self.device:<5} {self.dtype:<8} compile {'on ' if self.compile else 'off'}"

    @property
    def exact(self) -> bool:
        return self.dtype == "float64"


@dataclass
class Outcome:
    combo: Combo
    episodes: int = 0
    seconds: float = 0.0
    error: str | None = None
    rows: list | None = None

    @property
    def rate(self) -> float:
        return self.episodes / self.seconds if self.seconds > 0 else math.nan


# ------------------------------------------------------------------------------------------------ comparison
def numbers_close(a: object, b: object, tolerance: float) -> bool:
    """Equal values; numbers within `tolerance` (NaN equals NaN); lists and dicts element by element."""
    if isinstance(a, dict) and isinstance(b, dict):          # e.g. the calibrated thresholds a row records
        return a.keys() == b.keys() and all(numbers_close(a[k], b[k], tolerance) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(numbers_close(x, y, tolerance) for x, y in zip(a, b, strict=True))
    if e2e.is_number(a) and e2e.is_number(b):
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) <= tolerance * max(1.0, abs(a)) if tolerance else a == b
    return a == b


def compare_rows(reference: Sequence[Mapping], other: Sequence[Mapping], exact: bool) -> list[str]:
    """Differences between two runs of the same episodes. `exact`: every column identical. Otherwise the float32 rule
    (module docstring). Returns one line per problem; empty means the combination agrees with the reference."""
    a = dict(zip(episode_keys(reference), reference, strict=True))
    b = dict(zip(episode_keys(other), other, strict=True))
    problems = [f"episode {k} is only in the {'reference' if k in a else 'other'} run" for k in sorted(a.keys() ^ b.keys())]
    common = sorted(a.keys() & b.keys())
    if exact:
        for key in common:
            columns = [c for c in sorted(set(a[key]) | set(b[key])) if not c.startswith(IGNORED)
                       and not numbers_close(a[key].get(c), b[key].get(c), 0.0)]
            if columns:
                problems.append(f"episode {key}: {', '.join(columns)} differ "
                                f"({a[key].get(columns[0])!r} vs {b[key].get(columns[0])!r})")
        return problems
    bad_accounting, bad_other = [], []
    for key in common:
        columns = [c for c in sorted(set(a[key]) | set(b[key])) if not c.startswith(IGNORED)]
        if any(not numbers_close(a[key].get(c), b[key].get(c), 0.0) for c in columns if c in ACCOUNTING):
            bad_accounting.append(key)
        bad = [c for c in columns if c not in ACCOUNTING and not numbers_close(a[key].get(c), b[key].get(c), TOLERANCE)]
        if bad:
            bad_other.append((key, bad))
    limit = ALLOWED_SHARE * max(1, len(common))
    if len(bad_accounting) > limit:
        problems.append(f"budget accounting differs in {len(bad_accounting)} of {len(common)} episodes "
                        f"(allowed {limit:.1f}), e.g. {bad_accounting[0]}")
    if len(bad_other) > limit:
        key, columns = bad_other[0]
        problems.append(f"{len(bad_other)} of {len(common)} episodes leave the {TOLERANCE} tolerance "
                        f"(allowed {limit:.1f}), e.g. {key}: {', '.join(columns)}")
    return problems


def episode_keys(rows: Iterable[Mapping]) -> list[tuple]:
    seen: dict[tuple, int] = {}
    keys = []
    for row in rows:
        base = e2e.episode_key(row)
        seen[base] = seen.get(base, 0) + 1
        keys.append((*base, seen[base]))
    return keys


# ---------------------------------------------------------------------------------------------------- facts
def torch_facts() -> list[str]:
    import torch
    cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "n/a"
    lines = [f"torch {torch.__version__}, CUDA build {torch.version.cuda or 'none (CPU build)'}, cuDNN {cudnn}",
             f"cpu threads {torch.get_num_threads()} (interop {torch.get_num_interop_threads()})"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            lines.append(f"cuda:{i} {p.name}, compute capability {p.major}.{p.minor}, {p.total_memory / 2**30:.1f} GiB")
    else:
        lines.append("CUDA unavailable: the cuda combinations are skipped")
    return lines


def available_devices(wanted: Sequence[str]) -> tuple[list[str], list[str]]:
    import torch
    usable = [d for d in wanted if d == "cpu" or (d.startswith("cuda") and torch.cuda.is_available())]
    return usable, [d for d in wanted if d not in usable]


def combos(devices: Sequence[str], can_compile: Callable[[str], bool] = lambda device: True) -> list[Combo]:
    """Reference first (cpu, float64, compile off), then every other combination. compile on is listed only for a device
    where `can_compile(device)`: without a C++ compiler (CPU) or Triton (CUDA) Inductor cannot build kernels, and every
    attempt costs seconds before the engine falls back to eager."""
    every = [Combo(d, t, c) for d, t, c in itertools.product(devices, ("float64", "float32"), (False, True))
             if not c or can_compile(d)]
    return sorted(every, key=lambda c: (c.device != "cpu" or c.dtype != "float64" or c.compile, str(c)))


def inductor_usable(device: str) -> bool:
    from bazt_lab.core import public as core
    return core.inductor_available(device)



# --------------------------------------------------------------------------------------------------- running
def slice_sets(agents: int, seeds: int, combo: Combo) -> dict[str, object]:
    return {
        **e2e.ARM_TEST_VALUES, **e2e.runtime_sets(),
        "mock.batch_size": 64, "mock.topologies": TOPOLOGIES, "mock.families": ["infection", "insider_pi"],
        "mock.frontier_agents": [agents], "mock.epidemiology_agents": [agents], "mock.main_seeds": [1, seeds],
        "mock.tune_seeds": [9001, 9000 + seeds], "mock.uncapped_arms": ARMS_UNCAPPED, "mock.capped_arms": ARMS_CAPPED,
        "mock.allocation_arms": [], "mock.calibrated_arms": ["B", "C", "BAZT"], "budgets.levels_percent": [10],
        "compute.device": combo.device, "compute.dtype": combo.dtype, "compute.compile": combo.compile,
    }


def run_slice(combo: Combo, agents: int, seeds: int, root: Path) -> Outcome:
    """Calibrate, then run the evaluation block of the slice with the launcher's own block runner."""
    from bazt_lab import app
    from bazt_lab.features.runner import public as rx
    outcome = Outcome(combo)
    settings = rx.load_settings(overrides=e2e.override_items(slice_sets(agents, seeds, combo)))
    wiring = app.build_wiring()
    try:
        rx.run_block(settings, "mock_calibration", False, wiring=wiring, results_root=root)
        episodes = len(rx.units_mock_frontier(settings, True))
        start = time.perf_counter()
        rx.run_block(settings, "mock_frontier", False, wiring=wiring, results_root=root)
        outcome.seconds = time.perf_counter() - start
    except SystemExit as stop:
        outcome.error = str(stop.code)
        return outcome
    except Exception as error:      # a crash is a finding about this combination, not about the script
        outcome.error = f"{type(error).__name__}: {error}"
        return outcome
    outcome.episodes = episodes
    outcome.rows = e2e.all_rows(root / settings["profile"] / "mock_frontier" / "units")
    return outcome


def short_read(outcome: Outcome) -> str | None:
    """Why a combination that ran cannot be compared: it read fewer rows than the episodes it planned (a wrong rows
    path reads none, and compare_rows([], []) finds nothing to differ)."""
    read = len(outcome.rows or [])
    if outcome.episodes == 0 or read < outcome.episodes:
        return f"read {read} rows of {outcome.episodes} planned episodes"
    return None


def report(outcomes: Sequence[Outcome]) -> int:
    """Print the table and the comparisons; the exit code."""
    reference = outcomes[0]
    print(f"\n{'combination':<32}{'episodes':>9}{'rows':>7}{'seconds':>9}{'episodes/s':>12}  result")
    failures = 0
    for outcome in outcomes:
        short = None if outcome.error else short_read(outcome)
        if outcome.error:
            verdict = f"FAILED to run: {outcome.error}"
        elif short:
            verdict = f"FAILED: {short}"
            failures += 1
        elif outcome is reference:
            verdict = "reference"
        elif reference.error or short_read(reference):
            verdict = "not compared (the reference did not run or read too few rows)"
        else:
            problems = compare_rows(reference.rows, outcome.rows or [], outcome.combo.exact)
            verdict = "identical to the reference" if outcome.combo.exact and not problems else (
                "within the float32 tolerance" if not problems else f"MISMATCH ({len(problems)}): {problems[0]}")
            for problem in problems[1:4]:
                verdict += f"\n{'':<71}{problem}"
            failures += bool(problems)
        failures += bool(outcome.error)
        print(f"{outcome.combo!s:<32}{outcome.episodes:>9}{len(outcome.rows or []):>7}{outcome.seconds:>9.2f}"
              f"{outcome.rate:>12.1f}  {verdict}")
    print(f"\ngpu check: {failures} combination(s) failed")
    return 1 if failures else 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true", help="smaller slice (4 agents, 2 seeds)")
    ap.add_argument("--agents", type=int, default=None, help="agents per episode (default 6, 4 with --quick)")
    ap.add_argument("--seeds", type=int, default=None, help="seeds per cell (default 3, 2 with --quick)")
    ap.add_argument("--devices", default="cpu,cuda", help="comma separated: cpu, cuda, cuda:N (default cpu,cuda)")
    ap.add_argument("--keep", action="store_true", help="keep the temporary results directories")
    ap.add_argument("--require-cuda", action="store_true",
                    help="exit 2 before running anything when no cuda device is usable (a CPU build of torch, no GPU, "
                         "or --devices without cuda)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    with e2e.offline_environment():          # the block runner runs in this process: no real secrets, HF and W&B offline
        return run(args)


def run(args: argparse.Namespace) -> int:
    agents = args.agents or (4 if args.quick else 6)
    seeds = args.seeds or (2 if args.quick else 3)
    print("\n".join(torch_facts()))
    wanted = [d.strip() for d in args.devices.split(",") if d.strip()]
    if "cpu" not in wanted:
        wanted.insert(0, "cpu")                       # the reference is always the CPU float64 run
    devices, skipped = available_devices(wanted)
    for device in skipped:
        print(f"skipped: {device} (not available on this machine)")
    if args.require_cuda and not any(d.startswith("cuda") for d in devices):
        import torch
        print(f"gpu check: --require-cuda, but no cuda device is usable (torch {torch.__version__}, CUDA build "
              f"{torch.version.cuda or 'none'}, devices {','.join(wanted)}). Install the CUDA build and run without "
              "re-syncing (docs/EXPERIMENTS_HOWTO.md, Running on a GPU)")
        return 2
    tmp = Path(tempfile.mkdtemp(prefix="bazt_gpu_"))
    outcomes = []
    try:
        for device in devices:
            if not inductor_usable(device):
                print(f"skipped: {device} with compile on (Inductor cannot build kernels here: "
                      f"{'no Triton' if device.startswith('cuda') else 'no C++ compiler'})")
        for n, combo in enumerate(combos(devices, inductor_usable)):
            print(f"running {combo} ...", flush=True)
            outcomes.append(run_slice(combo, agents, seeds, tmp / f"{n}"))
        return report(outcomes)
    finally:
        if args.keep:
            print(f"results kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
