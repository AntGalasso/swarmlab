#!/usr/bin/env python3
"""End-to-end smoke run of the mock side through run_experiments.py on a tiny grid (see scripts/e2e_common.py).

    python scripts/e2e_mock.py                 # 2 topologies, 2 attack families, 3 agents, 2 seeds
    python scripts/e2e_mock.py --quick         # the smallest grid that can still be killed part-way
    python scripts/e2e_mock.py --seeds 4 --agents 5 --keep

The grid and the run-time variables are TEST VALUES passed with --set; they are not experiment settings. Exit code:
0 passed, 1 a check failed, 2 the run could not start.
"""
from __future__ import annotations

import argparse
import sys

import e2e_common as e2e


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    e2e.add_common_arguments(ap)
    ap.add_argument("--quick", action="store_true", help="smaller default grid (2 seeds, 3 agents)")
    ap.add_argument("--seeds", type=int, default=None, help="seeds per cell (default 3, 2 with --quick)")
    ap.add_argument("--agents", type=int, default=None, help="agents per episode (default 4, 3 with --quick)")
    return ap


def build_plan(args: argparse.Namespace) -> e2e.Plan:
    from bazt_lab.features.runner import public as rx
    seeds = args.seeds or (2 if args.quick else 3)
    agents = args.agents or (3 if args.quick else 4)
    sets = {
        **e2e.ARM_TEST_VALUES,
        **e2e.runtime_sets(),
        **e2e.tiny_mock_sets(agents, seeds),
    }
    settings = rx.load_settings(overrides=e2e.override_items(sets))
    calibration = rx.units_mock_calibration(settings, True)
    return e2e.Plan(
        "mock", sets,
        calibrations=[e2e.Calibration("mock_calibration", len(calibration), len(calibration.cells()))],
        evaluations=[e2e.Evaluation("mock_frontier", len(rx.units_mock_frontier(settings, True)), 1, interrupt=True)],
        analysis_sets=e2e.ANALYSIS_TEST_VALUES)


def main(argv=None) -> int:
    return e2e.main_for("mock", parser, build_plan, argv)


if __name__ == "__main__":
    sys.exit(main())
