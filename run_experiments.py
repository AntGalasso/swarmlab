#!/usr/bin/env python3
"""run_experiments.py -- launcher for the BAZT experiments (docs/EXPERIMENT_PLAN.md). Thin entry point: the code is in
bazt_lab (bazt_lab/app.py wires it, bazt_lab/features/runner implements it). Every setting comes from
experiments/profiles.toml; nothing has a default. Two profiles size the LLM grid (spec §1.1): `sprint` and `research`.

  python run_experiments.py --list                       blocks, unit counts, missing settings and features
  python run_experiments.py --dry-run                    sizes: units, conversations, LLM calls, hours
  python run_experiments.py --run --block mock_calibration --block mock_frontier    launch blocks (resumable)
  python run_experiments.py --profile research --set llm.conversations_per_cell=80 --dry-run

Rows go to results/experiments/<profile>/<block>/ (--results-root changes the base). Calibrate before evaluating:
the calibration blocks write thresholds.json, and the evaluation blocks of the platform refuse to run without it.
"""
import sys

from bazt_lab.app import main

if __name__ == "__main__":
    sys.exit(main())
