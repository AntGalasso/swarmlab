#!/usr/bin/env python3
"""End-to-end smoke run of the LLM side against a FAKE API, through run_experiments.py (see scripts/e2e_common.py).

    python scripts/e2e_llm_fake.py --kickoff-override 2000-01-01T00:00:00+00:00
    python scripts/e2e_llm_fake.py --with-gnn --kickoff-override ...     # also the GNN training block
    python scripts/e2e_llm_fake.py --fake-set llm.api='"fake"'           # how the fake API is selected

No network and no secret is used: the child process gets empty keys. The launcher still applies the kickoff guard to
every LLM block; before kickoff this script stops with exit code 3 unless --kickoff-override names a time in the past,
which is an explicit, visible acceptance of a run that makes no live call. Exit code: 0 passed, 1 a check failed,
2 the run could not start, 3 the kickoff guard refused.
"""
from __future__ import annotations

import argparse
import sys

import e2e_common as e2e

from bazt_lab.core import public as core

TOPOLOGIES = ["random_directed", "hub_star"]
BACKBONE = "openai/gpt-4o-mini"      # a priced name, so --dry-run can estimate; the fake API answers for it


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    e2e.add_common_arguments(ap)
    ap.add_argument("--quick", action="store_true", help="accepted for symmetry with e2e_mock.py; the grid is always tiny")
    ap.add_argument("--agents", type=int, default=3, help="agents per conversation (default 3)")
    ap.add_argument("--seeds", type=int, default=2, help="seeds per cell (default 2; at most the 2 test seeds)")
    ap.add_argument("--with-gnn", action="store_true", help="also run the gnn_train block and the GNN arm")
    ap.add_argument("--kickoff-override", default=None, metavar="ISO_TIME",
                    help="SWARMLAB_KICKOFF_UTC for the child process, a time in the past")
    ap.add_argument("--fake-set", action="append", default=None, metavar="KEY=VALUE",
                    help="how the harness is told to use its fake API (replaces the default llm.api=\"fake\"); repeatable")
    return ap


def fake_api_sets(items: list[str] | None) -> dict[str, object]:
    if not items:
        return {"llm.api": "fake"}
    return {key.strip(): core.parse_value(text.strip()) for key, _, text in (i.partition("=") for i in items)}


def build_plan(args: argparse.Namespace) -> e2e.Plan:
    from bazt_lab.features.runner import public as rx
    seeds = min(args.seeds, 2)
    sets = {
        **e2e.ARM_TEST_VALUES,
        **e2e.runtime_sets(llm_seeds=[1, 2][:seeds], llm_calibration_seeds=[101, 102][:seeds]),
        **fake_api_sets(args.fake_set),
        **e2e.tiny_mock_sets(args.agents, seeds),
        "llm.agents": args.agents, "llm.rounds": 2, "llm.seeds_per_cell": seeds,
        "llm.conversations_per_cell": 2 * seeds, "llm.calibration_conversations_per_cell": seeds,
        "llm.n_backbones": 1, "llm.backbones": [BACKBONE], "llm.topologies": TOPOLOGIES,
        "llm.attack_cells": [["insider_pi", "mmlu"]], "llm.attack_free_tasks": ["mmlu"],
        "llm.headline_arms": ["A", "B", "C", "BAZT", *(["GNN"] if args.with_gnn else [])],
        "llm.capped_arms": ["C", "BAZT"], "llm.capped_cells": [[10, 10]], "llm.calibrated_arms": ["B", "C", "BAZT"],
    }
    if args.with_gnn:
        sets.update({"gnn_train.families": [["insider_pi", "mmlu"]], "gnn_train.edge_probabilities": [0.5],
                     "gnn_train.attackers": [1], "gnn_train.conversations_per_setting": 2,
                     "gnn_train.agents": args.agents, "gnn_train.rounds": 2, "gnn_train.backbone": BACKBONE})
    settings = rx.load_settings(overrides=e2e.override_items(sets))
    headline, capped = rx.units_llm_headline(settings, True), rx.units_llm_capped(settings, True)
    calibration = rx.units_llm_calibration(settings, True)
    evaluations = [e2e.Evaluation("llm_headline", len(headline), headline[0]["conversations"], interrupt=True),
                   e2e.Evaluation("llm_capped", len(capped), capped[0]["conversations"])]
    if args.with_gnn:
        train = rx.units_gnn_train(settings, True)
        evaluations.insert(0, e2e.Evaluation("gnn_train", len(train), None, metrics=False))
    return e2e.Plan("llm-fake", sets, [e2e.Calibration("llm_calibration", len(calibration), len(calibration.cells()))],
                    evaluations, run_flags=["--allow-live-llm"], kickoff_override=args.kickoff_override,
                    analysis_sets=e2e.ANALYSIS_TEST_VALUES)


def main(argv=None) -> int:
    return e2e.main_for("llm-fake", parser, build_plan, argv)


if __name__ == "__main__":
    sys.exit(main())
