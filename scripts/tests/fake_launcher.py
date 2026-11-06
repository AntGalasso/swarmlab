#!/usr/bin/env python3
"""A stand-in for run_experiments.py that wires FAKE platforms into the real runner, to test scripts/e2e_common.py.

Not a test module. The fake mock and LLM platforms follow the Platform contract: rows are appended as they are
produced (metrics columns, valid values), a unit is marked done when finished, a restart continues after the last
saved row. FAKE_DELAY (seconds per row) makes a run slow enough to be killed. FAKE_BUG plants one defect so a test can
show the harness catches it:

  duplicate  a restart writes the rows of an unfinished unit again from the first row
  torn       a restart leaves half a line in the middle of a unit file
  range      a ratio metric above 1
  resume_shift  the part of a unit written after a restart differs from a straight run (benign_delivery_rate, in
                every row of an attack or attack-free unit; still a valid row)
  crn        one arm sees another number of attackers than the others in the same scenario
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bazt_lab.core import public as core  # noqa: E402
from bazt_lab.features.metrics import public as metrics  # noqa: E402
from bazt_lab.features.runner import public as rx  # noqa: E402

DELAY = float(os.environ.get("FAKE_DELAY", "0.02"))
BUG = os.environ.get("FAKE_BUG", "")
NAN = math.nan
LISTS = {"asr_by_round", "agents_reached_by_round", "accuracy_by_round"}
ATTACK_ONLY = ("asr_final", "agents_reached_final", "agents_reached_auc", "time_to_containment", "contained",
               "protected_sink_harm", "sink_exposed", "targeted_asr", "accuracy_attack", "nrp")
CLEAN_ONLY = ("accuracy_clean", "fpr_message", "fpr_agent")
ARMS = {
    "A": dict(), "B": dict(calibrated=True), "C": dict(calibrated=True), "BAZT": dict(calibrated=True),
    "BAZT_upstream": dict(calibrated=True), "GNN_standin": dict(platforms=("mock",), calibrated=True), "GNN": dict(platforms=("llm",)),
    "random_alloc": dict(calibrated=True), "oracle_alloc": dict(needs_truth=True), "bra_greedy": dict(calibrated=True),
    "BAZT_no_budget": dict(calibrated=True), "BAZT_no_topology": dict(calibrated=True),
    "BAZT_no_graduated": dict(calibrated=True),
}


def _no_defender():
    raise AssertionError("the fake launcher never builds a defender")


def fake_row(unit: dict) -> dict:
    """A row with every metrics column, valid for an attack-free unit (attack == 'none') or an attack unit."""
    row = {k: v for k, v in unit.items() if k not in ("id", "thresholds")}
    rounds = unit.get("rounds", 4)
    row.update(dict.fromkeys(metrics.COLUMNS, 0.5))
    row.update({c: [0.5] * rounds for c in LISTS})
    row.update(attack_free=unit["attack"] == "none", n_agents=unit["agents"], n_rounds=rounds, violations=0,
               exhausted_step=0, sink=unit.get("sink") or 0, spend_total=0.0 if unit["arm"] == "A" else 3.0,
               spend_audit_total=0.0, spend_intervention_total=0.0 if unit["arm"] == "A" else 3.0)
    for name in (ATTACK_ONLY if row["attack_free"] else CLEAN_ONLY):
        row[name] = NAN
    row["n_attackers"] = 0.0 if row["attack_free"] else 1.0
    if BUG == "range" and not row["attack_free"]:
        row["asr_final"] = 1.5
    return row


class FakePlatform:
    def __init__(self, name: str, rows_per_unit) -> None:
        self.name, self.rows_per_unit = name, rows_per_unit

    def run_batch(self, units, arm, settings, store, tracker):
        out = []
        for unit in units:
            uid = unit["id"]
            if store.is_done(uid):
                continue
            saved = store.resume_rows(uid)
            first = 0 if BUG == "duplicate" and saved else len(saved)
            for _ in range(first, self.rows_per_unit(unit)):
                row = fake_row(unit)
                if BUG == "resume_shift" and saved:
                    row["benign_delivery_rate"] = 0.9
                if BUG == "crn" and unit["arm"] == "C" and not row["attack_free"]:
                    row["n_attackers"] = 2.0
                store.append_rows(uid, [row])
                out.append(row)
                time.sleep(DELAY)
            if BUG == "torn" and saved:      # a restart that leaves half a line behind (after the store's own repair)
                with (store.root / "units" / f"{uid}.jsonl").open("a", encoding="utf-8") as f:
                    f.write('{"arm": "half a lin')
            store.mark_done(uid)
        return out

    def calibrate(self, units, arm, settings, store):
        time.sleep(DELAY)
        return {"deny_score": 0.5, "flag_score": 1.25, "audit_score": 1.0}


def wiring() -> rx.Wiring:
    specs = {name: core.ArmSpec(name, _no_defender, **flags) for name, flags in ARMS.items()}

    def arm(name):
        if name not in specs:
            raise LookupError(name)
        return specs[name]

    platforms = {"mock": FakePlatform("mock", lambda unit: 1),
                 "llm": FakePlatform("llm", lambda unit: unit["conversations"])}
    return rx.Wiring(arm=arm, platform=lambda name: platforms[name], tracker=lambda settings: core.NullTracker())


if __name__ == "__main__":
    sys.exit(rx.main(wiring=wiring()))
