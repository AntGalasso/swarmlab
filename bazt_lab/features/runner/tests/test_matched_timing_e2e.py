"""D116 end to end on the real mock engine: mock_calibration (BAZT), mock_frontier (BAZT), then mock_legacy with two
matched-timing controls. Each matched row carries the rounds of BAZT's row of the same scenario as its timing_schedule
and intervenes at no other round; the rows do not depend on the worker count. Every setting is a TEST value."""
import json
import math

import pytest

from bazt_lab.core import public as core
from bazt_lab.features.bazt_slot import public as slot
from bazt_lab.features.mock_engine import public as mock_engine
from bazt_lab.features.references import public as refs
from bazt_lab.features.runner import public as rx
from bazt_lab.features.runner.tests import fakes

MATCHED = ("centrality_only_matched", "random_capped_matched")
TEST = {
    "mock.uncapped_arms": ["BAZT"], "mock.capped_arms": ["BAZT"], "mock.allocation_arms": [],
    "budgets.levels_percent": [10], "mock.calibrated_arms": ["BAZT"], "calibration.mode": "open_loop",
    "mock.legacy.arms": list(MATCHED), "mock.legacy.cells": [[10, 10]],
    "mock.legacy.gamma": 0.7, "mock.legacy.dmax": 3, "mock.legacy.random_steps": 3,
    "mock.topologies": ["star_hub"], "mock.frontier_agents": [5], "mock.epidemiology_agents": [5],
    "runtime.insider_bias": 8.0, "runtime.aitm_rewrite_strength": 1.0, "costs.robustness.deny": 1,
    "mock.model.bias": 8.0, "mock.model.burst_span": 1,
    "conventional.c.judge_score_bins": {"mock": [[0.0, 0.0], [2.0, 4.0], [math.inf, 12.0]]},
    "conventional.c.presence_window_rounds": 3,
    "bazt.katz_decay": 0.5, "bazt.eigen_iterations": 50, "bazt.sink_weight": 2.0, "bazt.utility_weight": 1.0,
    "bazt.utility_loss.sanitize": 0.15, "bazt.utility_loss.delay": 0.1, "bazt.utility_loss.edge_quarantine": 0.3,
    "bazt.utility_loss.rate_limit": 0.3, "bazt.utility_loss.privilege_degradation": 0.35,
    "bazt.utility_loss.isolation": 1.0, "bazt.flag_logodds": 4.0, "bazt.clear_logodds": 3.0,
    "bazt.memory_decay": 0.8, "bazt.prior": 0.1, "bazt.inspect_conclusive": 0.8,
}
REBUILD = "bazt_lab.features.runner.tests.test_matched_timing_e2e:wiring"


def wiring():
    arms = {a.name: a for a in (*slot.ARMS, *refs.ARMS)}
    platform = mock_engine.MockPlatform()
    return rx.Wiring(arm=lambda name: arms[name], platform=lambda name: platform, tracker=lambda s: None,
                     rebuild=REBUILD)


def settings(workers=1):
    s = fakes.settings(fakes.RUNTIME_MOCK, fakes.TINY_MOCK, [f"compute.workers={workers}"])
    for key, value in TEST.items():
        core.put(s, key, value)
    return s


def rows_of(directory):
    files, uncommitted = core.committed_row_files(directory)
    assert uncommitted == 0
    return [json.loads(line) for path in files for line in path.read_text(encoding="utf-8").splitlines() if line]


def scenario(row):
    return tuple(row[k] for k in ("audit", "intervention", "topology", "attack", "agents", "seed", "sink"))


def run(root, workers=1):
    s = settings(workers)
    for block in ("mock_calibration", "mock_frontier", "mock_legacy"):
        rx.run_block(s, block, False, wiring=wiring(), results_root=root)
    return rows_of(root / "sprint" / "mock_frontier"), rows_of(root / "sprint" / "mock_legacy")


@pytest.fixture(scope="module")
def one_worker(tmp_path_factory):
    return run(tmp_path_factory.mktemp("w1"))


def test_bazt_runs_first_and_each_matched_row_acts_only_at_the_rounds_bazt_intervened(one_worker):
    frontier, legacy = one_worker
    bazt = {scenario(r): r["intervened_by_round"] for r in frontier if r["arm"] == "BAZT"}
    assert any(any(v) for v in bazt.values()) and not all(all(v) for v in bazt.values())
    matched = [r for r in legacy if r["arm"] in MATCHED]
    assert len(matched) == 2 * 2 * 2 * 5                     # arms x families x seeds x sinks at one cell
    for r in matched:
        assert r["timing_schedule"] == bazt[scenario(r)]
        outside = [t for t, (did, may) in enumerate(zip(r["intervened_by_round"], r["timing_schedule"], strict=True))
                   if did and not may]
        assert outside == [] and r["violations"] == 0
        assert "twin_timing_schedule" not in r or r["twin_timing_schedule"] == bazt[(*scenario(r)[:3], "none",
                                                                                      *scenario(r)[4:])]
    assert any(any(r["intervened_by_round"]) for r in matched)


def test_the_rows_do_not_depend_on_the_worker_count(one_worker, tmp_path):
    frontier, legacy = run(tmp_path, workers=2)
    pick = lambda rows: sorted((r["id"], json.dumps(r["intervened_by_round"]), json.dumps(r.get("timing_schedule")))  # noqa: E731
                               for r in rows)
    assert pick(frontier) == pick(one_worker[0]) and pick(legacy) == pick(one_worker[1])
