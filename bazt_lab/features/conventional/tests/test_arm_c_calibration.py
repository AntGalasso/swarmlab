"""Calibrated C on the real MockPlatform: with the threshold found closed-loop, the attack-free deny rate is within 10% of alpha.

Open-loop calibration (nobody flagged, cold histories) makes calibrated C deny 17-100% of attack-free deliveries. The
closed-loop mode is built by mock_engine behind the settings key `calibration.mode`; this test runs once that key is
read there and is skipped until then.
"""
import inspect
import math
from pathlib import Path

import pytest

from bazt_lab.core import public as core
from bazt_lab.features.conventional import public as conventional
from bazt_lab.features.mock_engine import public as mock

ALPHA = 0.02
MODE_KEY = "calibration.mode"
CLOSED_LOOP = "closed_loop"          # assumed value of calibration.mode; mock_engine's CONTRACT.md has the final name


def _reads_closed_loop_mode() -> bool:
    return MODE_KEY in Path(inspect.getsourcefile(mock.MockPlatform)).read_text(encoding="utf-8")


def _settings():
    s = core.load_settings(profile="sprint")           # test values for the run-time variables, none is an experiment decision
    for key, value in {
        "runtime.n_insiders": 2, "runtime.rate_limit_per_identity": 20, "runtime.horizon_constant": 3.0,
        "runtime.eps_correct": 2.0, "runtime.ba_m": 2, "runtime.ws_k": 4, "runtime.ws_rewire_p": 0.25,
        "runtime.sanitize_success": 1.0, "runtime.insider_bias": 8.0, "runtime.ta_detectability": 1.5,
        "runtime.ma_detectability": 1.2, "runtime.aitm_rewrite_strength": 1.0, "runtime.recovery_margin": 0.05,
        "mock.model.bias": 8.0, "mock.model.burst_span": 1, "mock.auditor.binormal_method": "shared_threshold",
        "costs.robustness.deny": 1, "costs.main.deny": 1, MODE_KEY: CLOSED_LOOP,
        "conventional.c.presence_window_rounds": 3,
        "conventional.c.judge_score_bins": {"mock": [[0.0, 0.0], [2.0, 4.0], [1e9, 12.0]]},
    }.items():
        core.put(s, key, value)
    return s


def _units(seeds, thresholds=None):
    out = []
    for seed in seeds:
        u = dict(block="test", platform="mock", arm="C", audit="uncapped", intervention="uncapped", topology="chain",
                 attack="none", agents=12, seed=seed, sink=seed % 12)
        u["id"] = f"C-chain-none-12-{seed}"
        if thresholds is not None:
            u["thresholds"] = thresholds
        out.append(u)
    return out


@pytest.mark.skipif(not _reads_closed_loop_mode(),
                    reason="mock_engine does not read calibration.mode yet (closed-loop calibration not landed)")
def test_c_calibrated_closed_loop_denies_about_alpha_of_attack_free_deliveries(tmp_path):
    spec = next(s for s in conventional.ARMS if s.name == "C")
    settings = _settings()
    thresholds = mock.MockPlatform(rows_fn=lambda trace, st: [{} for _ in trace.unit_ids]).calibrate(
        _units(range(9001, 9031)), spec, settings, core.CheckpointStore(tmp_path / "cal"))
    traces = []

    def rows(trace, st):
        traces.append(trace)
        return [{} for _ in trace.unit_ids]

    mock.MockPlatform(rows_fn=rows).run_batch(_units(range(1, 31), thresholds), spec, settings,
                                              core.CheckpointStore(tmp_path / "run"), core.NullTracker())
    restricted = sum(t.restricted.sum().item() for t in traces)
    attempts = sum(t.attempts.sum().item() for t in traces)
    assert math.isclose(restricted / attempts, ALPHA, rel_tol=0.1)
