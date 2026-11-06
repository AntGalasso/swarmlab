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


def _units(seeds, thresholds=None, tag="", topology="chain"):
    out = []
    for seed in seeds:
        u = dict(block="test", platform="mock", arm="C", audit="uncapped", intervention="uncapped", topology=topology,
                 attack="none", agents=12, seed=seed, sink=seed % 12)
        u["id"] = f"C-{topology}-none-12-{seed}{tag}"
        if thresholds is not None:
            u["thresholds"] = thresholds
        out.append(u)
    return out


def _rate(units, spec, settings, path):
    traces = []

    def rows(trace, st):
        traces.append(trace)
        return [{} for _ in trace.unit_ids]

    mock.MockPlatform(rows_fn=rows).run_batch(units, spec, settings, core.CheckpointStore(path), core.NullTracker())
    return sum(t.restricted.sum().item() for t in traces) / sum(t.attempts.sum().item() for t in traces)


def _closed_loop(tmp_path, topology):
    spec = next(s for s in conventional.ARMS if s.name == "C")
    settings = _settings()
    platform = mock.MockPlatform(rows_fn=lambda trace, st: [{} for _ in trace.unit_ids])
    thresholds = platform.calibrate(_units(range(9001, 9031), topology=topology), spec, settings,
                                    core.CheckpointStore(tmp_path / "cal"))
    return spec, settings, thresholds, platform.last_calibration


@pytest.mark.skipif(not _reads_closed_loop_mode(),
                    reason="mock_engine does not read calibration.mode yet (closed-loop calibration not landed)")
def test_c_calibrated_closed_loop_denies_about_alpha_of_attack_free_deliveries(tmp_path):
    spec, settings, thresholds, report = _closed_loop(tmp_path, "star_hub")
    assert report["converged"], report
    # the calibration episodes themselves hit alpha; the jitter streams of other episode ids are fresh draws, so a
    # held-out figure is averaged over several id tags instead of resting on one set of ids
    same = _rate(_units(range(9001, 9031), thresholds, topology="star_hub"), spec, settings, tmp_path / "same")
    assert math.isclose(same, ALPHA, rel_tol=0.1), same
    held = [_rate(_units(range(1, 31), thresholds, f"-t{k}", "star_hub"), spec, settings, tmp_path / f"held{k}")
            for k in range(3)]
    assert math.isclose(sum(held) / len(held), ALPHA, rel_tol=0.5), held      # other seeds: a looser, averaged check


@pytest.mark.skipif(not _reads_closed_loop_mode(), reason="closed-loop calibration not landed")
def test_c_closed_loop_reaches_alpha_on_a_chain(tmp_path):
    """Chain, N=12: the rate of C jumps with the threshold (threat-level feedback, D95), so this converges or not
    depending on where the steps fall; it is deterministic for the keyed draws."""
    assert _closed_loop(tmp_path, "chain")[3]["converged"]
