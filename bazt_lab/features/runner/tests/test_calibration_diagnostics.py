"""The platform's calibration diagnostics reach thresholds.json, the manifest and `status`."""
import json

from bazt_lab.features.runner import public as rx
from bazt_lab.features.runner.tests import fakes
from bazt_lab.features.runner.tests.test_launch import MOCK_B, block_dir, launch

GOOD = {"mode": "closed_loop", "passes": 3, "rate": 0.0201, "alpha": 0.02, "converged": True,
        "worst_episode_rate": 0.05}


def calibrated(tmp_path, diagnostics):
    plat = fakes.FakePlatform("mock")
    plat.diagnostics = diagnostics
    launch(fakes.settings(*MOCK_B), "mock_calibration", tmp_path, platform=plat)
    return block_dir(tmp_path, "mock_calibration")


def test_the_diagnostics_are_stored_with_the_thresholds_of_their_cell_and_in_the_manifest(tmp_path):
    out = calibrated(tmp_path, GOOD)
    (cell,) = json.loads((out / "thresholds.json").read_text())["cells"]
    assert cell["diagnostics"] == GOOD and "deny_score" in cell["thresholds"]
    (manifest,) = out.glob("manifest-*.json")
    assert list(json.loads(manifest.read_text())["calibration"].values()) == [GOOD]


def test_a_platform_without_diagnostics_leaves_the_record_as_it_was(tmp_path):
    out = calibrated(tmp_path, None)
    (cell,) = json.loads((out / "thresholds.json").read_text())["cells"]
    assert "diagnostics" not in cell
    (manifest,) = out.glob("manifest-*.json")
    assert "calibration" not in json.loads(manifest.read_text())


def status_out(tmp_path, capsys):
    argv = ["status", "--results-root", str(tmp_path)] + [x for g in [fakes.RUNTIME_MOCK, fakes.TINY_MOCK] for o in g
                                                          for x in ("--set", o)]
    rx.main(argv, wiring=fakes.wiring())
    return capsys.readouterr().out


def test_status_shows_a_cell_that_did_not_converge(tmp_path, capsys):
    calibrated(tmp_path, {**GOOD, "converged": False})
    out = status_out(tmp_path, capsys)
    assert "calibration warnings" in out and "arm=B topology=chain" in out and "did not converge" in out


def test_status_shows_a_cell_whose_worst_episode_rate_is_above_five_alpha(tmp_path, capsys):
    calibrated(tmp_path, {**GOOD, "worst_episode_rate": 0.11})
    out = status_out(tmp_path, capsys)
    assert "worst_episode_rate 0.11" in out and "5 x alpha" in out


def test_status_is_quiet_about_a_healthy_calibration(tmp_path, capsys):
    calibrated(tmp_path, GOOD)
    assert "calibration warnings" not in status_out(tmp_path, capsys)
    calibrated_within = {k: v for k, v in GOOD.items() if k != "converged"} | {"within_tolerance": False}
    other = tmp_path / "other"
    calibrated(other, calibrated_within)
    assert "did not converge" in status_out(other, capsys)
