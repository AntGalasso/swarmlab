"""D116 in the runner: the matched-timing controls of mock_legacy act at the rounds BAZT intervened in the same unit
scenario, read from mock_frontier's committed BAZT rows (`intervened_by_round`). BAZT runs first; a launch without its
rows, or with scenarios it never runs, is refused before anything is written; the source is pinned in the block."""
import json

import pytest

from bazt_lab.core import public as core
from bazt_lab.features.runner import public as rx
from bazt_lab.features.runner.private import blocks, runall
from bazt_lab.features.runner.tests import fakes

T = 4
ARMS = {**fakes.ARMS, "BAZT": fakes._arm("BAZT"),
        **{n: fakes._arm(n, platforms=("mock",)) for n in ("centrality_only", "centrality_only_matched")},
        **{n: fakes._arm(n, platforms=("mock",), calibrated=True) for n in ("fixed_early", "fixed_early_matched")}}
FRONTIER = ["mock.uncapped_arms=['BAZT']", "mock.capped_arms=['BAZT']", "mock.allocation_arms=[]",
            "budgets.levels_percent=[10]"]
LEGACY = ["mock.legacy.arms=['centrality_only', 'centrality_only_matched']", "mock.legacy.cells=[[10, 10]]"]


def schedule(unit, salt=0):
    """The fake BAZT record of a unit: T booleans that differ between scenarios (seed, sink, attack, budget cell)."""
    key = unit["seed"] + 3 * (unit["sink"] or 0) + 5 * (unit["attack"] != "none") + (unit["audit"] == "uncapped") + salt
    return [(key + r) % 3 == 0 for r in range(T)]


class RowsPlatform(fakes.FakePlatform):
    """Writes one row per unit as the mock does: the unit's fields plus the intervened_by_round column."""

    def __init__(self, salt=0, column=True):
        super().__init__("mock")
        self.salt, self.column = salt, column

    def run_batch(self, units, arm, settings, store, tracker):
        if store.root.name == "preflight-probe":
            return super().run_batch(units, arm, settings, store, tracker)
        self.batches.append((arm.name, [dict(u) for u in units]))
        for u in units:
            if not store.is_done(u["id"]):
                row = {**u, **({"intervened_by_round": schedule(u, self.salt)} if self.column else {})}
                store.append_rows(u["id"], [row])
                store.mark_done(u["id"])
        return []


def settings(*extra):
    return fakes.settings(fakes.RUNTIME_MOCK, fakes.TINY_MOCK, FRONTIER, LEGACY, *extra)


def launch(s, block, root, platform=None):
    plat = platform or RowsPlatform()
    rx.run_block(s, block, False, wiring=fakes.wiring({"mock": plat}, arms=ARMS), results_root=root)
    return plat


def bazt_of(unit, attack=None):
    """The mock_frontier BAZT unit of the same scenario (every field but the arm and the block)."""
    plain = {k: v for k, v in unit.items() if k not in ("id", "thresholds", "twin_id", "timing_schedule",
                                                        "twin_timing_schedule")}
    return {**plain, "arm": "BAZT", "block": "mock_frontier", **({"attack": attack} if attack else {})}


def sent(plat, arm):
    return [u for name, units in plat.batches if name == arm for u in units]


# ======================================================================================================= handover
def test_each_matched_unit_gets_exactly_the_rounds_bazt_recorded_for_its_scenario(tmp_path):
    s = settings()
    launch(s, "mock_frontier", tmp_path)
    plat = launch(s, "mock_legacy", tmp_path)
    matched = sent(plat, "centrality_only_matched")
    assert len(matched) == 12
    for u in matched:
        assert u["timing_schedule"] == schedule(bazt_of(u))
        if u["attack"] == "none":
            assert "twin_timing_schedule" not in u
        else:                                    # the attack-free twin acts at the attack-free BAZT unit's rounds
            assert u["twin_timing_schedule"] == schedule(bazt_of(u, attack="none"))
        assert u["id"] == rx.unit_id({k: v for k, v in u.items() if k not in ("timing_schedule", "twin_timing_schedule")})
    assert len({json.dumps(u["timing_schedule"]) for u in matched}) > 1
    assert all("timing_schedule" not in u for u in sent(plat, "centrality_only"))
    probes = [u for arm, units, _ in plat.probes if arm == "centrality_only_matched" for u in units]
    assert probes and all(u["timing_schedule"] == schedule(bazt_of(u)) for u in probes)


def test_the_schedule_does_not_take_part_in_the_unit_identity():
    unit = {"block": "mock_legacy", "arm": "x_matched", "seed": 1}
    assert rx.unit_id({**unit, "timing_schedule": [True], "twin_timing_schedule": [False]}) == rx.unit_id(unit)


# ======================================================================================================== refusals
def test_without_bazt_rows_the_matched_block_is_refused_before_anything_runs(tmp_path):
    plat = RowsPlatform()
    with pytest.raises(SystemExit, match=r"Refusing mock_legacy: .*BAZT.*run --block mock_frontier first"):
        launch(settings(), "mock_legacy", tmp_path, plat)
    assert plat.batches == [] and plat.probes == [] and not (tmp_path / "sprint" / "mock_legacy").exists()


def test_bazt_rows_missing_for_some_scenarios_are_refused_naming_how_many_and_one(tmp_path):
    s = settings()
    launch(s, "mock_frontier", tmp_path)
    units = tmp_path / "sprint" / "mock_frontier" / "units"
    victim = next(p for p in sorted(units.glob("BAZT-*.jsonl"))
                  if json.loads(p.read_text(encoding="utf-8"))["audit"] == 0.1
                  and json.loads(p.read_text(encoding="utf-8"))["intervention"] == 0.1)
    victim.with_suffix(".done").unlink()                         # a crashed or unfinished BAZT unit: not committed
    plat = RowsPlatform()
    with pytest.raises(SystemExit, match=r"Refusing mock_legacy: BAZT has no committed row for 1 of the 12 unit"):
        launch(s, "mock_legacy", tmp_path, plat)
    assert plat.batches == [] and plat.probes == []


def test_scenarios_bazt_never_runs_are_refused_naming_the_source_block(tmp_path):
    s = settings(["mock.legacy.cells=[[5, 5]]"])
    launch(s, "mock_frontier", tmp_path)
    with pytest.raises(SystemExit, match=r"Refusing mock_legacy: .*mock_frontier never runs BAZT.*audit=0\.05"):
        launch(s, "mock_legacy", tmp_path)


def test_a_bazt_row_without_the_column_is_refused(tmp_path):
    s = settings()
    launch(s, "mock_frontier", tmp_path, RowsPlatform(column=False))
    with pytest.raises(SystemExit, match=r"Refusing mock_legacy: .*intervened_by_round"):
        launch(s, "mock_legacy", tmp_path)


def test_a_matched_arm_in_a_block_without_a_timing_source_is_refused(tmp_path):
    s = settings(["mock.uncapped_arms=['BAZT', 'centrality_only_matched']"])
    plat = RowsPlatform()
    with pytest.raises(SystemExit, match=r"Refusing mock_frontier: arm 'centrality_only_matched'.*mock_legacy"):
        launch(s, "mock_frontier", tmp_path, plat)
    assert plat.batches == [] and not any(tmp_path.iterdir())


# =========================================================================================== pin and run-all order
def test_the_source_is_pinned_and_a_bazt_rerun_that_changes_the_rounds_is_refused(tmp_path):
    s = settings()
    launch(s, "mock_frontier", tmp_path)
    launch(s, "mock_legacy", tmp_path)
    pin = json.loads((tmp_path / "sprint" / "mock_legacy" / "timing_schedules.json").read_text(encoding="utf-8"))
    assert pin["source"] == "mock_frontier" and pin["arm"] == "BAZT" and pin["units"] == 12
    assert pin["settings"] == json.loads((tmp_path / "sprint" / "mock_frontier" / "settings.json").read_text(
        encoding="utf-8"))["settings"]
    launch(s, "mock_legacy", tmp_path)                           # the same rounds again: a rerun resumes
    import shutil
    shutil.rmtree(tmp_path / "sprint" / "mock_frontier")
    launch(s, "mock_frontier", tmp_path, RowsPlatform(salt=1))   # BAZT run again and intervening elsewhere
    with pytest.raises(SystemExit, match=r"Refusing mock_legacy: .*timing_schedules\.json.*other BAZT rounds"):
        launch(s, "mock_legacy", tmp_path)


def test_a_bazt_run_under_other_settings_is_refused(tmp_path):
    s = settings()
    launch(s, "mock_frontier", tmp_path)
    launch(s, "mock_legacy", tmp_path)
    path = tmp_path / "sprint" / "mock_frontier" / "settings.json"
    view = json.loads(path.read_text(encoding="utf-8"))
    view["settings"]["bazt.prior"] = 0.123                      # as if BAZT had been run again with another prior
    path.write_text(json.dumps(view), encoding="utf-8")
    with pytest.raises(SystemExit, match=r"Refusing mock_legacy: .*bazt\.prior"):
        launch(s, "mock_legacy", tmp_path)


def test_run_all_runs_the_bazt_block_before_the_matched_block():
    assert blocks.BLOCKS["mock_legacy"].timing_source == "mock_frontier"
    s = settings(fakes.SCOPED, ["mock.legacy.arms=['centrality_only_matched']"])
    order, _ = runall.plan(s, rx.RESULTS_ROOT, False, fakes.wiring(arms=ARMS))
    assert order.index("mock_frontier") < order.index("mock_legacy")


# ======================================================================== thresholds of a calibrated matched arm
CALIBRATED = ["mock.legacy.arms=['fixed_early_matched']", "calibration.mode='open_loop'"]


def calibrate(root, arms):
    rx.run_block(fakes.settings(fakes.RUNTIME_MOCK, fakes.TINY_MOCK, FRONTIER, LEGACY, CALIBRATED,
                                [f"mock.calibrated_arms={arms!r}"]),
                 "mock_calibration", False, wiring=fakes.wiring({"mock": RowsPlatform()}, arms=ARMS), results_root=root)


def without_choice(*extra):
    """Settings with the owner's threshold choice removed, whatever profiles.toml sets it to."""
    s = settings(*extra)
    s["mock"]["legacy"].pop("matched_thresholds", None)
    assert not core.is_set(s, "mock.legacy.matched_thresholds")
    return s


def test_a_calibrated_matched_arm_needs_the_owner_choice_of_its_thresholds(tmp_path):
    s = without_choice(CALIBRATED, ["mock.calibrated_arms=['fixed_early']"])
    calibrate(tmp_path, ["fixed_early"])
    launch(s, "mock_frontier", tmp_path)
    with pytest.raises(SystemExit, match=r"Refusing mock_legacy: setting not set .*mock\.legacy\.matched_thresholds"):
        launch(s, "mock_legacy", tmp_path)


def test_base_thresholds_are_the_legacy_arms_own_cell(tmp_path):
    s = settings(CALIBRATED, ["mock.calibrated_arms=['fixed_early']", "mock.legacy.matched_thresholds='base'"])
    calibrate(tmp_path, ["fixed_early"])
    launch(s, "mock_frontier", tmp_path)
    plat = launch(s, "mock_legacy", tmp_path)
    want = {"deny_score": fakes.threshold_of(("fixed_early", "chain", 3, None, None))}
    assert {json.dumps(u["thresholds"]) for u in sent(plat, "fixed_early_matched")} == {json.dumps(want)}


def test_own_thresholds_are_the_matched_arms_cell(tmp_path):
    s = settings(CALIBRATED, ["mock.calibrated_arms=['fixed_early_matched']", "mock.legacy.matched_thresholds='own'"])
    calibrate(tmp_path, ["fixed_early_matched"])
    launch(s, "mock_frontier", tmp_path)
    plat = launch(s, "mock_legacy", tmp_path)
    want = {"deny_score": fakes.threshold_of(("fixed_early_matched", "chain", 3, None, None))}
    assert {json.dumps(u["thresholds"]) for u in sent(plat, "fixed_early_matched")} == {json.dumps(want)}


def test_an_unknown_threshold_choice_is_refused(tmp_path):
    s = settings(CALIBRATED, ["mock.calibrated_arms=['fixed_early']", "mock.legacy.matched_thresholds='mine'"])
    calibrate(tmp_path, ["fixed_early"])
    launch(s, "mock_frontier", tmp_path)
    with pytest.raises(SystemExit, match=r"mock\.legacy\.matched_thresholds must be one of"):
        launch(s, "mock_legacy", tmp_path)


def test_a_matched_arm_cannot_be_calibrated_closed_loop(tmp_path):
    s = fakes.settings(fakes.RUNTIME_MOCK, fakes.TINY_MOCK, ["mock.calibrated_arms=['fixed_early_matched']",
                                                             "calibration.mode='closed_loop'"])
    plat = RowsPlatform()
    with pytest.raises(SystemExit, match=r"Refusing mock_calibration: arm 'fixed_early_matched'.*closed_loop"):
        rx.run_block(s, "mock_calibration", False, wiring=fakes.wiring({"mock": plat}, arms=ARMS),
                     results_root=tmp_path)
    assert plat.calibrations == []


def test_the_threshold_choice_is_not_part_of_the_threshold_identity():
    a = settings(["mock.legacy.matched_thresholds='base'"])
    assert core.get(a, "mock.legacy.matched_thresholds") == "base"
    from bazt_lab.features.runner.private import identity
    assert identity.threshold_view(a, "mock") == identity.threshold_view(without_choice(), "mock")


def test_bazt_rows_produced_under_other_settings_than_this_launch_are_refused(tmp_path):
    """Even on the first launch of the matched block: its rounds must come from BAZT under the settings it runs with."""
    launch(settings(), "mock_frontier", tmp_path)
    plat = RowsPlatform()
    with pytest.raises(SystemExit, match=r"Refusing mock_legacy: .*mock_frontier.*other settings.*runtime\.eps_correct"):
        launch(settings(["runtime.eps_correct=2.0"]), "mock_legacy", tmp_path, plat)
    assert plat.batches == [] and not (tmp_path / "sprint" / "mock_legacy").exists()
