"""Identity of a block's settings, seeds, preset expansion, probes: what keeps a long launch from failing hours in or
mixing rows from different settings."""
import json

import pytest

from bazt_lab.core import public as core
from bazt_lab.features.runner import public as rx
from bazt_lab.features.runner.tests import fakes
from bazt_lab.features.runner.tests.test_launch import (
    MOCK_A,
    MOCK_B,
    ORACLE,
    block_dir,
    calibrate_then,
    launch,
    past_kickoff,
)

LLM = [fakes.RUNTIME_LLM, fakes.TINY_LLM]
PRESETS = [fakes.RUNTIME_MOCK, fakes.TINY_MOCK, ["mock.presets_robustness.arms=['A']",
                                                 "mock.presets_robustness.cells=[[10, 10]]",
                                                 "auditor_presets.adaptive_tpr=[0.05, 0.3]"]]


# ============================================================================================== preset expansion
def test_the_adaptive_preset_expands_into_one_unit_set_per_tpr_point():
    s = fakes.settings(*PRESETS)
    units = list(rx.BLOCKS["mock_presets_robustness"].units(s, True))
    by_preset = {}
    for u in units:
        by_preset.setdefault(u["preset"], []).append(u)
    assert set(by_preset) == {"strong_model", "weak_judge", "adaptive"}
    assert {json.dumps(u["attack_params"]) for u in by_preset["adaptive"]} == {'{"auditor_tpr": 0.05}',
                                                                              '{"auditor_tpr": 0.3}'}
    assert len(by_preset["adaptive"]) == 2 * len(by_preset["weak_judge"])
    assert all("attack_params" not in u for u in by_preset["weak_judge"] + by_preset["strong_model"])
    assert len({rx.unit_id(u) for u in units}) == len(units)


def test_the_expanded_preset_grid_is_counted_without_expansion():
    grid = rx.BLOCKS["mock_presets_robustness"].units(fakes.settings(*PRESETS), True)
    assert len(grid) == len(list(grid))


@pytest.mark.parametrize("override, message", [
    ("auditor_presets.adaptive_tpr=[0.3, 0.05]", "adaptive_tpr"),
    ("auditor_presets.adaptive_tpr=[0.05]", "adaptive_tpr"),
    ("mock.presets_robustness.presets=['nonsense']", "nonsense"),
])
def test_a_preset_the_grid_cannot_build_is_refused_before_anything_runs(tmp_path, override, message):
    s = fakes.settings(*PRESETS, [override])
    with pytest.raises(SystemExit, match=rf"Refusing mock_presets_robustness: .*{message}"):
        launch(s, "mock_presets_robustness", tmp_path)
    assert not any(tmp_path.iterdir())


# ======================================================================================= settings identity
def test_a_second_launch_with_other_settings_into_the_same_block_is_refused_naming_the_keys(tmp_path, monkeypatch):
    past_kickoff(monkeypatch)
    s = fakes.settings(*LLM, ["llm.headline_arms=['A']"])
    launch(fakes.settings(*LLM, ["llm.headline_arms=['A']"]), "llm_headline", tmp_path, platform_name="llm",
           allow_live_llm=True)
    before = sorted(p.name for p in (block_dir(tmp_path, "llm_headline") / "units").iterdir())
    other = fakes.settings(*LLM, ["llm.headline_arms=['A']", "llm.conversations_per_cell=20"])
    with pytest.raises(SystemExit, match=r"Refusing llm_headline: .*llm\.conversations_per_cell"):
        launch(other, "llm_headline", tmp_path, platform_name="llm", allow_live_llm=True)
    assert sorted(p.name for p in (block_dir(tmp_path, "llm_headline") / "units").iterdir()) == before
    assert launch(s, "llm_headline", tmp_path, platform_name="llm", allow_live_llm=True).batches == []


def test_the_runtime_variables_and_the_model_are_part_of_a_blocks_identity(tmp_path):
    launch(fakes.settings(*MOCK_A), "mock_frontier", tmp_path)
    for override, key in [("runtime.horizon_constant=9", "runtime.horizon_constant"),
                          ("mock.model.beta=0.5", "mock.model.beta"), ("costs.main.sanitize=2", "costs.main.sanitize"),
                          ("calibration.mode='open_loop'", "calibration.mode")]:
        with pytest.raises(SystemExit, match=rf"Refusing mock_frontier: .*{key}"):
            launch(fakes.settings(*MOCK_A, [override]), "mock_frontier", tmp_path)


def test_performance_tracking_and_unrelated_blocks_settings_do_not_change_a_blocks_identity(tmp_path):
    launch(fakes.settings(*MOCK_A), "mock_frontier", tmp_path)
    again = fakes.settings(*MOCK_A, ["mock.batch_size=2", "tracking.mode='online'", "estimates.calls_per_second=1",
                                     "mock.ablations.arms=['BAZT_no_budget']", "llm.agents=20",
                                     "runtime.ni_margin_accuracy=0.02"])
    assert launch(again, "mock_frontier", tmp_path).batches == []


def test_thresholds_from_other_settings_are_refused_not_reused(tmp_path):
    s = fakes.settings(*MOCK_B)
    calibrate_then(tmp_path, s)
    changed = fakes.settings(*MOCK_B, ["runtime.ws_k=8"])
    with pytest.raises(SystemExit, match=r"Refusing mock_frontier: .*thresholds.*runtime\.ws_k"):
        launch(changed, "mock_frontier", tmp_path)
    assert not block_dir(tmp_path, "mock_frontier").exists()
    other_mode = fakes.settings(*MOCK_B, ["calibration.mode='open_loop'"])
    with pytest.raises(SystemExit, match=r"calibration\.mode"):
        launch(other_mode, "mock_frontier", tmp_path)
    grid_only = fakes.settings(*MOCK_B, ["mock.main_seeds=[3, 4]", "mock.capped_arms=[]"])
    assert launch(grid_only, "mock_frontier", tmp_path).batches            # evaluation-only keys do not matter


def test_a_calibration_is_not_extended_under_other_settings(tmp_path):
    chain_star = ["mock.topologies=['chain', 'star_hub']"]
    crashing = fakes.FakePlatform("mock", crash_on=lambda u: u["topology"] == "star_hub")
    with pytest.raises(RuntimeError, match="simulated crash"):
        calibrate_then(tmp_path, fakes.settings(*MOCK_B, chain_star), crashing)
    with pytest.raises(SystemExit, match=r"Refusing mock_calibration: .*runtime\.ws_k"):
        calibrate_then(tmp_path, fakes.settings(*MOCK_B, chain_star, ["runtime.ws_k=8"]))


def test_two_launches_in_the_same_second_leave_two_manifests(tmp_path):
    s = fakes.settings(*MOCK_A)
    launch(s, "mock_frontier", tmp_path)
    launch(s, "mock_frontier", tmp_path)
    assert len(list(block_dir(tmp_path, "mock_frontier").glob("manifest-*.json"))) == 2


# ===================================================================================== the adaptive thresholds
def test_the_adaptive_search_uses_the_same_thresholds_as_the_evaluation(tmp_path):
    s = fakes.settings(*ORACLE, ["mock.adaptive.arms=['oracle_alloc']", "mock.adaptive.cells=[[10, 10]]",
                                 "mock.adaptive.conditions=['gray_box']"])
    calibrate_then(tmp_path, s, fakes.FakePlatform("mock", calibrate_result={"audit_score": 0.7}))
    adaptive, plat = fakes.FakeAdaptive(), fakes.FakePlatform("mock")
    rx.run_block(s, "mock_adaptive", False, wiring=fakes.wiring({"mock": plat}, adaptive=adaptive),
                 results_root=tmp_path)
    ((_, searched, _),) = adaptive.searches
    sent = [u for _, us in plat.batches for u in us]
    assert searched == {"audit_score": 0.7} and {json.dumps(u["thresholds"]) for u in sent} == {'{"audit_score": 0.7}'}


# ================================================================================================== seeds
@pytest.mark.parametrize("override", ["mock.tune_seeds=[1, 2]", "mock.tune_seeds=[2, 40]",
                                      "mock.heldout_seeds=[2, 9]"])
def test_mock_seed_splits_must_be_disjoint(tmp_path, override):
    with pytest.raises(SystemExit, match=r"Refusing mock_frontier: (.*seeds.*overlap|.*not frozen)"):
        launch(fakes.settings(*MOCK_A, [override]), "mock_frontier", tmp_path)
    assert not any(tmp_path.iterdir())


def test_heldout_seeds_cannot_be_run_through_another_block_before_the_freeze(tmp_path):
    s = fakes.settings(*MOCK_A, ["mock.main_seeds=[31, 32]"])
    with pytest.raises(SystemExit, match="not frozen"):
        launch(s, "mock_frontier", tmp_path)
    assert not any(tmp_path.iterdir())


def test_llm_seed_sets_are_disjoint_and_as_many_as_seeds_per_cell(tmp_path, monkeypatch):
    past_kickoff(monkeypatch)
    for override, message in [("runtime.llm_calibration_seeds=[2, 3]", "overlap"),
                              ("runtime.llm_seeds=[1, 2, 3]", "llm.seeds_per_cell"),
                              ("runtime.llm_calibration_seeds=[5]", "llm.seeds_per_cell")]:
        s = fakes.settings(*LLM, [override])
        with pytest.raises(SystemExit, match=rf"Refusing llm_headline: .*{message}"):
            launch(s, "llm_headline", tmp_path, platform_name="llm", allow_live_llm=True)
    assert not any(tmp_path.iterdir())


def test_a_list_given_without_brackets_is_refused():
    with pytest.raises(SystemExit, match=r"runtime\.llm_seeds.*brackets"):
        rx.load_settings(overrides=["runtime.llm_seeds=1,2,3,4,5"])
    assert core.get(rx.load_settings(overrides=["mock.note=hello"]), "mock.note") == "hello"


# ================================================================================================= probes
def test_a_mock_block_runs_one_tiny_real_batch_per_arm_and_condition_before_it_starts(tmp_path):
    plat = launch(fakes.settings(*MOCK_A), "mock_frontier", tmp_path)
    assert len(plat.probes) == 2                                  # A x (infection, none)
    assert all(len(units) == 1 and root == "preflight-probe" for _, units, root in plat.probes)
    assert {u["attack"] for _, units, _ in plat.probes for u in units} == {"infection", "none"}
    assert all("id" in units[0] and "thresholds" in units[0] for _, units, _ in plat.probes)
    assert [len(u) for _, u in plat.batches] == [4, 2, 4, 2]       # the probes are not part of the run


def test_a_probe_that_fails_refuses_the_launch_naming_the_arm_before_anything_is_written(tmp_path):
    plat = fakes.FakePlatform("mock")
    plat.probe_raises = ValueError("unknown preset 'x'")
    with pytest.raises(SystemExit, match=r"Refusing mock_frontier: .*arm 'A'.*unknown preset 'x'"):
        launch(fakes.settings(*MOCK_A), "mock_frontier", tmp_path, platform=plat)
    assert plat.batches == [] and not any(tmp_path.iterdir())
    plat.probe_raises = core.Unset("bazt.risk_weight")
    with pytest.raises(SystemExit, match=r"Refusing mock_frontier: .*bazt\.risk_weight"):
        launch(fakes.settings(*MOCK_A), "mock_frontier", tmp_path, platform=plat)
    assert not any(tmp_path.iterdir())


def test_probes_cover_the_capped_and_uncapped_forms_of_each_arm(tmp_path):
    plat = launch(fakes.settings(*MOCK_B), "mock_calibration", tmp_path)
    assert {a for a, _, _ in plat.probes} == {"B"}
    calibrate_then(tmp_path, fakes.settings(*MOCK_B))
    ev = launch(fakes.settings(*MOCK_B), "mock_frontier", tmp_path)
    forms = {(u["audit"] == "uncapped", u["intervention"] == "uncapped") for _, units, _ in ev.probes
             for u in units if u["arm"] == "B"}
    assert forms == {(True, True), (False, False), (False, True), (True, False)}


def test_the_dry_run_hours_add_the_checkpoint_cost_per_unit(capsys):
    argv = ["--dry-run", "--block", "mock_frontier", "--set", "estimates.mock_checkpoint_seconds_per_unit=0.5"]
    rx.main(argv, wiring=fakes.wiring())
    s = rx.load_settings()
    grid = rx.units_mock_frontier(s, False)
    from bazt_lab.features.runner.private import sizing
    cpu = sum(c * (sizing.seconds_per_episode(s, a, kind, size) + 0.5) for (a, kind, size), c in grid.cost_groups().items())
    frontier = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("mock_frontier"))
    assert f"{cpu / 3600:.2f}" in frontier
