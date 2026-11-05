"""Offline tests for run_experiments.py: profiles, overrides, unit grids, refusals. No component is called."""
import json

import pytest

import run_experiments as rx

RUNTIME_MOCK = ["runtime.n_insiders=3", "runtime.rate_limit_per_identity=1", "runtime.horizon_constant=3",
                "runtime.eps_correct=1.0", "runtime.recovery_margin=0.01", "runtime.ba_m=2", "runtime.ws_k=4",
                "runtime.ws_rewire_p=0.25", "runtime.ni_margin_accuracy=0.01",
                "runtime.rank_agreement_threshold=0.8"]   # test values, not experiment settings


def test_profiles_differ_only_in_the_llm_grid():
    sprint, research = rx.load_settings(profile="sprint"), rx.load_settings(profile="research")
    assert rx.get(sprint, "llm.conversations_per_cell") == 60
    assert rx.get(sprint, "llm.capped_cells") == [[10, 10]]
    assert rx.get(research, "llm.conversations_per_cell") == 100
    assert rx.get(research, "llm.capped_cells") == [[2, 2], [10, 10], [50, 50]]
    strip = lambda s: {k: v for k, v in s.items() if k not in ("llm", "profile")}
    assert strip(sprint) == strip(research)


def test_active_profile_is_sprint():
    assert rx.load_settings()["profile"] == "sprint"


def test_set_overrides_one_value_with_toml_parsing():
    s = rx.load_settings(overrides=["llm.conversations_per_cell=80", "runtime.llm_seeds=[1, 2, 3, 4, 5]"])
    assert rx.get(s, "llm.conversations_per_cell") == 80
    assert rx.get(s, "runtime.llm_seeds") == [1, 2, 3, 4, 5]


def test_unknown_profile_is_refused():
    with pytest.raises(SystemExit):
        rx.load_settings(profile="nope")


def test_runtime_variables_have_no_value_in_the_settings():
    s = rx.load_settings()
    for platform in ("llm", "mock"):
        assert set(rx.PRE_REGISTERED) <= set(rx.missing_runtime(s, platform))


def test_llm_grid_sizes_follow_the_profile():
    sprint, research = rx.load_settings(profile="sprint"), rx.load_settings(profile="research")
    per_arm_cell = 2 * 4 * (11 + 5) * 5        # backbones x topologies x (attack + attack-free cells) x seeds
    assert len(rx.units_llm_headline(sprint, False)) == 5 * per_arm_cell
    assert len(rx.units_llm_capped(sprint, False)) == 2 * 1 * per_arm_cell
    assert len(rx.units_llm_capped(research, False)) == 2 * 3 * per_arm_cell
    assert {u["conversations"] for u in rx.units_llm_headline(sprint, False)} == {12}
    assert {u["conversations"] for u in rx.units_llm_headline(research, False)} == {20}
    assert {(u["audit"], u["intervention"]) for u in rx.units_llm_capped(sprint, False)} == {(0.10, 0.10)}


def test_strict_units_refuse_placeholders():
    s = rx.load_settings()
    with pytest.raises(rx.Unset):
        rx.units_llm_headline(s, True)


def test_conversations_must_split_evenly_across_seeds():
    s = rx.load_settings(overrides=["llm.conversations_per_cell=61"])
    with pytest.raises(SystemExit):
        rx.units_llm_headline(s, False)


def test_capped_cells_cross_both_budgets_without_the_uncapped_corner():
    cells = rx._capped_cells([1, 2, 5, 10, 20, 50])
    assert len(cells) == 7 * 7 - 1
    assert ("uncapped", "uncapped") not in cells
    assert (0.01, "uncapped") in cells and ("uncapped", 0.5) in cells


def test_mock_frontier_runs_every_sink():
    s = rx.load_settings(overrides=["mock.topologies=['chain']", "mock.families=['infection']",
                                    "mock.main_seeds=[1, 1]"])
    units = list(rx.units_mock_frontier(s, False))
    arms_cells = 5 + (3 + 3) * 48
    assert len(units) == arms_cells * 2 * 12           # families incl. attack-free, x 12 sinks
    assert {u["sink"] for u in units} == set(range(12))


def test_unscoped_mock_blocks_refuse_until_arms_and_cells_are_set():
    s = rx.load_settings()
    with pytest.raises(rx.Unset):
        rx.BLOCKS["mock_ablations"].units(s, False)
    s = rx.load_settings(overrides=["mock.ablations.arms=['BAZT_no_budget']", "mock.ablations.cells=[[10, 10]]",
                                    "mock.main_seeds=[1, 1]"])
    units = list(rx.BLOCKS["mock_ablations"].units(s, False))
    assert {(u["audit"], u["intervention"]) for u in units} == {(0.10, 0.10)}


def test_unit_ids_are_stable_and_distinct():
    s = rx.load_settings()
    units = rx.units_llm_capped(s, False)
    ids = [rx.unit_id(u) for u in units]
    assert len(set(ids)) == len(ids)
    assert ids == [rx.unit_id(json.loads(json.dumps(u))) for u in units]


def test_run_refuses_without_runtime_variables(tmp_path, monkeypatch):
    monkeypatch.setattr(rx, "RESULTS_ROOT", tmp_path)
    with pytest.raises(SystemExit, match="run-time variables not set"):
        rx.run_block(rx.load_settings(), "mock_calibration", allow_live_llm=False)
    assert not any(tmp_path.iterdir())


def test_run_refuses_when_the_component_is_not_built(tmp_path, monkeypatch):
    monkeypatch.setattr(rx, "RESULTS_ROOT", tmp_path)
    with pytest.raises(SystemExit, match="not available"):
        rx.run_block(rx.load_settings(overrides=RUNTIME_MOCK), "mock_calibration", allow_live_llm=False)


def test_heldout_refuses_before_the_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(rx, "FREEZE_MARKER", tmp_path / "ANALYSIS_FROZEN")
    with pytest.raises(SystemExit, match="not frozen"):
        rx.run_block(rx.load_settings(overrides=RUNTIME_MOCK), "mock_heldout", allow_live_llm=False)


def test_run_writes_rows_and_resumes(tmp_path, monkeypatch):
    calls = []

    def fake_unit(unit, settings):
        calls.append(unit)
        return [{"accuracy": 1.0}]

    monkeypatch.setattr(rx, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(rx, "_component", lambda s, key: fake_unit)
    s = rx.load_settings(overrides=RUNTIME_MOCK + ["mock.topologies=['chain']", "mock.tune_seeds=[9001, 9001]",
                                                   "mock.epidemiology_agents=[12]"])
    rx.run_block(s, "mock_calibration", allow_live_llm=False)
    first = len(calls)
    assert first == 3                                   # B, C, BAZT
    files = list((tmp_path / "sprint" / "mock_calibration" / "units").glob("*.jsonl"))
    assert len(files) == first
    row = json.loads(files[0].read_text().splitlines()[0])
    assert row["accuracy"] == 1.0 and row["unit"]["attack"] == "none"
    manifest = json.loads(next((tmp_path / "sprint" / "mock_calibration").glob("manifest-*.json")).read_text())
    assert manifest["pre_registered"] == {"ni_margin_accuracy": 0.01, "rank_agreement_threshold": 0.8}
    rx.run_block(s, "mock_calibration", allow_live_llm=False)
    assert len(calls) == first                          # resumed: nothing re-run
