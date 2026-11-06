"""Offline tests for the experiment launcher: profiles, overrides, unit grids, refusals, and the CLI end to end over
fake platforms. No live component is called."""
import json
from types import SimpleNamespace

import pytest

import run_experiments
from bazt_lab import app
from bazt_lab.core import public as core
from bazt_lab.features.runner import public as rx
from bazt_lab.features.runner.private import sizing
from bazt_lab.features.runner.private import units as units_mod
from bazt_lab.features.runner.tests import fakes

RUNTIME_MOCK = ["runtime.n_insiders=3", "runtime.rate_limit_per_identity=1", "runtime.horizon_constant=3",
                "runtime.eps_correct=1.0", "runtime.recovery_margin=0.01", "runtime.ba_m=2", "runtime.ws_k=4",
                "runtime.ws_rewire_p=0.25", "runtime.ni_margin_accuracy=0.01",
                "runtime.rank_agreement_threshold=0.8"] + fakes.NEEDED_MOCK   # test values, not experiment settings


def wired(platform=None):
    return fakes.wiring({"mock": platform or fakes.FakePlatform("mock")})


def test_profiles_differ_only_in_the_llm_grid():
    sprint, research = rx.load_settings(profile="sprint"), rx.load_settings(profile="research")
    assert core.get(sprint, "llm.conversations_per_cell") == 60
    assert core.get(sprint, "llm.capped_cells") == [[10, 10]]
    assert core.get(research, "llm.conversations_per_cell") == 100
    assert core.get(research, "llm.capped_cells") == [[2, 2], [10, 10], [50, 50]]
    def strip(s):
        return {k: v for k, v in s.items() if k not in ("llm", "profile")}
    assert strip(sprint) == strip(research)


def test_active_profile_is_sprint():
    assert rx.load_settings()["profile"] == "sprint"


def test_set_overrides_one_value_with_toml_parsing():
    s = rx.load_settings(overrides=["llm.conversations_per_cell=80", "runtime.llm_seeds=[1, 2, 3, 4, 5]"])
    assert core.get(s, "llm.conversations_per_cell") == 80
    assert core.get(s, "runtime.llm_seeds") == [1, 2, 3, 4, 5]


def test_unknown_profile_is_refused():
    with pytest.raises(SystemExit):
        rx.load_settings(profile="nope")


def test_a_malformed_override_is_refused():
    with pytest.raises(SystemExit, match="key=value"):
        rx.load_settings(overrides=["llm.conversations_per_cell"])


def test_runtime_variables_have_no_value_in_the_settings():
    s = rx.load_settings()
    for platform in ("llm", "mock"):
        assert set(rx.PRE_REGISTERED) <= set(rx.missing_runtime(s, platform))


def test_llm_grid_sizes_follow_the_profile():
    sprint, research = rx.load_settings(profile="sprint"), rx.load_settings(profile="research")
    # 640 units per arm and cell: 2 backbones x 4 topologies x (11 attack + 5 attack-free cells) x 5 seeds
    assert len(rx.units_llm_headline(sprint, False)) == 4480       # 7 arms, one sanitize mode
    assert len(rx.units_llm_capped(sprint, False)) == 1920         # 3 arms x 1 cell
    assert len(rx.units_llm_capped(research, False)) == 9600       # 3 cells x (C + BAZT x 2 modes + BAZT_upstream x 2)
    assert len(rx.units_llm_headline(research, False)) == 5760     # 5 arms + BAZT and BAZT_upstream x 2 modes
    assert {u["conversations"] for u in rx.units_llm_headline(sprint, False)} == {12}
    assert {u["conversations"] for u in rx.units_llm_headline(research, False)} == {20}
    assert {(u["audit"], u["intervention"]) for u in rx.units_llm_capped(sprint, False)} == {(0.10, 0.10)}


def test_strict_units_refuse_placeholders():
    s = rx.load_settings()
    with pytest.raises(core.Unset):
        rx.units_llm_headline(s, True)


def test_conversations_must_split_evenly_across_seeds():
    s = rx.load_settings(overrides=["llm.conversations_per_cell=61"])
    with pytest.raises(SystemExit):
        rx.units_llm_headline(s, False)


def test_capped_cells_cross_both_budgets_without_the_uncapped_corner():
    cells = rx.capped_cells([1, 2, 5, 10, 20, 50])
    assert len(cells) == 7 * 7 - 1
    assert ("uncapped", "uncapped") not in cells
    assert (0.01, "uncapped") in cells and ("uncapped", 0.5) in cells


def test_mock_frontier_runs_every_sink():
    s = rx.load_settings(overrides=["mock.topologies=['chain']", "mock.families=['infection']",
                                    "mock.main_seeds=[1, 1]"])
    units = list(rx.units_mock_frontier(s, False))
    assert len(units) == 342 * 2 * 12                  # 6 + (4 capped + 3 allocation) x 48 cells; 2 families; 12 sinks
    assert {u["sink"] for u in units} == set(range(12))


def test_unscoped_mock_blocks_refuse_until_arms_and_cells_are_set():
    s = rx.load_settings()
    with pytest.raises(core.Unset):
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


def test_run_refuses_without_runtime_variables(tmp_path):
    with pytest.raises(SystemExit, match="run-time variables not set"):
        rx.run_block(rx.load_settings(), "mock_calibration", allow_live_llm=False, wiring=wired(),
                     results_root=tmp_path)
    assert not any(tmp_path.iterdir())


def test_run_refuses_when_the_component_is_not_built(tmp_path):
    with pytest.raises(SystemExit, match="not available"):
        rx.run_block(rx.load_settings(overrides=RUNTIME_MOCK), "mock_calibration", allow_live_llm=False,
                     wiring=fakes.wiring({}), results_root=tmp_path)


def test_heldout_refuses_before_the_freeze(tmp_path):
    with pytest.raises(SystemExit, match="not frozen"):
        rx.run_block(rx.load_settings(overrides=RUNTIME_MOCK), "mock_heldout", allow_live_llm=False, wiring=wired(),
                     results_root=tmp_path)


def test_run_writes_rows_and_resumes(tmp_path):
    platform = fakes.FakePlatform("mock")
    s = rx.load_settings(overrides=RUNTIME_MOCK + ["mock.topologies=['chain']", "mock.tune_seeds=[9001, 9001]",
                                                   "mock.epidemiology_agents=[12]"])
    rx.run_block(s, "mock_calibration", allow_live_llm=False, wiring=wired(platform), results_root=tmp_path)
    first = len(platform.calibrations)
    assert first == len(core.get(s, "mock.calibrated_arms"))      # one cell per calibrated arm
    files = list((tmp_path / "sprint" / "mock_calibration" / "units").glob("*.jsonl"))
    assert len(files) == first
    row = json.loads(files[0].read_text().splitlines()[0])
    assert row["accuracy"] == 1.0 and row["unit"]["attack"] == "none"
    manifest = json.loads(next((tmp_path / "sprint" / "mock_calibration").glob("manifest-*.json")).read_text())
    assert manifest["pre_registered"] == {"ni_margin_accuracy": 0.01, "rank_agreement_threshold": 0.8}
    rx.run_block(s, "mock_calibration", allow_live_llm=False, wiring=wired(platform), results_root=tmp_path)
    assert len(platform.calibrations) == first          # resumed: nothing re-run


# ======================================================================================================== the CLI
def cli(argv, wiring=None):
    return rx.main(argv, wiring=wiring or wired())


def test_list_counts_blocks_without_expanding_them(monkeypatch, capsys):
    def no_expansion(self):
        raise AssertionError("--list must count arithmetically")
    monkeypatch.setattr(units_mod.MockGrid, "__iter__", no_expansion)
    assert cli(["--list"]) == 0
    out = capsys.readouterr().out
    assert "profile: sprint" in out
    for name in rx.BLOCKS:
        assert name in out
    frontier = next(line for line in out.splitlines() if line.startswith("mock_frontier"))
    assert f"{len(rx.units_mock_frontier(rx.load_settings(), False)):,}" in frontier


def test_list_says_what_each_block_still_needs(capsys):
    cli(["--list"], wiring=fakes.wiring({"mock": fakes.FakePlatform("mock")}, arms={"A": fakes.ARMS["A"]}))
    lines = {line.split()[0]: line for line in capsys.readouterr().out.splitlines()[3:]}
    assert "set mock.ablations.arms" in lines["mock_ablations"]
    assert "run-time variables" in lines["mock_frontier"]
    assert "not built: conventional" in lines["mock_frontier"]            # B, C, ... are missing from the registry
    assert "not built" in lines["llm_headline"] and "llm_harness" in lines["llm_headline"]


def test_dry_run_sizes_every_block(capsys):
    assert cli(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "LLM calls:" in out and "with the request cache" in out
    frontier = next(line for line in out.splitlines() if line.startswith("mock_frontier"))
    settings = rx.load_settings()
    grid = rx.units_mock_frontier(settings, False)
    n = len(grid)
    io = core.get(settings, "estimates.mock_checkpoint_seconds_per_unit")
    cpu = sum(c * (sizing.seconds_per_episode(settings, a, kind, size) + io)
              for (a, kind, size), c in grid.cost_groups().items()) / 3600
    assert f"{n:,}" in frontier and f"{cpu:.2f}" in frontier
    assert "scope not set: mock.ablations.arms" in next(ln for ln in out.splitlines() if ln.startswith("mock_ablations"))


def test_dry_run_follows_the_profile(capsys):
    cli(["--dry-run", "--block", "llm_capped"])
    sprint = capsys.readouterr().out
    cli(["--profile", "research", "--dry-run", "--block", "llm_capped"])
    assert capsys.readouterr().out != sprint


def test_run_needs_a_block():
    with pytest.raises(SystemExit, match="at least one --block"):
        cli(["--run"])


def test_exactly_one_mode_is_required():
    with pytest.raises(SystemExit):
        cli([])
    with pytest.raises(SystemExit):
        cli(["--list", "--dry-run"])


def test_run_through_the_cli_writes_into_the_results_root(tmp_path, capsys):
    platform = fakes.FakePlatform("mock")
    argv = ["--run", "--block", "mock_frontier", "--results-root", str(tmp_path)]
    argv += [x for o in fakes.RUNTIME_MOCK + fakes.TINY_MOCK + fakes.ONLY_A for x in ("--set", o)]
    assert cli(argv, wiring=wired(platform)) == 0
    assert "mock_frontier: 12 units run, 0 already done" in capsys.readouterr().out
    assert len(list((tmp_path / "sprint" / "mock_frontier" / "units").glob("*.done"))) == 12


def test_calibrate_then_evaluate_through_the_cli(tmp_path):
    platform = fakes.FakePlatform("mock")
    sets = [x for o in fakes.RUNTIME_MOCK + fakes.TINY_MOCK + fakes.CALIBRATED_B for x in ("--set", o)]
    root = ["--results-root", str(tmp_path)]
    with pytest.raises(SystemExit, match="no calibrated thresholds"):
        cli(["--run", "--block", "mock_frontier"] + root + sets, wiring=wired(platform))
    assert cli(["--run", "--block", "mock_calibration", "--block", "mock_frontier"] + root + sets,
               wiring=wired(platform)) == 0
    expected = {"deny_score": fakes.threshold_of(("B", "chain", 3, None, None))}
    assert {json.dumps(u["thresholds"]) for _, units in platform.batches for u in units if u["arm"] == "B"} == {
        json.dumps(expected)}


# ================================================================================ composition root and entry point
def test_the_entry_point_is_the_composition_roots_main():
    assert run_experiments.main is app.main


def test_a_feature_that_is_not_there_is_named_not_an_import_error():
    with pytest.raises(rx.FeatureMissing, match="feature definitely_not_built is not built") as e:
        app._feature("definitely_not_built")
    assert e.value.feature == "definitely_not_built"


def fake_features(monkeypatch, **features):
    def load(name):
        if name in features:
            return features[name]
        raise rx.FeatureMissing(name, f"No module named bazt_lab.features.{name}")
    monkeypatch.setattr(app, "_feature", load)


def test_wiring_builds_the_arm_registry_from_the_features_arms_tuples(monkeypatch):
    fake_features(monkeypatch, conventional=SimpleNamespace(ARMS=(fakes.ARMS["A"], fakes.ARMS["B"])),
                  references=SimpleNamespace(ARMS=(fakes.ARMS["random_alloc"],)))
    wiring = app.build_wiring()
    assert wiring.arm("A") is fakes.ARMS["A"] and wiring.arm("random_alloc") is fakes.ARMS["random_alloc"]
    with pytest.raises(rx.FeatureMissing) as e:
        wiring.arm("BAZT")
    assert "bazt_slot" in str(e.value) and "gnn" in str(e.value) and "'BAZT'" in str(e.value)


def test_wiring_rejects_an_arm_exported_twice(monkeypatch):
    both = SimpleNamespace(ARMS=(fakes.ARMS["A"],))
    fake_features(monkeypatch, conventional=both, references=both, bazt_slot=both, gnn=both)
    with pytest.raises(ValueError, match=r"'A'.*conventional.*references"):
        app.build_wiring().arm("A")


def test_wiring_reports_an_unknown_arm_when_every_feature_is_built(monkeypatch):
    empty = SimpleNamespace(ARMS=())
    fake_features(monkeypatch, conventional=empty, references=empty, bazt_slot=empty, gnn=empty)
    with pytest.raises(LookupError, match="no feature exports an arm named 'nope'"):
        app.build_wiring().arm("nope")


def test_wiring_builds_the_platforms_and_the_tracker_lazily(monkeypatch):
    built = []

    class Mock(fakes.FakePlatform):
        def __init__(self):
            super().__init__("mock")
            built.append("mock")

    fake_features(monkeypatch, mock_engine=SimpleNamespace(MockPlatform=Mock),
                  tracking=SimpleNamespace(WandbTracker=fakes.RecordingTracker))
    wiring = app.build_wiring()
    assert built == []
    assert isinstance(wiring.platform("mock"), Mock) and built == ["mock"]
    assert wiring.platform("mock") is wiring.platform("mock") and built == ["mock"]
    with pytest.raises(rx.FeatureMissing, match="llm_harness"):
        wiring.platform("llm")
    tracker = wiring.tracker({"tracking": {"mode": "offline"}})
    assert isinstance(tracker, fakes.RecordingTracker) and tracker.settings == {"tracking": {"mode": "offline"}}


def test_main_lists_with_every_feature_missing(monkeypatch, capsys):
    fake_features(monkeypatch)
    assert app.main(["--list"]) == 0
    assert "not built" in capsys.readouterr().out


def test_main_runs_a_block_over_the_composed_features(monkeypatch, tmp_path, capsys):
    fake_features(monkeypatch, conventional=SimpleNamespace(ARMS=(fakes.ARMS["A"],)),
                  mock_engine=SimpleNamespace(MockPlatform=lambda: fakes.FakePlatform("mock")))
    argv = ["--run", "--block", "mock_frontier", "--results-root", str(tmp_path)]
    argv += [x for o in fakes.RUNTIME_MOCK + fakes.TINY_MOCK + fakes.ONLY_A for x in ("--set", o)]
    assert app.main(argv) == 0
    assert "mock_frontier: 12 units run" in capsys.readouterr().out


# ========================================================================= crash, resume and LLM through the CLI
def cli_argv(*mode, groups, results_root):
    argv = [*mode, "--results-root", str(results_root)]
    return argv + [x for group in groups for o in group for x in ("--set", o)]


def test_a_crashed_cli_run_is_resumed_by_the_next_cli_run(tmp_path, capsys):
    argv = cli_argv("--run", "--block", "mock_frontier", groups=[fakes.RUNTIME_MOCK, fakes.TINY_MOCK, fakes.ONLY_A],
                    results_root=tmp_path)
    crashing = fakes.FakePlatform("mock", rows_per_unit=2, crash_on=lambda u: u["seed"] == 2 and u["sink"] == 1)
    with pytest.raises(RuntimeError, match="simulated crash"):
        cli(argv, wiring=wired(crashing))
    done = tmp_path / "sprint" / "mock_frontier" / "units"
    assert len(list(done.glob("*.done"))) == 4          # the first batch of 4 finished; the crash hit the 5th unit
    resumed = fakes.FakePlatform("mock", rows_per_unit=2)
    assert cli(argv, wiring=wired(resumed)) == 0
    assert "mock_frontier: 8 units run, 4 already done" in capsys.readouterr().out
    assert len(list(done.glob("*.done"))) == 12
    assert {len(f.read_text().splitlines()) for f in done.glob("*.jsonl")} == {2}   # no row lost, none repeated


def test_llm_blocks_run_through_the_cli_only_with_the_live_flag_after_kickoff(tmp_path, monkeypatch):
    monkeypatch.setenv("SWARMLAB_KICKOFF_UTC", "2020-01-01T00:00:00+00:00")
    platform = fakes.FakePlatform("llm")
    wiring = fakes.wiring({"llm": platform})
    base = dict(groups=[fakes.RUNTIME_LLM, fakes.TINY_LLM], results_root=tmp_path)
    with pytest.raises(SystemExit, match="allow-live-llm"):
        cli(cli_argv("--run", "--block", "llm_calibration", **base), wiring=wiring)
    assert platform.calibrations == []
    argv = cli_argv("--run", "--block", "llm_calibration", "--block", "llm_headline", "--allow-live-llm", **base)
    assert cli(argv, wiring=wiring) == 0
    assert len(platform.calibrations) == 2              # B on 2 backbones, task mmlu, one cell each
    sent = [u for _, units in platform.batches for u in units]
    assert len(sent) == 16 and {len(units) for _, units in platform.batches} == {1}
    assert all(u["thresholds"] == {"deny_score": fakes.threshold_of(rx.cell_key(u))} for u in sent if u["arm"] == "B")
    assert all(u["thresholds"] == {} for u in sent if u["arm"] == "A")


def test_wiring_hands_the_runner_the_adaptive_feature_lazily(monkeypatch):
    adaptive = SimpleNamespace(run_search=None)
    fake_features(monkeypatch, adaptive=adaptive)
    assert app.build_wiring().adaptive() is adaptive
    fake_features(monkeypatch)
    with pytest.raises(rx.FeatureMissing, match="feature adaptive is not built"):
        app.build_wiring().adaptive()


def test_a_worker_wiring_imports_only_the_feature_that_holds_the_arm(monkeypatch):
    """A spawned worker needs one arm: importing every arm feature (gnn, torch_geometric, bazt) cost it about 20 s."""
    loaded = []
    empty = SimpleNamespace(ARMS=())

    def load(name):
        loaded.append(name)
        return SimpleNamespace(ARMS=(fakes.ARMS["A"],)) if name == "conventional" else empty
    monkeypatch.setattr(app, "_feature", load)
    wiring = app.build_worker_wiring()
    assert wiring.arm("A") is fakes.ARMS["A"] and loaded == ["conventional"]
    assert app.build_wiring().rebuild == "bazt_lab.app:build_worker_wiring"
    assert wiring.rebuild == "bazt_lab.app:build_worker_wiring"
    with pytest.raises(LookupError, match="no feature exports an arm named 'nope'"):
        wiring.arm("nope")
