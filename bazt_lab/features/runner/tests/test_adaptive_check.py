"""llm_adaptive_check (D110): the LLM attacker replays the gray-box best responses of mock_adaptive as its schedule.

The runner maps each LLM unit onto one mock_adaptive cell (owner keys llm.adaptive_check.mock_cell, .mock_topology,
.start_clock, and .mock_arm.<arm> for an LLM arm the mock grid does not have), refuses the block before anything runs
when the table is missing or lacks a cell, attaches the schedule to every unit, and pins the entries it used."""
import json

import pytest

from bazt_lab.core import public as core
from bazt_lab.features.runner import public as rx
from bazt_lab.features.runner.private import runall
from bazt_lab.features.runner.tests import fakes
from bazt_lab.features.runner.tests.test_commands import LLM, MOCK, blocks_run, run, sets
from bazt_lab.features.runner.tests.test_launch import block_dir, past_kickoff

SCOPE = ["llm.agents=3", "llm.adaptive_check.arms=['A']", "llm.adaptive_check.families=['infection']",
         "llm.adaptive_check.condition='gray_box'", "llm.adaptive_check.tasks=['mmlu']", "mock.adaptive.arms=['A']", "mock.adaptive.cells=[[10, 10]]",
         "mock.adaptive.conditions=['gray_box']"]
REPLAY = ["llm.adaptive_check.mock_cell=[10, 10]", "llm.adaptive_check.mock_topology.hub_star='chain'",
          "llm.adaptive_check.start_clock='conversation'"]
CELL = ("mock", "A", 0.1, 0.1, "chain", "infection", 3, "gray_box")      # the mock_adaptive cell the units map to


def settings(*extra):
    return fakes.settings(fakes.TINY_MOCK, LLM, SCOPE, REPLAY, *extra)


def launch(s, tmp_path, adaptive, llm=None):
    llm = llm or fakes.FakePlatform("llm")
    rx.run_block(s, "llm_adaptive_check", True, wiring=fakes.wiring({"llm": llm}, adaptive=adaptive),
                 results_root=tmp_path)
    return llm


def refused(s, tmp_path, adaptive, match):
    llm = fakes.FakePlatform("llm")
    with pytest.raises(SystemExit, match=match) as e:
        launch(s, tmp_path, adaptive, llm)
    assert llm.probes == [] and llm.batches == [] and not block_dir(tmp_path, "llm_adaptive_check").exists()
    return e.value.code


def units_of(platform):
    return [u for _, us in platform.batches for u in us]


# ================================================================================================== run-all order
def test_run_all_runs_mock_adaptive_before_the_llm_adaptive_check_and_hands_it_the_found_schedule(tmp_path, capsys,
                                                                                                   monkeypatch):
    past_kickoff(monkeypatch)
    s = fakes.settings(MOCK, LLM, SCOPE, REPLAY)
    order, _ = runall.plan(s, tmp_path, True, fakes.wiring())
    assert order.index("mock_adaptive") < order.index("llm_adaptive_check")
    llm, adaptive = fakes.FakePlatform("llm"), fakes.FakeAdaptive()
    argv = ["run-all", "--allow-live-llm", "--results-root", str(tmp_path)] + sets(MOCK, LLM, SCOPE, REPLAY)
    assert run(argv, fakes.wiring({"llm": llm, "mock": fakes.FakePlatform("mock")}, adaptive=adaptive)) == 0
    ran = blocks_run(capsys.readouterr().out)
    assert ran.index("mock_adaptive") < ran.index("llm_adaptive_check")
    checked = [u for u in units_of(llm) if u["block"] == "llm_adaptive_check"]
    assert checked and all(u["attack_schedule"] == {"clock": "conversation", "start_frac": 0.5, "start_round": 2,
                                                    "rate": 1.0} for u in checked)


def test_run_all_skips_the_check_when_mock_adaptive_is_skipped(tmp_path, capsys, monkeypatch):
    past_kickoff(monkeypatch)
    no_mock_scope = [o for o in SCOPE if not o.startswith("mock.adaptive.arms")]
    _, skipped = runall.plan(fakes.settings(MOCK, LLM, no_mock_scope, REPLAY), tmp_path, True, fakes.wiring())
    reason = dict(skipped)["llm_adaptive_check"]
    assert "mock_adaptive" in reason and "mock.adaptive.arms" in reason


# ======================================================================================================= refusals
def test_the_check_is_refused_before_anything_runs_when_mock_adaptive_has_no_best_responses(tmp_path, monkeypatch):
    past_kickoff(monkeypatch)
    code = refused(settings(), tmp_path, fakes.FakeAdaptive(), r"Refusing llm_adaptive_check:.*--block mock_adaptive")
    assert "D110" in code


def test_the_check_is_refused_when_the_table_lacks_a_cell_it_replays(tmp_path, monkeypatch):
    past_kickoff(monkeypatch)
    other = ("mock", "A", 0.1, 0.1, "chain", "insider_pi", 3, "gray_box")
    code = refused(settings(), tmp_path, fakes.FakeAdaptive({other: {}}), r"lack 1 cell.*--block mock_adaptive")
    assert "infection" in code


@pytest.mark.parametrize("key", ["llm.adaptive_check.mock_cell", "llm.adaptive_check.mock_topology",
                                 "llm.adaptive_check.start_clock"])
def test_each_choice_of_the_mapping_is_an_owner_key_refused_while_unset(tmp_path, monkeypatch, key):
    past_kickoff(monkeypatch)
    s = fakes.settings(fakes.TINY_MOCK, LLM, SCOPE, [o for o in REPLAY if not o.startswith(key)])
    refused(s, tmp_path, fakes.FakeAdaptive({CELL: {}}), rf"setting not set in profiles.toml: {key}")


def test_the_tasks_of_the_check_are_an_owner_key_refused_while_unset(tmp_path, monkeypatch):
    past_kickoff(monkeypatch)
    s = fakes.settings(fakes.TINY_MOCK, LLM, [o for o in SCOPE if "adaptive_check.tasks" not in o], REPLAY)
    refused(s, tmp_path, fakes.FakeAdaptive({CELL: {}}), r"setting not set in profiles.toml: llm.adaptive_check.tasks")


def test_the_units_of_the_check_run_the_configured_tasks_not_a_fixed_one():
    s = settings(["llm.adaptive_check.tasks=['mmlu', 'csqa']"])
    units = list(rx.units_llm_adaptive_check(s, True))
    assert {u["task"] for u in units} == {"mmlu", "csqa"}
    assert len(units) == 2 * len([u for u in units if u["task"] == "mmlu"])
    only = list(rx.units_llm_adaptive_check(settings(["llm.adaptive_check.tasks=['gsm8k']"]), True))
    assert {u["task"] for u in only} == {"gsm8k"}


@pytest.mark.parametrize("bad,match", [
    ("llm.adaptive_check.mock_cell=[2, 2]", r"mock_cell.*mock.adaptive.cells"),
    ("llm.adaptive_check.mock_topology.hub_star='small_world'", r"mock_topology.hub_star.*mock.topologies"),
    ("llm.adaptive_check.start_clock='week'", r"start_clock must be one of"),
    ("llm.adaptive_check.mock_cell=[10]", r"mock_cell must be"),
])
def test_a_mapping_outside_the_mock_grid_is_refused(tmp_path, monkeypatch, bad, match):
    past_kickoff(monkeypatch)
    refused(settings([bad]), tmp_path, fakes.FakeAdaptive({CELL: {}}), match)


def test_an_llm_arm_the_mock_grid_lacks_needs_its_mock_arm_named(tmp_path, monkeypatch):
    past_kickoff(monkeypatch)
    s = settings(["llm.adaptive_check.arms=['B']"])
    refused(s, tmp_path, fakes.FakeAdaptive({CELL: {}}), r"setting not set in profiles.toml: llm.adaptive_check.mock_arm.B")
    s = settings(["llm.adaptive_check.arms=['B']", "llm.adaptive_check.mock_arm.B='A'"])
    rx.run_block(s, "llm_calibration", True, wiring=fakes.wiring({"llm": fakes.FakePlatform("llm")}),
                 results_root=tmp_path)                            # B is calibrated: its thresholds come first
    llm = launch(s, tmp_path, fakes.FakeAdaptive({CELL: {}}))
    assert {u["arm"] for u in units_of(llm)} == {"B"}


def test_an_llm_agent_count_the_mock_grid_lacks_is_refused(tmp_path, monkeypatch):
    past_kickoff(monkeypatch)
    refused(settings(["llm.agents=4"]), tmp_path, fakes.FakeAdaptive({CELL: {}}), r"mock.frontier_agents")


# ===================================================================================================== the schedule
@pytest.mark.parametrize("clock,start", [("conversation", 2), ("seed_run", 10)])
def test_every_unit_and_probe_carries_the_replayed_start_and_rate_and_nothing_else(tmp_path, monkeypatch, clock, start):
    past_kickoff(monkeypatch)
    table = {CELL: {"entry_ranks": [1], "start_frac": 0.5, "rate": 0.25, "detectability": 0.5, "potency": 0.75}}
    adaptive = fakes.FakeAdaptive(table)
    llm = launch(settings([f"llm.adaptive_check.start_clock='{clock}'"]), tmp_path, adaptive)
    units = units_of(llm)
    assert units and all(u["attack_schedule"] == {"clock": clock, "start_frac": 0.5, "start_round": start,
                                                  "rate": 0.25} for u in units)
    # 4 rounds per conversation, 10 conversations over 2 seeds: the seed-run clock has 20 rounds; a probe has one
    probe = [u for _, us, _ in llm.probes for u in us]
    assert probe and all(u["attack_schedule"]["start_round"] == 2 for u in probe)
    assert adaptive.loads[0][0] == tmp_path / "sprint" / "mock_adaptive"
    assert all(rx.unit_id(u) == u["id"] for u in units)          # the schedule is an annotation, not the unit's identity


def test_the_entries_used_are_pinned_and_a_changed_table_is_refused(tmp_path, monkeypatch):
    past_kickoff(monkeypatch)
    s = settings()
    launch(s, tmp_path, fakes.FakeAdaptive({CELL: {"start_frac": 0.5}}))
    pin = json.loads((block_dir(tmp_path, "llm_adaptive_check") / "best_responses.json").read_text(encoding="utf-8"))
    assert list(pin["entries"].values()) == [{"start_frac": 0.5, "rate": 1.0}]
    again = launch(s, tmp_path, fakes.FakeAdaptive({CELL: {"start_frac": 0.5, "entry_ranks": [2]}}))
    assert again.batches == []                                     # same schedule: every unit already done, nothing refused
    llm = fakes.FakePlatform("llm")
    with pytest.raises(SystemExit, match=r"Refusing llm_adaptive_check:.*best_responses.json.*start_frac"):
        launch(s, tmp_path, fakes.FakeAdaptive({CELL: {"start_frac": 0.25}}), llm)
    assert llm.probes == [] and llm.batches == []


# ===================================================================================== what the platform cannot run
def test_a_condition_the_llm_platform_does_not_run_is_said_before_anything_runs(tmp_path, monkeypatch):
    past_kickoff(monkeypatch)
    s = settings(["llm.adaptive_check.condition='white_box'"])
    reason = rx.BLOCKS["llm_adaptive_check"]
    from bazt_lab.features.runner.private import blocks
    why = blocks.unrunnable(reason, s, fakes.wiring())
    assert "runs the adaptive conditions 'gray_box' only" in why and "'white_box'" in why and "D110" in why
    with pytest.raises(SystemExit, match="white_box"):
        launch(s, tmp_path, fakes.FakeAdaptive({CELL: {}}))
    assert core.is_set(s, "llm.adaptive_check.condition")
