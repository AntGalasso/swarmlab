"""Block registry and unit expansion: same units as the old launcher, counted arithmetically, grouped contiguously."""
import itertools
import json

import pytest

from bazt_lab.core import public as core
from bazt_lab.features.runner import public as rx
from bazt_lab.features.runner.private import units as units_mod
from bazt_lab.features.runner.tests import fakes

EVERY_BLOCK = [fakes.TINY_MOCK, fakes.TINY_LLM, fakes.SCOPED]


def small():
    return fakes.settings(*EVERY_BLOCK)


def test_the_registry_lists_every_block_of_the_spec():
    assert list(rx.BLOCKS) == [
        "llm_calibration", "gnn_train", "llm_headline", "llm_capped", "mock_calibration",
        "mock_frontier", "mock_epidemiology", "mock_ablations", "mock_adaptive", "llm_adaptive_check",
        "mock_presets_robustness", "mock_costs_robustness", "mock_legacy", "mock_heldout"]
    # D110: the check replays mock_adaptive's best responses, so the registry (run-all's order inside a kind) has it after
    assert list(rx.BLOCKS).index("mock_adaptive") < list(rx.BLOCKS).index("llm_adaptive_check")
    assert {b.kind for b in rx.BLOCKS.values()} == {"calibrate", "train", "evaluate"}
    assert [n for n, b in rx.BLOCKS.items() if b.kind == "calibrate"] == ["llm_calibration", "mock_calibration"]
    assert [n for n, b in rx.BLOCKS.items() if b.requires_freeze] == ["mock_heldout"]


@pytest.mark.parametrize("name", list(rx.BLOCKS))
def test_count_is_arithmetic_and_equals_the_expansion(name):
    grid = rx.BLOCKS[name].units(small(), False)
    expanded = list(grid)
    assert len(grid) == len(expanded) > 0
    assert sum(grid.by_agents().values()) == len(expanded)
    assert grid.arms() == {u["arm"] for u in expanded}
    assert {rx.cell_key(u) for u in expanded} == grid.cells()


@pytest.mark.parametrize("name", list(rx.BLOCKS))
def test_unit_ids_are_distinct_and_stable(name):
    expanded = list(rx.BLOCKS[name].units(small(), False))
    ids = [rx.unit_id(u) for u in expanded]
    assert len(set(ids)) == len(ids)
    assert ids == [rx.unit_id(json.loads(json.dumps(u))) for u in expanded]


@pytest.mark.parametrize("name", [n for n, b in rx.BLOCKS.items() if b.platform == "mock"])
def test_mock_units_of_one_batch_group_are_consecutive(name):
    """A mock batch holds up to mock.batch_size units of one group, so a group must not be split by another one.
    An LLM batch is one unit, so the order of LLM units across groups does not matter."""
    expanded = list(rx.BLOCKS[name].units(small(), False))
    keys = [rx.group_key(u) for u in expanded]
    assert len([k for k, _ in itertools.groupby(keys)]) == len(set(keys))


def test_the_real_frontier_is_counted_without_expanding_it():
    s = rx.load_settings()
    grid = rx.units_mock_frontier(s, False)
    arms_cells = len(core.get(s, "mock.uncapped_arms")) + (
        len(core.get(s, "mock.capped_arms")) + len(core.get(s, "mock.allocation_arms"))) * 48
    assert len(grid) == arms_cells * 5 * 6 * 30 * 12            # topologies, families + none, seeds, sinks
    assert grid.by_agents() == {12: len(grid)}
    assert len(rx.units_mock_heldout(s, False)) == len(grid)


def test_agents_without_every_sink_take_no_sink_role():
    s = fakes.settings(fakes.TINY_MOCK, fakes.ONLY_A, ["mock.every_sink=false"])
    grid = rx.units_mock_frontier(s, False)
    assert len(grid) == 1 * 1 * 2 * 2 and {u["sink"] for u in grid} == {None}


def test_duplicate_agent_counts_still_count_what_they_expand_to():
    s = fakes.settings(fakes.TINY_MOCK, fakes.ONLY_A, ["mock.frontier_agents=[3, 3, 4]"])
    grid = rx.units_mock_frontier(s, False)
    assert len(grid) == len(list(grid)) and grid.by_agents() == {3: 2 * 2 * 2 * 3, 4: 2 * 2 * 4}


def test_an_empty_axis_is_an_empty_block():
    s = fakes.settings(fakes.TINY_MOCK, ["mock.uncapped_arms=[]", "mock.capped_arms=[]", "mock.allocation_arms=[]"])
    grid = rx.units_mock_frontier(s, False)
    assert len(grid) == 0 and list(grid) == []


def test_mock_unit_fields_are_those_of_the_old_launcher():
    s = fakes.settings(fakes.TINY_MOCK, fakes.ONLY_A)
    unit = next(iter(rx.units_mock_frontier(s, False)))
    assert unit == dict(block="mock_frontier", platform="mock", arm="A", audit="uncapped", intervention="uncapped",
                        topology="chain", attack="none", agents=3, seed=1, sink=0)


def test_variant_blocks_add_their_field_and_the_cost_block_names_its_cost_condition():
    s = small()
    adaptive = list(rx.BLOCKS["mock_adaptive"].units(s, False))
    assert {u["condition"] for u in adaptive} == set(core.get(s, "mock.adaptive.conditions"))
    presets = list(rx.BLOCKS["mock_presets_robustness"].units(s, False))
    assert {u["preset"] for u in presets} == set(core.get(s, "mock.presets_robustness.presets"))
    costs = list(rx.BLOCKS["mock_costs_robustness"].units(s, False))
    assert {u["cost_condition"] for u in costs} == {"robustness"}
    assert all("cost_condition" not in u for u in rx.BLOCKS["mock_frontier"].units(s, False))


def test_llm_units_keep_the_old_fields_and_split_across_seeds():
    s = fakes.settings(fakes.TINY_LLM, fakes.RUNTIME_LLM)
    units = rx.units_llm_headline(s, True)
    assert len(units) == 2 * 1 * 2 * 1 * (1 + 1) * 2          # arms, cells, backbones, topologies, attacks, seeds
    assert units[0] == dict(block="llm_headline", platform="llm", arm="A", audit="uncapped", intervention="uncapped",
                            backbone="bb1", topology="hub_star", attack="insider_pi", task="mmlu", seed=1,
                            conversations=5, agents=12, rounds=4, sanitize_mode="redaction")


def test_gnn_train_units_carry_the_training_setting():
    s = fakes.settings(["gnn_train.backbone='bb1'"])
    units = rx.units_gnn_train(s, True)
    assert len(units) == 5 * 5 * 4 and {u["arm"] for u in units} == {"GNN"}
    assert {u["conversations"] for u in units} == {40} and {u["agents"] for u in units} == {8}


def test_a_unit_id_is_wide_enough_that_no_two_units_of_the_frontier_share_one():
    """Two units with one id would share a rows file, and the second would be skipped as done. The birthday bound
    n^2 / 2^(bits + 1) over the whole block (stricter than per arm, since the arm is part of the id) must be tiny."""
    grid = rx.units_mock_frontier(rx.load_settings(), False)
    digest = rx.unit_id(next(iter(grid))).rsplit("-", 1)[1]
    bits = 4 * len(digest)
    assert len(grid) ** 2 / 2 ** (bits + 1) < 1e-6


def test_unit_id_ignores_what_the_runner_adds_to_a_unit():
    unit = next(iter(rx.units_mock_frontier(fakes.settings(fakes.TINY_MOCK, fakes.ONLY_A), False)))
    uid = rx.unit_id(unit)
    assert rx.unit_id({**unit, "id": uid, "thresholds": {"x": 1.0}}) == uid
    assert uid.startswith("A-")


def test_cell_key_of_a_mock_and_an_llm_unit():
    assert rx.cell_key({"arm": "B", "topology": "chain", "agents": 12}) == ("B", "chain", 12, None, None)
    assert rx.cell_key({"arm": "B", "topology": "chain", "agents": 12, "backbone": "m", "task": "mmlu"}) == (
        "B", "chain", 12, "m", "mmlu")


def test_group_key_names_the_fields_a_batch_shares():
    unit = {"arm": "B", "audit": 0.1, "intervention": "uncapped", "topology": "chain", "attack": "none",
            "agents": 12, "seed": 3, "sink": 4, "condition": "gray_box"}
    assert rx.group_key(unit) == ("B", 0.1, "uncapped", "chain", "none", 12, None, None, "gray_box", None)
    adaptive = {**unit, "attack_params": {"auditor_tpr": 0.05}}
    assert rx.group_key(adaptive)[-1] == '{"auditor_tpr": 0.05}' and rx.group_key(adaptive) != rx.group_key(unit)
    assert units_mod.GROUP_FIELDS == ("arm", "audit", "intervention", "topology", "attack", "agents", "cost_condition",
                                      "preset", "condition", "attack_params")


def test_sanitizing_arms_get_one_unit_per_sanitize_mode_and_the_others_the_first_mode():
    s = fakes.settings(fakes.TINY_LLM, fakes.RUNTIME_LLM, ["llm.sanitize_modes=['redaction', 'llm_rewrite']",
                                                           "llm.headline_arms=['A', 'BAZT', 'BAZT_no_budget', 'BAZT_upstream']"])
    units = rx.units_llm_headline(s, True)
    per_arm = 2 * 1 * (1 + 1) * 2                               # backbones, topologies, attacks, seeds
    assert len(units) == per_arm * (1 + 2 + 2 + 2)
    modes = {arm: {u["sanitize_mode"] for u in units if u["arm"] == arm} for arm in {u["arm"] for u in units}}
    assert modes == {"A": {"redaction"}, "BAZT": {"redaction", "llm_rewrite"},
                     "BAZT_no_budget": {"redaction", "llm_rewrite"}, "BAZT_upstream": {"redaction", "llm_rewrite"}}
    assert len({rx.unit_id(u) for u in units}) == len(units)


def test_calibration_and_training_units_carry_the_first_sanitize_mode():
    """The LLM probe and the platform read sanitize_mode on every LLM unit."""
    s = fakes.settings(fakes.TINY_LLM, fakes.RUNTIME_LLM, ["gnn_train.backbone='bb1'",
                                                           "llm.sanitize_modes=['redaction', 'llm_rewrite']"])
    assert {u["sanitize_mode"] for u in rx.units_llm_calibration(s, True)} == {"redaction"}
    assert {u["sanitize_mode"] for u in rx.units_gnn_train(s, True)} == {"redaction"}


def test_the_profiles_select_the_sanitize_and_memory_attack_modes():
    sprint, research = rx.load_settings(profile="sprint"), rx.load_settings(profile="research")
    assert core.get(sprint, "llm.sanitize_modes") == ["redaction"]
    assert core.get(sprint, "llm.ma_correctness") == ["rule"]
    assert core.get(research, "llm.sanitize_modes") == ["redaction", "llm_rewrite"]
    assert core.get(research, "llm.ma_correctness") == ["rule", "llm_judge"]
    assert core.get(sprint, "llm.ma_judge_model") == "openai/gpt-4o-mini"      # vendor/G-safeguard/MA/evaluate_output.py:31


def test_the_profile_lists_the_arms_the_decisions_calibrate():
    s = rx.load_settings()
    assert "GNN_recalibrated" in core.get(s, "llm.headline_arms")
    assert set(core.get(s, "llm.calibrated_arms")) == {"B", "C", "BAZT", "BAZT_upstream", "GNN_recalibrated"}
    assert {"B", "C", "BAZT", "BAZT_upstream", "GNN_standin", "BAZT_no_budget", "BAZT_no_topology",
            "BAZT_no_graduated"} <= set(core.get(s, "mock.calibrated_arms"))


def test_superseded_keys_are_gone_and_the_owner_placeholders_are_unset():
    s = rx.load_settings()
    for key in ("gnn.recalibrate", "gnn.checkpoint_family", "bazt_upstream.audit_verdict_threshold",
                "calibration.mode", "gnn.fallback_family", "analysis.topology_classes.mock",
                "analysis.topology_classes.llm"):
        assert not core.is_set(s, key), key


def test_the_ablation_block_always_includes_its_reference_arm():
    s = fakes.settings(fakes.TINY_MOCK, fakes.SCOPED)
    arms = {u["arm"] for u in rx.BLOCKS["mock_ablations"].units(s, False)}
    assert arms == {"BAZT", "BAZT_no_budget"}
    listed = fakes.settings(fakes.TINY_MOCK, fakes.SCOPED, ["mock.ablations.arms=['BAZT_no_budget', 'BAZT']"])
    grid = rx.BLOCKS["mock_ablations"].units(listed, False)
    assert len(grid) == len(rx.BLOCKS["mock_ablations"].units(s, False))
