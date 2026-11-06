"""The best-response table: keys, granularity, lookup refusals, atomic per-cell files, settings fingerprint."""
import json
import os

import pytest

from bazt_lab.features.adaptive.private import table as tb
from bazt_lab.features.adaptive.tests.fakes import unit

CELL = dict(platform="mock", arm="C", audit=0.1, intervention=0.2, topology="small_world", attack="infection",
            agents=6, condition="gray_box")
PARAMS = {"entry_ranks": [2], "start_frac": 0.2}
INPUTS = "thresholds-sha-1"


def choice(params=PARAMS, asr=0.8, baseline=0.3, candidates=17, episodes=36):
    return tb.Choice(attack_params=params, asr=asr, baseline_asr=baseline, candidates=candidates, episodes=episodes)


def eu(**over):
    return unit(**{**CELL, **over})


# ------------------------------------------------------------------------------------------------- identity
def test_cell_of_drops_the_episode_fields_and_the_annotations():
    u = dict(eu(), id="C-abc", thresholds={"x": 1.0}, attack_params={"entry_ranks": [0]})
    assert tb.cell_of(u) == CELL


def test_the_search_block_and_the_evaluation_block_share_a_cell():
    assert tb.cell_of(eu(block="mock_adaptive_search")) == tb.cell_of(eu(block="mock_adaptive"))


# ------------------------------------------------------------------------------------------------- lookup
def test_a_cell_level_entry_serves_every_seed_and_sink_of_the_cell():
    t = tb.BestResponses()
    t.add(CELL, None, None, choice())
    for seed, sink in [(1, 0), (7, 5), (30, 2)]:
        assert t.params_for(eu(seed=seed, sink=sink)) == PARAMS


def test_the_most_specific_entry_wins():
    t = tb.BestResponses()
    t.add(CELL, None, None, choice({"entry_ranks": [0]}))
    t.add(CELL, 3, None, choice({"entry_ranks": [1]}))
    t.add(CELL, 3, 4, choice({"entry_ranks": [2]}))
    assert t.params_for(eu(seed=1, sink=0)) == {"entry_ranks": [0]}
    assert t.params_for(eu(seed=3, sink=0)) == {"entry_ranks": [1]}
    assert t.params_for(eu(seed=3, sink=4)) == {"entry_ranks": [2]}


def test_each_budget_cell_topology_family_size_and_condition_has_its_own_entry():
    t = tb.BestResponses()
    t.add(CELL, None, None, choice())
    for change in (dict(audit=0.2), dict(audit="uncapped"), dict(intervention=0.1), dict(topology="chain"),
                   dict(attack="aitm"), dict(agents=7), dict(condition="white_box"), dict(arm="B")):
        with pytest.raises(LookupError):
            t.params_for(eu(**change))


def test_a_missing_entry_is_refused_and_names_the_cell():
    with pytest.raises(LookupError) as e:
        tb.BestResponses().params_for(eu())
    for part in ("small_world", "infection", "gray_box", '"C"'):
        assert part in str(e.value)


def test_the_attack_free_family_has_nothing_to_search():
    assert tb.BestResponses().params_for(eu(attack="none")) == {}


def test_a_returned_params_dict_is_a_copy():
    t = tb.BestResponses()
    t.add(CELL, None, None, choice({"entry_ranks": [2]}))
    t.params_for(eu())["entry_ranks"].append(9)
    assert t.params_for(eu()) == {"entry_ranks": [2]}


def test_annotate_adds_attack_params_and_leaves_the_unit_alone():
    t = tb.BestResponses()
    t.add(CELL, None, None, choice())
    u = eu(id="C-abc")
    out = t.annotate(u)
    assert out == {**u, "attack_params": PARAMS} and "attack_params" not in u


def test_missing_lists_each_uncovered_cell_once():
    t = tb.BestResponses()
    t.add(CELL, None, None, choice())
    units = [eu(seed=s, sink=k) for s in (1, 2) for k in (0, 1)] + [eu(audit=0.2, seed=s) for s in (1, 2)]
    assert t.missing(units) == [dict(CELL, audit=0.2)]
    assert t.missing([eu()]) == []


# ------------------------------------------------------------------------------------------ files on disk
def records(*rows):
    return [dict(seed=s, sink=k, attack_params=p, asr=0.5, baseline_asr=0.25, candidates=9, episodes=4)
            for s, k, p in rows]


def test_a_written_table_loads_back_exactly(tmp_path):
    tb.ensure_meta(tmp_path, "fp1", {"x": 1})
    tb.write_cell(tmp_path, CELL, records((None, None, PARAMS)), INPUTS)
    tb.write_cell(tmp_path, dict(CELL, audit="uncapped"), records((1, 2, {}), (1, 3, {"rate": 0.3, "entry_ranks": [0]})), INPUTS)
    t = tb.BestResponses.load(tmp_path, "fp1")
    assert len(t) == 3
    assert t.params_for(eu()) == PARAMS and t.params_for(eu(audit="uncapped", seed=1, sink=3)) == \
        {"rate": 0.3, "entry_ranks": [0]}
    c = t.choice(eu())
    assert (c.asr, c.baseline_asr, c.candidates, c.episodes) == (0.5, 0.25, 9, 4)


def test_budget_fractions_survive_the_round_trip_bit_for_bit(tmp_path):
    tb.ensure_meta(tmp_path, "fp1", {})
    cell = dict(CELL, audit=0.1 + 0.2, intervention=1 / 3)
    tb.write_cell(tmp_path, cell, records((None, None, PARAMS)), INPUTS)
    assert tb.BestResponses.load(tmp_path, "fp1").params_for(eu(audit=0.1 + 0.2, intervention=1 / 3)) == PARAMS


def test_loading_a_directory_without_meta_says_to_run_the_search(tmp_path):
    with pytest.raises(FileNotFoundError, match="search"):
        tb.BestResponses.load(tmp_path, "fp1")


def test_a_cell_file_is_written_atomically(tmp_path, monkeypatch):
    tb.ensure_meta(tmp_path, "fp1", {})
    tb.write_cell(tmp_path, CELL, records((None, None, PARAMS)), INPUTS)
    before = sorted(p.name for p in tmp_path.iterdir())

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        tb.write_cell(tmp_path, CELL, records((None, None, {"entry_ranks": [5]})), INPUTS)
    monkeypatch.undo()
    assert tb.BestResponses.load(tmp_path, "fp1").params_for(eu()) == PARAMS
    assert [n for n in sorted(p.name for p in tmp_path.iterdir()) if not n.endswith(".tmp")] == before


def test_the_cell_file_is_plain_json(tmp_path):
    tb.ensure_meta(tmp_path, "fp1", {})
    tb.write_cell(tmp_path, CELL, records((None, None, PARAMS)), INPUTS)
    path = next(p for p in tmp_path.iterdir() if p.name != "meta.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["cell"] == CELL and data["records"][0]["attack_params"] == PARAMS


def test_has_cell_tells_a_finished_cell(tmp_path):
    tb.ensure_meta(tmp_path, "fp1", {})
    assert not tb.has_cell(tmp_path, CELL, INPUTS)
    tb.write_cell(tmp_path, CELL, records((None, None, PARAMS)), INPUTS)
    assert tb.has_cell(tmp_path, CELL, INPUTS) and not tb.has_cell(tmp_path, dict(CELL, arm="B"), INPUTS)


def test_a_finished_cell_searched_against_other_thresholds_is_refused(tmp_path):
    tb.ensure_meta(tmp_path, "fp1", {})
    tb.write_cell(tmp_path, CELL, records((None, None, PARAMS)), INPUTS)
    with pytest.raises(ValueError, match="other thresholds"):
        tb.has_cell(tmp_path, CELL, "thresholds-sha-2")


def test_meta_is_written_once_and_a_different_fingerprint_is_refused(tmp_path):
    tb.ensure_meta(tmp_path, "fp1", {"evasion_cost": 1.0})
    tb.ensure_meta(tmp_path, "fp1", {"evasion_cost": 1.0})
    with pytest.raises(ValueError, match="other settings"):
        tb.ensure_meta(tmp_path, "fp2", {"evasion_cost": 0.5})
    assert json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))["fingerprint"] == "fp1"


def test_a_table_is_refused_by_a_load_with_another_fingerprint(tmp_path):
    tb.ensure_meta(tmp_path, "fp1", {})
    tb.write_cell(tmp_path, CELL, records((None, None, PARAMS)), INPUTS)
    with pytest.raises(ValueError, match="other settings"):
        tb.BestResponses.load(tmp_path, "fp2")
