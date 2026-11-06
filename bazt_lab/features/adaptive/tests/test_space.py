"""The strategy space: settings validation, entry sets, candidate enumeration."""
import json
import math
import time

import pytest

from bazt_lab.core import public as core
from bazt_lab.features.adaptive.private import space as sp
from bazt_lab.features.adaptive.tests.fakes import drop, make_settings

REQUIRED = [
    "runtime.n_insiders", "mock.every_sink", "mock.main_seeds", "mock.heldout_seeds",
    "mock.adaptive.conditions", "mock.adaptive.search_seeds", "mock.adaptive.insider_sets",
    "mock.adaptive.start_fractions", "mock.adaptive.rates", "mock.adaptive.detectabilities",
    "mock.adaptive.evasion_cost", "mock.adaptive.max_candidates", "mock.adaptive.knowledge.gray_box",
    "mock.adaptive.knowledge.white_box", "mock.adaptive.knowledge.embedding_evasion",
    "mock.adaptive.knowledge.attack_rate",
]


def space():
    return sp.load_space(make_settings())


# ------------------------------------------------------------------------------------------ settings
@pytest.mark.parametrize("key", REQUIRED)
def test_every_unset_key_is_refused_by_name(key):
    with pytest.raises(core.Unset) as e:
        sp.load_space(drop(make_settings(), key))
    assert e.value.args[0] == key


def test_load_reads_the_values():
    s = space()
    assert s.n_insiders == 3 and s.evasion_cost == 1.0 and s.insider_sets == "contiguous"
    assert s.search_seeds == (8001, 8003) and s.max_candidates == 5000
    assert s.knowledge["white_box"] == "episode" and s.knowledge["gray_box"] == "cell"


@pytest.mark.parametrize("over", [
    {"mock.adaptive.insider_sets": "some"},
    {"mock.adaptive.start_fractions": []},
    {"mock.adaptive.start_fractions": [0.0, 1.0]},                 # start at T + 1: never attacks
    {"mock.adaptive.start_fractions": [0.0, 0.2, 0.2]},            # duplicate
    {"mock.adaptive.rates": [0.5, 0.1]},                           # must contain 1.0 (nesting of conditions)
    {"mock.adaptive.rates": [1.0, 0.0]},
    {"mock.adaptive.detectabilities": [0.5, 0.1]},                 # must contain 1.0
    {"mock.adaptive.detectabilities": [1.0, 1.0]},
    {"mock.adaptive.evasion_cost": 1.5},
    {"mock.adaptive.evasion_cost": -0.1},
    {"mock.adaptive.max_candidates": 0},
    {"mock.adaptive.search_seeds": [8003, 8001]},
    {"mock.adaptive.search_seeds": [20, 40]},                      # overlaps main 1-30 and held-out 31-60
    {"mock.adaptive.search_seeds": [55, 70]},                      # overlaps held-out
    {"mock.adaptive.knowledge.gray_box": "episode"},               # gray-box does not know the allocation
    {"mock.adaptive.knowledge.white_box": "cell"},                 # white-box knows the allocation
    {"mock.adaptive.knowledge.attack_rate": "round"},
])
def test_invalid_settings_are_refused(over):
    with pytest.raises(ValueError):
        sp.load_space(make_settings(over))


def test_unknown_condition_in_conditions_is_refused():
    with pytest.raises(ValueError, match="stealth"):
        sp.load_space(make_settings({"mock.adaptive.conditions": ["gray_box", "stealth"]}))


def test_knowledge_is_only_required_for_listed_conditions():
    s = make_settings({"mock.adaptive.conditions": ["gray_box"]})
    s = drop(s, "mock.adaptive.knowledge.attack_rate")
    assert sp.load_space(s).knowledge == {"gray_box": "cell"}


# ----------------------------------------------------------------------------------------- entry sets
def test_single_entry_families_enumerate_every_degree_rank():
    for family in ("infection", "aitm"):
        assert sp.entry_sets(family, 5, space()) == [[0], [1], [2], [3], [4]]


def test_contiguous_insider_windows():
    for family in ("insider_pi", "insider_ta", "insider_ma"):
        assert sp.entry_sets(family, 6, space()) == [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]


def test_all_insider_subsets():
    sets = sp.entry_sets("insider_pi", 6, sp.load_space(make_settings({"mock.adaptive.insider_sets": "all"})))
    assert len(sets) == math.comb(6, 3) == 20
    assert sets[0] == [0, 1, 2] and sets[-1] == [3, 4, 5] and all(s == sorted(s) for s in sets)


def test_more_insiders_than_agents_is_refused():
    with pytest.raises(ValueError, match="n_insiders"):
        sp.entry_sets("insider_pi", 2, space())


def test_as_many_insiders_as_agents_leaves_no_victim_and_is_refused():
    with pytest.raises(ValueError, match="no victim"):
        sp.entry_sets("insider_pi", 3, space())
    assert sp.entry_sets("insider_pi", 4, space()) == [[0, 1, 2], [1, 2, 3]]


def test_unknown_family_is_refused():
    with pytest.raises(ValueError, match="worm"):
        sp.entry_sets("worm", 6, space())


# ------------------------------------------------------------------------------------------ candidates
def canon(params):
    return json.dumps(params, sort_keys=True)


def test_first_candidate_is_the_family_default():
    for cond in sp.CONDITIONS:
        assert sp.candidates("infection", 4, cond, space())[0] == {}


def test_gray_box_candidates_are_entry_by_start():
    c = sp.candidates("infection", 4, "gray_box", space())
    assert len(c) == 1 + 4 * 4
    assert c[1] == {"entry_ranks": [0]}                              # start_frac 0.0 is the identity: omitted
    assert c[2] == {"entry_ranks": [0], "start_frac": 0.2}
    assert c[-1] == {"entry_ranks": [3], "start_frac": 0.6}


def test_embedding_evasion_adds_detectability_and_potency():
    s = sp.load_space(make_settings({"mock.adaptive.evasion_cost": 0.5}))
    c = sp.candidates("infection", 2, "embedding_evasion", s)
    by = [p for p in c if p.get("entry_ranks") == [1] and "start_frac" not in p]
    d07 = next(p for p in by if p.get("detectability") == 0.7)
    assert d07 == {"entry_ranks": [1], "detectability": 0.7, "potency": 0.85}      # 1 - 0.5 * (1 - 0.7)
    assert {"entry_ranks": [1]} in by                                              # detectability 1.0 omitted
    assert all("rate" not in p for p in c)


def test_zero_evasion_cost_omits_potency():
    s = sp.load_space(make_settings({"mock.adaptive.evasion_cost": 0.0}))
    c = sp.candidates("infection", 2, "embedding_evasion", s)
    assert all("potency" not in p for p in c) and any("detectability" in p for p in c)


def test_attack_rate_adds_rate_only_below_one():
    c = sp.candidates("aitm", 2, "attack_rate", space())
    assert {"entry_ranks": [0], "rate": 0.3} in c
    assert all(p.get("rate", 0.5) < 1.0 for p in c if "rate" in p)
    assert all("detectability" not in p and "potency" not in p for p in c)


def test_candidate_order_breaks_ties_toward_the_least_adaptive_attack():
    s = sp.load_space(make_settings({"mock.adaptive.rates": [0.1, 1.0, 0.5], "mock.adaptive.start_fractions": [0.4, 0.0]}))
    c = sp.candidates("infection", 1, "attack_rate", s)
    assert c == [{}, {"entry_ranks": [0]},
                 {"entry_ranks": [0], "rate": 0.5}, {"entry_ranks": [0], "rate": 0.1},
                 {"entry_ranks": [0], "start_frac": 0.4},
                 {"entry_ranks": [0], "start_frac": 0.4, "rate": 0.5},
                 {"entry_ranks": [0], "start_frac": 0.4, "rate": 0.1}]


def test_conditions_are_nested():
    s = space()
    gray = {canon(p) for p in sp.candidates("insider_pi", 6, "gray_box", s)}
    assert gray == {canon(p) for p in sp.candidates("insider_pi", 6, "white_box", s)}
    for cond in ("embedding_evasion", "attack_rate"):
        assert gray < {canon(p) for p in sp.candidates("insider_pi", 6, cond, s)}


def test_candidates_are_unique_and_json_plain():
    for cond in sp.CONDITIONS:
        c = sp.candidates("insider_ta", 6, cond, space())
        assert len({canon(p) for p in c}) == len(c)
        json.dumps(c)                                                # plain ints, floats, lists
        for p in c:
            for v in p.values():
                assert type(v) in (int, float, list)


@pytest.mark.parametrize("family,N", [("infection", 5), ("aitm", 3), ("insider_pi", 7), ("insider_ma", 5)])
@pytest.mark.parametrize("cond", ["gray_box", "white_box", "embedding_evasion", "attack_rate"])
@pytest.mark.parametrize("mode", ["contiguous", "all"])
def test_count_matches_enumeration(family, N, cond, mode):
    s = sp.load_space(make_settings({"mock.adaptive.insider_sets": mode}))
    assert sp.candidate_count(family, N, cond, s) == len(sp.candidates(family, N, cond, s))


def test_cap_refuses_before_enumerating():
    s = sp.load_space(make_settings({"mock.adaptive.insider_sets": "all", "runtime.n_insiders": 10}))
    t0 = time.perf_counter()
    with pytest.raises(ValueError, match="max_candidates"):
        sp.candidates("insider_pi", 100, "embedding_evasion", s)     # C(100, 10) subsets: never materialised
    assert time.perf_counter() - t0 < 1.0


def test_cap_message_names_count_and_key():
    s = sp.load_space(make_settings({"mock.adaptive.max_candidates": 10}))
    with pytest.raises(ValueError) as e:
        sp.candidates("infection", 6, "gray_box", s)
    assert "mock.adaptive.max_candidates" in str(e.value) and str(sp.candidate_count("infection", 6, "gray_box", s)) in str(e.value)


def test_checked_count_returns_the_count_and_refuses_over_the_cap():
    s = sp.load_space(make_settings({"mock.adaptive.max_candidates": 10}))
    assert sp.checked_count("infection", 1, "gray_box", s) == 1 + 1 * 4
    with pytest.raises(ValueError, match="max_candidates"):
        sp.checked_count("infection", 6, "gray_box", s)
