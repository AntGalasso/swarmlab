"""The best response against a fake platform whose final ASR is a known function of the attack_params."""
import json
import math
import random
from fractions import Fraction

import pytest

from bazt_lab.features.adaptive.private.run import run_search
from bazt_lab.features.adaptive.tests.fakes import FakePlatform, arm_of, make_settings, unit, units_of

N = 6
SEARCH = [8001, 8002, 8003]


def go(asr_fn, tmp_path, cond="gray_box", over=None, units=None, platform=None, **kw):
    platform = platform or FakePlatform(asr_fn)
    units = units if units is not None else units_of(unit(agents=N, condition=cond), (1, 2), range(N))
    table = run_search(units, arm_of, platform, make_settings(over), tmp_path, **kw)
    return table, platform, units


def best(table, units, **over):
    return table.params_for({**units[0], **over})


def R(r, **kw):
    return {"entry_ranks": [r], **kw}


# ----------------------------------------------------------------------------------------- finding the optimum
def test_the_best_candidate_of_the_space_is_found(tmp_path):
    def asr(arm, u):
        p = u["attack_params"]
        return 0.9 if p == R(2, start_frac=0.2) else 0.3 if p == {} else 0.1
    table, platform, units = go(asr, tmp_path)
    assert best(table, units) == R(2, start_frac=0.2)
    c = table.choice(units[0])
    assert (c.asr, c.baseline_asr) == (pytest.approx(0.9), pytest.approx(0.3))
    assert (c.candidates, c.episodes) == (1 + N * 4, 3 * N)


def test_the_family_default_is_kept_unless_something_is_strictly_better(tmp_path):
    table, _, units = go(lambda arm, u: 0.5 if u["attack_params"] == {} else 0.4, tmp_path)
    assert best(table, units) == {} and table.choice(units[0]).asr == table.choice(units[0]).baseline_asr == 0.5


def test_when_everything_ties_the_default_wins(tmp_path):
    table, _, units = go(lambda arm, u: 0.5, tmp_path)
    assert best(table, units) == {}


def test_a_tie_goes_to_the_earlier_candidate_entry_before_start(tmp_path):
    def asr(arm, u):
        return 0.8 if u["attack_params"] in (R(3), R(1, start_frac=0.4)) else 0.1
    table, _, units = go(asr, tmp_path)
    assert best(table, units) == R(1, start_frac=0.4)


def test_equal_means_over_different_episodes_are_a_tie_not_a_rounding_accident(tmp_path):
    # Left to right, (0.1 + 0.2 + 0.3) / 3 is one ulp above (0.3 + 0.2 + 0.1) / 3, which would crown the later candidate.
    by_params = {json.dumps(R(0)): {8001: 0.3, 8002: 0.2, 8003: 0.1}, json.dumps(R(1)): {8001: 0.1, 8002: 0.2, 8003: 0.3}}

    def asr(arm, u):
        by_seed = by_params.get(json.dumps(u["attack_params"]))
        return by_seed[u["seed"]] if by_seed else 0.0
    units = units_of(unit(agents=N), (1, 2), [None])
    table, _, _ = go(asr, tmp_path, over={"mock.every_sink": False}, units=units)
    assert best(table, units) == R(0)


# ----------------------------------------------------------------------------------------------- conditions
def test_embedding_evasion_trades_detectability_against_potency(tmp_path):
    def asr(arm, u):
        p = u["attack_params"]
        return p.get("potency", 1.0) * (1.0 - p.get("detectability", 1.0) ** 2)
    table, _, units = go(asr, tmp_path, cond="embedding_evasion")
    assert best(table, units) == R(0, detectability=0.5, potency=0.5)


def test_without_an_evasion_cost_the_least_detectable_payload_wins(tmp_path):
    def asr(arm, u):
        return 1.0 - u["attack_params"].get("detectability", 1.0) ** 2
    table, _, units = go(asr, tmp_path, cond="embedding_evasion", over={"mock.adaptive.evasion_cost": 0.0})
    assert best(table, units) == R(0, detectability=0.1)


def test_attack_rate_finds_the_interior_rate(tmp_path):
    def asr(arm, u):
        return 1.0 - abs(u["attack_params"].get("rate", 1.0) - 0.3)
    table, _, units = go(asr, tmp_path, cond="attack_rate")
    assert best(table, units) == R(0, rate=0.3)


def test_conditions_do_not_leak_into_each_other(tmp_path):
    table, platform, _ = go(lambda arm, u: 0.5, tmp_path, cond="gray_box")
    for call in platform.calls:
        for u in call["units"]:
            assert not {"rate", "detectability", "potency"} & set(u["attack_params"])


# --------------------------------------------------------------------------------------------- knowledge levels
def per_seed_entry(arm, u):
    p = u["attack_params"]
    if p == {}:
        return 0.0
    return 1.0 if p == R(u["seed"] % 4) else 0.1


def test_a_gray_box_attacker_commits_to_one_strategy_for_the_cell(tmp_path):
    table, platform, units = go(per_seed_entry, tmp_path, units=units_of(unit(agents=N), (1, 2, 3), (0, 1)))
    assert {u["seed"] for c in platform.calls for u in c["units"]} == set(SEARCH)
    # ranks 1, 2 and 3 each win one search seed (8001 % 4 = 1, ...): a three-way tie, so the earliest
    assert {json.dumps(table.params_for({**units[0], "seed": s})) for s in (1, 2, 3, 30)} == {json.dumps(R(1))}


def test_a_white_box_attacker_picks_a_strategy_per_evaluated_episode(tmp_path):
    units = units_of(unit(agents=N, condition="white_box"), (1, 2, 3), (0, 1))
    table, platform, _ = go(per_seed_entry, tmp_path, units=units)
    assert {u["seed"] for c in platform.calls for u in c["units"]} == {1, 2, 3}
    for s in (1, 2, 3):
        for k in (0, 1):
            assert table.params_for({**units[0], "seed": s, "sink": k}) == R(s % 4)


def test_a_seed_level_white_box_averages_over_the_sinks_of_the_seed(tmp_path):
    def asr(arm, u):                     # sink 0 prefers rank 1, sinks 1 and 2 prefer rank 2
        p = u["attack_params"]
        want = 1 if u["sink"] == 0 else 2
        return 0.0 if p == {} else 1.0 if p == R(want) else 0.1
    units = units_of(unit(agents=N, condition="white_box"), (1, 2), (0, 1, 2))
    table, _, _ = go(asr, tmp_path, units=units, over={"mock.adaptive.knowledge.white_box": "seed"})
    for s in (1, 2):
        assert all(table.params_for({**units[0], "seed": s, "sink": k}) == R(2) for k in (0, 1, 2))


def test_candidates_are_compared_on_identical_episodes(tmp_path):
    _, platform, _ = go(lambda arm, u: 0.5, tmp_path)
    seen = {}
    for call in platform.calls:
        for u in call["units"]:
            seen.setdefault(json.dumps(u["attack_params"], sort_keys=True), set()).add((u["seed"], u["sink"]))
    assert len(seen) == 1 + N * 4 and len({frozenset(v) for v in seen.values()}) == 1
    assert len(next(iter(seen.values()))) == 3 * N


# --------------------------------------------------------------------------------------- what the platform sees
def test_search_units_are_the_cell_with_attack_params_ids_and_thresholds(tmp_path):
    _, platform, units = go(lambda arm, u: 0.5, tmp_path, thresholds_of=lambda cell: {"mon": 1.5, "arm": cell["arm"]})
    ids = set()
    for call in platform.calls:
        assert call["arm"] == "X" and call["thresholds"] == {"mon": 1.5, "arm": "X"}
        for u in call["units"]:
            assert u["block"] == "mock_adaptive_search" and u["platform"] == "mock"
            assert {k: u[k] for k in ("arm", "audit", "intervention", "topology", "attack", "agents", "condition")} == \
                {k: units[0][k] for k in ("arm", "audit", "intervention", "topology", "attack", "agents", "condition")}
            json.dumps(u["attack_params"])
            ids.add(u["id"])
    assert len(ids) == 3 * N                      # one id per search episode, shared by all its candidates


def test_without_thresholds_the_units_carry_empty_ones(tmp_path):
    _, platform, _ = go(lambda arm, u: 0.5, tmp_path)
    assert all(c["thresholds"] == {} for c in platform.calls)


def test_the_result_does_not_depend_on_the_batch_size(tmp_path):
    def asr(arm, u):
        p = json.dumps(u["attack_params"], sort_keys=True)
        return (sum(map(ord, p)) + u["seed"] * 7 + (u["sink"] or 0)) % 11 / 10
    total = (1 + N * 4) * 3 * N
    results, sizes = [], []
    for k, bs in enumerate((1, 7, 1000)):
        table, platform, units = go(asr, tmp_path / str(k), over={"mock.batch_size": bs})
        results.append(json.dumps(table.params_for(units[0])))
        sizes.append(max(c["n"] for c in platform.calls))
        assert platform.submitted == total
    assert len(set(results)) == 1 and sizes == [1, 7, 3 * N]


# --------------------------------------------------------------------------------------------- bad platforms
def test_a_nan_asr_is_refused(tmp_path):
    with pytest.raises(ValueError, match="asr_final"):
        go(lambda arm, u: math.nan, tmp_path)


def test_a_row_without_the_metric_is_refused(tmp_path):
    class NoMetric(FakePlatform):
        def run_batch(self, units, arm, settings, store, tracker):
            rows = super().run_batch(units, arm, settings, store, tracker)
            return [{k: v for k, v in r.items() if k != "asr_final"} for r in rows]
    with pytest.raises(ValueError, match="asr_final"):
        go(None, tmp_path, platform=NoMetric(lambda arm, u: 0.5))


def test_a_platform_that_returns_the_wrong_number_of_rows_is_refused(tmp_path):
    class Short(FakePlatform):
        def run_batch(self, units, arm, settings, store, tracker):
            return super().run_batch(units, arm, settings, store, tracker)[:-1]
    with pytest.raises(RuntimeError, match="rows"):
        go(None, tmp_path, platform=Short(lambda arm, u: 0.5))


def test_rows_in_another_order_than_the_units_are_refused(tmp_path):
    class Shuffled(FakePlatform):
        def run_batch(self, units, arm, settings, store, tracker):
            return list(reversed(super().run_batch(units, arm, settings, store, tracker)))
    with pytest.raises(RuntimeError, match="order"):
        go(None, tmp_path, platform=Shuffled(lambda arm, u: 0.5))


def test_rows_are_matched_to_their_units_by_either_id_field_the_metrics_rows_may_carry(tmp_path):
    class UnitIdRows(FakePlatform):                    # metrics rows name the unit `unit_id` when given an id string
        def run_batch(self, units, arm, settings, store, tracker):
            rows = super().run_batch(units, arm, settings, store, tracker)
            return [{"unit_id": r["id"], "asr_final": r["asr_final"]} for r in rows]

    class WrongUnitIdRows(FakePlatform):
        def run_batch(self, units, arm, settings, store, tracker):
            rows = super().run_batch(units, arm, settings, store, tracker)
            return [{"unit_id": "another-unit", "asr_final": r["asr_final"]} for r in rows]
    table, _, units = go(None, tmp_path / "ok", platform=UnitIdRows(lambda arm, u: 0.5 if u["attack_params"] == {} else 0.4))
    assert best(table, units) == {}
    with pytest.raises(RuntimeError, match="order"):
        go(None, tmp_path / "bad", platform=WrongUnitIdRows(lambda arm, u: 0.5))


# ----------------------------------------------------------------- against exact arithmetic, ties included
@pytest.mark.parametrize("level,cond", [("cell", "gray_box"), ("seed", "white_box"), ("episode", "white_box")])
def test_the_choice_equals_an_exact_rational_argmax_on_random_tables(tmp_path, level, cond):
    rng = random.Random(7)  # noqa: S311 - a fixture generator, not a secret
    table_of = {}                                   # (params json, seed, sink) -> Fraction in quarters

    def asr(arm, u):
        key = (json.dumps(u["attack_params"], sort_keys=True), u["seed"], u["sink"])
        if key not in table_of:
            table_of[key] = Fraction(rng.randrange(0, 5), 4)
        return float(table_of[key])
    units = units_of(unit(agents=4, condition=cond), (1, 2, 3), range(4))
    over = {"mock.adaptive.start_fractions": [0.0, 0.5]}
    if level != "cell":
        over["mock.adaptive.knowledge.white_box"] = level
    table, platform, _ = go(asr, tmp_path, over=over, units=units)

    from bazt_lab.features.adaptive.private import space as sp
    cands = sp.candidates("infection", 4, cond, sp.load_space(make_settings(over)))
    for u in units:
        seed, sink = u["seed"], u["sink"]
        if level == "cell":
            keys = [(s, k) for s in SEARCH for k in range(4)]
        elif level == "seed":
            keys = [(seed, k) for k in range(4)]
        else:
            keys = [(seed, sink)]
        means = [sum(table_of[(json.dumps(c, sort_keys=True), s, k)] for s, k in keys) / len(keys) for c in cands]
        want = cands[max(range(len(cands)), key=means.__getitem__)]
        assert table.params_for(u) == want
