"""H-efficiency: the smallest budget at which the challenger reaches the security of the uncapped baseline."""
import pytest

from bazt_lab.features.analysis.private import frontier
from bazt_lab.features.analysis.tests import synth
from bazt_lab.features.analysis.tests.helpers import load

U = "uncapped"


def rows(bazt_by_cell, c_asr=0.30, **kw):
    """C uncapped at c_asr; BAZT at each cell of bazt_by_cell (cell -> ASR). One topology, 30 seeds, no noise."""
    cells = list(bazt_by_cell)
    args = dict(seeds=range(1, 31), sinks=range(2), topologies=("chain",), noise_sd=0.0, seed_sd=0.0,
                cells_by_arm={"C": [(U, U)], "BAZT": cells})
    args.update(kw)
    arms = {"C": lambda cell, topo, n: c_asr, "BAZT": lambda cell, topo, n: bazt_by_cell[cell]}
    return synth.episodes(arms, **args)


def efficiency(tmp_path, data, **over):
    t, cfg = load(tmp_path, data, **over)
    return frontier.efficiency(t, cfg, "mock")["challengers"]["BAZT"]["frontiers"]


def test_the_smallest_budget_is_the_cheapest_cell_that_reaches_the_uncapped_baseline(tmp_path):
    ladder = {(0.01, 0.01): 0.45, (0.05, 0.05): 0.36, (0.10, 0.10): 0.25, (0.20, 0.20): 0.20, (U, U): 0.20}
    f, = efficiency(tmp_path, rows(ladder))
    assert f["smallest"] == {"audit": 0.10, "intervention": 0.10, "total": pytest.approx(0.20)}
    assert f["smallest_ci"] == [pytest.approx(0.20), pytest.approx(0.20)] and f["p_reached"] == 1.0
    reach = {(c["audit"], c["intervention"]): c["reaches"] for c in f["cells"]}
    assert reach == {(0.01, 0.01): False, (0.05, 0.05): False, (0.10, 0.10): True, (0.20, 0.20): True, (U, U): True}
    c = next(c for c in f["cells"] if c["audit"] == 0.10)
    assert c["delta"] == pytest.approx(-0.05) and c["ci"][1] < 0 and c["p_reach"] == 1.0
    assert f["context"]["topology"] == "chain" and f["reference_asr"] == pytest.approx(0.30)


def test_a_cell_exactly_at_the_baseline_widens_the_interval_of_the_smallest_budget(tmp_path):
    data = rows({(0.05, 0.05): 0.5, (0.10, 0.10): 0.25}, c_asr=0.5)   # powers of two: the mean below is exactly 0
    for r in data:
        if r["arm"] == "BAZT" and r["attack"] != "none" and r["audit"] == 0.05:
            r["asr_final"] = 0.5 + 0.25 * (1 if r["seed"] % 2 else -1)
    f, = efficiency(tmp_path, data)
    assert f["smallest"]["audit"] == 0.05                           # delta is 0 <= tolerance 0 in the data
    assert f["smallest_ci"] == [pytest.approx(0.10), pytest.approx(0.20)]
    edge = next(c for c in f["cells"] if c["audit"] == 0.05)
    assert 0.3 < edge["p_reach"] < 0.8 and edge["ci"][0] < 0 < edge["ci"][1]


def test_a_tolerance_lets_a_slightly_worse_cell_count_as_reaching(tmp_path):
    data = rows({(0.05, 0.05): 0.32, (0.10, 0.10): 0.25})
    strict, = efficiency(tmp_path / "a", data)
    lax, = efficiency(tmp_path / "b", data, analysis__efficiency_tolerance=0.03)
    assert strict["smallest"]["audit"] == 0.10 and lax["smallest"]["audit"] == 0.05


def test_a_frontier_that_never_reaches_has_no_smallest_budget(tmp_path):
    f, = efficiency(tmp_path, rows({(0.05, 0.05): 0.5, (0.10, 0.10): 0.45}))
    assert f["smallest"] is None and f["smallest_ci"] is None and f["p_reached"] == 0.0
    assert f["pareto_minimal"] == []


def test_a_cell_with_one_budget_uncapped_reaches_but_is_never_the_smallest_finite_budget(tmp_path):
    f, = efficiency(tmp_path, rows({(0.20, 0.20): 0.25, (0.01, U): 0.20}))
    assert f["smallest"]["audit"] == 0.20
    only, = efficiency(tmp_path / "x", rows({(0.01, U): 0.20, (0.20, 0.20): 0.5}))
    assert only["smallest"] == {"audit": 0.01, "intervention": U, "total": U}
    assert only["smallest_ci"] == [U, U]


def test_pareto_minimal_cells_are_the_reaching_cells_no_other_reaching_cell_undercuts(tmp_path):
    cells = {(0.10, 0.50): 0.2, (0.50, 0.10): 0.2, (0.20, 0.20): 0.2, (0.50, 0.50): 0.2, (0.05, 0.05): 0.5}
    f, = efficiency(tmp_path, rows(cells))
    assert {(c["audit"], c["intervention"]) for c in f["pareto_minimal"]} == {(0.10, 0.50), (0.50, 0.10), (0.20, 0.20)}
    assert f["smallest"]["audit"] == 0.20                           # the cheapest of them by total budget


def test_each_topology_and_agent_count_has_its_own_frontier_and_a_missing_reference_is_skipped(tmp_path):
    no_reference = [r for r in rows({(0.10, 0.10): 0.2}, topologies=("small_world",)) if r["arm"] != "C"]
    data = rows({(0.10, 0.10): 0.2}, topologies=("chain", "star_hub")) + no_reference
    t, cfg = load(tmp_path, data)
    r = frontier.efficiency(t, cfg, "mock")["challengers"]["BAZT"]
    assert [f["context"]["topology"] for f in r["frontiers"]] == ["chain", "star_hub"]
    assert len(r["skipped"]) == 1 and r["skipped"][0]["context"]["topology"] == "small_world"
    assert "uncapped" in r["skipped"][0]["reason"]


def test_the_interval_is_deterministic_and_does_not_depend_on_row_order(tmp_path):
    data = rows({(0.05, 0.05): 0.5, (0.10, 0.10): 0.25}, c_asr=0.5)
    for r in data:
        if r["arm"] == "BAZT" and r["attack"] != "none" and r["audit"] == 0.05:
            r["asr_final"] = 0.5 + 0.25 * (1 if r["seed"] % 2 else -1)
    first, = efficiency(tmp_path / "a", data)
    second, = efficiency(tmp_path / "b", list(reversed(data)))
    assert first == second


def test_episodes_pair_by_conversation_on_the_llm_side_and_the_seed_clusters_the_bootstrap(tmp_path):
    data = rows({(0.10, 0.10): 0.25}, block="llm_capped", platform="llm", conversations=range(12), seeds=range(1, 6))
    t, cfg = load(tmp_path, data, platform="llm")
    f, = frontier.efficiency(t, cfg, "llm")["challengers"]["BAZT"]["frontiers"]
    assert f["n_pairs"] == 60 and f["n_clusters"] == 5 and f["smallest"]["audit"] == 0.10


def test_the_reference_asr_and_split_combinations_come_from_every_cell_not_the_first(tmp_path):
    from bazt_lab.features.analysis.private import frontier
    rows = [r for r in frontier_rows_for_cells() if True]
    t, cfg = load(tmp_path, rows, analysis__split_by__mock=["attack"])
    fs = frontier.efficiency(t, cfg, "mock")["challengers"]["BAZT"]["frontiers"]
    assert {f["context"]["attack"] for f in fs} == {"insider_pi", "insider_ta"}


def frontier_rows_for_cells():
    from bazt_lab.features.analysis.tests import synth
    arms = {"C": lambda *a: 0.3, "BAZT": lambda *a: 0.1}
    cheap = synth.episodes({"BAZT": arms["BAZT"]}, cells=[(0.1, 0.1)], families=("insider_pi",), seeds=range(1, 6))
    dear = synth.episodes({"BAZT": arms["BAZT"]}, cells=[(0.5, 0.5)], families=("insider_pi", "insider_ta"),
                          seeds=range(1, 6))
    base = synth.episodes({"C": arms["C"]}, cells=[("uncapped", "uncapped")], families=("insider_pi", "insider_ta"),
                          seeds=range(1, 6))
    return cheap + dear + base
