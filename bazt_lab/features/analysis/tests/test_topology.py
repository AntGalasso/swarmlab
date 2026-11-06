"""H-topology: the advantage appears on heterogeneous graphs, not on Watts-Strogatz, and grows with N."""
import pytest

from bazt_lab.features.analysis.adapters import stats_check_adapter as sc
from bazt_lab.features.analysis.private import hypotheses as hyp
from bazt_lab.features.analysis.tests import synth
from bazt_lab.features.analysis.tests.helpers import load

BLOCK = "mock_epidemiology"


def graph_rows(gain, agents=(12, 100), topologies=("barabasi_albert", "small_world", "chain"), **kw):
    """C at ASR 0.4; BAZT gains gain(topology, N) below it. Accuracy is equal."""
    arms = {"C": lambda cell, topo, n: 0.4, "BAZT": lambda cell, topo, n: 0.4 - gain(topo, n)}
    args = dict(block=BLOCK, topologies=topologies, agents=agents, seeds=range(1, 31), sinks=range(2), noise_sd=0.01,
                utility={"C": lambda *a: 0.9, "BAZT": lambda *a: 0.9})
    args.update(kw)
    return synth.episodes(arms, **args)


def topology(tmp_path, rows, **over):
    t, cfg = load(tmp_path, rows, **over)
    return hyp.h_topology(t, cfg, "mock")["challengers"]["BAZT"]


def grows_on_hubs(topo, n):
    return {"barabasi_albert": {12: 0.05, 100: 0.20}}.get(topo, {}).get(n, 0.0)


def test_an_advantage_on_hubs_that_grows_with_n_and_none_on_small_world_supports_the_hypothesis(tmp_path):
    r = topology(tmp_path, graph_rows(grows_on_hubs))
    assert r["parts"] == {"appears_on_heterogeneous": True, "absent_on_homogeneous": True, "grows_with_n": True}
    assert r["supported"] is True and r["unevaluated"] == []
    assert r["classes"]["heterogeneous"]["n_comparisons"] == 2 and r["classes"]["homogeneous"]["n_comparisons"] == 2
    assert r["classes"]["other"]["n_comparisons"] == 2                    # chain is in neither class
    assert r["summary"]["n_comparisons"] == 6                             # one Holm family: topologies x N
    contrast, = r["growth_with_n"]["contrasts"]
    assert contrast["agents"] == [12, 100] and contrast["delta"] == pytest.approx(0.15, abs=0.02)
    assert contrast["p_holm"] < 0.05 and contrast["ci"][0] > 0 and contrast["test"] == "mann_whitney_u"


def test_an_advantage_on_small_world_too_refutes_the_not_on_small_world_part(tmp_path):
    r = topology(tmp_path, graph_rows(lambda topo, n: 0.2 if topo != "chain" else 0.0))
    assert r["parts"]["appears_on_heterogeneous"] is True and r["parts"]["absent_on_homogeneous"] is False
    assert r["supported"] is False


def test_an_advantage_that_does_not_grow_with_n_refutes_the_growth_part(tmp_path):
    r = topology(tmp_path, graph_rows(lambda topo, n: 0.1 if topo == "barabasi_albert" else 0.0))
    assert r["parts"]["appears_on_heterogeneous"] is True and r["parts"]["grows_with_n"] is False
    assert r["growth_with_n"]["n_grow"] == 0 and r["supported"] is False


def test_an_advantage_that_shrinks_with_n_is_counted_against_growth(tmp_path):
    r = topology(tmp_path, graph_rows(lambda topo, n: {12: 0.2, 100: 0.05}[n] if topo == "barabasi_albert" else 0.0))
    assert r["growth_with_n"]["n_shrink"] == 1 and r["parts"]["grows_with_n"] is False


def test_no_advantage_anywhere_does_not_appear_on_hubs(tmp_path):
    r = topology(tmp_path, graph_rows(lambda topo, n: 0.0))
    assert r["parts"]["appears_on_heterogeneous"] is False and r["supported"] is False
    assert {c["label"] for c in r["comparisons"]} == {sc.LABEL_NULL}


def test_a_single_agent_count_leaves_growth_unevaluated_and_the_other_parts_decide(tmp_path):
    r = topology(tmp_path, graph_rows(grows_on_hubs, agents=(12,)))
    assert r["parts"]["grows_with_n"] is None and r["unevaluated"] == ["grows_with_n"]
    assert r["growth_with_n"] == {"evaluated": False, "contrasts": [], "n_grow": 0, "n_shrink": 0}
    assert r["supported"] is True                                         # 0.05 at N=12 is still a significant gain


def test_without_a_homogeneous_topology_that_part_is_unevaluated(tmp_path):
    r = topology(tmp_path, graph_rows(grows_on_hubs, topologies=("barabasi_albert", "chain")))
    assert r["parts"]["absent_on_homogeneous"] is None and "absent_on_homogeneous" in r["unevaluated"]
    assert r["supported"] is True


def test_uncapped_cells_are_compared_too(tmp_path):
    r = topology(tmp_path, graph_rows(grows_on_hubs, cells=[synth.UNCAPPED, (0.1, 0.1)]))
    assert r["summary"]["n_comparisons"] == 12
    assert {c["context"]["audit"] for c in r["comparisons"]} == {"uncapped", 0.1}


def test_topology_classes_are_configured_per_platform(tmp_path):
    from bazt_lab.features.analysis.private import config
    from bazt_lab.features.analysis.tests import synth
    tree = synth.settings()
    tree["analysis"]["topology_classes"] = {"mock": {"heterogeneous": ["star_hub"], "homogeneous": ["small_world"]},
                                            "llm": {"heterogeneous": ["hub_star"], "homogeneous": ["watts_strogatz"]}}
    cfg = config.Cfg.from_settings(tree)
    assert cfg.heterogeneous == {"mock": ("star_hub",), "llm": ("hub_star",)}
    assert cfg.homogeneous == {"mock": ("small_world",), "llm": ("watts_strogatz",)}


def test_a_growth_contrast_with_a_nan_p_value_does_not_poison_the_holm_family():
    from bazt_lab.features.analysis.private import hypotheses
    assert hypotheses.holm_finite({0: float("nan"), 1: 0.01, 2: 0.04}) == {1: 0.02, 2: 0.04}
