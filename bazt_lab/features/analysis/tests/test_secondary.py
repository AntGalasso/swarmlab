"""Secondary analyses: the ladder A > B > C and the three ablations."""
import pytest

from bazt_lab.features.analysis.adapters import stats_check_adapter as sc
from bazt_lab.features.analysis.private import hypotheses as hyp
from bazt_lab.features.analysis.tests import synth
from bazt_lab.features.analysis.tests.helpers import load

TOPOS = ("chain", "barabasi_albert")


def rows(asr_by_arm, acc_by_arm=None, block="mock_frontier", cells=(synth.UNCAPPED,), **kw):
    arms = {a: (lambda cell, topo, n, v=v: v) for a, v in asr_by_arm.items()}
    acc = {a: (lambda *x, v=v: v) for a, v in (acc_by_arm or dict.fromkeys(asr_by_arm, 0.9)).items()}
    return synth.episodes(arms, utility=acc, block=block, cells=list(cells), topologies=TOPOS, seeds=range(1, 31),
                          sinks=range(2), noise_sd=0.01, **kw)


# --------------------------------------------------------------------------------------------- the ladder
def test_a_ladder_where_every_rung_beats_the_one_below_holds_in_every_cell(tmp_path):
    t, cfg = load(tmp_path, rows({"A": 0.6, "B": 0.45, "C": 0.3}))
    r = hyp.ladder(t, cfg, "mock")
    steps = {(c["challenger"], c["baseline"], c["context"]["topology"]): c for c in r["comparisons"]}
    assert set(steps) == {("B", "A", "chain"), ("C", "B", "chain"), ("B", "A", "barabasi_albert"),
                          ("C", "B", "barabasi_albert")}
    assert all(c["safety"]["delta"] == pytest.approx(-0.15, abs=0.02) and c["safety"]["p_holm"] < 0.05
               for c in steps.values())
    assert r["ordered_cells"] == 2 and r["n_cells"] == 2 and r["holds"] is True
    assert all(cell["ordered"] for cell in r["cells"])


def test_a_ladder_with_a_flat_step_does_not_hold(tmp_path):
    t, cfg = load(tmp_path, rows({"A": 0.6, "B": 0.3, "C": 0.3}))
    r = hyp.ladder(t, cfg, "mock")
    assert r["holds"] is False and r["ordered_cells"] == 0
    flat = [c for c in r["comparisons"] if c["challenger"] == "C"]
    assert {c["label"] for c in flat} == {sc.LABEL_NULL}


def test_a_ladder_in_the_wrong_order_is_not_ordered(tmp_path):
    t, cfg = load(tmp_path, rows({"A": 0.3, "B": 0.45, "C": 0.6}))
    r = hyp.ladder(t, cfg, "mock")
    assert r["ordered_cells"] == 0 and all(c["safety"]["delta"] > 0 for c in r["comparisons"])


def test_a_missing_rung_is_reported_as_skipped(tmp_path):
    t, cfg = load(tmp_path, rows({"A": 0.6, "C": 0.3}))
    r = hyp.ladder(t, cfg, "mock")
    assert r["comparisons"] == [] and len(r["skipped"]) == 4 and r["holds"] is None


# ------------------------------------------------------------------------------------------- ablations
def test_each_ablation_is_compared_with_the_full_arm_and_a_component_that_helps_wins(tmp_path):
    t, cfg = load(tmp_path, rows({"BAZT": 0.2, "BAZT_no_budget": 0.2, "BAZT_no_topology": 0.35,
                                  "BAZT_no_graduated": 0.2}, block="mock_ablations"))
    r = hyp.ablations(t, cfg, "mock")
    per = r["ablations"]
    assert set(per) == {"BAZT_no_budget", "BAZT_no_topology", "BAZT_no_graduated"}
    assert per["BAZT_no_topology"]["summary"]["supported"] is True          # removing topology hurts: the component helps
    assert per["BAZT_no_budget"]["summary"]["supported"] is False and per["BAZT_no_graduated"]["summary"]["n_wins"] == 0
    c = per["BAZT_no_topology"]["comparisons"][0]
    assert (c["challenger"], c["baseline"]) == ("BAZT", "BAZT_no_topology")
    assert c["safety"]["delta"] == pytest.approx(-0.15, abs=0.02)


def test_the_ablation_comparisons_share_one_holm_family(tmp_path):
    t, cfg = load(tmp_path, rows({"BAZT": 0.2, "BAZT_no_budget": 0.2005, "BAZT_no_topology": 0.2008,
                                  "BAZT_no_graduated": 0.2}, block="mock_ablations"))
    comps = [c for a in hyp.ablations(t, cfg, "mock")["ablations"].values() for c in a["comparisons"]]
    assert len(comps) == 6
    assert any(c["safety"]["p_holm"] > c["safety"]["p_raw"] for c in comps)


def test_an_ablation_arm_that_was_not_run_is_skipped(tmp_path):
    t, cfg = load(tmp_path, rows({"BAZT": 0.2, "BAZT_no_budget": 0.3}, block="mock_ablations"))
    per = hyp.ablations(t, cfg, "mock")["ablations"]
    assert per["BAZT_no_budget"]["summary"]["n_comparisons"] == 2
    assert per["BAZT_no_topology"]["summary"]["n_comparisons"] == 0 and len(per["BAZT_no_topology"]["skipped"]) == 2
    assert per["BAZT_no_topology"]["summary"]["complete"] is False and per["BAZT_no_budget"]["summary"]["complete"]
