"""H-budget on synthetic rows with planted effects: a real effect is detected, a null is not, Holm adjusts correctly,
mock pairs are seeds, LLM pairs are conversations with a seed-cluster bootstrap."""
import math

import numpy as np
import pytest

from bazt_lab.features.analysis.adapters import stats_check_adapter as sc
from bazt_lab.features.analysis.private import hypotheses as hyp
from bazt_lab.features.analysis.tests import synth
from bazt_lab.features.analysis.tests.helpers import independent_holm, load

CELLS = [(0.1, 0.1), (0.2, 0.2)]
TOPOS = ("barabasi_albert", "small_world")
WINS = (sc.LABEL_STRICT, sc.LABEL_SAFETY)


def frontier(effect=-0.15, acc_loss=0.0, cells=CELLS, **kw):
    """C at ASR 0.4 and accuracy 0.9; BAZT shifted by effect(cell, topology) (a number or a function) and acc_loss."""
    fx = effect if callable(effect) else (lambda cell, topo: effect)
    arms = {"C": lambda cell, topo, n: 0.4, "BAZT": lambda cell, topo, n: 0.4 + fx(cell, topo)}
    utility = {"C": lambda *a: 0.9, "BAZT": lambda *a: 0.9 - acc_loss}
    args = dict(cells=cells, topologies=TOPOS, seeds=range(1, 31), sinks=range(3), noise_sd=0.01, seed_sd=0.05)
    args.update(kw)
    return synth.episodes(arms, utility=utility, **args)


def budget(t, cfg, platform="mock", challenger="BAZT"):
    return hyp.h_budget(t, cfg, platform)["challengers"][challenger]


def test_a_planted_safety_gain_is_a_win_on_every_capped_cell(tmp_path):
    t, cfg = load(tmp_path, frontier(cells=CELLS + [synth.UNCAPPED]))
    r = budget(t, cfg)
    assert r["summary"]["n_comparisons"] == 4                       # 2 topologies x 2 capped cells; uncapped is not H-budget
    assert r["summary"]["supported"] is True and r["summary"]["complete"] is True
    for c in r["comparisons"]:
        s = c["safety"]
        assert c["label"] in WINS
        assert s["delta"] == pytest.approx(-0.15, abs=0.02) and s["p_holm"] < 0.05 and s["ci"][1] < 0
        assert s["test"] == "wilcoxon"
        assert (s["n_pairs"], s["n_clusters"]) == (30, 30)          # mock pairs are seeds, not seeds x sinks
        assert c["utility"]["delta"] == pytest.approx(0.0, abs=0.01)  # accuracy loss
        assert c["context"]["audit"] in (0.1, 0.2) and c["context"]["block"] == "mock_frontier"


def test_a_null_is_not_detected(tmp_path):
    t, cfg = load(tmp_path, frontier(effect=0.0))
    r = budget(t, cfg)
    assert [c["label"] for c in r["comparisons"]] == [sc.LABEL_NULL] * 4
    assert r["summary"]["supported"] is False and r["summary"]["n_wins"] == 0


def test_a_safety_gain_bought_with_lost_accuracy_is_a_utility_loss_never_a_win(tmp_path):
    t, cfg = load(tmp_path, frontier(effect=-0.15, acc_loss=0.05))
    r = budget(t, cfg)
    assert [c["label"] for c in r["comparisons"]] == [sc.LABEL_UTILITY_LOSS] * 4
    assert r["summary"]["n_utility_loss"] == 4 and r["summary"]["supported"] is False
    assert all(c["utility"]["delta"] == pytest.approx(0.05, abs=0.01) for c in r["comparisons"])


def test_holm_family_is_the_hypothesis_comparisons_and_matches_the_definition(tmp_path):
    spread = {(CELLS[0], TOPOS[0]): -0.004, (CELLS[1], TOPOS[0]): -0.0025, (CELLS[0], TOPOS[1]): -0.0015}
    t, cfg = load(tmp_path, frontier(effect=lambda cell, topo: spread.get((cell, topo), 0.0)))
    comps = budget(t, cfg)["comparisons"]
    raw = [c["safety"]["p_raw"] for c in comps]
    assert len(set(np.round(raw, 6))) == 4 and 0.0005 < min(raw) < 0.2 < max(raw)
    np.testing.assert_allclose([c["safety"]["p_holm"] for c in comps], independent_holm(raw), atol=1e-12)
    assert any(c["safety"]["p_holm"] > c["safety"]["p_raw"] for c in comps)
    utility_raw = [c["utility"]["p_raw"] for c in comps]                # its own family, per endpoint
    np.testing.assert_allclose([c["utility"]["p_holm"] for c in comps], independent_holm(utility_raw), atol=1e-12)


def test_the_binary_endpoint_is_tested_on_episodes_with_mcnemar(tmp_path):
    t, cfg = load(tmp_path, frontier())
    extra = budget(t, cfg)["comparisons"][0]["extras"]["protected_sink_harm"]
    assert extra["test"] == "mcnemar_exact" and extra["n_pairs"] == 90      # 30 seeds x 3 sinks
    assert extra["p_holm"] >= extra["p_raw"] and extra["delta"] < 0


def test_a_missing_challenger_cell_is_reported_as_skipped_and_the_hypothesis_is_incomplete(tmp_path):
    t, cfg = load(tmp_path, frontier(cells_by_arm={"BAZT": CELLS[:1], "C": CELLS}))
    r = budget(t, cfg)
    assert r["summary"]["n_comparisons"] == 2 and r["summary"]["complete"] is False
    assert len(r["skipped"]) == 2 and all("BAZT" in s["reason"] for s in r["skipped"])
    assert {s["context"]["audit"] for s in r["skipped"]} == {0.2}


def test_conditions_of_one_block_are_separate_comparisons_not_pooled(tmp_path):
    rows = (frontier(effect=-0.15) + frontier(effect=0.0, condition="gray_box", rng_seed=5))
    t, cfg = load(tmp_path, rows)
    comps = budget(t, cfg)["comparisons"]
    by_condition = {}
    for c in comps:
        by_condition.setdefault(c["context"]["condition"], set()).add(c["label"])
    assert len(comps) == 8 and by_condition[None] <= set(WINS) and by_condition["gray_box"] == {sc.LABEL_NULL}


def test_split_by_attack_gives_one_comparison_per_family_and_pooled_utility(tmp_path):
    effect = lambda arm, family: -0.2 if (arm == "BAZT" and family == "fa") else 0.0     # noqa: E731
    rows = frontier(effect=0.0, families=("fa", "fb"), family_effect=effect)
    pooled_t, pooled_cfg = load(tmp_path / "p", rows)
    assert len(budget(pooled_t, pooled_cfg)["comparisons"]) == 4
    t, cfg = load(tmp_path / "s", rows, analysis__split_by__mock=["attack"])
    comps = budget(t, cfg)["comparisons"]
    assert len(comps) == 8
    labels = {c["context"]["attack"]: set() for c in comps}
    for c in comps:
        labels[c["context"]["attack"]].add(c["label"])
        assert c["utility"] is not None and c["utility"]["n_pairs"] == 30    # attack-free rows have no family
    assert labels["fa"] <= set(WINS) and labels["fb"] == {sc.LABEL_NULL}


def llm_rows(seed_shift, challenger_effect=0.0):
    arms = {"C": lambda *a: 0.4, "BAZT": lambda *a: 0.4 + challenger_effect}
    rows = synth.episodes(arms, utility={"C": lambda *a: 0.9, "BAZT": lambda *a: 0.9}, block="llm_capped",
                          platform="llm", cells=[(0.1, 0.1)], topologies=("random_directed",), seeds=range(1, 6),
                          conversations=range(12), tasks=("mmlu",), backbones=("gpt",), noise_sd=0.01, seed_sd=0.0)
    for r in rows:
        if r["arm"] == "BAZT" and r["attack"] != "none":
            r["asr_final"] += seed_shift[r["seed"] - 1]
    return rows


def test_llm_pairs_are_conversations_and_the_ci_resamples_whole_seeds(tmp_path):
    shift = [-0.1, -0.05, 0.0, 0.05, 0.1]                               # a seed effect inside the difference
    rows = llm_rows(shift)
    t, cfg = load(tmp_path, rows, platform="llm")
    s = budget(t, cfg, "llm")["comparisons"][0]["safety"]
    assert (s["n_pairs"], s["n_clusters"]) == (60, 5)
    c_asr = {(r["seed"], r["conversation"]): r["asr_final"] for r in rows if r["arm"] == "C" and r["attack"] != "none"}
    d = np.array([r["asr_final"] - c_asr[(r["seed"], r["conversation"])]
                  for r in rows if r["arm"] == "BAZT" and r["attack"] != "none"])
    naive_half_width = 1.96 * d.std(ddof=1) / math.sqrt(len(d))
    assert (s["ci"][1] - s["ci"][0]) / 2 > 2 * naive_half_width         # clusters differ from the naive interval
    assert s["se_ratio_cluster_vs_naive"] > 2.0


def test_llm_real_effect_is_detected_on_conversation_pairs(tmp_path):
    t, cfg = load(tmp_path, llm_rows([0.0] * 5, challenger_effect=-0.15), platform="llm")
    c = budget(t, cfg, "llm")["comparisons"][0]
    assert c["safety"]["p_holm"] < 0.001 and c["safety"]["ci"][1] < 0 and c["label"] in WINS


def test_blocks_that_are_not_configured_or_not_on_disk_are_reported_not_invented(tmp_path):
    t, cfg = load(tmp_path, frontier(), analysis__blocks__budget__mock=[])
    assert hyp.h_budget(t, cfg, "mock") == {"status": "not_configured"}
    t, cfg = load(tmp_path, frontier(), analysis__blocks__budget__mock=["mock_nowhere"])
    r = hyp.h_budget(t, cfg, "mock")
    assert r["status"] == "no_data" and r["blocks_missing"] == ["mock_nowhere"]
