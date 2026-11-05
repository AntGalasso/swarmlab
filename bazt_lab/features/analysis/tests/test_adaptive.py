"""H-adaptive: under the adaptive attacker the challenger's advantage over the baseline shrinks less than the
baseline's own security does."""
import pytest

from bazt_lab.features.analysis.adapters import stats_check_adapter as sc
from bazt_lab.features.analysis.private import hypotheses as hyp
from bazt_lab.features.analysis.tests import synth
from bazt_lab.features.analysis.tests.helpers import load

CELLS = [(0.1, 0.1)]
TOPOS = ("barabasi_albert", "chain")


def rows(c, bazt, block, condition=None, acc=0.9, seed=0):
    arms = {"C": lambda *a: c, "BAZT": lambda *a: bazt}
    return synth.episodes(arms, block=block, cells=CELLS, topologies=TOPOS, seeds=range(1, 31), sinks=range(2),
                          condition=condition, noise_sd=0.01, rng_seed=seed,
                          utility={"C": lambda *a: 0.9, "BAZT": lambda *a: acc})


def adaptive(tmp_path, ref, ad, contrast="advantage_shrinks_less_than_c_security", acc=0.9, extra=()):
    """ref = (C, BAZT) ASR without the adaptive attacker; ad = the same under it."""
    data = rows(*ref, "mock_frontier", seed=1) + rows(*ad, "mock_adaptive", condition="gray_box", acc=acc, seed=2)
    t, cfg = load(tmp_path, data + list(extra), analysis__adaptive_contrast=contrast)
    return hyp.h_adaptive(t, cfg, "mock")["challengers"]["BAZT"]


def test_a_challenger_that_loses_less_than_the_baseline_supports_the_hypothesis(tmp_path):
    r = adaptive(tmp_path, ref=(0.30, 0.10), ad=(0.50, 0.15))      # delta_C = +0.20, delta_BAZT = +0.05
    assert r["summary"]["n_comparisons"] == 2 and r["summary"]["supported"] is True
    c = r["comparisons"][0]
    assert c["adaptive"]["delta_baseline"] == pytest.approx(0.20, abs=0.01)
    assert c["adaptive"]["delta_challenger"] == pytest.approx(0.05, abs=0.01)
    assert c["safety"]["delta"] == pytest.approx(0.05 - 2 * 0.20, abs=0.01) and c["safety"]["p_holm"] < 0.05
    assert c["label"] in (sc.LABEL_STRICT, sc.LABEL_SAFETY) and c["context"]["condition"] == "gray_box"


def test_a_challenger_whose_advantage_collapses_is_not_supported(tmp_path):
    r = adaptive(tmp_path, ref=(0.30, 0.10), ad=(0.32, 0.60))     # BAZT is wrecked, C hardly moves
    assert r["summary"]["supported"] is False and r["summary"]["n_wins"] == 0
    assert all(c["safety"]["delta"] > 0 for c in r["comparisons"])


def test_the_adaptive_contrast_setting_decides_between_the_two_readings(tmp_path):
    case = dict(ref=(0.30, 0.10), ad=(0.40, 0.25))     # delta_C = 0.10, delta_BAZT = 0.15
    literal = adaptive(tmp_path / "a", **case)                                  # 0.15 < 2 * 0.10: advantage shrinks less
    strict = adaptive(tmp_path / "b", **case, contrast="challenger_degrades_less_than_c")   # 0.15 < 0.10 is false
    assert literal["summary"]["supported"] is True and strict["summary"]["supported"] is False
    assert strict["comparisons"][0]["safety"]["delta"] == pytest.approx(0.05, abs=0.01)


def test_utility_is_the_attack_free_accuracy_of_the_adaptive_block(tmp_path):
    r = adaptive(tmp_path, ref=(0.30, 0.10), ad=(0.50, 0.15), acc=0.80)         # BAZT loses 0.10 accuracy
    assert {c["label"] for c in r["comparisons"]} == {sc.LABEL_UTILITY_LOSS} and r["summary"]["supported"] is False


def test_a_condition_without_a_non_adaptive_reference_is_skipped_with_the_reason(tmp_path):
    ref = rows(0.30, 0.10, "mock_frontier", seed=1)
    ad = rows(0.50, 0.15, "mock_adaptive", condition="gray_box", seed=2)
    ref = [r for r in ref if r["topology"] != "chain"]
    t, cfg = load(tmp_path, ref + ad)
    r = hyp.h_adaptive(t, cfg, "mock")["challengers"]["BAZT"]
    assert r["summary"]["n_comparisons"] == 1 and r["summary"]["complete"] is False
    assert "non-adaptive reference" in r["skipped"][0]["reason"] and r["skipped"][0]["context"]["topology"] == "chain"


def test_every_knowledge_condition_is_its_own_comparison_in_one_family(tmp_path):
    more = rows(0.50, 0.60, "mock_adaptive", condition="white_box", seed=3)      # BAZT collapses under white-box
    r = adaptive(tmp_path, ref=(0.30, 0.10), ad=(0.50, 0.15), extra=more)
    by = {(c["context"]["condition"], c["context"]["topology"]): c for c in r["comparisons"]}
    assert set(by) == {("gray_box", "barabasi_albert"), ("gray_box", "chain"),
                       ("white_box", "barabasi_albert"), ("white_box", "chain")}
    assert by[("gray_box", "chain")]["label"] in (sc.LABEL_STRICT, sc.LABEL_SAFETY)
    assert by[("white_box", "chain")]["label"] == sc.LABEL_NULL and by[("white_box", "chain")]["safety"]["delta"] > 0
    holm = [c["safety"]["p_holm"] for c in r["comparisons"]]
    assert all(h >= c["safety"]["p_raw"] for h, c in zip(holm, r["comparisons"], strict=True))


def test_an_adaptive_block_without_attack_free_rows_takes_the_utility_from_its_reference_block(tmp_path):
    ref = rows(0.30, 0.10, "mock_frontier", seed=1)
    ad = [r for r in rows(0.50, 0.15, "mock_adaptive", condition="gray_box", seed=2) if r["attack"] != "none"]
    t, cfg = load(tmp_path, ref + ad)
    r = hyp.h_adaptive(t, cfg, "mock")["challengers"]["BAZT"]
    assert all(c["utility"] is not None for c in r["comparisons"])
    assert r["summary"]["supported"] is True


def test_adaptive_rows_without_a_condition_are_skipped_with_the_reason_and_the_hypothesis_is_incomplete(tmp_path):
    ref = rows(0.30, 0.10, "mock_frontier", seed=1)
    ad = rows(0.50, 0.15, "mock_adaptive", condition=None, seed=2)
    t, cfg = load(tmp_path, ref + ad)
    r = hyp.h_adaptive(t, cfg, "mock")["challengers"]["BAZT"]
    assert r["summary"]["n_comparisons"] == 0 and r["summary"]["complete"] is False
    assert "condition" in r["skipped"][0]["reason"]
