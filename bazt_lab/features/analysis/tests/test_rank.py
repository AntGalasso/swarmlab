"""Rank agreement between the platforms: the arms ranked by ASR on each platform, compared with Spearman or Kendall."""
import math

import pytest

from bazt_lab.features.analysis.private import rank
from bazt_lab.features.analysis.tests import synth
from bazt_lab.features.analysis.tests.helpers import load

ARMS = ("A", "B", "C", "BAZT")


def platform_rows(asr, platform, noise=0.01):
    arms = {a: (lambda cell, topo, n, v=asr[a]: v) for a in asr}
    if platform == "mock":
        return synth.episodes(arms, block="mock_frontier", topologies=("chain", "star_hub"), sinks=range(2),
                              noise_sd=noise)
    return synth.episodes(arms, block="llm_headline", platform="llm", topologies=("random_directed",),
                          seeds=range(1, 6), conversations=range(6), tasks=("mmlu",), backbones=("gpt",),
                          noise_sd=noise)


def agreement(tmp_path, mock, llm, noise=0.01, **over):
    tm, cfg = load(tmp_path, platform_rows(mock, "mock", noise), "mock", **over)
    tl, _ = load(tmp_path, platform_rows(llm, "llm", noise), "llm", **over)
    return rank.rank_agreement({"mock": tm, "llm": tl}, cfg)


MOCK = {"A": 0.6, "B": 0.45, "C": 0.3, "BAZT": 0.15}


def mean_attacked_asr(rows, arm):
    """The mean ASR of an arm over the rows as written: the planted value plus the common seed effect of the draw."""
    values = [r["asr_final"] for r in rows if r["arm"] == arm and r["attack"] != "none"]
    return sum(values) / len(values)


def test_platforms_that_order_the_arms_alike_agree_perfectly(tmp_path):
    r = agreement(tmp_path, MOCK, {"A": 0.7, "B": 0.5, "C": 0.35, "BAZT": 0.2})
    assert r["status"] == "ok" and r["statistic"] == "spearman"
    assert r["value"] == pytest.approx(1.0) and r["supported"] is True and r["inversions"] == []
    assert r["ci"] == [pytest.approx(1.0), pytest.approx(1.0)]
    by = {a["mock"]: a for a in r["arms"]}
    assert (by["BAZT"]["mock_rank"], by["BAZT"]["llm_rank"], by["A"]["mock_rank"]) == (1.0, 1.0, 4.0)
    assert by["C"]["mock_asr"] == pytest.approx(mean_attacked_asr(platform_rows(MOCK, "mock"), "C"))
    assert by["C"]["llm_asr"] == pytest.approx(
        mean_attacked_asr(platform_rows({"A": 0.7, "B": 0.5, "C": 0.35, "BAZT": 0.2}, "llm"), "C"))
    assert r["threshold"] == 0.8 and r["contexts"] == {"mock": 2, "llm": 1}


def test_platforms_that_reverse_the_order_disagree(tmp_path):
    r = agreement(tmp_path, MOCK, {"A": 0.2, "B": 0.35, "C": 0.5, "BAZT": 0.7})
    assert r["value"] == pytest.approx(-1.0) and r["supported"] is False and len(r["inversions"]) == 6


def test_one_swapped_pair_is_one_inversion_and_lowers_the_statistic_by_the_hand_computed_amount(tmp_path):
    llm = {"A": 0.7, "B": 0.35, "C": 0.5, "BAZT": 0.2}                      # B and C swapped
    r = agreement(tmp_path / "s", MOCK, llm, runtime__rank_agreement_threshold=0.9)
    assert r["value"] == pytest.approx(1 - 6 * 2 / (4 * 15)) and r["supported"] is False     # 0.8
    # the safer arm is the one with the lower ASR: C (0.3 < 0.45) on the mock, B (0.35 < 0.5) on the LLM side
    assert r["inversions"] == [{"pair": ["B", "C"], "mock_better": "C", "llm_better": "B"}]
    k = agreement(tmp_path / "k", MOCK, llm, analysis__rank_agreement_statistic="kendall")
    assert k["statistic"] == "kendall" and k["value"] == pytest.approx((5 - 1) / 6)


def test_an_arm_missing_on_a_platform_makes_the_test_not_evaluable_and_says_which(tmp_path):
    r = agreement(tmp_path, MOCK, {"A": 0.7, "B": 0.5, "C": 0.35})
    assert r["status"] == "not_evaluable" and "BAZT" in r["reason"] and "llm" in r["reason"]


def test_a_platform_without_blocks_is_not_evaluable(tmp_path):
    r = agreement(tmp_path, MOCK, dict.fromkeys(ARMS, 0.5), analysis__blocks__rank__llm=[])
    assert r["status"] == "not_evaluable" and "llm" in r["reason"]


def test_the_ranking_does_not_depend_on_which_topologies_a_platform_ran(tmp_path):
    tm, cfg = load(tmp_path, platform_rows(MOCK, "mock"), "mock")
    one = [x for x in platform_rows(MOCK, "mock") if x["topology"] == "chain"]
    tm1, _ = load(tmp_path / "one", one, "mock")
    tl, _ = load(tmp_path, platform_rows(MOCK, "llm"), "llm")
    assert rank.rank_agreement({"mock": tm, "llm": tl}, cfg)["value"] == rank.rank_agreement({"mock": tm1, "llm": tl}, cfg)["value"]


def test_tied_arms_on_one_platform_give_an_undefined_statistic_that_is_not_supported(tmp_path):
    r = agreement(tmp_path, dict.fromkeys(ARMS, 0.4), MOCK, noise=0.0)
    assert r["status"] == "ok" and math.isnan(r["value"]) and r["supported"] is False
