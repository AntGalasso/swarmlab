"""Containment per cost against the paired reference run (A)."""
import math

import pytest

from bazt_lab.features.analysis.private import containment
from bazt_lab.features.analysis.tests import synth
from bazt_lab.features.analysis.tests.helpers import load

U = "uncapped"


def rows(spend_of=lambda r: (40.0, 60.0), arm_auc=None, **kw):
    """A uncapped (reached AUC 0.5, ASR 0.6, no spend); BAZT at two cells; every episode spends spend_of(row)."""
    arm_auc = arm_auc or {"A": 0.5, "BAZT": 0.3}
    arms = {"A": lambda cell, topo, n: 0.6, "BAZT": lambda cell, topo, n: 0.3}
    data = synth.episodes(arms, cells_by_arm={"A": [(U, U)], "BAZT": [(0.1, 0.1), (0.2, 0.2)]}, topologies=("chain",),
                          seeds=range(1, 31), sinks=range(2), noise_sd=0.0, seed_sd=0.0, **kw)
    for r in data:
        r["agents_reached_auc"] = arm_auc[r["arm"]] if r["attack"] != "none" else math.nan
        r["spend_audit_total"], r["spend_intervention_total"] = (0.0, 0.0) if r["arm"] == "A" else spend_of(r)
    return data


def per_cost(tmp_path, data, **over):
    t, cfg = load(tmp_path, data, **over)
    return containment.containment(t, cfg, "mock")


def pick(result, arm, audit, metric):
    return next(e for e in result["entries"] if e["arm"] == arm and e["context"]["audit"] == audit and e["metric"] == metric)


def test_containment_per_cost_is_the_gain_over_the_reference_per_unit_spent(tmp_path):
    r = per_cost(tmp_path, rows())
    e = pick(r, "BAZT", 0.1, "agents_reached_auc")
    assert e["mean_gain"] == pytest.approx(0.2) and e["mean_spend"] == pytest.approx(100.0)
    assert e["containment_per_cost"] == pytest.approx(0.002) and e["ci"] == [pytest.approx(0.002)] * 2
    assert (e["n_pairs"], e["n_clusters"]) == (60, 30)
    assert pick(r, "BAZT", 0.2, "asr_final")["containment_per_cost"] == pytest.approx(0.3 / 100)
    assert {e["arm"] for e in r["entries"]} == {"BAZT"}                    # the reference is not compared with itself


def test_the_ratio_is_total_gain_over_total_spend_not_the_mean_of_per_episode_ratios(tmp_path):
    r = per_cost(tmp_path, rows(spend_of=lambda row: (50.0, 0.0) if row["sink"] == 0 else (150.0, 0.0)))
    e = pick(r, "BAZT", 0.1, "agents_reached_auc")
    assert e["containment_per_cost"] == pytest.approx(0.2 / 100.0)         # a mean of ratios would be 0.2 * (1/50 + 1/150) / 2


def test_an_arm_that_spends_nothing_has_no_defined_containment_per_cost(tmp_path):
    e = pick(per_cost(tmp_path, rows(spend_of=lambda row: (0.0, 0.0))), "BAZT", 0.1, "agents_reached_auc")
    assert e["mean_gain"] == pytest.approx(0.2) and math.isnan(e["containment_per_cost"]) and math.isnan(e["ci"][0])


def test_a_cell_without_the_reference_run_is_skipped(tmp_path):
    data = [r for r in rows() if r["arm"] != "A"]
    r = per_cost(tmp_path, data)
    assert r["entries"] == [] and len(r["skipped"]) == 2 and "A" in r["skipped"][0]["reason"]


def test_the_gain_metrics_are_the_configured_ones(tmp_path):
    r = per_cost(tmp_path, rows(), analysis__containment_metrics=["asr_final"])
    assert {e["metric"] for e in r["entries"]} == {"asr_final"}


def test_the_reference_is_found_in_another_block_of_the_hypothesis(tmp_path):
    """A runs only in the headline block; the capped arms live in another block (llm_capped)."""
    data = rows()
    for r in data:
        r["block"] = "mock_frontier" if r["arm"] == "A" else "mock_capped"
    r = per_cost(tmp_path, data, analysis__blocks__containment__mock=["mock_frontier", "mock_capped"])
    e = pick(r, "BAZT", 0.1, "agents_reached_auc")
    assert e["containment_per_cost"] == pytest.approx(0.002) and not r["skipped"]


def test_a_reference_that_never_sanitizes_serves_every_sanitize_mode(tmp_path):
    data = rows()
    out = []
    for r in data:
        if r["arm"] == "A":
            out.append(dict(r, sanitize_mode="redaction"))
        else:
            out += [dict(r, sanitize_mode=m) for m in ("redaction", "llm_rewrite")]
    r = per_cost(tmp_path, out)
    modes = {e["context"]["sanitize_mode"] for e in r["entries"] if e["arm"] == "BAZT"}
    assert modes == {"redaction", "llm_rewrite"} and not r["skipped"]
