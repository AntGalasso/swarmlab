"""LLM rows cannot pair their attack-free run on the platform: analysis forms the clean reference, NRP and recovery."""
import math

import pytest

from bazt_lab.features.analysis.private import table as tb
from bazt_lab.features.analysis.tests import synth
from bazt_lab.features.analysis.tests.helpers import load

METRICS = ("asr_final", "accuracy_clean", "accuracy_attack")


def llm_pair(seed=1, conv=0, asr=0.2, clean=0.95, by_round=(0.5, 0.9, 0.95, 0.95), arm="C", twin=True):
    base = dict(block="b", platform="llm", arm=arm, audit="uncapped", intervention="uncapped", topology="chain",
                agents=12, seed=seed, sink=0, conversation=conv, task="mmlu", backbone="gpt")
    rows = [dict(base, attack="insider_pi", asr_final=asr, accuracy_attack=1 - asr, accuracy_clean=math.nan,
                 accuracy_by_round=list(by_round), accuracy_clean_reference=math.nan, nrp=math.nan,
                 recovery_time_accuracy=math.nan)]
    if twin:
        rows.append(dict(base, attack="none", asr_final=math.nan, accuracy_attack=math.nan, accuracy_clean=clean))
    return rows


def read(tmp_path, rows, margin=0.01):
    synth.write_rows(tmp_path, rows)
    return tb.load(tmp_path, ("b",), METRICS, clean_margin=margin)


def attacked(t, col):
    sel = [i for i in range(t.n) if t.vocab["attack"][t.codes["attack"][i]] != "none"]
    return t.metrics[col][sel]


def test_the_clean_reference_is_the_twin_accuracy_and_nrp_is_clean_times_one_minus_asr(tmp_path):
    t = read(tmp_path, llm_pair(asr=0.2, clean=0.95))
    assert attacked(t, "accuracy_clean_reference")[0] == pytest.approx(0.95)
    assert attacked(t, "nrp")[0] == pytest.approx(0.95 * 0.8)


def test_recovery_time_is_the_trailing_run_within_the_margin_of_the_twin(tmp_path):
    t = read(tmp_path, llm_pair(clean=0.95, by_round=(0.5, 0.9, 0.95, 0.95)))
    assert attacked(t, "recovery_time_accuracy")[0] == 2.0
    t2 = read(tmp_path / "wide", llm_pair(clean=0.95, by_round=(0.5, 0.9, 0.95, 0.95)), margin=0.1)
    assert attacked(t2, "recovery_time_accuracy")[0] == 1.0


def test_each_conversation_pairs_with_its_own_twin(tmp_path):
    rows = llm_pair(conv=0, clean=0.9, asr=0.1) + llm_pair(conv=1, clean=0.6, asr=0.1)
    t = read(tmp_path, rows)
    assert sorted(attacked(t, "nrp")) == pytest.approx([0.6 * 0.9, 0.9 * 0.9])


def test_a_missing_twin_refuses_and_names_the_unit(tmp_path):
    with pytest.raises(ValueError, match=r"no attack-free twin.*'arm': 'C'.*'seed': 3.*'conversation': 7"):
        read(tmp_path, llm_pair(seed=3, conv=7, twin=False))


def test_a_row_the_platform_already_paired_needs_no_twin(tmp_path):
    rows = llm_pair(twin=False)
    rows[0].update(accuracy_clean_reference=0.9, nrp=0.72, recovery_time_accuracy=1.0)
    t = read(tmp_path, rows)
    assert attacked(t, "nrp")[0] == pytest.approx(0.72)


def test_without_accuracy_by_round_the_recovery_time_is_undefined(tmp_path):
    rows = llm_pair()
    del rows[0]["accuracy_by_round"]
    t = read(tmp_path, rows)
    assert math.isnan(attacked(t, "recovery_time_accuracy")[0])
    assert attacked(t, "nrp")[0] == pytest.approx(0.95 * 0.8)


def test_insider_sink_episodes_are_excluded_from_the_harm_and_counted(tmp_path):
    rows = [r for r in synth.episodes({"C": lambda *a: 0.3, "BAZT": lambda *a: 0.1}, seeds=range(1, 7), sinks=range(2))
            if r["attack"] != "none"]
    for r in rows:
        r["sink_is_attacker"] = float(r["sink"] == 1)
        if r["sink"] == 1:
            r["protected_sink_harm"] = math.nan
    t, cfg = load(tmp_path, rows)
    from bazt_lab.features.analysis.private import hypotheses
    c = hypotheses.h_budget(t, cfg, "mock")
    assert c["status"] == "ok"
    from bazt_lab.features.analysis.private import report
    assert report.sink_attacker_excluded(t) == 12


def harness_rows(modes=("redaction", "llm_rewrite"), checks=("regex", "llm_judge"), twin_modes=None):
    """Research-profile LLM rows: one attacked row per ma_check and sanitize_mode, an attack-free twin per mode with
    ma_check None."""
    rows = []
    for mode in modes:
        for check in checks:
            rows.extend(dict(r, sanitize_mode=mode, ma_check=check)
                        for r in llm_pair(twin=False, asr=0.3 if mode == "redaction" else 0.1))
    for mode in (twin_modes if twin_modes is not None else modes):
        twin = llm_pair(clean=0.9 if mode == "redaction" else 0.8)[1]
        rows.append(dict(twin, sanitize_mode=mode, ma_check=None))
    return rows


def test_one_episode_repeated_per_sanitize_mode_and_ma_check_loads_as_separate_groups(tmp_path):
    t = read(tmp_path, harness_rows())
    assert {(k.sanitize_mode, k.ma_check) for k in t.index if not k.attack_free} == {
        (m, c) for m in ("redaction", "llm_rewrite") for c in ("regex", "llm_judge")}
    assert {(k.sanitize_mode, k.ma_check) for k in t.index if k.attack_free} == {("redaction", None),
                                                                                  ("llm_rewrite", None)}


def test_the_twin_matches_on_sanitize_mode_and_serves_every_ma_check(tmp_path):
    t = read(tmp_path, harness_rows())
    sel = [i for i in range(t.n) if t.vocab["attack"][t.codes["attack"][i]] != "none"]
    got = {(t.vocab["sanitize_mode"][t.codes["sanitize_mode"][i]], t.vocab["ma_check"][t.codes["ma_check"][i]]):
           t.metrics["accuracy_clean_reference"][i] for i in sel}
    assert got == {("redaction", "regex"): 0.9, ("redaction", "llm_judge"): 0.9,
                   ("llm_rewrite", "regex"): 0.8, ("llm_rewrite", "llm_judge"): 0.8}


def test_a_twin_of_another_sanitize_mode_is_no_twin(tmp_path):
    with pytest.raises(ValueError, match="no attack-free twin"):
        read(tmp_path, harness_rows(twin_modes=("redaction",)))


def test_comparisons_are_per_sanitize_mode_and_ma_check_and_use_the_attack_free_utility(tmp_path):
    from bazt_lab.features.analysis.private import hypotheses as hyp
    base = list(synth.episodes({"C": lambda *a: 0.3, "BAZT": lambda *a: 0.1}, platform="llm",
                                      block="llm_capped", cells=[(0.1, 0.1)], seeds=range(1, 6),
                                      conversations=range(4), tasks=("mmlu",), backbones=("gpt",)))
    rows = []
    for mode in ("redaction", "llm_rewrite"):
        for r in base:
            checks = ("regex", "llm_judge") if r["attack"] != "none" else (None,)
            rows.extend(dict(r, sanitize_mode=mode, ma_check=check) for check in checks)
    t, cfg = load(tmp_path, rows, platform="llm")
    out = hyp.h_budget(t, cfg, "llm")["challengers"]["BAZT"]
    assert out["summary"]["n_comparisons"] == 4 and out["summary"]["complete"] is True
    assert all(c["utility"] is not None for c in out["comparisons"])
    assert {(c["context"]["sanitize_mode"], c["context"]["ma_check"]) for c in out["comparisons"]} == {
        (m, c) for m in ("redaction", "llm_rewrite") for c in ("regex", "llm_judge")}
