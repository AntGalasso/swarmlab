"""Review-fix tests for stats_check.py and make_figures.py (H1, H2, M4, M5, m4, m5, m7)."""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.axes
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import make_figures
import stats_check

REAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "experiments")


def row(stage, scenario, arm, seed, asr, topo="barabasi_albert", n=12, audit=None, acc=0.5, exh=0):
    return {"platform": "mock", "split": "main", "stage": stage, "scenario": scenario, "topology": topo,
            "N": n, "arm": arm, "audit_pct": audit, "int_pct": audit, "seed": seed, "asr_final": asr,
            "accuracy": acc, "budget_exhausted": exh, "intervention_spend": 1.0}


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------------------------- H1
def test_h1_pair_arms_does_not_overwrite_across_stages():
    rows = []
    for seed in (1, 2, 3):
        rows += [row("mock_frontier", "infection", "arm_C", seed, 0.6),
                 row("mock_frontier", "infection", "bazt", seed, 0.2),
                 row("mock_topology_scale", "infection", "arm_C", seed, 0.5),
                 row("mock_topology_scale", "infection", "bazt", seed, 0.5)]
    pairs, incomplete = stats_check.pair_arms(rows)
    assert incomplete == 0
    assert len(pairs) == 6
    gains = sorted(round(p["a"]["asr_final"] - p["b"]["asr_final"], 6) for p in pairs)
    assert gains == [0.0] * 3 + [0.4] * 3


def test_h1_real_results_keep_frontier_rows():
    rows = stats_check.load_experiment_rows(REAL)
    pairs, _ = stats_check.pair_arms(rows)
    n_front = sum(1 for p in pairs if p["a"]["stage"] == "mock_frontier"
                  and p["a"]["scenario"] == "infection" and p["a"]["audit_pct"] is None
                  and p["a"]["topology"] == "barabasi_albert")
    assert n_front == 30         # frozen main split, seeds 1-30 (full suite committed in 32e92f4);
                                 # exact count still catches clobbering by other stages or reruns


# ---------------------------------------------------------------------------------------------- H2
def _did_rows():
    rows = []
    for seed in range(1, 6):
        e = 0.01 * seed
        st = "mock_frontier"
        rows += [row(st, "infection", "arm_C", seed, 0.2 + e), row(st, "infection", "bazt", seed, 0.1),
                 row(st, "adaptive", "arm_C", seed, 0.6 + e), row(st, "adaptive", "bazt", seed, 0.15 + 2 * e)]
        # other topology / stage rows that must never act as the "static" run
        rows += [row("mock_topology_scale", "infection", "arm_C", seed, 0.9, topo="watts_strogatz"),
                 row("mock_topology_scale", "infection", "bazt", seed, 0.9, topo="watts_strogatz"),
                 row("mock_adaptive", "infection", "arm_C", seed, 0.0),
                 row("mock_adaptive", "infection", "bazt", seed, 0.0)]
    return rows


def test_h2_did_compares_same_cell_not_topology():
    res = stats_check._did_adaptive(_did_rows())
    assert res["available"]
    exp = [0.4 - ((0.15 + 2 * 0.01 * s) - 0.1) for s in range(1, 6)]
    assert res["did_mean"] == pytest.approx(float(np.mean(exp)))
    assert res["n_seeds"] == 5


def test_h2_did_needs_both_runs_in_same_cell():
    rows = [r for r in _did_rows() if not (r["scenario"] == "adaptive" and r["seed"] == 1)]
    assert stats_check._did_adaptive(rows)["n_seeds"] == 4
    other = [r for r in _did_rows() if r["stage"] != "mock_frontier"]
    assert not stats_check._did_adaptive(other)["available"]


# ---------------------------------------------------------------------------------------------- M4
def test_m4_fig_exp3_separates_n(tmp_path, monkeypatch):
    rows = []
    for seed in (1, 2, 3):
        for n, c, z in ((12, 0.8, 0.4), (100, 0.2, 0.1)):
            rows += [row("mock_topology_scale", "infection", "arm_C", seed, c, n=n),
                     row("mock_topology_scale", "infection", "bazt", seed, z, n=n)]
    calls = []
    orig = matplotlib.axes.Axes.bar

    def spy(self, x, height, *a, **k):
        calls.append((list(np.atleast_1d(x)), list(np.atleast_1d(height))))
        return orig(self, x, height, *a, **k)
    monkeypatch.setattr(matplotlib.axes.Axes, "bar", spy)
    assert make_figures.fig_exp3(rows, str(tmp_path / "f.pdf"))
    xs, hs = calls[0]                    # first bar call = left panel, first arm
    assert len(set(round(x, 6) for x in xs)) == len(xs) == 2     # one bar per N, no shared x
    assert sorted(round(h, 3) for h in hs) == [0.2, 0.8]         # not pooled to 0.5


# ---------------------------------------------------------------------------------------------- M5
def test_m5_load_jsonl_skips_underscore_dirs(tmp_path):
    write_jsonl(str(tmp_path / "a.jsonl"), [row("mock_frontier", "clean", "arm_C", 1, 0.1)])
    write_jsonl(str(tmp_path / "_offline_llm_smoke" / "b.jsonl"), [row("llm_pilot", "clean", "arm_C", 9, 0.9)])
    write_jsonl(str(tmp_path / "paper_pack" / "c.jsonl"), [row("x", "clean", "arm_C", 8, 0.9)])
    assert [r["seed"] for r in make_figures._load_jsonl(str(tmp_path))] == [1]
    assert [r["seed"] for r in stats_check.load_experiment_rows(str(tmp_path))] == [1]


# ---------------------------------------------------------------------------------------------- m4
def test_m4_table2_markdown_has_no_latex_escape(tmp_path):
    rows = [row("mock_topology_scale", "infection", a, 1, 0.3, topo="watts_strogatz") for a in ("arm_C", "bazt")]
    make_figures.table2(rows, str(tmp_path))
    md = (tmp_path / "table2_ablations_and_scale.md").read_text()
    tex = (tmp_path / "table2_ablations_and_scale.tex").read_text()
    assert "\\_" not in md and "watts_strogatz" in md
    assert "watts\\_strogatz" in tex          # LaTeX keeps its escape


# ---------------------------------------------------------------------------------------------- m5
def test_m5_fig_exp1_error_bar_is_sem_of_paired_gain(tmp_path, monkeypatch):
    # arm_C varies across seeds but the paired gain is constant -> paired SEM is 0
    rows = []
    for seed, c in zip((1, 2, 3, 4), (0.2, 0.4, 0.6, 0.8)):
        rows += [row("mock_frontier", "infection", "arm_C", seed, c),
                 row("mock_frontier", "infection", "bazt", seed, c - 0.1)]
    got = []
    orig = matplotlib.axes.Axes.errorbar

    def spy(self, x, y, yerr=None, **k):
        got.append((list(y), list(yerr)))
        return orig(self, x, y, yerr=yerr, **k)
    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", spy)
    make_figures.fig_exp1(rows, str(tmp_path / "f.pdf"))
    ys, es = got[0]
    assert ys[0] == pytest.approx(10.0)
    assert es[0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------------------------- m7
def test_m7_mcnemar_key_renamed_and_old_key_readable():
    rep = stats_check.confirmatory_bazt(REAL, bootstrap_draws=200)
    assert "mcnemar_exhaustion" in rep and "mcnamara_exhaustion" not in rep
    old = {k: v for k, v in rep.items() if k != "mcnemar_exhaustion"}
    old["mcnamara_exhaustion"] = rep["mcnemar_exhaustion"]
    stats_check.print_confirmatory(old)       # old result files still print
# ------------------------------------------------------------------------- extra: _topology_scale cell keying
def test_n1_topology_scale_ignores_other_stages_and_scenarios():
    rows = []
    for seed in (1, 2, 3):
        e = 0.01 * seed
        for topo in ("barabasi_albert", "watts_strogatz"):
            for n in (12, 100):
                rows += [row("mock_topology_scale", "infection", "arm_C", seed, 0.5 + e, topo=topo, n=n),
                         row("mock_topology_scale", "infection", "bazt", seed, 0.4 + e, topo=topo, n=n)]
        # same topology/N/seed in other stages and scenarios, with a very different gain
        rows += [row("mock_frontier", "clean", "arm_C", seed, 0.9, topo="watts_strogatz"),
                 row("mock_frontier", "clean", "bazt", seed, 0.0, topo="watts_strogatz")]
    rows = rows[-2:] + rows[:-2]            # frontier rows first: a first-wins key would take them
    res = stats_check._topology_scale(rows)
    assert res["watts_strogatz"]["mean_gain_pp"] == pytest.approx(10.0)
