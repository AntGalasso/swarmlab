"""scripts/check_gpu.py: the comparison rules, the combinations, and the CPU part end to end (CUDA is absent here)."""
from __future__ import annotations

import math

import check_gpu as gpu
import pytest
import torch

NAN = math.nan


def row(seed=1, **changes) -> dict:
    base = {"arm": "B", "audit": 0.1, "intervention": 0.1, "topology": "chain", "attack": "infection", "agents": 4,
            "seed": seed, "sink": 0, "spend_audit_total": 3.0, "spend_intervention_total": 2.0, "spend_total": 5.0,
            "violations": 0, "exhausted_step": 0, "asr_final": 0.25, "accuracy_attack": 0.75, "nrp": 0.5625,
            "asr_by_round": [0.0, 0.25], "time_to_containment": NAN, "latency": 1.0}
    base.update(changes)
    return base


def rows(n=200, **changes) -> list:
    return [row(seed=i, **changes) for i in range(n)]


# ----------------------------------------------------------------------------------------------- comparison
def test_float64_combinations_must_be_identical_in_every_column_but_the_timings():
    a = rows(5)
    b = [{**r, "latency": 9.0} for r in rows(5)]
    assert gpu.compare_rows(a, b, exact=True) == []
    b[2]["asr_final"] = 0.2500001
    (problem,) = gpu.compare_rows(a, b, exact=True)
    assert "asr_final" in problem
    assert gpu.compare_rows(rows(3), rows(2), exact=True)[0].startswith("episode")


def test_float32_may_move_a_number_by_the_tolerance_in_a_few_episodes_and_nothing_else():
    a = rows()
    b = rows()
    b[3]["asr_final"] = 0.25 + gpu.TOLERANCE / 2
    assert gpu.compare_rows(a, b, exact=False) == []
    for i in range(5):                                                   # 5 of 200 episodes beyond the tolerance
        b[i]["accuracy_attack"] = 0.5
    assert any("tolerance" in p for p in gpu.compare_rows(a, b, exact=False))


def test_float32_budget_accounting_must_be_identical_for_99_percent_of_the_episodes():
    a = rows()
    b = rows()
    b[0]["spend_total"] = 5.5
    b[1]["violations"] = 1
    assert gpu.compare_rows(a, b, exact=False) == []
    b[2]["exhausted_step"] = 3
    assert any("accounting" in p for p in gpu.compare_rows(a, b, exact=False))


def test_dicts_such_as_calibrated_thresholds_are_compared_by_key_and_value():
    a = rows(3, thresholds={"judge": 1.25})
    assert gpu.compare_rows(a, rows(3, thresholds={"judge": 1.2500001}), exact=False) == []
    assert gpu.compare_rows(a, rows(3, thresholds={"judge": 1.2500001}), exact=True)
    assert gpu.compare_rows(a, rows(3, thresholds={"other": 1.25}), exact=False)


def test_the_runs_own_provenance_columns_are_not_results():
    a = rows(5, meta_dtype="float64", meta_threads=8)
    b = rows(5, meta_dtype="float32", meta_threads=4)
    assert gpu.compare_rows(a, b, exact=False) == [] and gpu.compare_rows(a, b, exact=True) == []


def test_nan_equals_nan_and_lists_are_compared_elementwise():
    assert gpu.numbers_close(NAN, NAN, 0.0)
    assert gpu.numbers_close([0.0, 0.25], [0.0, 0.2504], gpu.TOLERANCE)
    assert not gpu.numbers_close([0.0, 0.25], [0.0, 0.26], gpu.TOLERANCE)
    assert not gpu.numbers_close([0.0], [0.0, 0.1], 0.0)


def test_rows_of_one_episode_are_matched_in_order():
    a = [row(seed=1, asr_final=0.1), row(seed=1, asr_final=0.2)]
    b = [row(seed=1, asr_final=0.1), row(seed=1, asr_final=0.9)]
    assert len(gpu.compare_rows(a, b, exact=True)) == 1


# ------------------------------------------------------------------------------------------ combinations
def test_the_reference_comes_first_and_every_combination_is_listed_once():
    every = gpu.combos(["cpu"])
    assert every[0] == gpu.Combo("cpu", "float64", False)
    assert len(every) == len(set(every)) == 4
    both = gpu.combos(["cpu", "cuda"])
    assert both[0] == gpu.Combo("cpu", "float64", False) and len(both) == 8
    assert [c.exact for c in both].count(True) == 4


def test_compile_on_is_listed_only_where_inductor_can_build_kernels():
    cpu_only = gpu.combos(["cpu", "cuda"], can_compile=lambda device: device == "cuda")
    assert gpu.Combo("cpu", "float64", True) not in cpu_only and gpu.Combo("cuda", "float32", True) in cpu_only
    assert len(cpu_only) == 6 and cpu_only[0] == gpu.Combo("cpu", "float64", False)
    assert len(gpu.combos(["cpu"], can_compile=lambda device: False)) == 2


def test_cuda_is_skipped_when_this_machine_has_none():
    usable, skipped = gpu.available_devices(["cpu", "cuda", "cuda:1"])
    if torch.cuda.is_available():
        assert usable == ["cpu", "cuda", "cuda:1"] and skipped == []
    else:
        assert usable == ["cpu"] and skipped == ["cuda", "cuda:1"]


def test_the_facts_name_torch_and_the_cuda_state():
    facts = "\n".join(gpu.torch_facts())
    assert f"torch {torch.__version__}" in facts
    assert ("CUDA unavailable" in facts) or ("compute capability" in facts)


def test_the_slice_sets_pass_the_combination_to_the_launcher_as_compute_settings():
    sets = gpu.slice_sets(5, 2, gpu.Combo("cuda", "float32", True))
    assert (sets["compute.device"], sets["compute.dtype"], sets["compute.compile"]) == ("cuda", "float32", True)
    assert sets["mock.topologies"] == ["chain", "small_world"]
    assert {"A", "B", "C", "BAZT"} <= set(sets["mock.uncapped_arms"])


# -------------------------------------------------------------------------------------------- end to end
def test_a_mismatch_between_combinations_ends_the_report_with_exit_code_1(capsys):
    reference = gpu.Outcome(gpu.Combo("cpu", "float64", False), 10, 1.0, rows=rows(10))
    same = gpu.Outcome(gpu.Combo("cpu", "float64", True), 10, 0.5, rows=rows(10))
    assert gpu.report([reference, same]) == 0
    broken = gpu.Outcome(gpu.Combo("cpu", "float64", True), 10, 0.5, rows=rows(10, asr_final=0.3))
    assert gpu.report([reference, broken]) == 1
    assert "MISMATCH" in capsys.readouterr().out
    failed = gpu.Outcome(gpu.Combo("cpu", "float32", False), error="RuntimeError: boom")
    assert gpu.report([reference, failed]) == 1


@pytest.mark.skipif(torch.cuda.is_available(), reason="this test is about a machine without CUDA")
def test_the_cpu_part_runs_on_a_machine_without_cuda_and_reports_it(capsys):
    code = gpu.main(["--quick", "--devices", "cpu,cuda"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "CUDA unavailable" in out and "skipped: cuda" in out
    assert out.count("episodes/s") == 1 and "reference" in out
    from bazt_lab.core import public as core
    table = [line for line in out.splitlines() if line.startswith("cpu ") and "float" in line]
    assert any("compile on" in line for line in table) == core.inductor_available("cpu")
    assert len(table) == (4 if core.inductor_available("cpu") else 2)
    assert "MISMATCH" not in out and "float32" in out
    for combination in ("float64", "float32"):
        assert combination in out
    assert "gpu check: 0 combination(s) failed" in out


def test_a_combination_that_read_fewer_rows_than_it_planned_is_a_failure(capsys):
    """compare_rows([], []) finds nothing to differ: a wrong rows path must not pass as 'identical'."""
    reference = gpu.Outcome(gpu.Combo("cpu", "float64", False), 10, 1.0, rows=rows(10))
    short = gpu.Outcome(gpu.Combo("cpu", "float64", True), 10, 0.5, rows=rows(9))
    assert gpu.report([reference, short]) == 1
    assert "read 9 rows of 10" in capsys.readouterr().out
    empty_reference = gpu.Outcome(gpu.Combo("cpu", "float64", False), 10, 1.0, rows=[])
    empty_other = gpu.Outcome(gpu.Combo("cpu", "float32", False), 10, 1.0, rows=[])
    assert gpu.report([empty_reference, empty_other]) == 1
    out = capsys.readouterr().out
    assert "read 0 rows of 10" in out and "identical" not in out and "within the float32" not in out


def test_the_table_shows_the_rows_each_combination_read(capsys):
    reference = gpu.Outcome(gpu.Combo("cpu", "float64", False), 10, 1.0, rows=rows(10))
    assert gpu.report([reference]) == 0
    out = capsys.readouterr().out
    header = next(line for line in out.splitlines() if line.startswith("combination"))
    assert "rows" in header.split()


def test_require_cuda_fails_at_once_on_a_machine_without_cuda(capsys, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(gpu, "run_slice", lambda *a: pytest.fail("the slice must not run"))
    assert gpu.main(["--quick", "--require-cuda"]) != 0
    assert "--require-cuda" in capsys.readouterr().out


def test_the_slices_run_in_the_offline_environment_of_the_e2e_children_and_restore_the_callers(monkeypatch):
    """The block runner runs in this process: no real secret may reach it, and HF and W&B stay offline, as in
    e2e_common.child_env. The caller's environment is back when main returns."""
    import os
    monkeypatch.setenv("HF_TOKEN", "real-secret")
    monkeypatch.setenv("WANDB_API_KEY", "real-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "real-secret")
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    seen = []

    def fake_slice(combo, agents, seeds, root):
        seen.append({k: os.environ.get(k) for k in ("HF_TOKEN", "WANDB_API_KEY", "OPENROUTER_API_KEY", "WANDB_MODE",
                                                    "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")})
        return gpu.Outcome(combo, 4, 1.0, rows=rows(4))

    monkeypatch.setattr(gpu, "run_slice", fake_slice)
    monkeypatch.setattr(gpu, "inductor_usable", lambda device: False)
    assert gpu.main(["--quick", "--devices", "cpu"]) == 0
    assert seen and all(s == {"HF_TOKEN": "", "WANDB_API_KEY": "", "OPENROUTER_API_KEY": "", "WANDB_MODE": "offline",
                              "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"} for s in seen), seen
    assert os.environ["HF_TOKEN"] == "real-secret" and os.environ["WANDB_MODE"] == "online"
    assert "HF_HUB_OFFLINE" not in os.environ
