"""run_search: progressive checkpoints, resume after a crash, settings fingerprint, refusals, tracking."""
import json

import pytest

from bazt_lab.core import public as core
from bazt_lab.features.adaptive.private import run as rn
from bazt_lab.features.adaptive.tests.fakes import FakePlatform, arm_of, drop, make_settings, unit, units_of

N = 6
PER_CELL = (1 + N * 4) * 3 * N              # candidates x search pairs of one gray-box cell


def asr(arm, u):
    p = u["attack_params"]
    return 0.9 if p == {"entry_ranks": [2], "start_frac": 0.2} else 0.3 if p == {} else 0.1


SETTINGS = make_settings({"mock.batch_size": 40})


def cells(*audits):
    return [u for a in audits for u in units_of(unit(agents=N, audit=a), (1, 2), range(N))]


def run(tmp_path, units, platform=None, over=None, **kw):
    platform = platform or FakePlatform(asr)
    return rn.run_search(units, arm_of, platform, make_settings({"mock.batch_size": 40, **(over or {})}), tmp_path, **kw), \
        platform


def progress_files(tmp_path):
    return list((tmp_path / rn.PROGRESS_DIR).glob("*.jsonl")) if (tmp_path / rn.PROGRESS_DIR).exists() else []


# ---------------------------------------------------------------------------------------------------- output
def test_the_table_is_saved_and_loads_back(tmp_path):
    units = cells(0.1, 0.2)
    table, _ = run(tmp_path, units)
    loaded = rn.load_best_responses(tmp_path, SETTINGS)
    assert len(loaded) == len(table) == 2
    for u in units:
        assert loaded.params_for(u) == table.params_for(u) == {"entry_ranks": [2], "start_frac": 0.2}


def test_a_finished_run_leaves_no_progress_files(tmp_path):
    run(tmp_path, cells(0.1))
    assert progress_files(tmp_path) == []


def test_a_finished_cell_is_not_searched_again(tmp_path):
    units = cells(0.1, 0.2)
    run(tmp_path, units)
    table, platform = run(tmp_path, units)
    assert platform.calls == [] and len(table) == 2


def test_the_attack_free_family_runs_nothing(tmp_path):
    table, platform = run(tmp_path, units_of(unit(agents=N, attack="none"), (1, 2), range(N)))
    assert platform.calls == [] and len(table) == 0


def test_two_runs_write_identical_tables(tmp_path):
    for k in (0, 1):
        run(tmp_path / str(k), cells(0.1, 0.2))
    def files(k):
        return {p.name: p.read_text(encoding="utf-8") for p in (tmp_path / str(k) / rn.TABLE_DIR).iterdir()}
    assert files(0) == files(1)


# ------------------------------------------------------------------------------------------------ resume
def total_submitted(*platforms):
    return sum(p.submitted for p in platforms)


def test_a_crash_inside_a_cell_resumes_after_the_last_saved_chunk(tmp_path):
    units = cells(0.1)
    crashing = FakePlatform(asr, fail_after_calls=3)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run(tmp_path, units, crashing)
    assert crashing.submitted == 3 * 18 and len(progress_files(tmp_path)) == 1
    assert not (tmp_path / rn.TABLE_DIR).exists() or len(rn.load_best_responses(tmp_path, SETTINGS)) == 0
    table, resumed = run(tmp_path, units)
    assert total_submitted(crashing, resumed) == PER_CELL
    clean, _ = run(tmp_path / "clean", units)
    assert table.params_for(units[0]) == clean.params_for(units[0]) and table.choice(units[0]) == clean.choice(units[0])
    assert progress_files(tmp_path) == []


def test_a_crash_between_cells_keeps_the_finished_cells(tmp_path):
    units = cells(0.1, 0.2, 0.3)
    crashing = FakePlatform(asr, fail_after_calls=2 * 25 + 2)           # 25 chunks per cell: one per candidate (18 episodes)
    with pytest.raises(RuntimeError):
        run(tmp_path, units, crashing)
    assert len(rn.load_best_responses(tmp_path, SETTINGS)) == 2
    table, resumed = run(tmp_path, units)
    assert {u["audit"] for c in resumed.calls for u in c["units"]} == {0.3}
    assert total_submitted(crashing, resumed) == 3 * PER_CELL and len(table) == 3


def test_a_torn_last_progress_line_is_dropped(tmp_path):
    units = cells(0.1)
    with pytest.raises(RuntimeError):
        run(tmp_path, units, FakePlatform(asr, fail_after_calls=3))
    (path,) = progress_files(tmp_path)
    with path.open("a", encoding="utf-8") as f:
        f.write('{"f": [1, 2')
    table, resumed = run(tmp_path, units)
    assert total_submitted(resumed) == PER_CELL - 3 * 18 and table.params_for(units[0]) == \
        {"entry_ranks": [2], "start_frac": 0.2}


def test_progress_that_does_not_fit_the_plan_is_refused(tmp_path):
    units = cells(0.1)
    with pytest.raises(RuntimeError):
        run(tmp_path, units, FakePlatform(asr, fail_after_calls=2))
    (path,) = progress_files(tmp_path)
    path.write_text(json.dumps({"f": [PER_CELL + 5], "asr": [0.5]}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="progress"):
        run(tmp_path, units)


# ---------------------------------------------------------------------------------------------- refusals
def test_results_of_other_settings_are_not_mixed_in(tmp_path):
    run(tmp_path, cells(0.1))
    with pytest.raises(ValueError, match="other settings"):
        run(tmp_path, cells(0.1), over={"mock.adaptive.evasion_cost": 0.5})


def test_batch_size_does_not_change_the_fingerprint(tmp_path):
    run(tmp_path, cells(0.1))
    run(tmp_path, cells(0.1), over={"mock.batch_size": 3})


def test_an_unset_batch_size_is_refused_by_name(tmp_path):
    s = drop(make_settings(), "mock.batch_size")
    with pytest.raises(core.Unset) as e:
        rn.run_search(cells(0.1), arm_of, FakePlatform(asr), s, tmp_path)
    assert e.value.args[0] == "mock.batch_size"


@pytest.mark.parametrize("bad", [0, -3, 2.5, "64", True])
def test_an_invalid_batch_size_is_refused(tmp_path, bad):
    with pytest.raises(ValueError, match=r"mock\.batch_size"):
        run(tmp_path, cells(0.1), over={"mock.batch_size": bad})


def test_nothing_is_written_when_planning_refuses(tmp_path):
    with pytest.raises(ValueError, match="max_candidates"):
        run(tmp_path, cells(0.1), over={"mock.adaptive.max_candidates": 5})
    assert not any(tmp_path.iterdir())


def test_an_unknown_arm_is_refused(tmp_path):
    def arm_of_none(name):
        raise LookupError(f"unknown arm {name!r}")
    with pytest.raises(LookupError, match="unknown arm"):
        rn.run_search(cells(0.1), arm_of_none, FakePlatform(asr), make_settings({"mock.batch_size": 40}), tmp_path)


# ---------------------------------------------------------------------------------------------- tracking
class Recorder:
    def __init__(self):
        self.logged = []

    def log(self, metrics, step):
        self.logged.append((dict(metrics), step))

    def start(self, *a):
        raise AssertionError("the search does not own the tracking run")

    finish = log_rows = artifact = start


def test_progress_is_logged_to_the_tracker_per_chunk(tmp_path):
    rec = Recorder()
    _, platform = run(tmp_path, cells(0.1), tracker=rec)
    assert len(rec.logged) == len(platform.calls)
    last, _ = rec.logged[-1]
    assert last["adaptive/episodes_done"] == PER_CELL and last["adaptive/episodes_total"] == PER_CELL


def test_recalibrated_thresholds_do_not_reuse_a_finished_cell(tmp_path):
    run(tmp_path, cells(0.1), thresholds_of=lambda cell: {"mon": 1.0})
    run(tmp_path, cells(0.1), thresholds_of=lambda cell: {"mon": 1.0})
    with pytest.raises(ValueError, match="other thresholds"):
        run(tmp_path, cells(0.1), thresholds_of=lambda cell: {"mon": 1.1})


def fp(over=None):
    return rn.fingerprint(make_settings(over))


def test_the_fingerprint_changes_with_every_setting_that_changes_results():
    base = fp()
    for over in ({"mock.adaptive.evasion_cost": 0.5}, {"mock.adaptive.rates": [1.0, 0.5]}, {"runtime.n_insiders": 2},
                 {"mock.adaptive.search_seeds": [8001, 8004]}, {"mock.adaptive.knowledge.white_box": "seed"},
                 {"mock.adaptive.start_fractions": [0.0]}, {"mock.adaptive.detectabilities": [1.0, 0.5]},
                 {"mock.adaptive.insider_sets": "all"}, {"mock.every_sink": False},
                 # what the platform reads: a stale best response must not survive a changed model or auditor
                 {"mock.model.beta": 0.4}, {"runtime.insider_bias": 3.0}, {"auditor_presets.good_classifier": [0.9, 0.1, 0.2]},
                 {"costs.main.sanitize": 2}, {"runtime.rate_limit_per_identity": 1}):
        assert fp(over) != base, over


def test_the_fingerprint_ignores_what_cannot_change_an_episode():
    base = fp()
    for over in ({"mock.adaptive.max_candidates": 99}, {"mock.batch_size": 3}, {"mock.main_seeds": [1, 20]},
                 {"mock.heldout_seeds": [31, 50]}, {"mock.adaptive.arms": ["C"]}, {"mock.adaptive.cells": [[10, 10]]},
                 {"mock.adaptive.conditions": ["gray_box"]}, {"mock.legacy.arms": ["fixed_early"]},
                 {"mock.ablations.cells": [[1, 1]]}, {"llm.agents": 8}, {"gnn_train.agents": 4},
                 {"tracking.mode": "online"}, {"estimates.calls_per_second": 9}, {"profile": "research"},
                 {"runtime.llm_seeds": [1, 2, 3, 4, 5]}, {"runtime.ni_margin_accuracy": 0.01}):
        assert fp(over) == base, over


def test_a_table_of_other_settings_is_refused_when_loaded_for_the_evaluation(tmp_path):
    run(tmp_path, cells(0.1))
    assert len(rn.load_best_responses(tmp_path, SETTINGS)) == 1
    with pytest.raises(ValueError, match="other settings"):
        rn.load_best_responses(tmp_path, make_settings({"mock.model.beta": 0.4}))
