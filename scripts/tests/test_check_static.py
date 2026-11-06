"""scripts/check_static.py: ruff, compileall and the import gate, run in turn from the repository root."""
from __future__ import annotations

import sys

import check_static as static


def test_the_default_steps_are_ruff_compileall_and_the_import_gate():
    steps = static.steps(fix=False, only=None)
    assert [s.name for s in steps] == ["ruff", "compileall", "import gate"]
    ruff, compile_all, gate = (s.argv for s in steps)
    assert ruff[:4] == [sys.executable, "-m", "ruff", "check"] and "--fix" not in ruff
    assert compile_all[:3] == [sys.executable, "-m", "compileall"] and "bazt_lab" in compile_all
    assert gate[-1] == "bazt_lab.tools.check_architecture"


def test_ruff_gets_no_path_so_that_ruff_toml_decides_what_is_linted():
    (ruff,) = static.steps(fix=False, only="ruff")
    assert ruff.argv == [sys.executable, "-m", "ruff", "check"]


def test_fix_is_passed_to_ruff_only():
    steps = {s.name: s.argv for s in static.steps(fix=True, only=None)}
    assert "--fix" in steps["ruff"] and "--fix" not in steps["compileall"] and "--fix" not in steps["import gate"]


def test_only_selects_one_step_and_an_unknown_name_is_refused():
    assert [s.name for s in static.steps(fix=False, only="import gate")] == ["import gate"]
    try:
        static.steps(fix=False, only="mypy")
    except ValueError as error:
        assert "mypy" in str(error) and "ruff" in str(error)
    else:
        raise AssertionError("an unknown step must be refused")


def test_every_step_runs_even_after_a_failure_and_the_exit_code_is_1(capsys):
    steps = [static.Step("passes", [sys.executable, "-c", "print('fine')"]),
             static.Step("fails", [sys.executable, "-c", "import sys; print('broken'); sys.exit(3)"]),
             static.Step("also passes", [sys.executable, "-c", "pass"])]
    assert static.run(steps) == 1
    out = capsys.readouterr().out
    assert "fine" in out and "broken" in out
    assert "ok    passes" in out and "FAIL  fails (exit 3)" in out and "ok    also passes" in out


def test_all_steps_passing_exits_0(capsys):
    assert static.run([static.Step("one", [sys.executable, "-c", "pass"])]) == 0
    assert "static checks: 0 failed" in capsys.readouterr().out


def test_a_missing_program_is_a_failure_not_a_crash(capsys):
    assert static.run([static.Step("ghost", ["definitely-not-a-program-xyz"])]) == 1
    assert "FAIL  ghost" in capsys.readouterr().out


def test_compileall_step_passes_on_the_repository(capsys):
    assert static.main(["--only", "compileall"]) == 0, capsys.readouterr().out
