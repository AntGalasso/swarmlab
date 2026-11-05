"""scripts/check_env.py: pins read from pyproject.toml, installed versions, the lock file, torch, and secrets."""
from __future__ import annotations

import tomllib
from pathlib import Path

import check_env as env
import pytest

REPO = Path(__file__).resolve().parents[2]
SECRET = "hf_ThisIsAFakeTokenValueThatMustNeverBePrinted0123"

PYPROJECT = """
[project]
name = "x"
requires-python = "==3.11.*"
dependencies = ["torch==2.14.0", "python-dotenv==1.2.2", "numpy==2.4.6"]
[dependency-groups]
dev = ["pytest==9.1.1"]
"""
LOCK = """
version = 1
[[package]]
name = "torch"
version = "2.14.0"
[[package]]
name = "torch"
version = "2.14.0+cpu"
[[package]]
name = "python-dotenv"
version = "1.2.2"
[[package]]
name = "numpy"
version = "2.4.6"
[[package]]
name = "pytest"
version = "9.1.1"
"""


def statuses(results) -> set[str]:
    return {r.status for r in results}


# ------------------------------------------------------------------------------------------------- the pins
def test_pins_are_read_from_dependencies_and_the_dev_group():
    pins, problems = env.parse_pins(tomllib.loads(PYPROJECT))
    assert pins == {"torch": "2.14.0", "python-dotenv": "1.2.2", "numpy": "2.4.6", "pytest": "9.1.1"}
    assert problems == []


@pytest.mark.parametrize("spec", ["numpy>=2.4", "numpy", "numpy~=2.4.6", "numpy==2.*", "numpy==2.4.6,<3"])
def test_a_dependency_that_is_not_an_exact_pin_is_a_problem(spec):
    pins, problems = env.parse_pins({"project": {"dependencies": [spec]}})
    assert "numpy" not in pins and len(problems) == 1 and spec in problems[0]


def test_pin_names_are_normalised():
    pins, _ = env.parse_pins({"project": {"dependencies": ["Python_Dotenv==1.2.2", "torch_geometric==2.8.1"]}})
    assert pins == {"python-dotenv": "1.2.2", "torch-geometric": "2.8.1"}


def test_the_repository_pins_everything_exactly():
    pins, problems = env.parse_pins(tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8")))
    assert problems == []
    assert {"torch", "numpy", "wandb", "pytest", "ruff"} <= set(pins)


# ---------------------------------------------------------------------------------------------- the versions
@pytest.mark.parametrize("installed, pin, same", [
    ("2.14.0", "2.14.0", True), ("2.14.0+cpu", "2.14.0", True), ("2.14.1", "2.14.0", False),
    ("2.14.0rc1", "2.14.0", False), ("2.14", "2.14.0", False),
])
def test_a_local_version_label_does_not_change_the_release(installed, pin, same):
    assert env.same_release(installed, pin) is same


def test_installed_packages_must_equal_their_pins():
    installed = {"torch": "2.14.0+cpu", "python-dotenv": "1.2.2", "numpy": "2.4.5"}.get
    results = env.check_packages({"torch": "2.14.0", "python-dotenv": "1.2.2", "numpy": "2.4.6", "wandb": "0.31.0"},
                                 installed)
    by_name = {r.name: r for r in results}
    assert by_name["torch"].status == by_name["python-dotenv"].status == "ok"
    assert by_name["numpy"].status == "fail" and "2.4.5" in by_name["numpy"].detail and "2.4.6" in by_name["numpy"].detail
    assert by_name["wandb"].status == "fail" and "not installed" in by_name["wandb"].detail


def test_python_must_be_3_11():
    assert env.check_python((3, 11, 15)).status == "ok"
    assert env.check_python((3, 12, 0)).status == "fail"
    assert env.check_python((3, 10, 9)).status == "fail"


# ------------------------------------------------------------------------------------------------ the lock
def test_the_lock_must_agree_with_the_pins():
    pins, _ = env.parse_pins(tomllib.loads(PYPROJECT))
    assert statuses(env.check_lock(pins, env.lock_versions(tomllib.loads(LOCK)))) == {"ok"}
    stale = tomllib.loads(LOCK.replace('version = "2.4.6"', 'version = "2.4.5"'))
    results = env.check_lock(pins, env.lock_versions(stale))
    assert [r.name for r in results if r.status == "fail"] == ["uv.lock numpy"]
    missing = tomllib.loads(LOCK.replace('name = "numpy"', 'name = "scipy"'))
    assert [r.name for r in env.check_lock(pins, env.lock_versions(missing)) if r.status == "fail"] == ["uv.lock numpy"]


def test_the_repository_lock_agrees_with_the_repository_pins():
    pins, _ = env.parse_pins(tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8")))
    lock = env.lock_versions(tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8")))
    failed = [r for r in env.check_lock(pins, lock) if r.status == "fail"]
    assert failed == [], failed


# ------------------------------------------------------------------------------------------------- torch
def test_torch_runs_on_cpu_with_a_known_thread_count():
    results = env.check_torch(expected_threads=None)
    assert statuses(results) == {"ok"}, results
    assert any("threads" in r.name for r in results)


def test_a_thread_count_that_differs_from_the_requested_one_fails():
    import torch
    assert statuses(env.check_torch(expected_threads=torch.get_num_threads())) == {"ok"}
    bad = env.check_torch(expected_threads=torch.get_num_threads() + 1)
    assert "fail" in statuses(bad)


# ----------------------------------------------------------------------------------------------- secrets
def test_secrets_are_reported_as_present_or_missing_and_never_by_value():
    results = env.check_secrets({"HF_TOKEN": SECRET, "WANDB_API_KEY": ""}, require=False)
    by_name = {r.name: r for r in results}
    assert by_name["HF_TOKEN"].status == "ok"
    assert by_name["WANDB_API_KEY"].status == by_name["OPENROUTER_API_KEY"].status == "warn"
    assert SECRET not in " ".join(f"{r.name} {r.detail}" for r in results)


def test_missing_secrets_fail_only_when_required():
    assert statuses(env.check_secrets({}, require=False)) == {"warn"}
    assert statuses(env.check_secrets({}, require=True)) == {"fail"}
    full = {"HF_TOKEN": "a", "WANDB_API_KEY": "b", "OPENROUTER_API_KEY": "c"}
    assert statuses(env.check_secrets(full, require=True)) == {"ok"}


def test_the_dot_env_file_fills_gaps_but_never_overrides_the_environment(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"HF_TOKEN={SECRET}\nWANDB_API_KEY=from-file\n", encoding="utf-8")
    merged = env.read_environment(dotenv, {"WANDB_API_KEY": "from-environment"})
    assert merged["HF_TOKEN"] == SECRET and merged["WANDB_API_KEY"] == "from-environment"
    assert env.read_environment(tmp_path / "missing.env", {"A": "1"}) == {"A": "1"}


# --------------------------------------------------------------------------------------------- the command
def project(tmp_path: Path, pyproject: str = PYPROJECT, lock: str = LOCK) -> Path:
    (tmp_path / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    (tmp_path / "uv.lock").write_text(lock, encoding="utf-8")
    return tmp_path


def test_a_wrong_pin_ends_with_exit_code_1(tmp_path, capsys):
    root = project(tmp_path, PYPROJECT.replace("numpy==2.4.6", "numpy==0.0.1"), LOCK.replace("2.4.6", "0.0.1"))
    assert env.main(["--root", str(root), "--env-file", str(tmp_path / "none.env")]) == 1
    out = capsys.readouterr().out
    assert "[FAIL] numpy" in out and "0.0.1" in out and "failure" in out


def test_the_current_environment_passes_and_prints_no_secret_value(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", SECRET)
    code = env.main(["--env-file", str(tmp_path / "none.env")])
    out = capsys.readouterr().out
    assert code == 0, out
    assert SECRET not in out and "HF_TOKEN" in out


def test_require_secrets_fails_when_one_is_missing(tmp_path, monkeypatch, capsys):
    for name in env.SECRETS:
        monkeypatch.delenv(name, raising=False)
    assert env.main(["--env-file", str(tmp_path / "none.env")]) == 0
    assert env.main(["--env-file", str(tmp_path / "none.env"), "--require-secrets"]) == 1
    assert "[FAIL] HF_TOKEN" in capsys.readouterr().out
