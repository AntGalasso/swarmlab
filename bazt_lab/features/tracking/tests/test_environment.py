"""The two things this feature reads from the process environment: the W&B key and the kickoff time."""
from __future__ import annotations

import datetime as dt
import sys
import types

import pytest

from bazt_lab.features.tracking.adapters import env, kickoff
from bazt_lab.features.tracking.private.errors import TrackingError

NOT_A_KEY = "not-a-real-key-for-tests"
UTC = dt.UTC


def test_a_missing_key_is_a_clear_error_that_names_the_variable():
    with pytest.raises(TrackingError, match="WANDB_API_KEY") as error:
        env.require_api_key()
    assert "profiles.toml" in str(error.value) and ".env" in str(error.value)   # where it does not / does come from


def test_an_empty_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "")
    with pytest.raises(TrackingError, match="WANDB_API_KEY"):
        env.require_api_key()


def test_the_key_from_the_process_is_accepted(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", NOT_A_KEY)
    env.require_api_key()


def test_the_key_can_come_from_the_dotenv_file_and_the_process_wins_over_it(monkeypatch, tmp_path):
    dotenv = tmp_path / "present.env"
    dotenv.write_text("WANDB_API_KEY=from-the-file\n", encoding="utf-8")
    monkeypatch.setattr(env, "ENV_PATH", dotenv)
    env.require_api_key()
    assert env.redact("from-the-file") == "[redacted]"        # the key is now the one in the file
    monkeypatch.setenv("WANDB_API_KEY", NOT_A_KEY)
    env.load()
    assert env.redact(NOT_A_KEY) == "[redacted]" and env.redact("from-the-file") == "from-the-file"


def test_redact_removes_the_key_and_leaves_other_text(monkeypatch):
    assert env.redact("nothing to hide") == "nothing to hide"
    monkeypatch.setenv("WANDB_API_KEY", NOT_A_KEY)
    assert env.redact(f"401 for {NOT_A_KEY} at {NOT_A_KEY}") == "401 for [redacted] at [redacted]"


def test_the_kickoff_defaults_to_the_swarmlab_constant():
    assert kickoff.kickoff_time() == dt.datetime(2026, 11, 6, 20, 0, tzinfo=UTC)


def test_the_kickoff_is_overridden_by_the_environment(monkeypatch):
    monkeypatch.setenv("SWARMLAB_KICKOFF_UTC", "2026-11-07T08:30:00+01:00")
    assert kickoff.kickoff_time() == dt.datetime(2026, 11, 7, 7, 30, tzinfo=UTC)


def test_a_kickoff_without_a_zone_is_utc_as_in_swarmlab(monkeypatch):
    monkeypatch.setenv("SWARMLAB_KICKOFF_UTC", "2026-11-07T08:30:00")
    assert kickoff.kickoff_time() == dt.datetime(2026, 11, 7, 8, 30, tzinfo=UTC)


def test_the_kickoff_is_also_read_from_the_dotenv_file(monkeypatch, tmp_path):
    dotenv = tmp_path / "present.env"
    dotenv.write_text("SWARMLAB_KICKOFF_UTC=2030-01-01T00:00:00+00:00\n", encoding="utf-8")
    monkeypatch.setattr(env, "ENV_PATH", dotenv)
    assert kickoff.kickoff_time() == dt.datetime(2030, 1, 1, tzinfo=UTC)


def test_a_kickoff_that_is_not_a_time_is_an_error(monkeypatch):
    monkeypatch.setenv("SWARMLAB_KICKOFF_UTC", "soon")
    with pytest.raises(TrackingError, match="SWARMLAB_KICKOFF_UTC"):
        kickoff.kickoff_time()


def test_a_kickoff_that_swarmlab_cannot_give_is_a_clear_error_naming_the_variable(monkeypatch):
    monkeypatch.setitem(sys.modules, "swarmlab", None)                       # `import swarmlab` raises ImportError
    with pytest.raises(TrackingError, match="SWARMLAB_KICKOFF_UTC"):
        kickoff.kickoff_time()
    monkeypatch.setitem(sys.modules, "swarmlab", types.ModuleType("swarmlab"))   # a namespace-package lookalike
    with pytest.raises(TrackingError, match="SWARMLAB_KICKOFF_UTC"):
        kickoff.kickoff_time()
