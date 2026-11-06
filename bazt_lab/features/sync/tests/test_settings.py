"""Reading the sync.* settings: every key required, nothing defaulted, every bad value refused by name."""
from __future__ import annotations

import pytest

from bazt_lab.core import public as core
from bazt_lab.features.sync.private.errors import SyncError
from bazt_lab.features.sync.private.settings import SyncSettings, read


def settings(**sync):
    base = {"enabled": True, "repo_id": "me/bazt-ckpt", "repo_type": "dataset", "private": True, "every_minutes": 10,
            "include": [], "exclude": [".launch.lock"], "priority": "below_normal", "final_push_timeout_seconds": 600}
    return {"sync": {**base, **sync}}


def test_read_returns_the_resolved_values():
    assert read(settings()) == SyncSettings(
        repo_id="me/bazt-ckpt", repo_type="dataset", private=True, every_minutes=10.0, include=(),
        exclude=(".launch.lock",), priority="below_normal", final_push_timeout_seconds=600.0)


def test_enabled_is_read_first_and_the_rest_is_not_needed_when_off():
    assert read({"sync": {"enabled": False}}) is None


@pytest.mark.parametrize("key", ["enabled", "repo_id", "repo_type", "private", "every_minutes", "include", "exclude",
                                 "priority", "final_push_timeout_seconds"])
def test_a_missing_key_raises_core_unset_naming_it(key):
    s = settings()
    del s["sync"][key]
    with pytest.raises(core.Unset) as error:
        read(s)
    assert error.value.args[0] == f"sync.{key}"


@pytest.mark.parametrize("key, value", [("enabled", "yes"), ("repo_id", ""), ("repo_id", 3), ("repo_type", "bucket"),
                                        ("private", "true"), ("every_minutes", 0), ("every_minutes", -1),
                                        ("every_minutes", True), ("every_minutes", "10"), ("include", "*.json"),
                                        ("exclude", [1]), ("priority", "high"), ("final_push_timeout_seconds", 0)])
def test_a_bad_value_is_refused_and_names_the_key(key, value):
    with pytest.raises(SyncError, match=rf"sync\.{key}"):
        read(settings(**{key: value}))


@pytest.mark.parametrize("key", ["every_minutes", "final_push_timeout_seconds"])
@pytest.mark.parametrize("value", [float("inf"), float("nan"), float("-inf")])
def test_a_number_that_is_not_finite_is_refused(key, value):
    with pytest.raises(SyncError, match=rf"sync\.{key}"):
        read(settings(**{key: value}))
