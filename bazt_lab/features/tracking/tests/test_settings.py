"""Reading tracking.* settings and the offline-before-kickoff rule (D87)."""
from __future__ import annotations

import datetime as dt

import pytest

from bazt_lab.core import public as core
from bazt_lab.features.tracking.private.errors import TrackingError
from bazt_lab.features.tracking.private.settings import TrackingSettings, effective_mode, read

KICKOFF = dt.datetime(2026, 11, 6, 20, 0, tzinfo=dt.UTC)
SECOND = dt.timedelta(seconds=1)


def settings(**tracking):
    return {"profile": "sprint", "tracking": {"project": "bazt-experiments", "mode": "offline", "flush_rows": 500, **tracking}}


def test_read_returns_the_resolved_values_and_marks_optional_keys_as_absent():
    got = read(settings())
    assert got == TrackingSettings(mode="offline", project="bazt-experiments", entity=None, dir=None,
                                   allow_online_before_kickoff=False, flush_rows=500)


def test_read_takes_the_optional_keys_when_they_are_set():
    got = read(settings(entity="vittoria", dir="runs/wandb", allow_online_before_kickoff=True, mode="online"))
    assert (got.entity, got.dir, got.allow_online_before_kickoff, got.mode) == ("vittoria", "runs/wandb", True, "online")


@pytest.mark.parametrize("key", ["mode", "project", "flush_rows"])
def test_read_names_the_required_key_that_is_missing(key):
    s = settings()
    del s["tracking"][key]
    with pytest.raises(core.Unset) as error:
        read(s)
    assert error.value.args[0] == f"tracking.{key}"


def test_read_names_the_tracking_table_when_it_is_absent():
    with pytest.raises(core.Unset) as error:
        read({"profile": "sprint"})
    assert error.value.args[0] == "tracking.mode"


@pytest.mark.parametrize("mode", ["Online", "shared", "", 1, None])
def test_read_refuses_a_mode_that_is_not_online_offline_or_disabled(mode):
    with pytest.raises(TrackingError, match=r"tracking\.mode"):
        read(settings(mode=mode))


@pytest.mark.parametrize("key, value", [("project", ""), ("project", 3), ("entity", ""), ("entity", 7),
                                        ("dir", ""), ("allow_online_before_kickoff", "yes"),
                                        ("flush_rows", 0), ("flush_rows", 1.5), ("flush_rows", True)])
def test_read_refuses_a_value_of_the_wrong_kind_and_names_the_key(key, value):
    with pytest.raises(TrackingError, match=rf"tracking\.{key}"):
        read(settings(**{key: value}))


def online(**kw):
    return TrackingSettings(mode="online", project="p", entity="e", dir=None,
                            allow_online_before_kickoff=kw.get("allow", False), flush_rows=1)


def test_online_is_forced_offline_before_the_kickoff_and_says_why():
    mode, notice = effective_mode(online(), lambda: KICKOFF, KICKOFF - SECOND)
    assert mode == "offline"
    assert "kickoff" in notice and KICKOFF.isoformat() in notice and "allow_online_before_kickoff" in notice


@pytest.mark.parametrize("now", [KICKOFF, KICKOFF + SECOND, KICKOFF + dt.timedelta(days=30)])
def test_online_stays_online_from_the_kickoff_on(now):
    assert effective_mode(online(), lambda: KICKOFF, now) == ("online", None)


def test_online_before_the_kickoff_is_kept_when_explicitly_allowed():
    assert effective_mode(online(allow=True), lambda: KICKOFF, KICKOFF - 5 * SECOND) == ("online", None)


@pytest.mark.parametrize("mode", ["offline", "disabled"])
def test_other_modes_never_consult_the_kickoff_time(mode):
    def boom():
        raise AssertionError("the kickoff time is only needed to decide on online")

    ts = TrackingSettings(mode=mode, project="p", entity=None, dir=None, allow_online_before_kickoff=False,
                          flush_rows=1)
    assert effective_mode(ts, boom, KICKOFF - SECOND) == (mode, None)


def test_the_kickoff_time_is_not_consulted_when_online_is_allowed_anyway():
    def boom():
        raise AssertionError("not needed")

    assert effective_mode(online(allow=True), boom, KICKOFF) == ("online", None)
