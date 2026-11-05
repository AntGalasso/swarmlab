"""The tracking.* settings and the rule that W&B stays offline until the hackathon kickoff (D87).

Required keys (mode, project, flush_rows) raise core.Unset naming the key. The optional keys are `entity`, `dir` and `allow_online_before_kickoff`
(unset means: default W&B entity is refused for online runs, W&B's own directory, online not allowed before kickoff).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Tuple

from bazt_lab.core import public as core
from bazt_lab.features.tracking.private.errors import TrackingError

MODES = ("online", "offline", "disabled")


@dataclass(frozen=True)
class TrackingSettings:
    mode: str
    project: str
    entity: Optional[str]
    dir: Optional[str]
    allow_online_before_kickoff: bool
    flush_rows: int                          # rows buffered per table before one wandb.Table part is logged


def _text(settings: Mapping, key: str) -> str:
    value = core.get(settings, f"tracking.{key}")
    if not isinstance(value, str) or not value:
        raise TrackingError(f"tracking.{key} must be a non-empty string, got {value!r}")
    return value


def _optional(settings: Mapping, key: str) -> Optional[str]:
    return _text(settings, key) if core.is_set(settings, f"tracking.{key}") else None


def read(settings: Mapping) -> TrackingSettings:
    mode = core.get(settings, "tracking.mode")
    if mode not in MODES:
        raise TrackingError(f"tracking.mode must be one of {', '.join(MODES)}, got {mode!r}")
    allow = False
    if core.is_set(settings, "tracking.allow_online_before_kickoff"):
        allow = core.get(settings, "tracking.allow_online_before_kickoff")
        if not isinstance(allow, bool):
            raise TrackingError(f"tracking.allow_online_before_kickoff must be true or false, got {allow!r}")
    flush = core.get(settings, "tracking.flush_rows")
    if isinstance(flush, bool) or not isinstance(flush, int) or flush < 1:
        raise TrackingError(f"tracking.flush_rows must be a whole number of rows, at least 1, got {flush!r}")
    return TrackingSettings(mode, _text(settings, "project"), _optional(settings, "entity"),
                            _optional(settings, "dir"), allow, flush)


def effective_mode(ts: TrackingSettings, kickoff: Callable[[], dt.datetime],
                   now: dt.datetime) -> Tuple[str, Optional[str]]:
    """(mode to use, notice to tell the user or None). "online" before the kickoff becomes "offline" unless the
    settings allow it; the other modes are used as asked, and the kickoff time is read only when it decides."""
    if ts.mode != "online" or ts.allow_online_before_kickoff:
        return ts.mode, None
    start = kickoff()
    if now < start:
        return "offline", (f"tracking: tracking.mode is online but the kickoff is {start.isoformat()} (D87); this run "
                           "is recorded offline, upload it later with `wandb sync`. Set "
                           "tracking.allow_online_before_kickoff = true to go online anyway.")
    return "online", None
