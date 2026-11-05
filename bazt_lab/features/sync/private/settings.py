"""The sync.* settings. Every key is required once `sync.enabled` is true (no default lives in code); a missing key
raises core.Unset naming it, a bad value raises SyncError naming it."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from bazt_lab.core import public as core
from bazt_lab.features.sync.private.errors import SyncError

REPO_TYPES = ("dataset", "model")
PRIORITIES = ("below_normal", "idle")


@dataclass(frozen=True)
class SyncSettings:
    repo_id: str
    repo_type: str
    private: bool
    every_minutes: float
    include: Tuple[str, ...]
    exclude: Tuple[str, ...]
    priority: str
    final_push_timeout_seconds: float


def _get(settings: Mapping, key: str):
    return core.get(settings, f"sync.{key}")


def _bad(key: str, why: str, value) -> SyncError:
    return SyncError(f"sync.{key} {why}, got {value!r}")


def _number(settings: Mapping, key: str) -> float:
    value = _get(settings, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        raise _bad(key, "must be a number greater than 0", value)
    return float(value)


def _patterns(settings: Mapping, key: str) -> Tuple[str, ...]:
    value = _get(settings, key)
    if not isinstance(value, list) or not all(isinstance(p, str) and p for p in value):
        raise _bad(key, "must be a list of non-empty glob patterns", value)
    return tuple(value)


def _choice(settings: Mapping, key: str, allowed: Tuple[str, ...]) -> str:
    value = _get(settings, key)
    if value not in allowed:
        raise _bad(key, f"must be one of {', '.join(allowed)}", value)
    return value


def read(settings: Mapping) -> Optional[SyncSettings]:
    """None when sync is disabled (nothing else is needed then); the resolved settings otherwise."""
    enabled = _get(settings, "enabled")
    if not isinstance(enabled, bool):
        raise _bad("enabled", "must be true or false", enabled)
    if not enabled:
        return None
    repo_id = _get(settings, "repo_id")
    if not isinstance(repo_id, str) or not repo_id:
        raise _bad("repo_id", "must be the Hub repository as a non-empty string such as user/name", repo_id)
    private = _get(settings, "private")
    if not isinstance(private, bool):
        raise _bad("private", "must be true or false", private)
    return SyncSettings(repo_id, _choice(settings, "repo_type", REPO_TYPES), private, _number(settings, "every_minutes"),
                        _patterns(settings, "include"), _patterns(settings, "exclude"),
                        _choice(settings, "priority", PRIORITIES), _number(settings, "final_push_timeout_seconds"))
