"""The one error type of the sync feature."""
from __future__ import annotations


class SyncError(RuntimeError):
    """Sync cannot be started as the settings ask: raised by `start` before anything runs. Messages name the setting
    or the variable and never contain a secret."""
