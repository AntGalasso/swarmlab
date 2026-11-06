"""The one error type of the tracking feature."""
from __future__ import annotations


class TrackingError(RuntimeError):
    """Tracking cannot be set up as the settings ask: raised at start (or when the tracker is built from settings),
    never swallowed and never replaced by a tracker that records nothing. Messages name the setting or the variable
    and never contain a secret."""
