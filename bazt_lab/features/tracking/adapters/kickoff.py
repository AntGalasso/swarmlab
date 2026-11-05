"""The hackathon kickoff time, as swarmlab's own guard reads it: SWARMLAB_KICKOFF_UTC when set (ISO time, no zone means
UTC), else swarmlab.KICKOFF_UTC. swarmlab is a third-party service for bazt_lab, so only an adapter imports it."""
from __future__ import annotations

import datetime as dt
import os

from bazt_lab.features.tracking.adapters import env
from bazt_lab.features.tracking.private.errors import TrackingError

VARIABLE = "SWARMLAB_KICKOFF_UTC"


def kickoff_time() -> dt.datetime:
    env.load()                              # the variable may be in .env, as .env.example says
    text = os.getenv(VARIABLE)
    if text:
        try:
            when = dt.datetime.fromisoformat(text)
        except ValueError as error:
            raise TrackingError(f"{VARIABLE}={text!r} is not an ISO time such as 2026-11-06T20:00:00+00:00") from error
    else:
        try:
            import swarmlab
            when = swarmlab.KICKOFF_UTC
        except (ImportError, AttributeError) as error:   # e.g. run outside the repo root, where `swarmlab` is a folder
            raise TrackingError(f"cannot read the kickoff time from swarmlab.KICKOFF_UTC ({type(error).__name__}); run "
                                f"from the repo root or set {VARIABLE} to an ISO time such as "
                                "2026-11-06T20:00:00+00:00") from error
    return when if when.tzinfo else when.replace(tzinfo=dt.UTC)
