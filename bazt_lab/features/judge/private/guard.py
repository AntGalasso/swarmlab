"""Kickoff guard (README compliance rule): no live LLM call before the hackathon kickoff, and none without an
explicit allow flag. Pure: the clock and the kickoff time are arguments, so tests never depend on the date."""
from __future__ import annotations

import datetime as dt

from bazt_lab.features.judge.private.errors import KickoffGuardError


def parse_kickoff(text: str) -> dt.datetime:
    """An ISO-8601 time; a time without an offset is UTC. Anything else refuses: a guard that cannot read its time
    stays closed."""
    try:
        kickoff = dt.datetime.fromisoformat(text)
    except ValueError as e:
        raise KickoffGuardError(f"SWARMLAB_KICKOFF_UTC={text!r} is not an ISO-8601 time") from e
    return kickoff if kickoff.tzinfo is not None else kickoff.replace(tzinfo=dt.UTC)


def check_live_allowed(*, allow_live: bool, kickoff: dt.datetime, now: dt.datetime) -> None:
    if now < kickoff:
        raise KickoffGuardError(f"live judge call refused before kickoff ({kickoff.isoformat()}); "
                                "set SWARMLAB_KICKOFF_UTC if the official start differs")
    if allow_live is not True:
        raise KickoffGuardError("live judge call refused: it needs an explicit allow_live=True "
                                "(the --allow-live-llm flag of the launcher)")
