"""The hackathon kickoff guard: no live LLM call before the official start (README compliance rule).

swarmlab is a third-party service as far as bazt_lab is concerned, so only this adapter imports it.
"""
from __future__ import annotations

from types import SimpleNamespace


def guard(allow_live_llm: bool) -> None:
    """Ends the launch (SystemExit) unless --allow-live-llm was given and the kickoff time has passed."""
    import swarmlab
    swarmlab._kickoff_guard(SimpleNamespace(mode="llm", allow_live_llm=allow_live_llm))
