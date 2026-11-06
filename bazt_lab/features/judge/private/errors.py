"""Errors of the judge feature. A caller that must stop a run catches `JudgeError`."""
from __future__ import annotations


class JudgeError(Exception):
    """Base class."""


class JudgeConfigError(JudgeError, ValueError):
    """A setting has a value the judge does not support (a missing setting raises core.Unset instead)."""


class KickoffGuardError(JudgeError):
    """A live call was attempted before the hackathon kickoff or without the explicit allow flag."""


class TransientError(JudgeError):
    """A call failed in a way that is worth retrying (rate limit, upstream 5xx, timeout, connection loss)."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class JudgeCallFailed(JudgeError):
    """A call failed for good: a non-retryable error, or retries exhausted."""


class JudgeProtocolError(JudgeCallFailed):
    """The provider answered but did not return what the score needs (for example no logprobs)."""
