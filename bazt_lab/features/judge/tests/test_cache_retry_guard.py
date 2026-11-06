"""Disk cache, retry with backoff, kickoff guard."""
import logging
from datetime import UTC, datetime, timedelta

import pytest

from bazt_lab.features.judge.private import cache as cache_mod
from bazt_lab.features.judge.private.config import JudgeConfig, build_request
from bazt_lab.features.judge.private.errors import JudgeCallFailed, KickoffGuardError, TransientError
from bazt_lab.features.judge.private.guard import check_live_allowed, parse_kickoff
from bazt_lab.features.judge.private.retry import RetryPolicy, call_with_retries
from bazt_lab.features.judge.tests.helpers import make_settings


# ------------------------------------------------------------------------------------------------- cache
def test_cache_roundtrip_persists_and_first_write_wins(tmp_path):
    path = tmp_path / "judge.sqlite"
    c = cache_mod.DiskCache(path)
    assert c.get("k") is None
    c.put("k", {"top": [["Yes", -0.1]], "sampled": None, "prompt_tokens": 7, "completion_tokens": 1})
    c.put("k", {"top": [["No", -0.1]], "sampled": None, "prompt_tokens": 9, "completion_tokens": 1})
    c.close()
    again = cache_mod.DiskCache(path)
    assert again.get("k")["top"] == [["Yes", -0.1]]
    again.close()


def test_cache_key_covers_model_provider_prompt_text_and_every_request_field():
    base = build_request(JudgeConfig.from_settings(make_settings()), "text")
    key = cache_mod.cache_key(base)
    assert key == cache_mod.cache_key(dict(reversed(list(base.items()))))          # field order is irrelevant
    variants = [
        build_request(JudgeConfig.from_settings(make_settings()), "other text"),
        build_request(JudgeConfig.from_settings(make_settings(provider="elsewhere")), "text"),
        build_request(JudgeConfig.from_settings(make_settings(model="other/model")), "text"),
        build_request(JudgeConfig.from_settings(make_settings(top_logprobs=5)), "text"),
        {**base, "temperature": 0.7},
    ]
    keys = {cache_mod.cache_key(v) for v in variants}
    assert key not in keys and len(keys) == len(variants)


# ------------------------------------------------------------------------------------------------- retry
class Flaky:
    def __init__(self, failures, then="ok"):
        self.failures, self.then, self.calls = list(failures), then, 0

    def __call__(self):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return self.then


POLICY = RetryPolicy(attempts=4, base_delay=1.0, max_delay=8.0)


def test_transient_errors_are_retried_with_exponential_backoff_and_jitter():
    sleeps = []
    fn = Flaky([TransientError("a"), TransientError("b"), TransientError("c")])
    result, retries = call_with_retries(fn, POLICY, key="k", sleep=sleeps.append)
    assert (result, retries, fn.calls) == ("ok", 3, 4)
    caps = [1.0, 2.0, 4.0]
    assert all(0.5 * cap <= s <= cap for s, cap in zip(sleeps, caps, strict=True))          # equal jitter inside the cap


def test_backoff_jitter_is_deterministic_per_key_and_differs_across_keys():
    def run(key):
        sleeps = []
        call_with_retries(Flaky([TransientError("x"), TransientError("y")]), POLICY, key=key, sleep=sleeps.append)
        return sleeps
    assert run("a") == run("a") and run("a") != run("b")


def test_delay_is_capped_by_max_delay():
    sleeps = []
    policy = RetryPolicy(attempts=8, base_delay=1.0, max_delay=3.0)
    call_with_retries(Flaky([TransientError("x")] * 6), policy, key="k", sleep=sleeps.append)
    assert max(sleeps) <= 3.0


def test_retry_after_from_the_server_is_honoured():
    sleeps = []
    call_with_retries(Flaky([TransientError("429", retry_after=6.0)]), POLICY, key="k", sleep=sleeps.append)
    assert sleeps == [6.0]
    sleeps.clear()
    call_with_retries(Flaky([TransientError("429", retry_after=600.0)]), POLICY, key="k", sleep=sleeps.append)
    assert sleeps == [POLICY.max_delay]                                        # never waits longer than the cap


def test_exhaustion_raises_call_failed_chained_from_the_last_error():
    fn = Flaky([TransientError(str(i)) for i in range(10)])
    with pytest.raises(JudgeCallFailed) as e:
        call_with_retries(fn, POLICY, key="k", sleep=lambda s: None)
    assert fn.calls == POLICY.attempts and isinstance(e.value.__cause__, TransientError)


def test_a_fatal_error_is_not_retried():
    fn = Flaky([JudgeCallFailed("HTTP 401")])
    with pytest.raises(JudgeCallFailed):
        call_with_retries(fn, POLICY, key="k", sleep=lambda s: pytest.fail("slept"))
    assert fn.calls == 1


def test_retry_policy_rejects_nonsense():
    for bad in (dict(attempts=0, base_delay=1, max_delay=2), dict(attempts=2, base_delay=0, max_delay=2),
                dict(attempts=2, base_delay=3, max_delay=2)):
        with pytest.raises(ValueError):
            RetryPolicy(**bad)


# ------------------------------------------------------------------------------------------------- guard
KICKOFF = datetime(2026, 11, 6, 20, 0, tzinfo=UTC)


def test_guard_refuses_before_kickoff_even_with_explicit_allow():
    with pytest.raises(KickoffGuardError, match="kickoff"):
        check_live_allowed(allow_live=True, kickoff=KICKOFF, now=KICKOFF - timedelta(seconds=1))


def test_guard_passes_from_kickoff_on_with_explicit_allow():
    check_live_allowed(allow_live=True, kickoff=KICKOFF, now=KICKOFF)
    check_live_allowed(allow_live=True, kickoff=KICKOFF, now=KICKOFF + timedelta(days=3))


def test_guard_needs_the_explicit_allow_flag_after_kickoff():
    with pytest.raises(KickoffGuardError, match="allow"):
        check_live_allowed(allow_live=False, kickoff=KICKOFF, now=KICKOFF + timedelta(days=1))


def test_parse_kickoff_treats_naive_times_as_utc_and_accepts_offsets():
    assert parse_kickoff("2026-11-06T20:00:00") == KICKOFF
    assert parse_kickoff("2026-11-06T21:00:00+01:00") == KICKOFF
    assert parse_kickoff("2026-11-06T20:00:00Z") == KICKOFF


@pytest.mark.parametrize("text", ["tomorrow", "2026-13-45", "", "not a time"])
def test_a_malformed_kickoff_time_refuses_instead_of_opening_the_guard(text):
    with pytest.raises(KickoffGuardError, match="SWARMLAB_KICKOFF_UTC"):
        parse_kickoff(text)


def test_every_retry_is_logged_with_its_error_and_attempt_number(caplog):
    caplog.set_level(logging.WARNING, logger="bazt_lab.judge")
    call_with_retries(Flaky([TransientError("HTTP 429 slow down"), TransientError("HTTP 503")]), POLICY, key="k",
                      sleep=lambda s: None)
    lines = [r.getMessage() for r in caplog.records]
    assert len(lines) == 2 and "attempt 1/4" in lines[0] and "HTTP 429 slow down" in lines[0] and "attempt 2/4" in lines[1]


@pytest.mark.parametrize("flag", ["false", 1, "yes", None, 0])
def test_allow_live_must_be_the_boolean_true(flag):
    with pytest.raises(KickoffGuardError, match="allow"):
        check_live_allowed(allow_live=flag, kickoff=KICKOFF, now=KICKOFF + timedelta(days=1))
