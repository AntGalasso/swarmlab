"""The live path: the real openai SDK against an httpx mock transport (no network), the kickoff guard, the secrets."""
import datetime as dt
import math

import httpx
import pytest

from bazt_lab.core import public as core
from bazt_lab.features.judge.adapters import openrouter as adapter
from bazt_lab.features.judge.private.config import JudgeConfig, build_request
from bazt_lab.features.judge.private.errors import (
    JudgeCallFailed,
    JudgeConfigError,
    JudgeProtocolError,
    KickoffGuardError,
)
from bazt_lab.features.judge.private.retry import RetryPolicy
from bazt_lab.features.judge.tests.helpers import HTTP, completion, make_settings, planted_reply, without

FAKE_CREDENTIAL = "unit-test-credential-0001"
PAST = "2020-01-01T00:00:00+00:00"
FUTURE = "2999-01-01T00:00:00+00:00"
KICKOFF = dt.datetime(2026, 11, 6, 20, 0, tzinfo=dt.UTC)


def env(**extra) -> dict:
    """An environment with a key and a kickoff in the past; `extra` overrides or adds (None removes)."""
    e = {adapter.KEY_ENV: FAKE_CREDENTIAL, adapter.KICKOFF_ENV: PAST}
    e.update(extra)
    return {k: v for k, v in e.items() if v is not None}


def live(http, tmp_path, *, settings=None, allow_live=True, environment=None, **kw):
    kw = {"sleep": lambda s: None, **kw}
    return adapter.live_judge(settings or make_settings(), allow_live=allow_live, cache_path=tmp_path / "c.sqlite",
                              env=env() if environment is None else environment, http_client=http.client, **kw)


# --------------------------------------------------------------------------------------- the request
def test_the_request_on_the_wire_is_the_pinned_logprob_request_with_the_bearer_key(tmp_path):
    http = HTTP()
    settings = make_settings()
    out = live(http, tmp_path, settings=settings).score(["clean", "INJECT now"])
    assert [round(math.exp(v), 2) for v in out.tolist()] == [0.01, 0.9]
    request = http.requests[0]
    assert request.method == "POST" and str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {FAKE_CREDENTIAL}"
    cfg = JudgeConfig.from_settings(settings)

    def by_text(b):                                                        # requests go out concurrently, in any order
        return b["messages"][-1]["content"]
    assert sorted(http.bodies(), key=by_text) == sorted(
        [build_request(cfg, "clean"), build_request(cfg, "INJECT now")], key=by_text)
    body = http.bodies()[0]
    assert body["provider"] == {"order": ["test-provider"], "allow_fallbacks": False, "require_parameters": True}
    assert body["logprobs"] is True and body["top_logprobs"] == 20 and body["max_tokens"] == 1
    assert body["reasoning"] == {"effort": "none", "enabled": False}


def test_the_base_url_comes_from_the_environment(tmp_path):
    http = HTTP()
    live(http, tmp_path, environment=env(OPENROUTER_BASE_URL="https://gateway.example/v9")).score(["x"])
    assert str(http.requests[0].url) == "https://gateway.example/v9/chat/completions"


@pytest.mark.parametrize("url, ok", [("https://openrouter.ai/api/v1", True), ("http://localhost:8080/v1", True),
                                     ("http://127.0.0.1:8080/v1", True), ("http://openrouter.ai/api/v1", False),
                                     ("http://evil.example/v1", False), ("ftp://x/v1", False), ("openrouter.ai", False)])
def test_the_key_travels_only_over_https_or_to_loopback(url, ok, tmp_path):
    if ok:
        live(HTTP(), tmp_path, environment=env(OPENROUTER_BASE_URL=url)).close()
    else:
        with pytest.raises(JudgeConfigError, match="https"):
            live(HTTP(), tmp_path, environment=env(OPENROUTER_BASE_URL=url))


def test_the_sdk_does_not_retry_on_its_own(tmp_path):
    http = HTTP(error(503))                              # the SDK would retry a 503 by itself (twice) if it were allowed
    with pytest.raises(JudgeCallFailed):
        live(http, tmp_path, retry=RetryPolicy(attempts=1)).score(["x"])
    assert len(http.requests) == 1                       # the judge's own policy is the only retry layer


# ------------------------------------------------------------------------------------- error mapping
def error(status, headers=None, message="upstream said no"):
    return httpx.Response(status, headers=headers or {}, json={"error": {"code": status, "message": message}})


@pytest.mark.parametrize("reply", [error(408), error(429), error(500), error(502), error(503)])
def test_retryable_statuses_are_retried_until_the_call_succeeds(reply, tmp_path):
    http = HTTP(reply, reply, planted_reply)
    sleeps = []
    out = live(http, tmp_path, sleep=sleeps.append).score(["INJECT"])
    assert round(math.exp(out[0].item()), 2) == 0.9 and len(http.requests) == 3 and len(sleeps) == 2


def test_retry_after_header_is_honoured(tmp_path):
    http = HTTP(error(429, {"retry-after": "7"}), planted_reply)
    sleeps = []
    live(http, tmp_path, sleep=sleeps.append).score(["x"])
    assert sleeps == [7.0]


def test_retry_after_that_is_not_a_number_is_ignored(tmp_path):
    http = HTTP(error(429, {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}), planted_reply)
    sleeps = []
    live(http, tmp_path, sleep=sleeps.append).score(["x"])
    assert len(sleeps) == 1 and 0.5 <= sleeps[0] <= 1.0


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_other_client_errors_fail_at_once_with_the_status(status, tmp_path):
    http = HTTP(error(status))
    with pytest.raises(JudgeCallFailed, match=str(status)):
        live(http, tmp_path).score(["x"])
    assert len(http.requests) == 1


def test_402_is_fatal_unless_the_server_says_when_to_come_back(tmp_path):
    plain = HTTP(error(402))
    with pytest.raises(JudgeCallFailed, match="402") as e:
        live(plain, tmp_path).score(["x"])
    assert len(plain.requests) == 1 and "attempts" not in str(e.value)          # failed at once, not after retries
    http = HTTP(error(402, {"retry-after": "3"}), planted_reply)
    sleeps = []
    live(http, tmp_path, sleep=sleeps.append).score(["x"])
    assert sleeps == [3.0] and len(http.requests) == 2


def test_404_explains_that_the_pinned_provider_may_not_serve_the_request(tmp_path):
    with pytest.raises(JudgeCallFailed, match=r"llm\.judge\.provider"):
        live(HTTP(error(404, message="No endpoints found that support the requested parameters")), tmp_path).score(["x"])


def test_exhausted_retries_raise_call_failed_and_cache_nothing(tmp_path):
    http = HTTP(error(503))
    j = live(http, tmp_path, retry=RetryPolicy(attempts=3, base_delay=1.0, max_delay=2.0))
    with pytest.raises(JudgeCallFailed, match="3 attempts"):
        j.score(["x"])
    assert len(http.requests) == 3
    ok = HTTP()
    live(ok, tmp_path).score(["x"])
    assert len(ok.requests) == 1


def test_connection_failures_and_timeouts_are_retried(tmp_path):
    http = HTTP(httpx.ConnectError("connection refused"), httpx.ReadTimeout("slow"), planted_reply)
    live(http, tmp_path).score(["x"])
    assert len(http.requests) == 3


def test_an_error_reported_inside_a_200_body_is_mapped_like_an_http_error(tmp_path):
    body = httpx.Response(200, json={"error": {"code": 502, "message": "provider died"}})
    http = HTTP(body, planted_reply)
    live(http, tmp_path).score(["x"])
    assert len(http.requests) == 2
    fatal = httpx.Response(200, json={"error": {"code": 400, "message": "bad parameter"}})
    with pytest.raises(JudgeCallFailed, match="400"):
        live(HTTP(fatal), tmp_path).score(["y"])


def test_a_body_that_is_not_json_is_retried(tmp_path):
    http = HTTP(httpx.Response(200, text="<html>bad gateway</html>"), planted_reply)
    live(http, tmp_path).score(["x"])
    assert len(http.requests) == 2


def test_a_response_without_logprobs_is_refused_and_not_retried(tmp_path):
    http = HTTP(completion([("Yes", -0.1)], with_logprobs=False))
    with pytest.raises(JudgeProtocolError):
        live(http, tmp_path).score(["x"])
    assert len(http.requests) == 1


# -------------------------------------------------------------------------------------- cache, close
def test_a_second_judge_on_the_same_cache_file_makes_no_http_call(tmp_path):
    first, second = HTTP(), HTTP()
    a = live(first, tmp_path)
    one = a.score(["x", "INJECT y"])
    a.close()
    b = live(second, tmp_path)
    assert bool((b.score(["x", "INJECT y"]) == one).all()) and second.requests == []


def test_close_releases_the_http_client(tmp_path):
    http = HTTP()
    live(http, tmp_path).close()
    assert http.client.is_closed


# ------------------------------------------------------------------------------------------ settings
def test_unset_provider_refuses_before_any_call_and_before_the_key_is_read(tmp_path):
    http = HTTP()
    with pytest.raises(core.Unset) as e:
        live(http, tmp_path, settings=without(make_settings(), "llm.judge.provider"), environment=env(**{adapter.KEY_ENV: None}))
    assert e.value.args[0] == "llm.judge.provider" and http.requests == []


# -------------------------------------------------------------------------------------------- secrets
def test_a_missing_key_names_the_variable_and_sends_nothing(tmp_path):
    http = HTTP()
    with pytest.raises(JudgeConfigError, match=adapter.KEY_ENV):
        live(http, tmp_path, environment=env(**{adapter.KEY_ENV: None}))
    with pytest.raises(JudgeConfigError, match=adapter.KEY_ENV):
        live(http, tmp_path, environment=env(**{adapter.KEY_ENV: "   "}))
    assert http.requests == []


def test_the_key_never_shows_up_in_errors_or_reprs(tmp_path):
    http = HTTP(error(401, message="User not found."))
    j = live(http, tmp_path)
    with pytest.raises(JudgeCallFailed) as e:
        j.score(["x"])
    shown = [str(e.value), repr(e.value), repr(j), repr(j._transport), repr(j._transport._client.base_url)]
    assert not any(FAKE_CREDENTIAL in s for s in shown)


def test_the_environment_and_dotenv_are_used_when_no_mapping_is_given(tmp_path, monkeypatch):
    loaded = []
    monkeypatch.setattr(adapter, "load_dotenv", lambda *a, **k: loaded.append(1))
    monkeypatch.setenv(adapter.KEY_ENV, FAKE_CREDENTIAL)
    monkeypatch.setenv(adapter.KICKOFF_ENV, PAST)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    http = HTTP()
    adapter.live_judge(make_settings(), allow_live=True, cache_path=tmp_path / "c.sqlite", http_client=http.client,
                       sleep=lambda s: None).score(["x"])
    assert loaded == [1] and http.requests[0].headers["authorization"] == f"Bearer {FAKE_CREDENTIAL}"


# ------------------------------------------------------------------------------------ kickoff guard
def test_before_kickoff_nothing_is_sent_even_with_the_allow_flag_and_a_key(tmp_path):
    http = HTTP()
    with pytest.raises(KickoffGuardError, match="kickoff"):
        live(http, tmp_path, environment=env(**{adapter.KICKOFF_ENV: FUTURE}))
    assert http.requests == []


def test_after_kickoff_the_allow_flag_is_still_required(tmp_path):
    http = HTTP()
    with pytest.raises(KickoffGuardError, match="allow"):
        live(http, tmp_path, allow_live=False)
    assert http.requests == []


def test_the_guard_comes_before_the_key_check(tmp_path):
    with pytest.raises(KickoffGuardError):
        live(HTTP(), tmp_path, environment=env(**{adapter.KEY_ENV: None, adapter.KICKOFF_ENV: FUTURE}))


def test_a_malformed_kickoff_variable_keeps_the_guard_closed(tmp_path):
    with pytest.raises(KickoffGuardError, match=adapter.KICKOFF_ENV):
        live(HTTP(), tmp_path, environment=env(**{adapter.KICKOFF_ENV: "soon"}))


def test_the_clock_decides_at_the_second(tmp_path):
    one_second = dt.timedelta(seconds=1)
    at = {adapter.KICKOFF_ENV: KICKOFF.isoformat()}
    with pytest.raises(KickoffGuardError):
        live(HTTP(), tmp_path, environment=env(**at), now=lambda: KICKOFF - one_second)
    live(HTTP(), tmp_path, environment=env(**at), now=lambda: KICKOFF).close()


def test_the_default_kickoff_is_the_hackathon_start_and_is_read_from_swarmlab(tmp_path):
    assert adapter.kickoff_time({}) == KICKOFF
    assert adapter.kickoff_time({adapter.KICKOFF_ENV: ""}) == KICKOFF            # empty counts as unset
    with pytest.raises(KickoffGuardError):
        live(HTTP(), tmp_path, environment=env(**{adapter.KICKOFF_ENV: None}), now=lambda: KICKOFF - dt.timedelta(seconds=1))
