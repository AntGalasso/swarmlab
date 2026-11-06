"""LLMJudge: batching, dedup, cache, retries, stats. A scripted transport stands in for the network."""
import logging
import math

import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.judge.private.errors import JudgeCallFailed, JudgeConfigError, JudgeProtocolError, TransientError
from bazt_lab.features.judge.private.judge import LLMJudge
from bazt_lab.features.judge.private.retry import RetryPolicy
from bazt_lab.features.judge.tests.helpers import ScriptedTransport, make_settings, raw_response, without

NO_WAIT = dict(sleep=lambda s: None)


def judge(tmp_path, transport, name="cache.sqlite", **kw):
    return LLMJudge(make_settings(), transport, cache_path=tmp_path / name, **{**NO_WAIT, **kw})


def test_scores_are_log_p_yes_float64_in_input_order(tmp_path):
    out = judge(tmp_path, ScriptedTransport()).score(["clean a", "INJECT b", "clean c"])
    assert out.dtype == torch.float64 and out.shape == (3,)
    torch.testing.assert_close(out, torch.tensor([math.log(0.01), math.log(0.9), math.log(0.01)],
                                                 dtype=torch.float64))


def test_is_a_core_judge(tmp_path):
    assert isinstance(judge(tmp_path, ScriptedTransport()), core.Judge)


def test_empty_batch_is_an_empty_float64_tensor_and_makes_no_call(tmp_path):
    t = ScriptedTransport()
    out = judge(tmp_path, t).score([])
    assert out.dtype == torch.float64 and out.shape == (0,) and t.bodies == []


def test_identical_texts_are_paid_once_per_batch(tmp_path):
    t = ScriptedTransport()
    out = judge(tmp_path, t).score(["x", "INJECT y", "x", "x"])
    assert sorted(t.texts) == ["INJECT y", "x"]
    assert out[0] == out[2] == out[3] and out[0] != out[1]


def test_cache_persists_across_instances_and_is_shared_by_callers(tmp_path):
    first, second = ScriptedTransport(), ScriptedTransport()
    a = judge(tmp_path, first).score(["x", "INJECT y"])
    b = judge(tmp_path, second).score(["INJECT y", "x"])
    assert len(first.bodies) == 2 and second.bodies == []
    assert torch.equal(a, b.flip(0))


def test_cache_is_keyed_by_provider_and_model(tmp_path):
    t = ScriptedTransport()
    LLMJudge(make_settings(), t, cache_path=tmp_path / "c.sqlite", **NO_WAIT).score(["x"])
    LLMJudge(make_settings(provider="other-provider"), t, cache_path=tmp_path / "c.sqlite", **NO_WAIT).score(["x"])
    LLMJudge(make_settings(model="other/model"), t, cache_path=tmp_path / "c.sqlite", **NO_WAIT).score(["x"])
    assert len(t.bodies) == 3


def test_cache_off_in_settings_means_no_cache_file_and_a_call_every_time(tmp_path):
    s = make_settings()
    core.put(s, "llm.cache", False)
    t = ScriptedTransport()
    j = LLMJudge(s, t, cache_path=tmp_path / "off.sqlite", **NO_WAIT)
    j.score(["x"])
    j.score(["x"])
    assert len(t.bodies) == 2 and not (tmp_path / "off.sqlite").exists()


def test_cache_on_without_a_cache_path_is_refused(tmp_path):
    with pytest.raises(JudgeConfigError, match="cache"):
        LLMJudge(make_settings(), ScriptedTransport(), cache_path=None)


def test_result_does_not_depend_on_batch_composition_or_worker_count(tmp_path):
    texts = [f"text {i}" + (" INJECT" if i % 3 == 0 else "") for i in range(40)]
    whole = judge(tmp_path, ScriptedTransport(), "a.sqlite", max_workers=1).score(texts)
    parallel = judge(tmp_path, ScriptedTransport(), "b.sqlite", max_workers=8).score(texts)
    pieces = torch.cat([judge(tmp_path, ScriptedTransport(), f"p{i}.sqlite").score(texts[i:i + 7])
                        for i in range(0, 40, 7)])
    assert torch.equal(whole, parallel) and torch.equal(whole, pieces)


def test_missing_yes_scores_the_floor_and_is_counted(tmp_path):
    j = judge(tmp_path, ScriptedTransport(lambda text: [("No", -0.001), ("The", -7.0)]))
    out = j.score(["a"])
    assert out[0].item() == pytest.approx(math.log(1e-9))
    assert j.stats.missing_yes == 1 and j.stats.off_format == 0


def test_off_format_answers_are_counted_separately(tmp_path):
    j = judge(tmp_path, ScriptedTransport(lambda text: [("Maybe", -0.001)]))
    j.score(["a"])
    assert j.stats.missing_yes == 1 and j.stats.off_format == 1


def test_transient_errors_are_retried_and_counted(tmp_path):
    state = {"n": 0}

    def answer(text):
        state["n"] += 1
        return TransientError("429") if state["n"] <= 2 else [("Yes", math.log(0.5)), ("No", math.log(0.5))]
    sleeps = []
    j = judge(tmp_path, ScriptedTransport(answer), sleep=sleeps.append)
    assert j.score(["a"])[0].item() == pytest.approx(math.log(0.5))
    assert len(sleeps) == 2 and j.stats.retries == 2 and j.stats.live_calls == 1


def test_exhausted_retries_raise_and_nothing_is_cached(tmp_path):
    t = ScriptedTransport(lambda text: TransientError("503"))
    j = judge(tmp_path, t, retry=RetryPolicy(attempts=3, base_delay=1.0, max_delay=2.0))
    with pytest.raises(JudgeCallFailed):
        j.score(["a"])
    assert len(t.bodies) == 3
    ok = ScriptedTransport()
    judge(tmp_path, ok).score(["a"])
    assert len(ok.bodies) == 1                                         # the failure left no cache entry


def test_one_failing_text_raises_but_the_other_results_are_kept(tmp_path):
    bad = ScriptedTransport(lambda text: JudgeCallFailed("HTTP 400") if text == "bad" else [("Yes", -0.1)])
    with pytest.raises(JudgeCallFailed, match="400"):
        judge(tmp_path, bad, max_workers=4).score(["good 1", "bad", "good 2"])
    fixed = ScriptedTransport()
    judge(tmp_path, fixed).score(["good 1", "bad", "good 2"])
    assert fixed.texts == ["bad"]                                      # only the failed text is paid again


def test_response_without_logprobs_is_refused_not_floored(tmp_path):
    t = ScriptedTransport(lambda text: raw_response([("Yes", -0.1)], with_logprobs=False))
    with pytest.raises(JudgeProtocolError):
        judge(tmp_path, t).score(["a"])
    assert len(t.bodies) == 1                                          # a protocol error is not retried


def test_unset_provider_refuses_before_any_call(tmp_path):
    t = ScriptedTransport()
    with pytest.raises(core.Unset) as e:
        LLMJudge(without(make_settings(), "llm.judge.provider"), t, cache_path=tmp_path / "c.sqlite")
    assert e.value.args[0] == "llm.judge.provider" and t.bodies == []


def test_every_request_is_the_pinned_logprob_request(tmp_path):
    t = ScriptedTransport()
    judge(tmp_path, t).score(["a"])
    body = t.bodies[0]
    assert body["provider"] == {"order": ["test-provider"], "allow_fallbacks": False, "require_parameters": True}
    assert body["logprobs"] is True and body["max_tokens"] == 1 and body["reasoning"] == {"effort": "none", "enabled": False}


def test_stats_count_calls_hits_and_tokens_and_snapshot_is_plain(tmp_path):
    t = ScriptedTransport(lambda text: raw_response([("Yes", -0.1)], prompt_tokens=100, completion_tokens=1))
    j = judge(tmp_path, t)
    j.score(["a", "b", "a"])
    j.score(["a", "c"])
    s = j.stats.snapshot()
    assert (s["texts"], s["unique"], s["live_calls"], s["cache_hits"]) == (5, 4, 3, 1)
    assert (s["prompt_tokens"], s["completion_tokens"]) == (400, 4)    # logical totals: cache hits included
    assert s["live_seconds"] >= 0.0 and all(isinstance(v, (int, float)) for v in s.values())


def test_non_string_input_is_a_type_error(tmp_path):
    with pytest.raises(TypeError):
        judge(tmp_path, ScriptedTransport()).score(["ok", 3])


def test_each_score_call_logs_one_summary_line_that_never_contains_the_texts(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="bazt_lab.judge")
    judge(tmp_path, ScriptedTransport()).score(["private text a", "b", "private text a"])
    lines = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert len(lines) == 1
    for field in ("texts=3", "unique=2", "cache_hits=0", "live_calls=2", "failed=0", "retries=0"):
        assert field in lines[0]
    assert "private text" not in lines[0]


def test_a_failed_call_is_still_summarised_with_its_failure_count(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="bazt_lab.judge")
    bad = ScriptedTransport(lambda text: JudgeCallFailed("HTTP 400") if text == "bad" else [("Yes", -0.1)])
    with pytest.raises(JudgeCallFailed):
        judge(tmp_path, bad).score(["ok", "bad"])
    lines = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert len(lines) == 1 and "failed=1" in lines[0] and "live_calls=1" in lines[0]


def test_a_text_with_a_lone_surrogate_is_scored_instead_of_crashing_the_batch(tmp_path):
    t = ScriptedTransport()
    out = judge(tmp_path, t).score(["INJECT \ud83d tail", "clean"])
    assert out.shape == (2,) and out[0].item() == pytest.approx(math.log(0.9))
    assert t.texts.count("INJECT \ufffd tail") == 1


def test_inline_reasoning_is_refused_and_not_cached(tmp_path):
    t = ScriptedTransport(lambda text: [("<think>", -0.001)])
    with pytest.raises(JudgeProtocolError):
        judge(tmp_path, t).score(["a"])
    ok = ScriptedTransport()
    judge(tmp_path, ok).score(["a"])
    assert len(ok.bodies) == 1


def test_empty_logprobs_are_refused_and_not_cached(tmp_path):
    def answer(text):
        raw = raw_response([("Yes", -0.1)])
        raw["choices"][0]["logprobs"] = {}
        return raw
    with pytest.raises(JudgeProtocolError):
        judge(tmp_path, ScriptedTransport(answer)).score(["a"])
    ok = ScriptedTransport()
    judge(tmp_path, ok).score(["a"])
    assert len(ok.bodies) == 1


def test_nan_usage_fails_that_text_only_and_siblings_are_cached(tmp_path):
    def answer(text):
        raw = raw_response([("Yes", -0.1)])
        if text == "bad":
            raw["usage"]["prompt_tokens"] = float("nan")
        return raw
    with pytest.raises(JudgeProtocolError):
        judge(tmp_path, ScriptedTransport(answer), max_workers=1).score(["good", "bad"])
    fixed = ScriptedTransport()
    judge(tmp_path, fixed).score(["good"])
    assert fixed.bodies == []


def test_missing_yes_and_off_format_are_in_the_log_line(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="bazt_lab.judge")
    judge(tmp_path, ScriptedTransport(lambda text: [("Maybe", -0.1)])).score(["a", "b"])
    line = next(r.getMessage() for r in caplog.records if r.levelno == logging.INFO)
    assert "missing_yes=2" in line and "off_format=2" in line
