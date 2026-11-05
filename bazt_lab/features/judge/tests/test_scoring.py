"""logprob('yes') of the first answer token: variants, missing token, malformed responses."""
import math

import pytest

from bazt_lab.features.judge.private import scoring
from bazt_lab.features.judge.private.errors import JudgeCallFailed, JudgeProtocolError, TransientError
from bazt_lab.features.judge.tests.helpers import raw_response

FLOOR = 1e-9


def score(top, sampled="first"):
    rec = scoring.extract_record(raw_response(top, sampled))
    return scoring.score_record(rec, FLOOR)


def test_single_yes_token():
    s = score([("Yes", math.log(0.9)), ("No", math.log(0.1))])
    assert s.value == pytest.approx(math.log(0.9)) and s.yes_found and s.no_found


def test_tokenisation_variants_of_yes_add_their_mass():
    top = [("Yes", math.log(0.5)), (" yes", math.log(0.25)), ("YES", math.log(0.05)), ("\nYes", math.log(0.01)),
           ("No", math.log(0.15))]
    assert score(top).value == pytest.approx(math.log(0.81))


@pytest.mark.parametrize("token", ["Yesterday", "Y", "yes.", "Yeah", "Yes!", "no", "Nope", ""])
def test_only_exact_yes_variants_count(token):
    s = score([("No", math.log(0.9)), (token, math.log(0.1))])
    assert not s.yes_found and s.value == pytest.approx(math.log(FLOOR))


def test_missing_yes_is_log_of_the_floor_probability():
    s = score([("No", math.log(0.999)), ("The", math.log(0.001))])
    assert s.value == pytest.approx(math.log(FLOOR)) and not s.yes_found and s.no_found


def test_off_format_answer_is_floor_and_flagged():
    s = score([("Maybe", -0.01), ("\n", -5.0)])
    assert s.value == pytest.approx(math.log(FLOOR)) and not s.yes_found and not s.no_found


def test_empty_completion_is_floor_and_flagged():
    rec = scoring.extract_record(raw_response([], None, completion_tokens=0))
    s = scoring.score_record(rec, FLOOR)
    assert s.value == pytest.approx(math.log(FLOOR)) and not s.yes_found and not s.no_found


def test_sampled_token_outside_top_list_is_included_once():
    # temperature > 0 can sample a token that is not in top_logprobs; at temperature 0 it is the argmax
    assert score([("No", math.log(0.9))], sampled=("Yes", math.log(0.1))).value == pytest.approx(math.log(0.1))
    both = score([("Yes", math.log(0.9)), ("No", math.log(0.1))], sampled=("Yes", math.log(0.9)))
    assert both.value == pytest.approx(math.log(0.9))            # not 2 * 0.9


def test_very_negative_logprob_does_not_underflow():
    assert score([("Yes", -900.0), (" yes", -901.0)]).value == pytest.approx(-900.0 + math.log(1 + math.exp(-1)))


def test_rounding_cannot_push_a_score_above_log_one():
    assert score([("Yes", 0.0), (" yes", -1e-12)]).value == 0.0


@pytest.mark.parametrize("floor", [0.0, 1.0, -0.5, 2.0])
def test_floor_must_be_a_probability_in_the_open_unit_interval(floor):
    rec = scoring.extract_record(raw_response([("No", -0.1)]))
    with pytest.raises(ValueError):
        scoring.score_record(rec, floor)


def test_record_carries_usage_and_is_json_plain():
    import json
    rec = scoring.extract_record(raw_response([("Yes", -0.1)], prompt_tokens=123, completion_tokens=1))
    assert rec["prompt_tokens"] == 123 and rec["completion_tokens"] == 1
    assert json.loads(json.dumps(rec)) == rec


def test_response_without_logprobs_is_a_protocol_error_not_a_floor():
    with pytest.raises(JudgeProtocolError):
        scoring.extract_record(raw_response([("Yes", -0.1)], with_logprobs=False))


def test_no_choices_and_no_error_is_transient():
    with pytest.raises(TransientError):
        scoring.extract_record({"id": "x", "choices": []})


@pytest.mark.parametrize("code", [408, 429, 500, 502, 503])
def test_error_body_with_retryable_code_is_transient(code):
    with pytest.raises(TransientError):
        scoring.extract_record({"error": {"code": code, "message": "upstream"}})


@pytest.mark.parametrize("code", [400, 401, 402, 403, 404])
def test_error_body_with_other_code_is_fatal(code):
    with pytest.raises(JudgeCallFailed) as e:
        scoring.extract_record({"error": {"code": code, "message": "no endpoints support logprobs"}})
    assert not isinstance(e.value, TransientError) and str(code) in str(e.value)


# ------------------------------------------------------------------ untrusted response shapes (OpenRouter)
def with_choice(**fields):
    raw = raw_response([("Yes", -0.1)])
    raw["choices"][0].update(fields)
    return raw


@pytest.mark.parametrize("code, exc", [(429, TransientError), (502, TransientError), (400, JudgeCallFailed)])
def test_choice_level_error_is_mapped_like_a_body_level_error(code, exc):
    raw = with_choice(error={"code": code, "message": "provider failed mid-generation"}, finish_reason="error")
    with pytest.raises(exc) as e:
        scoring.extract_record(raw)
    assert (code == 400) != isinstance(e.value, TransientError)


def test_finish_reason_error_without_detail_is_transient():
    with pytest.raises(TransientError):
        scoring.extract_record(with_choice(finish_reason="error"))


def test_a_reasoning_trace_makes_the_answer_unusable_and_is_refused():
    raw = raw_response([("Yes", -0.1)])
    raw["usage"]["completion_tokens_details"] = {"reasoning_tokens": 37}
    with pytest.raises(JudgeProtocolError, match="reasoning"):
        scoring.extract_record(raw)


def test_zero_reasoning_tokens_is_fine():
    raw = raw_response([("Yes", -0.1)])
    raw["usage"]["completion_tokens_details"] = {"reasoning_tokens": 0}
    assert scoring.extract_record(raw)["top"] == [["Yes", -0.1]]


@pytest.mark.parametrize("bad", [
    {"logprob": -0.1},                                   # no token
    {"token": None, "logprob": -0.1},
    {"token": "Yes", "logprob": None},
    {"token": "Yes", "logprob": "high"},
    {"token": "Yes", "logprob": float("nan")},
    {"token": "Yes", "logprob": float("inf")},
])
def test_malformed_candidates_are_a_protocol_error_not_a_crash_or_a_wrong_score(bad):
    raw = raw_response([("No", -0.1)])
    raw["choices"][0]["logprobs"]["content"][0]["top_logprobs"].append(bad)
    with pytest.raises(JudgeProtocolError):
        scoring.extract_record(raw)


@pytest.mark.parametrize("raw", [[], "oops", None, 3])
def test_a_response_that_is_not_a_json_object_is_a_protocol_error(raw):
    with pytest.raises(JudgeProtocolError):
        scoring.extract_record(raw)


def test_candidate_with_probability_zero_is_absent_not_a_nan_score():
    s = score([("Yes", float("-inf")), ("No", 0.0)])          # P(yes) = 0 exactly
    assert not s.yes_found and s.value == pytest.approx(math.log(FLOOR))
    mixed = score([("Yes", float("-inf")), (" yes", math.log(0.2)), ("No", math.log(0.8))])
    assert mixed.value == pytest.approx(math.log(0.2))


def test_error_body_without_a_code_is_retried_not_fatal():
    with pytest.raises(TransientError):
        scoring.extract_record({"error": {"message": "boom"}})


@pytest.mark.parametrize("logprobs", ["x", {"content": ["not a mapping"]}, {"content": "abc"}])
def test_malformed_logprobs_container_is_a_protocol_error(logprobs):
    raw = raw_response([("Yes", -0.1)])
    raw["choices"][0]["logprobs"] = logprobs
    with pytest.raises(JudgeProtocolError):
        scoring.extract_record(raw)


# ------------------------------- a provider that ignores top_logprobs must not turn every "No" into the same floor
@pytest.mark.parametrize("top", [[], None, "absent"])
def test_logprobs_without_top_logprobs_are_refused_not_scored_as_a_wall_of_floors(top):
    # Sampled "No" with P(No) ~ 1 and nothing else: scoring it would give every benign text the identical floor, and a
    # threshold at a deny rate alpha cannot be read off a distribution made of ties.
    raw = raw_response([("No", -0.01)])
    first = raw["choices"][0]["logprobs"]["content"][0]
    if top == "absent":
        del first["top_logprobs"]
    else:
        first["top_logprobs"] = top
    with pytest.raises(JudgeProtocolError, match="top_logprobs"):
        scoring.extract_record(raw)


def test_a_sampled_yes_without_top_logprobs_is_refused_too_so_the_scale_never_mixes():
    raw = raw_response([("Yes", -0.1)])
    raw["choices"][0]["logprobs"]["content"][0]["top_logprobs"] = []
    with pytest.raises(JudgeProtocolError, match="top_logprobs"):
        scoring.extract_record(raw)


def test_top_logprobs_that_are_all_probability_zero_are_still_a_returned_list():
    # -inf is a returned candidate with probability 0: the provider did honour top_logprobs
    s = score([("Yes", float("-inf")), ("No", float("-inf"))])
    assert not s.yes_found and s.value == pytest.approx(math.log(FLOOR))


# ------------------------------- a response is untrusted input: wrong container types are a protocol error
@pytest.mark.parametrize("field, bad", [("message", "oops"), ("message", [1]), ("usage", [1]), ("usage", "x"),
                                        ("choices", {"a": 1}), ("choices", "abc")])
def test_wrong_container_types_in_the_response_are_a_protocol_error_not_a_crash(field, bad):
    raw = raw_response([("Yes", -0.1)])
    if field == "choices":
        raw["choices"] = bad
    elif field == "usage":
        raw["usage"] = bad
    else:
        raw["choices"][0]["message"] = bad
    with pytest.raises(JudgeProtocolError):
        scoring.extract_record(raw)


# ------------------------------------------------ an answer that was generated but whose logprobs are empty
@pytest.mark.parametrize("logprobs", [{}, {"content": []}, {"content": None}])
def test_generated_token_with_empty_logprobs_is_refused_not_floored(logprobs):
    raw = raw_response([("Yes", -0.1)])                       # completion_tokens = 1
    raw["choices"][0]["logprobs"] = logprobs
    with pytest.raises(JudgeProtocolError, match="logprobs"):
        scoring.extract_record(raw)


def test_a_really_empty_completion_stays_a_floor():
    raw = raw_response([], None, completion_tokens=0)
    raw["choices"][0]["logprobs"] = {}
    assert scoring.score_record(scoring.extract_record(raw), FLOOR).value == pytest.approx(math.log(FLOOR))


@pytest.mark.parametrize("token", ["<think>", "<think", " <THINK>", "\n<reasoning>", "<thought>"])
def test_an_inline_reasoning_start_is_refused(token):
    with pytest.raises(JudgeProtocolError, match="reasoning"):
        scoring.extract_record(raw_response([(token, -0.01), ("Yes", -5.0)]))


def test_inline_reasoning_is_refused_when_only_the_sampled_token_shows_it():
    with pytest.raises(JudgeProtocolError, match="reasoning"):
        scoring.extract_record(raw_response([("No", -0.5)], sampled=("<think>", -0.9)))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_usage_counts_are_a_protocol_error(bad):
    raw = raw_response([("Yes", -0.1)])
    raw["usage"]["prompt_tokens"] = bad
    with pytest.raises(JudgeProtocolError):
        scoring.extract_record(raw)
