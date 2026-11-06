"""Response -> first-token record -> score = log P("yes").

Rule. The judge answers "Yes" or "No" as its first generated token (thinking is off and max_tokens is 1, so that
token is the only one returned). The candidates at that position are the response's top_logprobs plus the sampled
token. A candidate is a "yes" variant when, after stripping whitespace and case-folding, it equals "yes": "Yes",
"yes", " yes", " Yes", "YES", "\\nYes" all count; "Y", "Yes.", "Yesterday" do not. Variants are distinct vocabulary
entries, so their probabilities add: score = log(sum of exp(logprob)) over the variants, computed with the
log-sum-exp shift and clamped at 0 (provider rounding can put the sum a hair above 1). A candidate whose logprob is
-inf has probability 0 and is dropped.

Missing "yes". When no variant is among the returned candidates, P(yes) is below the smallest returned
probability and is not observed. The score is then log(floor_probability), the settings value
llm.judge.floor_probability. `no_found` tells apart a confident "No" (benign, expected) from an answer that is
neither (off format: the first token is "<think>", a newline, a refusal...), which the caller counts.

The response is untrusted input. A response that cannot be scored raises instead of yielding a score: transient
errors (rate limit, upstream failure, an error reported inside a 200 body or a choice) are TransientError, a
non-retryable error body is JudgeCallFailed, and a body that is not what the method needs (no logprobs, logprobs
without top_logprobs, malformed candidates or containers, a reasoning trace, not a JSON object) is JudgeProtocolError.
Without top_logprobs only the sampled token is known: every "No" would score the same floor, so the response is refused.
"""
from __future__ import annotations

import math
from typing import Mapping, NamedTuple, Optional

from bazt_lab.features.judge.private.errors import JudgeCallFailed, JudgeProtocolError, TransientError


class Scored(NamedTuple):
    value: float                 # log P("yes"), float
    yes_found: bool              # a yes variant was among the candidates
    no_found: bool               # a "no" variant was among the candidates


def _is(token: str, word: str) -> bool:
    return token.strip().casefold() == word


def _error_to_exception(error) -> Exception:
    code = error.get("code") if isinstance(error, Mapping) else None
    message = str(error.get("message", ""))[:200] if isinstance(error, Mapping) else str(error)[:200]
    try:
        code = int(code)
    except (TypeError, ValueError):
        code = None
    if code is None or code in (408, 429) or code >= 500:         # no code at all: retry, a failure stays loud
        return TransientError(f"upstream error {code}: {message}")
    return JudgeCallFailed(f"upstream error {code}: {message}")


def _count(value) -> int:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return 0
    if isinstance(value, float) and not math.isfinite(value):
        raise JudgeProtocolError(f"malformed usage: non-finite count {value!r}")
    return int(value)


INLINE_REASONING = ("<think", "<reasoning", "<thought")       # a trace that starts inside the content, no structured field


def _starts_reasoning(token: str) -> bool:
    return token.strip().casefold().startswith(INLINE_REASONING)


def _candidate(entry) -> Optional[list]:
    """[token, logprob] of one logprob entry, None for probability 0 (-inf). Anything malformed raises."""
    try:
        token, logprob = entry["token"], entry["logprob"]
    except (KeyError, TypeError) as e:
        raise JudgeProtocolError(f"malformed logprob entry: {e!r}") from e
    if not isinstance(token, str) or isinstance(logprob, bool) or not isinstance(logprob, (int, float)):
        raise JudgeProtocolError("malformed logprob entry: a token must be a str and a logprob a number")
    logprob = float(logprob)
    if math.isnan(logprob) or logprob == math.inf:
        raise JudgeProtocolError(f"impossible logprob {logprob!r} for token {token!r}")
    return None if logprob == -math.inf else [token, logprob]


def _mapping(value, what: str) -> Mapping:
    """`value` when it is a JSON object, {} when it is absent or empty, else a protocol error (untrusted input)."""
    if not value:
        return {}
    if not isinstance(value, Mapping):
        raise JudgeProtocolError(f"malformed {what}: expected a JSON object, got {type(value).__name__}")
    return value


def extract_record(raw: Mapping) -> dict:
    """The JSON-plain record the cache stores: first-token candidates and usage. Raises on a response that cannot
    be scored (see the module docstring)."""
    if not isinstance(raw, Mapping):
        raise JudgeProtocolError(f"the response is not a JSON object but {type(raw).__name__}")
    choices = raw.get("choices") or []
    if not isinstance(choices, list):
        raise JudgeProtocolError(f"malformed choices: expected a list, got {type(choices).__name__}")
    if not choices:
        if raw.get("error") is not None:                        # OpenRouter reports upstream failures as HTTP 200 + error
            raise _error_to_exception(raw["error"])
        raise TransientError("response without choices")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise JudgeProtocolError("malformed choice")
    if choice.get("error") is not None:                         # a failure during generation, embedded in the choice
        raise _error_to_exception(choice["error"])
    if choice.get("finish_reason") == "error":
        raise TransientError("generation ended with finish_reason 'error'")
    usage = _mapping(raw.get("usage"), "usage")
    message = _mapping(choice.get("message"), "message")
    if _count(_mapping(usage.get("completion_tokens_details"), "completion_tokens_details").get("reasoning_tokens")) > 0 \
            or message.get("reasoning") or message.get("reasoning_content"):
        raise JudgeProtocolError("the provider returned a reasoning trace: the first generated token is then not "
                                 "the answer's first token and P(yes) is ill-defined (judge thinking must be off)")
    logprobs = choice.get("logprobs")
    if logprobs is None:
        raise JudgeProtocolError("the response has no logprobs: the provider ignored logprobs / top_logprobs")
    content = logprobs.get("content") if isinstance(logprobs, Mapping) else None
    first = content[0] if isinstance(content, list) and content else None
    generated = _count(usage.get("completion_tokens"))
    if first is None and generated >= 1 and isinstance(logprobs, Mapping) and not content:
        raise JudgeProtocolError(f"the response generated {generated} token(s) but its logprobs are empty: the "
                                 "provider dropped the logprobs, so no score exists")
    if (content and first is None) or not isinstance(logprobs, Mapping) or (
            first is not None and not isinstance(first, Mapping)):
        raise JudgeProtocolError("malformed logprobs")
    top, sampled = [], None
    if first is not None:
        returned = first.get("top_logprobs")
        if not isinstance(returned, list) or not returned:
            # Only the sampled token's logprob came back. Every "No" would then share one floor score (and a "Yes" a
            # real one), so no distribution exists to calibrate a threshold on: refuse instead of scoring.
            raise JudgeProtocolError("the response has logprobs but no top_logprobs: the provider ignored top_logprobs")
        top = [c for c in map(_candidate, returned) if c is not None]
        if "token" in first and first.get("logprob") is not None:
            sampled = _candidate(first)
        leading = [sampled] if sampled else []
        if top:
            leading.append(max(top, key=lambda c: c[1]))
        if any(_starts_reasoning(token) for token, _ in leading):
            raise JudgeProtocolError("the first generated token starts an inline reasoning trace: P(yes) is ill-defined "
                                     "(judge thinking must be off)")
    return {"top": top, "sampled": sampled,
            "prompt_tokens": _count(usage.get("prompt_tokens")),
            "completion_tokens": _count(usage.get("completion_tokens"))}


def score_record(record: Mapping, floor_probability: float) -> Scored:
    if not 0.0 < floor_probability < 1.0:
        raise ValueError(f"floor_probability must be in (0, 1), got {floor_probability!r}")
    candidates = dict(record["top"])      # one entry per distinct token string
    if record["sampled"] is not None:
        candidates.setdefault(record["sampled"][0], record["sampled"][1])
    yes = [lp for token, lp in candidates.items() if _is(token, "yes")]
    no_found = any(_is(token, "no") for token in candidates)
    if not yes:
        return Scored(math.log(floor_probability), False, no_found)
    peak = max(yes)
    value = peak + math.log(sum(math.exp(lp - peak) for lp in yes))
    return Scored(min(0.0, value), True, no_found)
