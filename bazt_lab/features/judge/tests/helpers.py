"""Shared test fixtures: settings, OpenAI-shaped responses and a scripted transport. No network."""
from __future__ import annotations

import copy
import json as _json
import math
from typing import Callable

import httpx

from bazt_lab.core import public as core

TEST_PROVIDER = "test-provider"          # a made-up slug; no test talks to a real provider


def make_settings(**judge) -> dict:
    """profiles.toml, plus the judge keys that the file leaves unset, plus `judge` overrides."""
    s = core.load_settings()
    core.put(s, "llm.judge.provider", TEST_PROVIDER)
    core.put(s, "llm.judge.top_logprobs", 20)
    core.put(s, "llm.judge.floor_probability", 1e-9)
    for key, value in judge.items():
        core.put(s, f"llm.judge.{key}", value)
    return s


def without(settings: dict, dotted: str) -> dict:
    """A copy of `settings` with one dotted key removed."""
    s = copy.deepcopy(settings)
    *path, leaf = dotted.split(".")
    node = s
    for part in path:
        node = node[part]
    del node[leaf]
    return s


def yes_no(p_yes: float) -> list:
    """First-token candidates of a judge that answers Yes with probability p_yes and No otherwise."""
    return [("Yes", math.log(p_yes)), ("No", math.log(1.0 - p_yes))]


def raw_response(top, sampled="first", *, prompt_tokens: int = 400, completion_tokens: int = 1,
                 with_logprobs: bool = True) -> dict:
    """An OpenAI chat-completion dict whose first generated token has the given top_logprobs."""
    if sampled == "first":
        sampled = top[0] if top else None
    content = []
    if sampled is not None:
        content.append({"token": sampled[0], "logprob": sampled[1], "bytes": None,
                        "top_logprobs": [{"token": t, "logprob": lp, "bytes": None} for t, lp in top]})
    choice = {"index": 0, "finish_reason": "length",
              "message": {"role": "assistant", "content": sampled[0] if sampled else ""},
              "logprobs": {"content": content, "refusal": None} if with_logprobs else None}
    return {"id": "gen-test", "object": "chat.completion", "created": 1, "model": "qwen/qwen3.8-27b",
            "choices": [choice],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens}}


def planted(text: str) -> list:
    """Default answer: the judge says Yes (0.9) on texts that contain INJECT, No (P(yes)=0.01) otherwise."""
    return yes_no(0.9) if "INJECT" in text else yes_no(0.01)


class ScriptedTransport:
    """transport(body) -> raw response. `answer(text)` returns top candidates, a raw dict, or an Exception."""

    def __init__(self, answer: Callable = planted) -> None:
        self.answer = answer
        self.bodies: list = []

    def __call__(self, body: dict) -> dict:
        self.bodies.append(body)
        out = self.answer(body["messages"][-1]["content"])
        if isinstance(out, Exception):
            raise out
        return out if isinstance(out, dict) else raw_response(out)

    @property
    def texts(self) -> list:
        return [b["messages"][-1]["content"] for b in self.bodies]


# ------------------------------------------------------------------------------------------ HTTP level


def completion(top, **kw) -> httpx.Response:
    """HTTP 200 with an OpenAI-shaped chat completion whose first token has the given candidates."""
    return httpx.Response(200, json=raw_response(top, **kw))


def planted_reply(request: httpx.Request) -> httpx.Response:
    return completion(planted(_json.loads(request.content)["messages"][-1]["content"]))


class HTTP:
    """What the network would do: replies are consumed in order and the last one repeats. A reply is an
    httpx.Response, an exception to raise, or a callable(request) -> response. `requests` records what was sent."""

    def __init__(self, *replies) -> None:
        self.replies, self.requests = list(replies) or [planted_reply], []
        self.client = httpx.Client(transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply(request) if callable(reply) else reply

    def bodies(self) -> list:
        return [_json.loads(r.content) for r in self.requests]
