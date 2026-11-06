"""Judge settings (validated once) and the wire body of one judge request.

Every experiment value comes from settings; a missing key raises core.Unset naming it. The method itself is fixed
and lives here as constants: one generated token (the answer's first token carries the score), logprobs on,
reasoning off, provider pinned.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from bazt_lab.core import public as core
from bazt_lab.features.judge.private.errors import JudgeConfigError
from bazt_lab.features.judge.private.prompt import build_messages

_LONE_SURROGATE = re.compile("[\ud800-\udfff]")      # in a Python str every code point in this range is unpaired
MAX_TOKENS = 1                       # only the first answer token is scored; the "Injection: ..." tail is never needed
MAX_TOP_LOGPROBS = 20                # OpenRouter accepts top_logprobs 0..20
# Thinking off. OpenRouter documents reasoning.effort "none" as "disables reasoning entirely" and reasoning.enabled as
# a boolean; both say off, so a provider that reads only one of them still gets the instruction. The guarantee is
# on the response side: a reasoning trace in the answer is refused (scoring.extract_record).
REASONING_OFF = {"effort": "none", "enabled": False}

KEYS = ("llm.judge.model", "llm.judge.provider", "llm.judge.thinking", "llm.judge.prompt", "llm.judge.score",
        "llm.judge.top_logprobs", "llm.judge.floor_probability", "llm.temperature", "llm.cache")


def _number(value, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JudgeConfigError(f"{key} must be a number, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class JudgeConfig:
    model: str
    provider: str
    temperature: float
    top_logprobs: int
    floor_probability: float
    use_cache: bool

    @classmethod
    def from_settings(cls, settings: Mapping) -> JudgeConfig:
        v = {k: core.get(settings, k) for k in KEYS}            # raises core.Unset(<first missing key>)
        provider = v["llm.judge.provider"]
        if not isinstance(provider, str) or not provider.strip():
            raise core.Unset("llm.judge.provider")             # "" is the placeholder in profiles.toml: unset
        model = v["llm.judge.model"]
        if not isinstance(model, str) or not model.strip():
            raise JudgeConfigError(f"llm.judge.model must be a model slug, got {model!r}")
        if v["llm.judge.thinking"] is not False:
            raise JudgeConfigError("llm.judge.thinking must be false: with a reasoning trace the first answer token "
                                   "is not the first generated token and P(yes) is ill-defined (D55)")
        if v["llm.judge.prompt"] != "promptarmor":
            raise JudgeConfigError(f"llm.judge.prompt {v['llm.judge.prompt']!r} is not supported (only 'promptarmor')")
        if v["llm.judge.score"] != "logprob_yes":
            raise JudgeConfigError(f"llm.judge.score {v['llm.judge.score']!r} is not supported (only 'logprob_yes')")
        top = v["llm.judge.top_logprobs"]
        if isinstance(top, bool) or not isinstance(top, int) or not 1 <= top <= MAX_TOP_LOGPROBS:
            raise JudgeConfigError(f"llm.judge.top_logprobs must be an int in 1..{MAX_TOP_LOGPROBS}, got {top!r}")
        floor = _number(v["llm.judge.floor_probability"], "llm.judge.floor_probability")
        if not 0.0 < floor < 1.0:
            raise JudgeConfigError(f"llm.judge.floor_probability must be in (0, 1), got {floor!r}")
        temperature = _number(v["llm.temperature"], "llm.temperature")
        if temperature < 0.0:
            raise JudgeConfigError(f"llm.temperature must be >= 0, got {temperature!r}")
        if not isinstance(v["llm.cache"], bool):
            raise JudgeConfigError(f"llm.cache must be true or false, got {v['llm.cache']!r}")
        return cls(model=model, provider=provider.strip(), temperature=temperature, top_logprobs=top,
                   floor_probability=floor, use_cache=v["llm.cache"])


def build_request(cfg: JudgeConfig, text: str) -> dict:
    """The JSON body sent to OpenRouter's chat completions endpoint (also the cache key's content). The text goes
    unchanged except that an unpaired surrogate, which has no UTF-8 form and so cannot be sent or hashed, becomes
    U+FFFD."""
    if not isinstance(text, str):
        raise TypeError(f"the judge scores str, got {type(text).__name__}")
    text = _LONE_SURROGATE.sub("�", text)
    return {
        "model": cfg.model,
        "messages": build_messages(text),
        "temperature": cfg.temperature,
        "max_tokens": MAX_TOKENS,
        "logprobs": True,
        "top_logprobs": cfg.top_logprobs,
        "provider": {"order": [cfg.provider], "allow_fallbacks": False, "require_parameters": True},
        "reasoning": dict(REASONING_OFF),
    }
