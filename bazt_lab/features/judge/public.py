"""Public interface of the judge feature: the content check of B, C and BAZT on the LLM side. See CONTRACT.md.

    judge = live_judge(settings, allow_live=..., cache_path=...)    # PromptArmor over OpenRouter, after kickoff only
    judge = FakeJudge({"[PLANT]": 0.95})                            # offline: tests and the end-to-end fake run
    scores = judge.score(texts)                                     # float64 [len(texts)], log P("yes"), higher = injected
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

from bazt_lab.features.judge.private.errors import (
    JudgeCallFailed,
    JudgeConfigError,
    JudgeError,
    JudgeProtocolError,
    KickoffGuardError,
    TransientError,
)
from bazt_lab.features.judge.private.fake import FakeJudge
from bazt_lab.features.judge.private.judge import JudgeStats, LLMJudge, Transport
from bazt_lab.features.judge.private.prompt import SOURCE as PROMPT_SOURCE
from bazt_lab.features.judge.private.prompt import SYSTEM_PROMPT
from bazt_lab.features.judge.private.retry import DEFAULT_RETRY, RetryPolicy


def live_judge(settings: Mapping, *, allow_live: bool, cache_path: Optional[Path], max_workers: int = 16,
               retry: RetryPolicy = DEFAULT_RETRY) -> LLMJudge:
    """The PromptArmor judge over OpenRouter's OpenAI-compatible API, provider pinned from settings.

    Refuses (nothing is sent) with KickoffGuardError before the hackathon kickoff or without allow_live=True, with
    core.Unset naming `llm.judge.provider` (or another missing key), and with JudgeConfigError for an unusable setting,
    a missing OPENROUTER_API_KEY or an insecure OPENROUTER_BASE_URL. `cache_path` is required when `llm.cache` is true.
    """
    from bazt_lab.features.judge.adapters import openrouter  # imports the openai SDK only when a live judge is built
    return openrouter.live_judge(settings, allow_live=allow_live, cache_path=cache_path, max_workers=max_workers,
                                 retry=retry)


__all__ = ["FakeJudge", "JudgeCallFailed", "JudgeConfigError", "JudgeError", "JudgeProtocolError", "JudgeStats",
           "KickoffGuardError", "LLMJudge", "PROMPT_SOURCE", "RetryPolicy", "SYSTEM_PROMPT", "TransientError",
           "Transport", "live_judge"]
