"""The LLM seam of the harness: any object with `async complete(messages, model) -> Completion`.

The live backend (llm_client, OpenRouter, SQLite cache) lives in adapters/live.py; FakeBackend (private/fake.py) runs
the whole platform offline. A backend may also define `begin_conversation(info)`, called with what a scripted backend
needs (the case's correct and wrong answers, attacker tool); live backends ignore it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    provider: Optional[str] = None             # the provider and quantization the response reports (spec 4.1)
    quantization: Optional[str] = None


class Backend(Protocol):
    async def complete(self, messages: List[dict], model: str) -> Completion: ...


class Meter:
    """Tokens per part ("agent", "adversary", "sanitizer", "ma_judge") and the seconds of one conversation."""

    def __init__(self) -> None:
        self.tokens: Dict[str, int] = {"agent": 0, "adversary": 0, "sanitizer": 0, "ma_judge": 0}
        self.empty: Dict[str, int] = dict.fromkeys(self.tokens, 0)      # D114: replies whose text is empty, per part
        self.start = time.perf_counter()
        self.providers: set = set()
        self.quantizations: set = set()

    def seconds(self) -> float:
        return time.perf_counter() - self.start

    def wrap(self, backend: Backend, part: str):
        """An async `invoke(messages, model)` returning the text and counting the tokens under `part`."""
        async def invoke(messages, model):
            done = await backend.complete(messages, model)
            self.tokens[part] += int(done.prompt_tokens) + int(done.completion_tokens)
            if not str(done.text).strip():
                self.empty[part] += 1
            if done.provider:
                self.providers.add(done.provider)
            if done.quantization:
                self.quantizations.add(done.quantization)
            return done.text
        return invoke
