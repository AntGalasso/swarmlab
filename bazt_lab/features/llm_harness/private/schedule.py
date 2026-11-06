"""The attacker's schedule of the held-out LLM adaptive check (D110): the mock's gray-box best response, replayed.

The runner attaches `attack_schedule = {clock, start_frac, start_round, rate}` to every unit of the block (start_frac is
the mock's, start_round its round on this unit's clock). Rounds are the trace's rounds (round 0 = the initial answers);
`clock` says how they are counted: `conversation` = r inside each conversation, `seed_run` = c x rounds + r, the
conversations of the seed run in sequence (the LLM episode, D54). Round r of conversation c is an attack round when its
clock index is at least start_round and, for rate < 1, a keyed draw of (topology, N, seed, c, r) is below rate: one
draw per seed run and round, never of the arm, so every arm faces the same attack rounds (common random numbers).

In an attack round a scheduled attacker writes as the plain attack does; in any other round it writes as a benign agent.
What an attacker is given beyond a benign agent is its system prompt and its first user message (an insider's attacker
prompt and the user's wrong answer; the infection's external document with the entry agent's question) and, for an
insider, upstream's attacker branch of the round's prompt (`attacker_idxes`). Those parts are switched round by round;
the rest of its chat memory (what it wrote and was shown) stays. The AiTM adversary rewrites only in attack rounds.
Starting at round 0 with rate 1 is the plain attack, request for request. Detectability does not transfer (D110).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from bazt_lab.core import public as core

ADAPTIVE_CONDITIONS = ("gray_box",)          # the conditions the LLM harness implements (D110)
CLOCKS = ("seed_run", "conversation")


@dataclass(frozen=True)
class Schedule:
    clock: str
    start: int
    rate: float

    def rounds(self, key: tuple, conversation: int, rounds: int) -> List[bool]:
        """The attack rounds of one conversation, index r = round r. `key` = (topology, N, seed) of the seed run."""
        out = []
        for r in range(rounds):
            index = r if self.clock == "conversation" else conversation * rounds + r
            on = index >= self.start
            if on and self.rate < 1.0:
                u = core.uniforms(core.stream("llm_harness", "adaptive_schedule", *key, conversation, r), (1,))
                on = float(u[0]) < self.rate
            out.append(on)
        return out


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse(value, unit_id: str) -> Schedule:
    """The unit's attack_schedule, validated."""
    where = f"unit {unit_id}: attack_schedule"
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} is missing: an adaptive unit replays the mock's best response as its attack "
                         f"schedule (D110), which the runner attaches; got {value!r}")
    clock, start, rate = value.get("clock"), value.get("start_round"), value.get("rate")
    if clock not in CLOCKS:
        raise ValueError(f"{where}['clock'] must be one of {CLOCKS}, got {clock!r}")
    if not isinstance(start, int) or isinstance(start, bool) or start < 0:
        raise ValueError(f"{where}['start_round'] must be an integer >= 0, got {start!r}")
    if not _number(rate) or not math.isfinite(rate) or not 0.0 < rate <= 1.0:
        raise ValueError(f"{where}['rate'] must be in (0, 1], got {rate!r}")
    return Schedule(clock, start, float(rate))


class _Capture:
    """A stand-in agent: keeps the prompt upstream builds for it and answers nothing (no call is made)."""

    prompt: Optional[str] = None

    async def achat(self, prompt: str) -> str:
        self.prompt = prompt
        return ""


class Roles:
    """The two versions (attack, benign) of the scheduled attackers of one conversation, switched round by round.

    `systems[version][i]` is agent i's system prompt; the first user messages are rendered by upstream's own first round
    on stand-in agents (`capture`), so both versions are upstream's text, not a copy of it."""

    def __init__(self, ag: Any, agents: Sequence[int], insiders: Sequence[int], systems: Mapping[bool, Sequence[str]],
                 context: Mapping[int, str]) -> None:
        self.ag, self.agents, self.insiders = ag, list(agents), list(insiders)
        self.systems, self.context = systems, dict(context)
        self.firsts: Dict[bool, List[Optional[str]]] = {}

    async def capture(self, args: tuple) -> None:
        ag = self.ag
        saved = ag.agents, ag.attacker_idxes, getattr(ag, "context", {})
        try:
            for version in (True, False):
                stand_ins = [_Capture() for _ in saved[0]]
                ag.agents, ag.context = stand_ins, (self.context if version else {})
                ag.attacker_idxes = list(self.insiders) if version else []
                await ag.afirst_generate(*args)
                self.firsts[version] = [s.prompt for s in stand_ins]
        finally:
            ag.agents, ag.attacker_idxes, ag.context = saved

    def start(self, on: bool) -> None:
        """Round 0: upstream's first round reads the attacker branch and the external document from the graph."""
        self.ag.attacker_idxes = list(self.insiders) if on else []
        self.ag.context = self.context if on else {}

    def apply(self, on: bool) -> None:
        """Before an update round: the version of every scheduled attacker for that round."""
        for i in self.agents:
            memory = self.ag.agents[i].memory
            memory[0] = {"role": "system", "content": self.systems[on][i]}
            if len(memory) > 1:
                memory[1] = {"role": "user", "content": self.firsts[on][i]}
        self.ag.attacker_idxes = list(self.insiders) if on else []
