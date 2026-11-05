"""The conventional pipeline inside BAZT: A's identity-tier rules, B's content check, C's contextual trust score, run
through the Defender protocol of the conventional feature (its public ARMS) and nothing else.

BAZT decides which senders to audit (the INSPECT action) and pays for the audits once, so the arms are served a
cached view: `obs.audit` returns the scores BAZT bought and NaN elsewhere, and debits nothing. They are told the
intervention budget is unlimited, because BAZT does not use their denies: it takes A's free denies and the decision
statistics B and C report (the judge score, the trust score minus the risk level), and decides the actions itself.
B's own rate limit is not applied; BAZT's RATE_LIMIT action takes its place.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, Mapping

import torch

from bazt_lab.core import public as core

Tensor = torch.Tensor
NAMES = ("A", "B", "C")


@dataclass
class Result:
    deny_a: Tensor                     # bool [B, N, N]: deliveries A's rules drop (free)
    statistics: Dict[str, Tensor]      # decision statistics of B and C, name -> float64 [B, N, N]


class AuditCache:
    """obs.audit as the inner arms see it: the scores BAZT bought, for the senders they ask about."""

    def __init__(self, scores: Tensor) -> None:
        self.scores = scores

    def __call__(self, mask: Tensor) -> Tensor:
        return self.scores.masked_fill(~mask, float("nan"))


def default_factories() -> Dict[str, Callable[[], core.Defender]]:
    from bazt_lab.features.conventional import public as conventional
    found = {spec.name: spec.factory for spec in conventional.ARMS}
    return {name: found[name] for name in NAMES}


class Pipeline:
    def __init__(self, factories: Mapping[str, Callable[[], core.Defender]] | None = None) -> None:
        self.factories = dict(default_factories() if factories is None else factories)
        for name in NAMES:
            self.factories[name]                                      # KeyError names the arm that is missing
        self.arms: Dict[str, core.Defender] = {}

    def reset(self, ctx: core.EpisodeContext) -> None:
        self.arms = {name: self.factories[name]() for name in NAMES}
        for arm in self.arms.values():
            arm.reset(ctx)

    def step(self, obs: core.Observation, scores: Tensor) -> Result:
        view = dataclasses.replace(obs, audit=AuditCache(scores),
                                   remaining_intervention=torch.full_like(obs.remaining_intervention, float("inf")))
        deny_a = self.arms["A"].decide(view).deny
        statistics: Dict[str, Tensor] = {}
        for name in ("B", "C"):
            statistics.update(self.arms[name].decide(view).statistics)
        return Result(deny_a, statistics)

    def state_dict(self) -> dict:
        return {name: arm.state_dict() for name, arm in self.arms.items()}

    def load_state_dict(self, state: Mapping) -> None:
        for name, arm in self.arms.items():
            arm.load_state_dict(state[name])
