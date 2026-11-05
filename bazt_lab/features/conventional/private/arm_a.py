"""Arm A: identity-tier zero trust (spec 2.1). BeyondCorp-style allowlist plus authenticated delivery, one trust
tier, no content reading. Its rules are preset and free, so it runs uncapped and never emits budget events."""
from __future__ import annotations

from bazt_lab.core import public as core
from bazt_lab.features.conventional.private.common import a_gate


class ArmA:
    name = "A"

    def __init__(self) -> None:
        self.ctx: core.EpisodeContext | None = None

    def reset(self, ctx: core.EpisodeContext) -> None:
        self.ctx = ctx

    def decide(self, obs: core.Observation) -> core.Actions:
        return core.Actions(deny=obs.attempts & ~a_gate(self.ctx.adj, obs.attempts, obs.authentic))

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state) -> None:
        pass
