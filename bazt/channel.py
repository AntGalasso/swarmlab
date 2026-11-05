"""The cost of filtering: a calibration rationale rides on every message and is lost when it is stripped.

`swarmlab.trusted_sanitize` keeps only `est=<number>`. In the mock, a benign message is *exactly* that
schema, so stripping would be free and over-sanitizing could never hurt the task. This module gives every
message a short rationale (`sens=hp|lp`, the sender's sensor class) and an edge-weight rule that trusts
an intact message by its sensor precision but a schema-only message only by `stripped_edge_weight`.
Sanitizing a benign node therefore degrades consensus for real (measurable MAE), which is the price the
budget optimizer has to weigh. Everything here is simulator-side: no model writes or reads the rationale.
"""
from __future__ import annotations

from typing import Callable, List, Sequence

import swarmlab as sl
from bazt.config import BaztConfig

HP, LP = "hp", "lp"
RATIONALE_PREFIX = "sens="


def sensor_classes(topology: str, n: int, seed: int, noisy_frac: float) -> List[str]:
    u = sl.stream_rng("bazt-sens", f"{topology}:{n}", seed).random(n)
    return [LP if x < noisy_frac else HP for x in u]


class RationaleAgent:
    """Wraps any swarm agent and appends the sender's calibration rationale to each message."""

    def __init__(self, inner, classes: Sequence[str]):
        self.inner, self.classes = inner, list(classes)

    def step(self, ctxs: Sequence[sl.AgentCtx]) -> List[str]:
        texts = self.inner.step(ctxs)
        return [f"{tx} {RATIONALE_PREFIX}{self.classes[c.i]}" for c, tx in zip(ctxs, texts)]


def make_edge_weight_modifier(cfg: BaztConfig) -> Callable[[int, int, int, str], float]:
    """f(sender, receiver, t, delivered_text) -> scale of the Metropolis weight of that edge."""
    hp_tag, lp_tag = f"{RATIONALE_PREFIX}{HP}", f"{RATIONALE_PREFIX}{LP}"

    def modifier(_j: int, _i: int, _t: int, msg: str) -> float:
        if hp_tag in msg:
            return cfg.precision_hp
        if lp_tag in msg:
            return cfg.precision_lp
        return cfg.stripped_edge_weight                  # rationale gone: schema-only message
    return modifier
