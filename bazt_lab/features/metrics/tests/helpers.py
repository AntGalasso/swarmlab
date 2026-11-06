"""Test helpers for metrics: a clean trace to edit, row accessors and a random trace generator.

`recovery_margin` here is a TEST value. The experiment value is a run-time variable of spec section 10.
"""
import dataclasses
import math

import torch

from bazt_lab.core import public as core
from bazt_lab.features.metrics import public as metrics

SETTINGS = {"runtime": {"recovery_margin": 0.1}}
KINDS = [k.value for k in core.ActionKind]


def nan(x):
    return isinstance(x, float) and math.isnan(x)


def same(a, b):
    """Exact equality where NaN equals NaN, recursing into dicts and lists."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    return (nan(a) and nan(b)) or a == b


def make(B=1, T=3, N=4, **over):
    """A clean attack trace on the complete directed graph: nobody infected, everything delivered and correct."""
    z = lambda *s: torch.zeros(*s, dtype=torch.bool)  # noqa: E731
    attempts = (~torch.eye(N, dtype=torch.bool)).expand(B, T, N, N).clone()
    tr = core.EpisodeTrace(
        unit_ids=tuple(f"u{b}" for b in range(B)), infected=z(B, T, N), attackers=z(B, N),
        active=torch.ones(B, T, N, dtype=torch.bool), attempts=attempts, delivered=attempts.clone(),
        restricted=z(B, T, N, N), payload_delivered=z(B, T, N, N), correct=torch.ones(B, T, N, dtype=torch.bool),
        goal_reached=z(B), sink=torch.zeros(B, dtype=torch.int64),
        spend_audit=torch.zeros(B, T, dtype=torch.float64), spend_intervention=torch.zeros(B, T, dtype=torch.float64),
        exhausted_step=torch.zeros(B, dtype=torch.int64), violations=torch.zeros(B, dtype=torch.int64),
        action_counts=torch.zeros(B, len(core.ActionKind), dtype=torch.int64), attack_free=False, meta={})
    for k, v in over.items():
        setattr(tr, k, v)
    return tr


def rows(tr, settings=SETTINGS):
    return metrics.rows_from_trace(tr, settings)


def one(tr, settings=SETTINGS):
    (r,) = rows(tr, settings)
    return r


def random_trace(B, T, N, seed, attack_free=False):
    """A random but internally consistent trace: delivered and restricted are disjoint subsets of attempts, and an
    attack-free trace has no attackers, no infection, no payload and no goal."""
    u = lambda *shape, key: core.uniforms(core.stream("trace", seed, key), shape)  # noqa: E731
    attempts = u(B, T, N, N, key="a") < 0.8
    attempts &= ~torch.eye(N, dtype=torch.bool)
    restricted = attempts & (u(B, T, N, N, key="r") < 0.2)
    delivered = attempts & ~restricted & (u(B, T, N, N, key="d") < 0.9)
    attack = not attack_free
    return core.EpisodeTrace(
        unit_ids=tuple({"seed": s} for s in range(B)), infected=(u(B, T, N, key="i") < 0.3) & attack,
        attackers=(u(B, N, key="k") < 0.2) & attack, active=u(B, T, N, key="v") < 0.9, attempts=attempts,
        delivered=delivered, restricted=restricted, payload_delivered=(u(B, T, N, N, key="p") < 0.1) & attempts & attack,
        correct=u(B, T, N, key="c") < 0.7, goal_reached=(u(B, key="g") < 0.5) & attack, sink=(u(B, key="s") * N).long(),
        spend_audit=u(B, T, key="sa") * 5, spend_intervention=u(B, T, key="si") * 5,
        exhausted_step=(u(B, key="e") * (T + 1)).long(), violations=(u(B, key="x") * 3).long(),
        action_counts=(u(B, len(KINDS), key="n") * 9).long(), attack_free=attack_free,
        meta={"tokens": (u(B, key="t") * 100).tolist(), "latency": u(B, key="l").tolist(),
              "clean_accuracy": (0.5 + 0.5 * u(B, key="q")).tolist(), "seeds": [1, 2, 3]})


def with_masks(tr, seed):
    """Add the optional meta masks: agents flagged by the defender and, on attack traces, tampered deliveries."""
    B, T, N = tr.infected.shape
    u = lambda *shape, key: core.uniforms(core.stream("meta", seed, key), shape)  # noqa: E731
    tr.meta["flagged"] = u(B, T, N, key="f") < 0.1
    if not tr.attack_free:
        tr.meta["tampered"] = u(B, T, N, N, key="t") < 0.1
    return tr


def take(tr, b):
    """Episode b of a batched trace as a B = 1 trace (per-episode meta sliced, batch-wide meta kept)."""
    B = tr.infected.shape[0]
    kw = {f.name: getattr(tr, f.name) for f in dataclasses.fields(tr)}
    for name, v in kw.items():
        if isinstance(v, torch.Tensor):
            kw[name] = v[b:b + 1]
    kw["unit_ids"] = tr.unit_ids[b:b + 1]
    per_episode = ("tokens", "latency", "clean_accuracy")
    kw["meta"] = {k: (v[b:b + 1] if k in per_episode or (isinstance(v, torch.Tensor) and v.ndim and v.shape[0] == B)
                      else v) for k, v in tr.meta.items()}
    return core.EpisodeTrace(**kw)
