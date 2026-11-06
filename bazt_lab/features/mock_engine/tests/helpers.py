"""Test helpers for mock_engine: test-only settings, a scripted defender, a fake metrics rows function, a fake store.

Every number in `make_settings` is a TEST value chosen to make a scenario small and checkable. None of them is an
experiment decision (those are the run-time variables of spec section 10, set by the owner in profiles.toml).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch

from bazt_lab.core import public as core

KINDS = [k.value for k in core.ActionKind]


def make_settings(**over) -> dict:
    """profiles.toml plus test values for every key the engine reads. `over` maps dotted keys (with `__` for `.`)."""
    s = core.load_settings(profile="sprint")
    tests = {
        "runtime.n_insiders": 2, "runtime.rate_limit_per_identity": 1, "runtime.horizon_constant": 3.0,
        "runtime.eps_correct": 2.0, "runtime.ba_m": 2, "runtime.ws_k": 4, "runtime.ws_rewire_p": 0.25,
        "runtime.sanitize_success": 1.0, "runtime.insider_bias": 8.0, "runtime.ta_detectability": 1.5,
        "runtime.ma_detectability": 1.2, "runtime.aitm_rewrite_strength": 1.0, "runtime.recovery_margin": 0.05,
        "mock.model.bias": 8.0, "mock.model.burst_span": 1, "mock.auditor.binormal_method": "shared_threshold",
        "costs.robustness.deny": 1, "calibration.mode": "open_loop",
    }
    tests.update({k.replace("__", "."): v for k, v in over.items()})
    for key, value in tests.items():
        core.put(s, key, value)
    return s


def unit(arm="A", topology="chain", attack="none", agents=6, seed=1, sink=0, audit="uncapped",
         intervention="uncapped", **extra) -> dict:
    u = dict(block="test", platform="mock", arm=arm, audit=audit, intervention=intervention, topology=topology,
             attack=attack, agents=agents, seed=seed, sink=sink, **extra)
    u.setdefault("id", "-".join(str(u[k]) for k in ("arm", "topology", "attack", "agents", "seed", "sink", "audit",
                                                    "intervention")) + str(extra.get("tag", "")))
    return u


def mask(B, N, *pairs) -> torch.Tensor:
    """bool [B, N, N] with the given (b, i, j) entries set."""
    m = torch.zeros(B, N, N, dtype=torch.bool)
    for b, i, j in pairs:
        m[b, i, j] = True
    return m


def agents(B, N, *pairs) -> torch.Tensor:
    """bool [B, N] with the given (b, i) entries set."""
    m = torch.zeros(B, N, dtype=torch.bool)
    for b, i in pairs:
        m[b, i] = True
    return m


class Scripted:
    """A defender that returns `script[t]` (an Actions, or a callable obs -> Actions) at round t, else nothing.

    It records every Observation it was shown in `seen`, so a test can check what a defender may see.
    """
    name = "scripted"

    def __init__(self, script: Optional[Dict[int, object]] = None, calibrated: bool = False):
        self.script = script or {}
        self.seen = []
        self.ctx = None

    def reset(self, ctx):
        self.ctx = ctx
        self.seen = []

    def decide(self, obs):
        self.seen.append(obs)
        a = self.script.get(obs.t)
        a = a(obs) if callable(a) else a
        return a if a is not None else core.Actions()

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


def spec(defender_factory: Callable = Scripted, name="scripted", **kw) -> core.ArmSpec:
    return core.ArmSpec(name=name, factory=defender_factory, **kw)


class FakeStore:
    """In-memory CheckpointStore with the same surface."""

    def __init__(self):
        self.rows, self.done, self.appends = {}, set(), []

    def is_done(self, uid):
        return uid in self.done

    def append_rows(self, uid, rows):
        self.rows.setdefault(uid, []).extend(rows)
        self.appends.append(uid)

    def resume_rows(self, uid):
        return list(self.rows.get(uid, []))

    def mark_done(self, uid):
        self.done.add(uid)

    def save_state(self, uid, state):
        pass

    def load_state(self, uid):
        return None


def fake_rows(trace, settings):
    """Stands in for metrics.public.rows_from_trace: one plain row per episode, in trace order."""
    return [{"unit_id": u["id"], "asr_final": float((~trace.correct[b, -1]).double().mean()),
             "n_rounds": int(trace.correct.shape[1])} for b, u in enumerate(trace.unit_ids)]


class Fn(Scripted):
    """A defender that applies `fn(obs) -> Actions` at every round."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def decide(self, obs):
        self.seen.append(obs)
        return self.fn(obs)


def sim(attack="none", defender=None, T=6, topology="chain", N=6, settings=None, edit=None, needs_truth=False,
        seeds=(1,), sink=0, collect=None, **unit_kw):
    """Build a world, optionally edit its tensors in place (to force a scenario), run it. -> (world, trace, est)."""
    from bazt_lab.features.mock_engine.private import engine
    from bazt_lab.features.mock_engine.private import world as W
    from bazt_lab.features.mock_engine.private.model import load_model
    s = settings or make_settings()
    units = [unit(arm="scripted", attack=attack, topology=topology, agents=N, seed=sd, sink=sink, **unit_kw) for sd in seeds]
    adj = W.make_graphs(units, s, topology, N)
    w = W.build_world(units, s, topology, N, attack, adj, T)
    if edit:
        edit(w)
    ctx = engine.make_context(units, w, "scripted", s, {})
    trace, est = engine.simulate(w, load_model(s), defender, ctx, needs_truth, collect, units)
    return w, trace, est


def seed_at(*agents_):
    """edit: the payload starts at exactly these agents."""
    def edit(w):
        w.seed.zero_()
        w.seed[:, list(agents_)] = True
        if w.family in ("insider_pi", "insider_ta", "insider_ma"):
            w.attackers.copy_(w.seed)
    return edit


def forced(**values):
    """edit: set world tensors (by name) to constants, in place."""
    def edit(w):
        for name, v in values.items():
            getattr(w, name).fill_(v)
    return edit


def chain(*edits):
    def edit(w):
        for e in edits:
            e(w)
    return edit
