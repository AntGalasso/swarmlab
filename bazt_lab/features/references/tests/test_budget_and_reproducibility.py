"""Every arm, through its public ArmSpec, on random synthetic worlds: it never asks for more than the budgets hold
(the fake platform fails the test on an intervention-budget violation), never requests an audit it cannot pay for,
and decides identically when run twice and when run in a different batch."""
import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.references import public as refs
from bazt_lab.features.references.tests.synth import INF, FakePlatform, make_ctx
from bazt_lab.features.references.tests.test_legacy import SETTINGS

THRESHOLDS = {"audit_score": 0.0, "gnn_standin": 0.5, "monitor": 0.5, "risk": 0.2}
CAPS = [0.0, 1.0, 3.0, 7.0, 15.0, 40.0, INF]
SPECS = [a for a in refs.ARMS if "mock" in a.platforms]
FIELDS = ("deny", "sanitize", "delay", "edge_quarantine", "rate_limit", "privilege_degradation", "isolation",
          "reinstatement")


def world(seed: int, B: int, unit_ids, spec):
    g = core.stream("budget-test", seed)
    N = int(torch.randint(4, 10, (), generator=g))
    T = int(torch.randint(3, 9, (), generator=g))
    adj = (core.uniforms(g, (N, N)) < 0.45) & ~torch.eye(N, dtype=torch.bool)
    caps = [CAPS[int(torch.randint(0, len(CAPS), (), generator=g))] for _ in range(2)]
    mon = core.uniforms(g, (T, N))
    inf = core.uniforms(g, (T, N)) < 0.3
    sched = core.uniforms(g, (T,)) < 0.5
    ctx = make_ctx(adj, T, B=B, audit_cap=caps[0], intervention_cap=caps[1], thresholds=THRESHOLDS,
                   settings=SETTINGS, arm=spec.name, unit_ids=unit_ids)
    object.__setattr__(ctx, "timing_schedule", sched.expand(B, T).clone())
    platform = FakePlatform(ctx, monitor=lambda t: mon[t - 1].expand(B, N), infected=lambda t: inf[t - 1].expand(B, N))
    return ctx, platform


def run(spec, seed, B=3, unit_ids=("a", "b", "c")):
    ctx, platform = world(seed, B, unit_ids, spec)
    platform.run(spec.factory())                  # raises on an intervention-budget violation
    return platform


@pytest.mark.parametrize("spec", SPECS, ids=lambda a: a.name)
def test_budgets_are_never_exceeded_on_random_worlds(spec):
    for seed in range(12):
        fp = run(spec, seed)
        assert (fp.rem_int >= 0).all() and (fp.rem_audit >= 0).all()
        for r in fp.log:
            asked_not_done = r.requested & ~r.audited
            assert not asked_not_done.any(), f"{spec.name} asked for an audit it could not pay for (seed {seed})"


@pytest.mark.parametrize("spec", SPECS, ids=lambda a: a.name)
def test_decisions_are_reproducible_and_do_not_depend_on_the_batch(spec):
    ids = ("a", "b", "c")
    for seed in range(6):
        full, again, single = run(spec, seed, 3, ids), run(spec, seed, 3, ids), run(spec, seed, 1, ("b",))
        for rf, ra, rs in zip(full.log, again.log, single.log, strict=True):
            for k in FIELDS:
                x, y, z = getattr(rf.actions, k), getattr(ra.actions, k), getattr(rs.actions, k)
                assert (x is None) == (y is None) == (z is None)
                if x is not None:
                    assert torch.equal(x, y) and torch.equal(x[1], z[0]), f"{spec.name}.{k} seed {seed}"
