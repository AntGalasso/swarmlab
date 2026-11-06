"""Compute layer: keyed batched draws, ctx.device / ctx.dtype, and the sync-free budgeted greedy selection."""
import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.references import public as refs
from bazt_lab.features.references.private import common
from bazt_lab.features.references.tests.synth import FakePlatform, make_ctx, star
from bazt_lab.features.references.tests.test_budget_and_reproducibility import CAPS, FIELDS, SPECS, THRESHOLDS
from bazt_lab.features.references.tests.test_legacy import SETTINGS

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"))]


def sequential_fit(order, cost, valid, budget):
    """The definition: walk the order, take each valid agent whose cost still fits, skip the rest."""
    out = torch.zeros_like(valid)
    for b in range(order.shape[0]):
        left = float(budget[b])
        for i in order[b].tolist():
            if valid[b, i] and float(cost[b, i]) <= left:
                out[b, i] = True
                left -= float(cost[b, i])
    return out


def test_fit_in_order_equals_the_sequential_definition_on_random_cases():
    g = core.stream("fit-test")
    for case in range(60):
        B, N = 1 + case % 5, 2 + case % 9
        cost = torch.randint(0, 6, (B, N), generator=g).double()
        valid = core.uniforms(g, (B, N)) < 0.7
        order = core.uniforms(g, (B, N)).argsort(-1)
        order = order.gather(-1, (~valid.gather(-1, order)).to(torch.int8).argsort(dim=-1, stable=True))   # valid agents first
        budget = torch.where(core.uniforms(g, (B,)) < 0.2, torch.tensor(float("inf"), dtype=torch.float64),
                             torch.randint(0, 20, (B,), generator=g).double())
        assert torch.equal(common.fit_in_order(order, cost, valid, budget), sequential_fit(order, cost, valid, budget))


def test_fit_in_order_runs_no_python_loop_over_ranks_when_everything_fits(monkeypatch):
    order = torch.arange(6).expand(2, 6).clone()
    cost, valid = torch.ones(2, 6, dtype=torch.float64), torch.ones(2, 6, dtype=torch.bool)

    def boom(*a, **k):
        raise AssertionError("per-rank loop")
    got = common.fit_in_order(order, cost, valid, torch.full((2,), 100.0, dtype=torch.float64))
    assert got.all()
    monkeypatch.setattr(torch.Tensor, "item", boom)             # the happy path never syncs with .item()
    assert common.fit_in_order(order, cost, valid, torch.full((2,), 100.0, dtype=torch.float64)).all()


def no_arm_stream(ctx):
    def refuse(ns, b):
        raise AssertionError("per-episode Generator used; draws must be keyed and batched")
    object.__setattr__(ctx, "arm_stream", refuse)
    return ctx


@pytest.mark.parametrize("name", ["random_alloc", "oracle_alloc", "bra_greedy", "random_capped", "random_capped_matched"])
def test_random_arms_draw_from_the_keyed_batched_streams_only(name):
    spec = next(a for a in refs.ARMS if a.name == name)
    ctx = no_arm_stream(make_ctx(star(5), 4, B=2, settings=SETTINGS, arm=name))
    object.__setattr__(ctx, "timing_schedule", refs.schedule_from_rounds([[1], [2]], 4))
    spec.factory().reset(ctx)


def test_draws_are_arm_specific_and_keyed_by_episode_id_not_position():
    def picks(arm, ids):
        ctx = make_ctx(star(8), 6, B=len(ids), audit_cap=6, unit_ids=ids, arm=arm)
        fp = FakePlatform(ctx)
        fp.run(next(a for a in refs.ARMS if a.name == arm).factory())
        return torch.stack([r.audited for r in fp.log])
    a = picks("random_alloc", ("x", "y", "z"))
    assert torch.equal(a[:, 2], picks("random_alloc", ("z",))[:, 0])        # position and batch size are irrelevant
    assert torch.equal(a[:, 1], picks("random_alloc", ("q", "y"))[:, 1])
    assert not torch.equal(a, picks("bra_greedy", ("x", "y", "z")))         # a different arm draws differently


def world32(spec, seed, dtype):
    g = core.stream("dtype-test", seed)
    N, T = int(torch.randint(4, 10, (), generator=g)), int(torch.randint(3, 9, (), generator=g))
    adj = (core.uniforms(g, (N, N)) < 0.45) & ~torch.eye(N, dtype=torch.bool)
    caps = [CAPS[int(torch.randint(0, len(CAPS), (), generator=g))] for _ in range(2)]
    mon = core.uniforms(g, (T, N))
    inf = core.uniforms(g, (T, N)) < 0.3
    sched = core.uniforms(g, (T,)) < 0.5
    ctx = make_ctx(adj, T, B=3, audit_cap=caps[0], intervention_cap=caps[1], thresholds=THRESHOLDS, settings=SETTINGS,
                   arm=spec.name, unit_ids=("a", "b", "c"), dtype=dtype)
    object.__setattr__(ctx, "timing_schedule", sched.expand(3, T).clone())
    return FakePlatform(ctx, monitor=lambda t: mon[t - 1].expand(3, N), infected=lambda t: inf[t - 1].expand(3, N))


@pytest.mark.parametrize("spec", SPECS, ids=lambda a: a.name)
def test_float32_agrees_with_float64_on_decisions_and_accounting(spec):
    """float32 scores may flip a decision only where a score sits within rounding of a threshold: at most 2% of the
    mask entries differ, the integer accounting (audit and intervention budgets) never leaves float64, and no run
    violates a budget (the fake platform asserts)."""
    differ = total = 0
    for seed in range(8):
        runs = {}
        for dt in (torch.float64, torch.float32):
            fp = world32(spec, seed, dt)
            fp.run(spec.factory())
            assert fp.rem_int.dtype == torch.float64 and fp.rem_audit.dtype == torch.float64
            runs[dt] = fp
        for r64, r32 in zip(runs[torch.float64].log, runs[torch.float32].log, strict=True):
            for k in FIELDS:
                x, y = getattr(r64.actions, k), getattr(r32.actions, k)
                assert (x is None) == (y is None)
                if x is not None:
                    differ += int((x != y).sum())
                    total += x.numel()
            for name, s32 in r32.actions.statistics.items():
                assert s32.dtype == torch.float32, f"{spec.name}.{name} must follow ctx.dtype"
    assert differ <= 0.02 * max(total, 1), f"{spec.name}: {differ}/{total} mask entries differ"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("spec", SPECS, ids=lambda a: a.name)
def test_every_tensor_an_arm_returns_lives_on_ctx_device(spec, device):
    ctx = make_ctx(star(5), 4, B=2, settings=SETTINGS, thresholds=THRESHOLDS, intervention_cap=20, audit_cap=8,
                   arm=spec.name, device=device)
    object.__setattr__(ctx, "timing_schedule", refs.schedule_from_rounds([[1, 3], [2]], 4).to(device))
    fp = FakePlatform(ctx, monitor=lambda t: torch.tensor([[0.1, 0.9, 0.2, 0.7, 0.3]] * 2, dtype=torch.float64),
                      infected=lambda t: torch.tensor([[False, True, False, False, False]] * 2))
    fp.run(spec.factory())
    for r in fp.log:
        for k in FIELDS:
            v = getattr(r.actions, k)
            if v is not None:
                assert v.device.type == torch.device(device).type
        for s in r.actions.statistics.values():
            assert s.device.type == torch.device(device).type
