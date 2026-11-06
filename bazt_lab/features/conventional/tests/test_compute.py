"""Device and precision of the conventional arms (settings [compute]): float tensors follow ctx.dtype, tensors live on
ctx.device, accounting stays exact, and float32 agrees with float64 on the outcome metrics."""
import pytest
import torch

from bazt_lab.features.conventional.private.arm_b import ArmB
from bazt_lab.features.conventional.private.arm_c import ArmC
from bazt_lab.features.conventional.tests.testkit import BINS, Fake, make_ctx, make_settings

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"))]


def _episodes(arm_cls, name, dtype, device="cpu", B=6, N=8, T=6, cap=40.0):
    settings = make_settings(rate_limit=3, bins=BINS, window=2)
    g = torch.Generator().manual_seed(5)
    adj = (torch.rand(B, N, N, generator=g) < 0.8) & ~torch.eye(N, dtype=torch.bool)
    thresholds = {"judge": 0.3} if name == "B" else {"rl_minus_ts": -22.0}
    ctx = make_ctx(adj.to(device), name, dtype=dtype, device=device, thresholds=thresholds, settings=settings, T=T,
                   audit_cap=cap, intervention_cap=cap)
    arm = arm_cls()
    arm.reset(ctx)
    fake = Fake(ctx)
    out = []
    for t in range(1, T + 1):
        att = (adj & (torch.rand(B, N, N, generator=g) < 0.85)).to(device)
        sc = (torch.rand(B, N, generator=g) * 4 - 3).to(device)
        a = arm.decide(fake.obs(t, att, scores=sc))
        fake.settle(a)
        out.append(a)
    return ctx, arm, out


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("arm_cls, name", [(ArmB, "B"), (ArmC, "C")])
def test_float_tensors_follow_ctx_dtype_and_everything_lives_on_ctx_device(arm_cls, name, dtype, device):
    ctx, arm, out = _episodes(arm_cls, name, dtype, device)
    for a in out:
        for stat in a.statistics.values():
            assert stat.dtype == dtype and stat.device.type == device
        assert a.deny.dtype == torch.bool and a.deny.device.type == device
        assert a.threshold_restricted.device.type == device
    for value in arm.state_dict().values():
        if isinstance(value, torch.Tensor):
            assert value.device.type == device
            assert value.dtype in (torch.bool, torch.int64, dtype)


@pytest.mark.parametrize("arm_cls, name", [(ArmB, "B"), (ArmC, "C")])
def test_float32_agrees_with_float64_on_outcomes_and_accounting_is_exact(arm_cls, name):
    _, _, wide = _episodes(arm_cls, name, torch.float64)
    _, _, narrow = _episodes(arm_cls, name, torch.float32)
    denies = lambda out: sum(a.deny.sum().item() for a in out)                       # noqa: E731
    assert abs(denies(wide) - denies(narrow)) <= 0.02 * max(denies(wide), 1)         # tolerance: 2% of the denies
    assert [a.events for a in wide] == [a.events for a in narrow]                     # budget accounting: identical
    for a, b in zip(wide, narrow, strict=True):
        x, y = a.statistics.popitem()[1], b.statistics.popitem()[1]
        assert torch.equal(x.isnan(), y.isnan())
        assert (x.nan_to_num(0).round() - y.double().nan_to_num(0).round()).abs().max() <= 1.0   # same lattice value
