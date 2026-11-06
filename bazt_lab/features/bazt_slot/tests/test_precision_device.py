"""compute.dtype and compute.device: float32 agrees with float64 on outcomes, accounting is exact, tensors stay on the device."""
import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.bazt_slot.private import bazt as bz
from bazt_lab.features.bazt_slot.tests.test_bazt import THRESHOLDS, random_episode
from bazt_lab.features.bazt_slot.tests.testkit import Fake, FakePipeline, make_ctx

F64 = torch.float64
FIELDS = ("sanitize", "delay", "edge_quarantine", "privilege_degradation", "isolation", "rate_limit", "reinstatement")
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"))]


def play(seed, dtype=F64, device="cpu", variant="BAZT"):
    adj, caps, acaps, mon, scores = random_episode(seed)
    ctx = make_ctx(adj, arm=variant, thresholds=THRESHOLDS, T=6)
    ctx = core.EpisodeContext(**{**ctx.__dict__, "audit_cap": acaps, "intervention_cap": caps, "dtype": dtype,
                                 "device": torch.device(device)})
    for name in ("adj", "sink", "audit_cap", "intervention_cap"):
        object.__setattr__(ctx, name, getattr(ctx, name).to(device))
    d = bz.Bazt(variant, pipeline=_Pipe(dtype))
    d.reset(ctx)
    fake = Fake(ctx)
    fake.scores = scores.to(device=device, dtype=dtype)
    out = []
    for t in range(1, 7):
        a = d.decide(fake.obs(t, monitor=mon[t - 1].to(device=device, dtype=dtype)))
        fake.settle(a)
        out.append(a)
    return fake, out


class _Pipe(FakePipeline):
    def __init__(self, dtype):
        self.dtype = dtype

    def step(self, obs, scores):
        res = super().step(obs, scores)
        res.statistics = {k: v.to(self.dtype) for k, v in res.statistics.items()}
        return res


@pytest.mark.parametrize("seed", range(4))
def test_float32_keeps_the_accounting_exact_and_the_outcome_close_to_float64(seed):
    f64, a64 = play(seed, F64)
    f32, a32 = play(seed, torch.float32)
    caps = f32.ctx.intervention_cap
    assert (f32.spent_intervention <= caps).all() and (f32.spent_audit <= f32.ctx.audit_cap).all()
    assert f32.spent_intervention.dtype == torch.float64 and f32.remaining_audit.dtype == F64        # accounting stays float64
    n64 = sum(int(getattr(a, f).sum()) for a in a64 for f in FIELDS if getattr(a, f) is not None)
    n32 = sum(int(getattr(a, f).sum()) for a in a32 for f in FIELDS if getattr(a, f) is not None)
    assert abs(n64 - n32) <= max(3, 0.25 * n64)       # documented tolerance: the number of actions agrees within 25%
    for a in a32:
        assert all(v.dtype == torch.float32 for v in a.statistics.values())


@pytest.mark.parametrize("device", DEVICES)
def test_every_output_tensor_lives_on_the_device(device):
    fake, out = play(1, F64, device)
    for a in out:
        for f in FIELDS + ("deny", "threshold_restricted"):
            m = getattr(a, f)
            if m is not None:
                assert m.device.type == device and m.dtype == torch.bool, f
        for v in a.statistics.values():
            assert v.device.type == device
