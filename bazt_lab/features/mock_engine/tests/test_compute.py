"""Device, precision and the uint8 mask algebra: results must not depend on how the tensors are computed."""
import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.mock_engine.private import bits
from bazt_lab.features.mock_engine.tests.helpers import Fn, make_settings, sim

Actions = core.Actions


def test_uint8_mask_algebra_equals_the_bool_operators_on_contiguous_expanded_and_transposed_masks():
    g = torch.Generator().manual_seed(0)
    a, b, c = (torch.rand(5, 6, 6, generator=g) < p for p in (0.3, 0.6, 0.5))
    row = torch.rand(5, 6, generator=g) < 0.5
    for x, y, z in ((a, b, c), (a.transpose(1, 2), b, c), (row[:, :, None].expand(5, 6, 6), b, c.transpose(1, 2))):
        assert torch.equal(bits.band(x, y, z), x & y & z)
        assert torch.equal(bits.bor(x, y, z), x | y | z)
        assert torch.equal(bits.bnot(x), ~x)
        assert torch.equal(bits.bandnot(x, y), x & ~y)
        for r in (bits.band(x, y), bits.bnot(x), bits.bor(x, y)):
            assert r.dtype == torch.bool


def _deny_hot(o):
    return Actions(deny=o.attempts & (o.monitor > 0.6)[:, :, None])


def test_the_context_carries_the_device_and_dtype_of_the_settings():
    seen = []

    def spy(o):
        seen.append(o)
        return Actions()
    for name, dt in (("float64", torch.float64), ("float32", torch.float32)):
        seen.clear()
        s = make_settings(compute__dtype=name)
        d = Fn(spy)
        sim("infection", d, T=3, settings=s)
        assert d.ctx.dtype == dt and d.ctx.device == torch.device("cpu")
        assert seen[0].monitor.dtype == dt and seen[0].remaining_audit.dtype == torch.float64
        assert d.ctx.audit_cap.dtype == torch.float64 and d.ctx.intervention_cap.dtype == torch.float64


def test_a_missing_compute_key_refuses_naming_it():
    s = make_settings()
    del s["compute"]["dtype"]
    with pytest.raises(core.Unset) as e:
        sim("infection", Fn(lambda o: Actions()), T=2, settings=s)
    assert e.value.args[0] == "compute.dtype"


def test_float32_agrees_with_float64_on_the_outcome_metrics():
    """Documented tolerance: over 96 episodes the infection and correctness traces agree on at least 99.9% of the entries,
    the final estimates differ by less than 1e-3, accuracy by less than 0.005, and the budget accounting (float64/int64
    whatever the dtype) is identical for at least 99% of the episodes. float32 rounds the comparisons u < p and
    |est - mean| <= eps, so a rare entry may flip; nothing else may change."""
    seeds = tuple(range(1, 97))
    kw = dict(attack="infection", T=8, topology="small_world", N=10, seeds=seeds, audit=0.2, intervention=0.2)
    _, t64, e64 = sim(defender=Fn(_deny_hot), settings=make_settings(compute__dtype="float64"), **kw)
    _, t32, e32 = sim(defender=Fn(_deny_hot), settings=make_settings(compute__dtype="float32"), **kw)
    assert e32.dtype == torch.float64 and t32.spend_intervention.dtype == torch.float64
    assert (t64.infected == t32.infected).double().mean() >= 0.999
    assert (t64.correct == t32.correct).double().mean() >= 0.999
    assert (e64 - e32).abs().max() < 1e-3
    assert abs(t64.correct[:, -1].double().mean() - t32.correct[:, -1].double().mean()) < 0.005
    same = (t64.spend_intervention == t32.spend_intervention).all(1) & (t64.spend_audit == t32.spend_audit).all(1)
    assert same.double().mean() >= 0.99


def test_the_compiled_kernels_are_the_eager_functions_when_compile_is_off():
    from bazt_lab.features.mock_engine.private import engine
    s = make_settings()
    assert core.compiled(engine._consensus, s, "mock_engine.consensus") is engine._consensus
    assert core.compiled(engine._write, s, "mock_engine.write") is engine._write


def test_trace_tensors_and_estimates_are_returned_on_the_cpu_in_float64():
    _, tr, est = sim("infection", Fn(lambda o: Actions()), T=3, settings=make_settings(compute__dtype="float32"))
    assert est.device.type == "cpu" and est.dtype == torch.float64
    assert all(t.device.type == "cpu" for t in (tr.infected, tr.active, tr.delivered, tr.spend_audit, tr.action_counts))
