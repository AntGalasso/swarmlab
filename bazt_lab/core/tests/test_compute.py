"""Compute layer: device, dtype, rounding-safe jitter and the verified torch.compile helper."""
import os
import threading

import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.core.private import compute


def cfg(device="cpu", use_compile=False, deterministic=True, dtype="float64"):
    return {"compute": {"device": device, "compile": use_compile, "deterministic": deterministic, "dtype": dtype}}


REAL_INDUCTOR_CHECK = compute.inductor_available


@pytest.fixture(autouse=True)
def compile_seam(monkeypatch):
    """Every test starts with no cached wrapper, on a machine where Inductor can build kernels: the seam tests replace
    `_compile` with what torch.compile would return there. Tests about the real backend put the real check back."""
    getattr(compute, "_WRAPPERS", {}).clear()
    monkeypatch.setattr(compute, "inductor_available", lambda device: True)
    yield
    getattr(compute, "_WRAPPERS", {}).clear()


@pytest.fixture
def fake_cuda(monkeypatch):
    """A machine that claims two CUDA devices; records the determinism switches instead of touching global state."""
    calls = {"deterministic": []}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda mode, **kw: calls["deterministic"].append(mode))
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    return calls


# ------------------------------------------------------------------------------------------------ device
def test_cpu_device_needs_no_cuda_and_touches_no_global_state(monkeypatch):
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *a, **k: pytest.fail("CPU must not switch determinism"))
    assert core.device(cfg("cpu")) == torch.device("cpu")


def test_cuda_is_refused_on_the_cpu_build_with_install_instructions():
    if torch.cuda.is_available():
        pytest.skip("this machine has CUDA")
    with pytest.raises(RuntimeError) as err:
        core.device(cfg("cuda"))
    msg = str(err.value)
    assert "compute.device" in msg and "'cuda'" in msg
    assert "CPU build" in msg and torch.__version__ in msg
    assert "uv" in msg and "download.pytorch.org/whl/cu" in msg and "nvidia-smi" not in msg


def test_cuda_with_a_cuda_build_but_no_gpu_says_so(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    with pytest.raises(RuntimeError, match="nvidia-smi"):
        core.device(cfg("cuda:0"))


def test_cuda_is_canonicalised_to_an_indexed_device_and_checked_against_the_count(fake_cuda):
    assert core.device(cfg("cuda")) == torch.device("cuda", 0)
    assert core.device(cfg("cuda:1")) == torch.device("cuda", 1)
    with pytest.raises(RuntimeError, match="only 2"):
        core.device(cfg("cuda:2"))


@pytest.mark.parametrize("bad", ["gpu", "CUDA", "cuda:", "cuda:-1", "cuda:x", "cpu:0", "mps", "", " cpu"])
def test_malformed_device_names_are_refused(bad):
    with pytest.raises(ValueError, match=r"compute\.device"):
        core.device(cfg(bad))


def test_device_setting_must_be_a_string_and_every_key_must_be_set():
    with pytest.raises(TypeError, match=r"compute\.device"):
        core.device(cfg(0))
    with pytest.raises(core.Unset):
        core.device({"compute": {"device": "cpu"}})                # compute.deterministic missing: no hidden defaults
    with pytest.raises(core.Unset):
        core.device({})


def test_deterministic_cuda_sets_the_cublas_workspace_before_any_op(fake_cuda, monkeypatch):
    core.device(cfg("cuda", deterministic=True))
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert fake_cuda["deterministic"] == [True]
    assert torch.backends.cudnn.benchmark is False and torch.backends.cudnn.deterministic is True


def test_a_valid_cublas_workspace_chosen_by_the_user_is_kept_and_a_bad_one_replaced(fake_cuda, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    core.device(cfg("cuda"))
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":0:0")
    core.device(cfg("cuda"))
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_late_determinism_warns_when_cuda_was_already_initialised(fake_cuda, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.warns(UserWarning, match="CUBLAS_WORKSPACE_CONFIG"):
        core.device(cfg("cuda"))


def test_non_deterministic_cuda_switches_the_algorithms_off_and_leaves_the_environment(fake_cuda):
    core.device(cfg("cuda", deterministic=False))
    assert fake_cuda["deterministic"] == [False]
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_deterministic_flag_must_be_a_bool():
    with pytest.raises(TypeError, match=r"compute\.deterministic"):
        core.device(cfg(deterministic="yes"))


# ------------------------------------------------------------------------------------------------ dtype
def test_dtype_reads_float64_and_float32():
    assert core.dtype(cfg(dtype="float64")) is torch.float64
    assert core.dtype(cfg(dtype="float32")) is torch.float32


@pytest.mark.parametrize("name", ["bfloat16", "float16", "bf16", "half"])
def test_half_precision_is_refused_with_the_reason(name):
    with pytest.raises(ValueError) as err:
        core.dtype(cfg(dtype=name))
    msg = str(err.value)
    assert "compute.dtype" in msg and name in msg
    assert "256" in msg and "2048" in msg and "budget" in msg and "eps_correct" in msg


@pytest.mark.parametrize("bad", ["float128", "int64", "double", "", "Float64"])
def test_unknown_dtypes_list_the_allowed_ones(bad):
    with pytest.raises(ValueError, match=r"float64.*float32"):
        core.dtype(cfg(dtype=bad))
    with pytest.raises(TypeError, match=r"compute\.dtype"):
        core.dtype(cfg(dtype=32))
    with pytest.raises(core.Unset):
        core.dtype({"compute": {}})


# ------------------------------------------------------------------------------------------------ jitter
def test_tie_jitter_scale_is_eight_eps_times_the_magnitude():
    assert core.tie_jitter_scale(torch.float64, 50.0) == pytest.approx(8 * torch.finfo(torch.float64).eps * 50.0, rel=0, abs=0)
    assert core.tie_jitter_scale(torch.float32, 1.0) == pytest.approx(8 * torch.finfo(torch.float32).eps, rel=0, abs=0)
    assert core.tie_jitter_scale(torch.float32, -3) == core.tie_jitter_scale(torch.float32, 3)
    assert core.tie_jitter_scale(torch.float32, 0.0) == 0.0
    scaled = core.tie_jitter_scale(torch.float32, torch.tensor([1.0, -4.0], dtype=torch.float64))
    assert torch.equal(scaled, torch.tensor([1.0, 4.0], dtype=torch.float64) * (8 * torch.finfo(torch.float32).eps))


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_tie_jitter_survives_rounding_at_every_magnitude(dtype):
    for magnitude in (1e-3, 0.37, 1.0, 1.999, 2.0, 50.0, 63.99, 64.0, 1000.0, 123456.0):
        v = torch.tensor(magnitude, dtype=dtype)
        jitter = torch.tensor(core.tie_jitter_scale(dtype, magnitude), dtype=dtype)
        assert v + jitter > v and v - jitter < v, magnitude
        assert (v + jitter) - v >= jitter / 2, magnitude


def test_tie_jitter_refuses_non_float_and_half_dtypes():
    for bad in (torch.float16, torch.bfloat16, torch.int64):
        with pytest.raises(ValueError, match="dtype"):
            core.tie_jitter_scale(bad, 1.0)


# ------------------------------------------------------------------------------------------------ mask helpers
def test_as_uint8_and_as_bool_reinterpret_without_copying():
    mask = torch.tensor([[True, False, True], [False, False, True]])
    u8 = core.as_uint8(mask)
    assert u8.dtype == torch.uint8 and u8.shape == mask.shape and u8.data_ptr() == mask.data_ptr()
    assert u8.tolist() == [[1, 0, 1], [0, 0, 1]]
    back = core.as_bool(u8)
    assert back.dtype == torch.bool and back.data_ptr() == mask.data_ptr() and torch.equal(back, mask)


def test_uint8_logic_on_zero_one_masks_equals_bool_logic():
    g = torch.Generator().manual_seed(3)
    a, b = torch.rand(1000, generator=g) < 0.5, torch.rand(1000, generator=g) < 0.5
    ua, ub = core.as_uint8(a), core.as_uint8(b)
    assert torch.equal(core.as_bool(ua & ub), a & b)
    assert torch.equal(core.as_bool(ua | ub), a | b)
    assert torch.equal(core.as_bool(ua ^ ub), a ^ b)
    assert torch.equal(core.as_bool(ua ^ 1), ~a)                           # not: xor 1, never ~ (which gives 254 and 255)
    assert torch.equal(core.as_bool(ua * ub), a & b)


def test_as_uint8_works_on_views_and_as_bool_can_check_its_input():
    mask = torch.rand(6, 4, generator=torch.Generator().manual_seed(1)) < 0.5
    column = mask[:, 1]
    assert core.as_uint8(column).tolist() == column.long().tolist()
    with pytest.raises(ValueError, match="0 or 1"):
        core.as_bool(torch.tensor([0, 1, 2], dtype=torch.uint8), checked=True)
    assert core.as_bool(torch.tensor([0, 1, 1], dtype=torch.uint8), checked=True).tolist() == [False, True, True]


def test_mask_helpers_refuse_the_wrong_dtype():
    with pytest.raises(TypeError, match="bool"):
        core.as_uint8(torch.ones(2, dtype=torch.uint8))
    with pytest.raises(TypeError, match="uint8"):
        core.as_bool(torch.ones(2, dtype=torch.bool))
    with pytest.raises(TypeError, match="uint8"):
        core.as_bool(torch.ones(2, dtype=torch.int64))


# ------------------------------------------------------------------------------------------------ contracts
def test_episode_context_defaults_to_cpu_float64_and_accepts_overrides():
    base = dict(B=1, N=2, T=3, adj=torch.ones(1, 2, 2, dtype=torch.bool), sink=torch.zeros(1, dtype=torch.int64),
                audit_cap=torch.ones(1, dtype=torch.float64), intervention_cap=torch.ones(1, dtype=torch.float64),
                costs={}, thresholds={}, settings={}, arm="A", unit_ids=("u",), arm_stream=lambda ns, b: None)
    ctx = core.EpisodeContext(**base)
    assert ctx.device == torch.device("cpu") and ctx.dtype is torch.float64
    other = core.EpisodeContext(**base, device=torch.device("cuda", 0), dtype=torch.float32)
    assert other.device == torch.device("cuda", 0) and other.dtype is torch.float32


def test_the_contract_docstring_says_which_tensors_follow_the_dtype():
    from bazt_lab.core.private import contracts
    doc = contracts.__doc__
    assert "ctx.dtype" in doc and "ctx.device" in doc and "tie_jitter_scale" in doc
    assert "never follow" in doc.lower() or "never follows" in doc.lower()
    assert "float64" in doc and "int64" in doc and "budget" in doc


# ------------------------------------------------------------------------------------------------ compiled
class Counter:
    """A function that counts its calls, so tests can see how often the eager path ran."""

    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.fn(*args, **kwargs)


def double(x):
    return x * 2.0


def test_compile_off_returns_the_function_itself_and_needs_no_dtype():
    assert core.compiled(double, {"compute": {"compile": False}}, "double") is double


def test_compile_flag_must_be_a_bool_and_set():
    with pytest.raises(TypeError, match=r"compute\.compile"):
        core.compiled(double, cfg(use_compile="true"), "double")
    with pytest.raises(core.Unset):
        core.compiled(double, {}, "double")


def test_a_correct_compiled_function_is_verified_once_and_used_afterwards(monkeypatch, capsys):
    built = []
    monkeypatch.setattr(compute, "_compile", lambda fn: built.append(fn) or fn)
    fn = Counter(double)
    wrapped = core.compiled(fn, cfg(use_compile=True), "double")
    assert wrapped is not fn and len(built) == 0                          # nothing compiled before the first call
    x = torch.arange(4.0, dtype=torch.float64)
    assert torch.equal(wrapped(x), x * 2) and len(built) == 1
    after_first = fn.calls
    for _ in range(5):
        assert torch.equal(wrapped(x), x * 2)
    assert fn.calls - after_first == 5 and len(built) == 1                # later calls: one call each, no re-check
    assert capsys.readouterr().err == ""


def test_the_wrapper_keeps_the_name_of_the_function(monkeypatch):
    monkeypatch.setattr(compute, "_compile", lambda fn: fn)
    assert core.compiled(double, cfg(use_compile=True), "double").__name__ == "double"


def test_a_compile_error_falls_back_to_eager_and_logs_one_line(monkeypatch, capsys):
    def broken_compile(fn):
        def run(*a, **k):
            raise RuntimeError("InvalidCxxCompiler: cl is not found\nsecond line")
        return run

    monkeypatch.setattr(compute, "_compile", broken_compile)
    fn = Counter(double)
    wrapped = core.compiled(fn, cfg(use_compile=True), "double_kernel")
    x = torch.ones(3, dtype=torch.float64)
    for _ in range(4):
        assert torch.equal(wrapped(x), x * 2)
    err = capsys.readouterr().err
    assert len(err.strip().splitlines()) == 1
    assert "double_kernel" in err and "InvalidCxxCompiler" in err and "eager" in err
    assert "second line" not in err                                       # one line: only the first line of the reason


def test_an_error_while_building_the_compiled_function_also_falls_back(monkeypatch, capsys):
    def explode(fn):
        raise ImportError("no triton")

    monkeypatch.setattr(compute, "_compile", explode)
    wrapped = core.compiled(double, cfg(use_compile=True), "k")
    assert torch.equal(wrapped(torch.ones(2, dtype=torch.float64)), torch.full((2,), 2.0, dtype=torch.float64))
    err = capsys.readouterr().err
    assert "k" in err and "no triton" in err and len(err.strip().splitlines()) == 1


def test_a_wrong_compiled_result_is_detected_and_eager_is_used_from_then_on(monkeypatch, capsys):
    monkeypatch.setattr(compute, "_compile", lambda fn: (lambda x: fn(x) + 1e-3))
    wrapped = core.compiled(double, cfg(use_compile=True), "drifting")
    x = torch.ones(3, dtype=torch.float64)
    for _ in range(3):
        assert torch.equal(wrapped(x), x * 2)                             # always the eager answer
    err = capsys.readouterr().err
    assert len(err.strip().splitlines()) == 1 and "drifting" in err and "differs" in err


def test_integers_and_bools_must_match_exactly(monkeypatch, capsys):
    monkeypatch.setattr(compute, "_compile", lambda fn: (lambda x: tuple(o + (o.dtype != torch.bool) for o in fn(x))))
    wrapped = core.compiled(lambda x: (x > 0, x.long()), cfg(use_compile=True), "exact")
    out = wrapped(torch.tensor([1.0, -1.0], dtype=torch.float64))
    assert out[0].tolist() == [True, False] and out[1].tolist() == [1, -1]
    assert "differs" in capsys.readouterr().err
    monkeypatch.setattr(compute, "_compile", lambda fn: (lambda x: (~fn(x)[0], fn(x)[1])))
    flip = core.compiled(lambda x: (x > 0, x.long()), cfg(use_compile=True), "flip")
    assert flip(torch.tensor([1.0, -1.0], dtype=torch.float64))[0].tolist() == [True, False]
    assert "flip" in capsys.readouterr().err


def _perturb(rel):
    return lambda fn: (lambda x: fn(x) * (1 + rel))


def test_float64_tolerance_is_rtol_1e9_atol_1e12(monkeypatch, capsys):
    x = torch.full((3,), 2.0, dtype=torch.float64)
    monkeypatch.setattr(compute, "_compile", _perturb(1e-10))
    ok = core.compiled(double, cfg(use_compile=True, dtype="float64"), "tight_ok")
    ok(x)
    assert capsys.readouterr().err == ""
    monkeypatch.setattr(compute, "_compile", _perturb(1e-8))
    bad = core.compiled(double, cfg(use_compile=True, dtype="float64"), "tight_bad")
    bad(x)
    assert "tight_bad" in capsys.readouterr().err
    tiny = torch.zeros(3, dtype=torch.float64)                            # near zero atol decides: 1e-13 passes, 1e-11 does not
    monkeypatch.setattr(compute, "_compile", lambda fn: (lambda v: fn(v) + 1e-13))
    core.compiled(double, cfg(use_compile=True), "atol_ok")(tiny)
    assert capsys.readouterr().err == ""
    monkeypatch.setattr(compute, "_compile", lambda fn: (lambda v: fn(v) + 1e-11))
    core.compiled(double, cfg(use_compile=True), "atol_bad")(tiny)
    assert "atol_bad" in capsys.readouterr().err


def test_float32_setting_uses_rtol_1e5_atol_1e6(monkeypatch, capsys):
    x = torch.full((3,), 2.0, dtype=torch.float32)
    monkeypatch.setattr(compute, "_compile", _perturb(2e-6))
    core.compiled(double, cfg(use_compile=True, dtype="float32"), "loose_ok")(x)
    assert capsys.readouterr().err == ""
    monkeypatch.setattr(compute, "_compile", _perturb(1e-4))
    core.compiled(double, cfg(use_compile=True, dtype="float32"), "loose_bad")(x)
    assert "loose_bad" in capsys.readouterr().err
    zero = torch.zeros(3, dtype=torch.float32)
    monkeypatch.setattr(compute, "_compile", lambda fn: (lambda v: fn(v) + 5e-7))
    core.compiled(double, cfg(use_compile=True, dtype="float32"), "atol32_ok")(zero)
    assert capsys.readouterr().err == ""


def test_the_looser_of_the_setting_and_the_tensor_dtype_applies(monkeypatch, capsys):
    monkeypatch.setattr(compute, "_compile", _perturb(2e-6))
    core.compiled(double, cfg(use_compile=True, dtype="float64"), "f64_out_f64_set")(torch.full((2,), 2.0, dtype=torch.float64))
    assert "f64_out_f64_set" in capsys.readouterr().err                   # float64 output, float64 setting: strict
    core.compiled(double, cfg(use_compile=True, dtype="float64"), "f32_out")(torch.full((2,), 2.0, dtype=torch.float32))
    assert capsys.readouterr().err == ""                                   # float32 output is judged at float32 resolution
    core.compiled(double, cfg(use_compile=True, dtype="float32"), "f64_out_f32_set")(torch.full((2,), 2.0, dtype=torch.float64))
    assert capsys.readouterr().err == ""                                   # float32 run: float64 outputs may carry float32 error


def test_nan_in_both_results_counts_as_equal_but_one_sided_nan_does_not(monkeypatch, capsys):
    monkeypatch.setattr(compute, "_compile", lambda fn: fn)
    core.compiled(lambda x: x / 0.0, cfg(use_compile=True), "nan_same")(torch.zeros(2, dtype=torch.float64))
    assert capsys.readouterr().err == ""
    monkeypatch.setattr(compute, "_compile", lambda fn: (lambda x: torch.full_like(x, float("nan"))))
    core.compiled(double, cfg(use_compile=True), "nan_only_compiled")(torch.ones(2, dtype=torch.float64))
    assert "nan_only_compiled" in capsys.readouterr().err


def test_shape_dtype_structure_and_scalars_are_compared(monkeypatch, capsys):
    cases = {
        "shape": (lambda x: x * 2, lambda x: (x * 2).reshape(1, -1)),
        "dtype": (lambda x: x * 2, lambda x: (x * 2).to(torch.float32)),
        "length": (lambda x: (x, x), lambda x: (x,)),
        "type": (lambda x: [x], lambda x: (x,)),
        "keys": (lambda x: {"a": x}, lambda x: {"b": x}),
        "scalar": (lambda x: 3, lambda x: 4),
        "float_scalar": (lambda x: 1.0, lambda x: 1.1),
        "none": (lambda x: None, lambda x: x),
    }
    x = torch.ones(2, dtype=torch.float64)
    for label, (eager, fake) in cases.items():
        monkeypatch.setattr(compute, "_compile", lambda fn, fake=fake: fake)
        core.compiled(eager, cfg(use_compile=True), label)(x)
        assert label in capsys.readouterr().err, label
    monkeypatch.setattr(compute, "_compile", lambda fn: fn)
    same = core.compiled(lambda x: {"a": [x, 1, 2.5, "s", None, True]}, cfg(use_compile=True), "nested")
    out = same(x)
    assert out["a"][1:] == [1, 2.5, "s", None, True] and capsys.readouterr().err == ""


def test_outputs_that_cannot_be_compared_fall_back_instead_of_crashing(monkeypatch, capsys):
    class Opaque:
        def __eq__(self, other):
            raise RuntimeError("no comparison for this object")

        __hash__ = None

    monkeypatch.setattr(compute, "_compile", lambda fn: fn)
    wrapped = core.compiled(lambda x: Opaque(), cfg(use_compile=True), "opaque")
    assert isinstance(wrapped(torch.ones(1)), Opaque) and isinstance(wrapped(torch.ones(1)), Opaque)
    err = capsys.readouterr().err
    assert len(err.strip().splitlines()) == 1 and "opaque" in err and "could not compare" in err


def test_inputs_mutated_in_place_are_mutated_exactly_once_per_call(monkeypatch):
    monkeypatch.setattr(compute, "_compile", lambda fn: fn)

    def spend(budget, amount):
        budget -= amount                                                   # in-place accounting, like the platform does
        return budget.clone()

    wrapped = core.compiled(spend, cfg(use_compile=True), "spend")
    budget = torch.full((3,), 10.0, dtype=torch.float64)
    for expected in (9.0, 8.0, 7.0, 6.0):                                  # the first call is the verified one
        out = wrapped(budget, 1.0)
        assert out.tolist() == [expected] * 3 and budget.tolist() == [expected] * 3


def test_an_error_in_the_eager_function_itself_is_raised_not_swallowed(monkeypatch, capsys):
    monkeypatch.setattr(compute, "_compile", lambda fn: fn)

    def bad(x):
        raise ValueError("shape bug in the user's function")

    wrapped = core.compiled(bad, cfg(use_compile=True), "bad")
    with pytest.raises(ValueError, match="shape bug"):
        wrapped(torch.ones(1))
    assert capsys.readouterr().err == ""
    with pytest.raises(ValueError, match="shape bug"):
        wrapped(torch.ones(1))


def test_a_compiled_function_that_fails_on_a_later_call_falls_back_once(monkeypatch, capsys):
    state = {"n": 0}

    def flaky(fn):
        def run(x):
            state["n"] += 1
            if state["n"] > 2:                                             # verification passes, a later recompile fails
                raise RuntimeError("recompile limit hit")
            return fn(x)
        return run

    monkeypatch.setattr(compute, "_compile", flaky)
    wrapped = core.compiled(double, cfg(use_compile=True), "late")
    x = torch.ones(2, dtype=torch.float64)
    for _ in range(5):
        assert torch.equal(wrapped(x), x * 2)
    err = capsys.readouterr().err
    assert len(err.strip().splitlines()) == 1 and "late" in err and "recompile limit hit" in err


def test_concurrent_first_calls_verify_once(monkeypatch, capsys):
    built = []
    gate = threading.Barrier(6)

    def slow_compile(fn):
        built.append(1)
        return fn

    monkeypatch.setattr(compute, "_compile", slow_compile)
    wrapped = core.compiled(double, cfg(use_compile=True), "threads")
    results = []

    def worker():
        gate.wait()
        results.append(wrapped(torch.ones(2, dtype=torch.float64)))

    threads = [threading.Thread(target=worker) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(results) == 6 and all(torch.equal(r, torch.full((2,), 2.0, dtype=torch.float64)) for r in results)
    assert len(built) == 1 and capsys.readouterr().err == ""


# real torch.compile -----------------------------------------------------------------------------------------
def test_a_real_dynamo_trace_with_the_eager_backend_is_verified_and_used(monkeypatch, capsys):
    """Dynamo itself (graph capture, dynamic shapes) without Inductor, so this runs on machines with no C++ compiler."""
    traced = []
    monkeypatch.setattr(compute, "_compile", lambda fn: traced.append(fn) or torch.compile(fn, backend="eager", dynamic=True))

    def kernel(x, y):
        return torch.sqrt(x * x + y) - 0.5 * x, (x > y)

    wrapped = core.compiled(kernel, cfg(use_compile=True), "dynamo_kernel")
    for size in (64, 7, 1000):                                            # dynamic=True: new sizes, one compile
        x, y = torch.rand(size, dtype=torch.float64), torch.rand(size, dtype=torch.float64)
        got, want = wrapped(x, y), kernel(x, y)
        assert torch.allclose(got[0], want[0], rtol=1e-12, atol=1e-12) and torch.equal(got[1], want[1])
    assert len(traced) == 1 and capsys.readouterr().err == ""             # verified once, no fallback line


@pytest.mark.skipif(not compute.inductor_available("cpu"), reason="Inductor cannot build CPU kernels here (no C++ compiler: cl.exe/g++)")
def test_real_torch_compile_matches_eager_and_logs_nothing(monkeypatch, capsys):
    monkeypatch.setattr(compute, "inductor_available", REAL_INDUCTOR_CHECK)

    def kernel(x, y):
        return torch.sqrt(x * x + y) - 0.5 * x

    wrapped = core.compiled(kernel, cfg(use_compile=True), "kernel")
    x, y = torch.rand(64, dtype=torch.float64), torch.rand(64, dtype=torch.float64)
    assert torch.allclose(wrapped(x, y), kernel(x, y), rtol=1e-9, atol=1e-12)
    assert torch.allclose(wrapped(x[:7], y[:7]), kernel(x[:7], y[:7]), rtol=1e-9, atol=1e-12)     # dynamic=True: new size
    assert capsys.readouterr().err == ""


@pytest.mark.skipif(compute.inductor_available("cpu"), reason="Inductor works here: the fallback is covered by the seam tests")
def test_real_torch_compile_without_a_compiler_is_never_tried_and_logs_one_line(monkeypatch, capsys):
    monkeypatch.setattr(compute, "inductor_available", REAL_INDUCTOR_CHECK)
    monkeypatch.setattr(compute, "_compile", lambda fn: pytest.fail("torch.compile tried on a machine without a compiler"))
    x = torch.arange(3.0, dtype=torch.float64)
    for _ in range(3):                                                    # one call per batch, as mock_engine.simulate does
        wrapped = core.compiled(double, cfg(use_compile=True), "no_compiler")
        assert torch.equal(wrapped(x), x * 2) and torch.equal(wrapped(x), x * 2)
    err = capsys.readouterr().err
    assert len(err.strip().splitlines()) == 1 and "no_compiler" in err and "eager" in err


# one wrapper per kernel and settings, per process -------------------------------------------------------------
def test_the_same_kernel_name_and_settings_give_the_same_wrapper_verified_once(monkeypatch, capsys):
    built = []
    monkeypatch.setattr(compute, "_compile", lambda fn: built.append(fn) or fn)
    fn = Counter(double)
    x = torch.arange(4.0, dtype=torch.float64)
    first = core.compiled(fn, cfg(use_compile=True), "mock_engine.consensus")
    first(x)
    after_check = fn.calls
    second = core.compiled(fn, cfg(use_compile=True), "mock_engine.consensus")       # the next batch
    assert second is first
    assert torch.equal(second(x), x * 2) and len(built) == 1 and fn.calls - after_check == 1    # no second check
    assert capsys.readouterr().err == ""


def test_a_fallback_is_decided_and_logged_once_for_every_later_batch(monkeypatch, capsys):
    attempts = []

    def broken_compile(fn):
        attempts.append(fn)
        raise RuntimeError("InvalidCxxCompiler: cl is not found")

    monkeypatch.setattr(compute, "_compile", broken_compile)
    x = torch.ones(3, dtype=torch.float64)
    for _ in range(4):
        assert torch.equal(core.compiled(double, cfg(use_compile=True), "mock_engine.write")(x), x * 2)
    err = capsys.readouterr().err
    assert len(attempts) == 1 and len(err.strip().splitlines()) == 1 and "mock_engine.write" in err


def test_without_inductor_for_the_device_compile_is_never_tried_and_one_line_names_the_reason(monkeypatch, capsys):
    monkeypatch.setattr(compute, "inductor_available", lambda device: torch.device(device).type == "cuda")
    monkeypatch.setattr(compute, "_compile", lambda fn: pytest.fail("torch.compile tried without Inductor"))
    x = torch.ones(2, dtype=torch.float64)
    for _ in range(3):
        assert torch.equal(core.compiled(double, cfg(use_compile=True), "k")(x), x * 2)
    err = capsys.readouterr().err
    assert len(err.strip().splitlines()) == 1 and "'k'" in err and "Inductor" in err and "cpu" in err and "eager" in err


def test_a_different_name_dtype_or_device_gets_its_own_wrapper(monkeypatch):
    monkeypatch.setattr(compute, "_compile", lambda fn: fn)
    base = core.compiled(double, cfg(use_compile=True), "k")
    others = [core.compiled(double, cfg(use_compile=True), "k2"),
              core.compiled(double, cfg(use_compile=True, dtype="float32"), "k"),
              core.compiled(double, cfg(device="cuda", use_compile=True), "k")]
    assert all(o is not base for o in others) and len({id(o) for o in others}) == 3
    assert core.compiled(double, cfg(use_compile=True, deterministic=False), "k") is base    # determinism: same kernel
    with pytest.raises(ValueError, match=r"compute\.device"):
        core.compiled(double, cfg(device="gpu", use_compile=True), "k")


def test_a_kernel_that_cannot_be_hashed_still_gets_a_working_wrapper(monkeypatch):
    class Unhashable(Counter):
        __hash__ = None

    monkeypatch.setattr(compute, "_compile", lambda fn: fn)
    fn = Unhashable(double)
    x = torch.ones(2, dtype=torch.float64)
    assert torch.equal(core.compiled(fn, cfg(use_compile=True), "u")(x), x * 2)
