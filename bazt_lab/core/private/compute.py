"""Compute layer: device, precision and a verified torch.compile, all driven by the settings table [compute].

    [compute]
    device        "cpu" | "cuda" | "cuda:N"
    dtype         "float64" | "float32"   (half precision is refused, see `dtype`)
    compile       bool: wrap hot kernels in torch.compile, verified against eager on the first call
    deterministic bool: with a CUDA device, deterministic algorithms and the cuBLAS workspace setting

There are no defaults here: a missing key raises `Unset` (settings.py). Call `device(settings)` once, at the start of
a run and before the first CUDA tensor is created, because the determinism switches below only take effect for
operations that run after them. References (PyTorch 2.14): notes/randomness, torch.use_deterministic_algorithms,
torch.compile, torch.device. The cuBLAS requirement (CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8 for deterministic matmuls)
is the one torch itself names in its RuntimeError when the variable is missing.
"""
from __future__ import annotations

import functools
import importlib.util
import math
import os
import re
import shutil
import sys
import threading
import warnings
from typing import Any, Callable, Mapping

import torch

from bazt_lab.core.private.settings import get

_DEVICE_NAME = re.compile(r"cpu|cuda(?::([0-9]+))?")
_DTYPES = {"float64": torch.float64, "float32": torch.float32}
_REFUSED_DTYPES = {"bfloat16", "float16", "bf16", "fp16", "half"}
_CUBLAS_VALID = (":4096:8", ":16:8")
_CUBLAS_DEFAULT = ":4096:8"

# (rtol, atol) of the compiled-versus-eager check per dtype. float64 is the experiment's reference precision (eps 2.2e-16);
# float32 has eps 1.2e-7, so rtol 1e-5 allows about 80 ulp of reordering error from fused kernels.
_TOLERANCE = {
    torch.float64: (1e-9, 1e-12), torch.complex128: (1e-9, 1e-12),
    torch.float32: (1e-5, 1e-6), torch.complex64: (1e-5, 1e-6),
    torch.float16: (1e-3, 1e-4), torch.bfloat16: (1e-2, 1e-3),
}


def _flag(settings: Mapping, key: str) -> bool:
    value = get(settings, key)
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be true or false, got {value!r}")
    return value


# ------------------------------------------------------------------------------------------------ device
def _cuda_unavailable_message(spec: str) -> str:
    version = torch.__version__
    if torch.version.cuda is None:
        pin = version.split("+")[0]
        return (
            f"compute.device is {spec!r} but the installed torch ({version}) is the CPU build, which cannot use a GPU. "
            f"Install the CUDA build with uv: in pyproject.toml point [[tool.uv.index]] / [tool.uv.sources] for torch at "
            f"https://download.pytorch.org/whl/cu<XXX> (pick the <XXX> that matches your driver on "
            f"https://pytorch.org/get-started/locally/) and run `uv sync`; or, without editing the project, "
            f"`uv pip install torch=={pin} --index-url https://download.pytorch.org/whl/cu<XXX>` (a later `uv sync` "
            f"reinstalls the CPU wheel while pyproject.toml names the CPU index). Or set compute.device = \"cpu\"."
        )
    return (
        f"compute.device is {spec!r} but torch ({version}, built with CUDA {torch.version.cuda}) reports no usable GPU: "
        f"torch.cuda.is_available() is False. Check `nvidia-smi`, that the driver supports CUDA {torch.version.cuda}, "
        f"and CUDA_VISIBLE_DEVICES. Or set compute.device = \"cpu\"."
    )


def _make_deterministic(on: bool) -> None:
    """CUDA determinism switches. cuBLAS reads CUBLAS_WORKSPACE_CONFIG when its workspace is created (first matmul),
    so it must be in the environment before the first CUDA op; torch refuses cuBLAS ops in deterministic mode without it."""
    if on:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in _CUBLAS_VALID:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_DEFAULT
            if torch.cuda.is_initialized():
                warnings.warn(
                    "CUDA was already initialised when CUBLAS_WORKSPACE_CONFIG was set to ':4096:8'; if a matmul raises "
                    "a determinism error, export CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python.",
                    stacklevel=3,
                )
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(on)


def _device_spec(settings: Mapping) -> tuple[str, re.Match]:
    """compute.device checked for form only (no CUDA query, no global switch)."""
    spec = get(settings, "compute.device")
    if not isinstance(spec, str):
        raise TypeError(f"compute.device must be a string ('cpu', 'cuda' or 'cuda:N'), got {spec!r}")
    match = _DEVICE_NAME.fullmatch(spec)
    if match is None:
        raise ValueError(f"compute.device must be 'cpu', 'cuda' or 'cuda:N', got {spec!r}")
    return spec, match


def resolve_device(settings: Mapping) -> torch.device:
    """The torch.device named by compute.device. "cuda" becomes "cuda:<current>" so it equals `tensor.device`.

    Refuses CUDA with an explanation when torch cannot use it. On CUDA, compute.deterministic switches
    torch.use_deterministic_algorithms and the cuBLAS workspace setting on (or the algorithms off); on CPU it changes nothing."""
    spec, match = _device_spec(settings)
    deterministic = _flag(settings, "compute.deterministic")
    if spec == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(_cuda_unavailable_message(spec))
    index = int(match.group(1)) if match.group(1) is not None else torch.cuda.current_device()
    count = torch.cuda.device_count()
    if index >= count:
        raise RuntimeError(f"compute.device is {spec!r} but this machine has only {count} CUDA device(s)")
    _make_deterministic(deterministic)
    return torch.device("cuda", index)


# ------------------------------------------------------------------------------------------------ dtype
def resolve_dtype(settings: Mapping) -> torch.dtype:
    """torch.float64 or torch.float32 from compute.dtype; half precision is refused: bfloat16 and float16 hold integers
    exactly only up to 256 and 2048, so audit and intervention budget sums break, and their resolution near 50 breaks the
    eps_correct test."""
    name = get(settings, "compute.dtype")
    if not isinstance(name, str):
        raise TypeError(f"compute.dtype must be 'float64' or 'float32', got {name!r}")
    if name in _REFUSED_DTYPES:
        raise ValueError(
            f"compute.dtype={name!r} is refused: bfloat16 and float16 represent integers exactly only up to 256 and 2048, "
            f"so audit and intervention budget sums break, and their resolution near 50 breaks the eps_correct test. "
            f"Use 'float64' or 'float32'."
        )
    if name not in _DTYPES:
        raise ValueError(f"compute.dtype must be 'float64' or 'float32', got {name!r}")
    return _DTYPES[name]


def tie_jitter_scale(dtype: torch.dtype, magnitude):
    """Amplitude of a tie-breaking jitter that survives rounding at values of size `magnitude`: 8 x eps(dtype) x |magnitude|
    (at least eight ulp, so v + jitter differs from v in `dtype`). `magnitude` may be a number or a tensor."""
    if dtype not in _DTYPES.values():
        raise ValueError(f"dtype must be torch.float64 or torch.float32, got {dtype}")
    scale = 8 * torch.finfo(dtype).eps
    return scale * magnitude.abs() if isinstance(magnitude, torch.Tensor) else scale * abs(magnitude)


# ------------------------------------------------------------------------------------------------ masks
def as_uint8(mask: torch.Tensor) -> torch.Tensor:
    """A bool mask as uint8 (values 0 and 1), no copy: same memory, any shape or strides, any device. On CPU torch the
    bitwise `&`, `|`, `^` and `*` of uint8 masks are many times faster than the bool kernels. Use `x ^ 1` for not (`~`
    would give 254 and 255) and convert back with `as_bool` only where a bool mask is needed (indexing, torch.where)."""
    if mask.dtype != torch.bool:
        raise TypeError(f"as_uint8 needs a bool tensor, got {mask.dtype}")
    return mask.view(torch.uint8)


def as_bool(u8: torch.Tensor, checked: bool = False) -> torch.Tensor:
    """A uint8 tensor holding only 0 and 1 as bool, no copy. Other values in a bool tensor are undefined behaviour, so
    anything that may hold them (`~`, sums, arithmetic) must go through `u8 != 0` instead; `checked=True` verifies the
    input (one pass, and a device sync on CUDA)."""
    if u8.dtype != torch.uint8:
        raise TypeError(f"as_bool needs a uint8 tensor, got {u8.dtype}")
    if checked and bool((u8 > 1).any()):
        raise ValueError("as_bool needs values 0 or 1")
    return u8.view(torch.bool)


# ------------------------------------------------------------------------------------------------ compile
def inductor_available(device) -> bool:
    """A cheap, conservative check that torch.compile (Inductor) can build kernels for `device`: Triton for CUDA, a C++
    compiler (cl on Windows, g++ or clang++ elsewhere, or $CXX) for CPU. It does not compile anything."""
    if torch.device(device).type == "cuda":
        return torch.cuda.is_available() and importlib.util.find_spec("triton") is not None
    candidates = [os.environ["CXX"]] if os.environ.get("CXX") else (["cl"] if sys.platform == "win32" else ["g++", "clang++"])
    return any(shutil.which(c) for c in candidates)


def _compile(fn: Callable) -> Callable:
    """The one place torch.compile is called (tests replace it to inject a wrong or failing 'compiled' function)."""
    return torch.compile(fn, dynamic=True, fullgraph=False)


def _clone(obj: Any) -> Any:
    """A copy to run on: tensors, generators and plain list/tuple/dict containers are copied, other objects are shared."""
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, torch.Generator):
        twin = torch.Generator(device=obj.device)
        twin.set_state(obj.get_state())
        return twin
    if type(obj) in (list, tuple):
        return type(obj)(_clone(item) for item in obj)
    if type(obj) is dict:
        return {key: _clone(value) for key, value in obj.items()}
    return obj


def _tolerance(tensor_dtype: torch.dtype, setting_dtype: torch.dtype) -> tuple[float, float]:
    """The looser of the setting's tolerance and the tensor's own: a float32 run may carry float32 error in float64
    outputs, and a float32 tensor is never judged finer than float32 resolution."""
    rtol, atol = _TOLERANCE[setting_dtype]
    own = _TOLERANCE.get(tensor_dtype, (rtol, atol))
    return max(rtol, own[0]), max(atol, own[1])


def _mismatch(eager: Any, fast: Any, setting_dtype: torch.dtype, path: str = "result") -> str | None:
    """None when `fast` equals `eager` (bool and int exactly, floats within tolerance), else where they differ."""
    if isinstance(eager, torch.Tensor) or isinstance(fast, torch.Tensor):
        if not (isinstance(eager, torch.Tensor) and isinstance(fast, torch.Tensor)):
            return f"{path}: {type(eager).__name__} vs {type(fast).__name__}"
        if eager.shape != fast.shape or eager.dtype != fast.dtype or eager.device != fast.device:
            return f"{path}: {tuple(eager.shape)} {eager.dtype} {eager.device} vs {tuple(fast.shape)} {fast.dtype} {fast.device}"
        if eager.is_floating_point() or eager.is_complex():
            rtol, atol = _tolerance(eager.dtype, setting_dtype)
            if not torch.allclose(eager, fast, rtol=rtol, atol=atol, equal_nan=True):
                worst = (eager - fast).abs().nan_to_num(nan=0.0).max().item()
                return f"{path}: max abs difference {worst:.3g} beyond rtol {rtol:g}, atol {atol:g}"
        elif not torch.equal(eager, fast):
            return f"{path}: {eager.dtype} values differ"
        return None
    if type(eager) is not type(fast):
        return f"{path}: {type(eager).__name__} vs {type(fast).__name__}"
    if isinstance(eager, (list, tuple)):
        if len(eager) != len(fast):
            return f"{path}: length {len(eager)} vs {len(fast)}"
        return next((m for i, (a, b) in enumerate(zip(eager, fast, strict=True))
                     if (m := _mismatch(a, b, setting_dtype, f"{path}[{i}]"))), None)
    if isinstance(eager, dict):
        if eager.keys() != fast.keys():
            return f"{path}: keys {sorted(map(str, eager))} vs {sorted(map(str, fast))}"
        return next((m for k in eager if (m := _mismatch(eager[k], fast[k], setting_dtype, f"{path}[{k!r}]"))), None)
    if isinstance(eager, float):
        rtol, atol = _TOLERANCE[setting_dtype]
        same = (math.isnan(eager) and math.isnan(fast)) or math.isclose(eager, fast, rel_tol=rtol, abs_tol=atol)
        return None if same else f"{path}: {eager!r} vs {fast!r}"
    return None if eager == fast else f"{path}: {eager!r} vs {fast!r}"


def _one_line(text: object) -> str:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    return (lines[0] if lines else "")[:240]


class _VerifiedCompile:
    """`fn` under torch.compile once the first call has shown that it agrees with eager; eager otherwise.

    First call (under a lock): run eager and compiled on copies of the inputs, compare every output, then serve the call
    from whichever path won, on the real inputs (so a function that updates its inputs in place does so exactly once).
    A disagreement, a compile error or an error while building the compiled function logs one line to stderr and switches
    to eager for the rest of the process. Later calls are not re-checked; if the compiled path raises later while eager
    succeeds on the same inputs, the same fallback happens (an error that eager raises too is the caller's and propagates)."""

    def __init__(self, fn: Callable, name: str, setting_dtype: torch.dtype):
        self._fn, self._name, self._dtype = fn, name, setting_dtype
        self._fast: Callable | None = None
        self._mode = "unverified"                       # unverified | compiled | eager
        self._lock = threading.Lock()
        functools.update_wrapper(self, fn)

    def _fall_back(self, reason: str) -> None:
        self._fast, self._mode = None, "eager"
        print(f"[bazt_lab.compute] torch.compile off for {self._name!r}: {reason}; using eager for the rest of this process",
              file=sys.stderr, flush=True)

    def _verify(self, args: tuple, kwargs: dict) -> None:
        reference = self._fn(*_clone(args), **_clone(kwargs))               # a bug in fn itself propagates from here
        try:
            fast = _compile(self._fn)
            candidate = fast(*_clone(args), **_clone(kwargs))
        except Exception as exc:                                             # compile error of any kind
            self._fall_back(f"compile error {type(exc).__name__}: {_one_line(exc)}")
            return
        try:
            problem = _mismatch(reference, candidate, self._dtype)
        except Exception as exc:                                             # an output type we cannot compare is not verified
            problem = f"could not compare the outputs ({type(exc).__name__}: {_one_line(exc)})"
        if problem is not None:
            self._fall_back(f"compiled output differs from eager ({_one_line(problem)})")
            return
        self._fast, self._mode = fast, "compiled"

    def __call__(self, *args, **kwargs):
        if self._mode == "unverified":
            with self._lock:
                if self._mode == "unverified":
                    self._verify(args, kwargs)
        fast = self._fast                                                    # read once: another thread may fall back
        if fast is None:
            return self._fn(*args, **kwargs)
        try:
            return fast(*args, **kwargs)
        except Exception as exc:
            result = self._fn(*args, **kwargs)                               # raises the caller's own error if there is one
            with self._lock:
                if self._mode == "compiled":
                    self._fall_back(f"compiled call failed {type(exc).__name__}: {_one_line(exc)}")
            return result


_WRAPPERS: dict = {}                    # (fn, name, dtype, device) -> its wrapper, for the life of the process
_WRAPPERS_LOCK = threading.Lock()


def _no_inductor_reason(device_type: str) -> str:
    if device_type == "cuda":
        return "Inductor cannot build kernels for cuda here (needs Triton and a usable GPU)"
    compiler = "$CXX" if os.environ.get("CXX") else ("cl" if sys.platform == "win32" else "g++ or clang++")
    return f"Inductor cannot build kernels for cpu here (no C++ compiler: {compiler} not found)"


def compiled(fn: Callable, settings: Mapping, name: str) -> Callable:
    """`fn` itself when compute.compile is false; otherwise `fn` under torch.compile(dynamic=True, fullgraph=False),
    verified against eager on the first call (see `_VerifiedCompile`). Equivalence tolerance follows compute.dtype:
    float64 rtol 1e-9 atol 1e-12, float32 rtol 1e-5 atol 1e-6; bool and int outputs must match exactly.
    `name` identifies the kernel in the one log line written on a fallback. Inputs given as tensors, lists, tuples, dicts and
    generators are copied for the check; other arguments are shared, so pass functions that do not mutate them.

    One wrapper per (fn, name, compute.dtype, compute.device) per process: a caller that asks again for every batch gets
    the same wrapper, so the check, the compile and a fallback line happen once. Pass a module-level function, not a
    closure made per call (each new closure is a new key). When Inductor cannot run on compute.device
    (`inductor_available`), torch.compile is not tried: the wrapper is eager from the start, with one log line."""
    if not _flag(settings, "compute.compile"):
        return fn
    dtype = resolve_dtype(settings)
    spec, _ = _device_spec(settings)
    key = (fn, name, dtype, spec)
    try:
        hash(key)
    except TypeError:                                   # an unhashable callable: a wrapper of its own, not cached
        key = None
    with _WRAPPERS_LOCK:
        wrapper = _WRAPPERS.get(key) if key is not None else None
        if wrapper is None:
            wrapper = _VerifiedCompile(fn, name, dtype)
            device_type = torch.device(spec).type
            if not inductor_available(device_type):
                wrapper._fall_back(_no_inductor_reason(device_type))
            if key is not None:
                _WRAPPERS[key] = wrapper
    return wrapper
