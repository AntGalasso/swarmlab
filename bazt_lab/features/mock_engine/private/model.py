"""The mock model's parameters, read once per batch from the settings. Nothing here has a default: a missing key raises
core.Unset naming it, a value outside its range raises ValueError naming it."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from bazt_lab.core import public as core


@dataclass(frozen=True)
class Model:
    m: int                    # rolling window of the cheap monitor
    a: float                  # monitor: p = sigmoid(a * rolling - b)
    b: float
    mu_inf: float             # mean monitor score of a payload-carrying message
    mu_dec: float             # mean monitor score of a decoy header message
    sd: float                 # monitor noise
    beta: float               # per-exposure adoption probability
    p_stay: float             # probability an infected agent keeps the payload
    p_decoy: float            # probability a decoy writes its header inside its burst
    decoys: int               # decoy agents per episode
    burst_span: int           # decoy bursts start at round 1 + (agent index % burst_span)
    reading_mean: float       # private readings ~ N(reading_mean, reading_sd^2)
    reading_sd: float
    bias: float               # reported-value shift of an agent carrying the infection / AiTM payload
    eps_correct: float        # an agent is correct iff |estimate - mean(readings)| <= eps_correct
    sanitize_success: float   # probability a sanitized or degraded delivery loses its payload
    rate_limit: int           # receivers a rate-limited sender reaches per round


def _number(settings: Mapping, key: str, lo: float | None = None, hi: float | None = None) -> float:
    value = core.get(settings, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number, got {value!r}")
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise ValueError(f"{key}={value!r} is outside [{'-inf' if lo is None else lo}, {'inf' if hi is None else hi}]")
    return float(value)


def _integer(settings: Mapping, key: str, lo: int) -> int:
    value = core.get(settings, key)
    if isinstance(value, bool) or not isinstance(value, int) or value < lo:
        raise ValueError(f"{key} must be an integer >= {lo}, got {value!r}")
    return value


def load_model(settings: Mapping) -> Model:
    p = "mock.model."
    return Model(
        m=_integer(settings, p + "m", 1), a=_number(settings, p + "a"), b=_number(settings, p + "b"),
        mu_inf=_number(settings, p + "mu_inf"), mu_dec=_number(settings, p + "mu_dec"),
        sd=_number(settings, p + "sd", 0.0), beta=_number(settings, p + "beta", 0.0, 1.0),
        p_stay=_number(settings, p + "p_stay", 0.0, 1.0), p_decoy=_number(settings, p + "p_decoy", 0.0, 1.0),
        decoys=_integer(settings, p + "decoys", 0), burst_span=_integer(settings, p + "burst_span", 1),
        reading_mean=_number(settings, p + "reading_mean"), reading_sd=_number(settings, p + "reading_sd", 0.0),
        bias=_number(settings, p + "bias"), eps_correct=_number(settings, "runtime.eps_correct", 0.0),
        sanitize_success=_number(settings, "runtime.sanitize_success", 0.0, 1.0),
        rate_limit=_integer(settings, "runtime.rate_limit_per_identity", 1))
