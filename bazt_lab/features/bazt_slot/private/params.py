"""The settings BAZT reads, validated once. No default: a missing key raises core.Unset naming it (spec section 10)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from bazt_lab.core import public as core

UTILITY_KINDS = ("sanitize", "delay", "edge_quarantine", "rate_limit", "privilege_degradation", "isolation")


@dataclass(frozen=True)
class Params:
    decay: float                  # bazt.katz_decay: per-hop transmission probability of the truncated Katz reach
    iterations: int               # bazt.eigen_iterations: power iterations of the leading eigenpair
    sink_weight: float            # bazt.sink_weight: loss weight of the protected sink, an ordinary agent weighs 1
    utility_weight: float         # bazt.utility_weight: lambda, utility loss against loss reduction
    rho: Mapping[str, float]      # bazt.utility_loss.<kind>: utility lost per application, as a share of centrality
    flag_logodds: float           # bazt.flag_logodds: evidence of a delivery the A+B+C pipeline flags
    clear_logodds: float          # bazt.clear_logodds: evidence of an audited sender the pipeline lets through
    memory_decay: float           # bazt.memory_decay: per-round factor on the verdict memory
    prior: float                  # bazt.prior: P(compromised) with no monitor reading
    q_inspect: float              # bazt.inspect_conclusive: probability an audit settles the question
    sanitize_success: float       # runtime.sanitize_success
    rate_limit: int               # runtime.rate_limit_per_identity


def _number(settings: Mapping, key: str, low: float, high: float, *, low_open: bool, high_open: bool) -> float:
    value = core.get(settings, key)
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and value == value
    if ok and (value > low if low_open else value >= low) and (value < high if high_open else value <= high):
        return float(value)
    raise ValueError(f"{key} must be a number in {'(' if low_open else '['}{low}, {high}{')' if high_open else ']'}, "
                     f"got {value!r}")


def _integer(settings: Mapping, key: str) -> int:
    value = core.get(settings, key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be an integer >= 1, got {value!r}")
    return value


def read(settings: Mapping) -> Params:
    inf = float("inf")
    return Params(
        decay=_number(settings, "bazt.katz_decay", 0.0, 1.0, low_open=True, high_open=False),
        iterations=_integer(settings, "bazt.eigen_iterations"),
        sink_weight=_number(settings, "bazt.sink_weight", 1.0, inf, low_open=False, high_open=False),
        utility_weight=_number(settings, "bazt.utility_weight", 0.0, inf, low_open=False, high_open=False),
        rho={k: _number(settings, f"bazt.utility_loss.{k}", 0.0, inf, low_open=False, high_open=False)
             for k in UTILITY_KINDS},
        flag_logodds=_number(settings, "bazt.flag_logodds", 0.0, inf, low_open=True, high_open=False),
        clear_logodds=_number(settings, "bazt.clear_logodds", 0.0, inf, low_open=True, high_open=False),
        memory_decay=_number(settings, "bazt.memory_decay", 0.0, 1.0, low_open=False, high_open=True),
        prior=_number(settings, "bazt.prior", 0.0, 1.0, low_open=True, high_open=True),
        q_inspect=_number(settings, "bazt.inspect_conclusive", 0.0, 1.0, low_open=True, high_open=False),
        sanitize_success=_number(settings, "runtime.sanitize_success", 0.0, 1.0, low_open=False, high_open=False),
        rate_limit=_integer(settings, "runtime.rate_limit_per_identity"),
    )
