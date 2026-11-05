"""Settings handling: loading with CLI-style refusals, and the run-time variables a platform needs (spec §10)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Mapping

from bazt_lab.core import public as core

# The two pre-registered ones are required for every run, so they are recorded in the manifest before any result.
PRE_REGISTERED = ("ni_margin_accuracy", "rank_agreement_threshold")
RUNTIME_BY_PLATFORM = {
    "llm": ("n_insiders", "rate_limit_per_identity", "recovery_margin", "random_edge_probability", "ba_m", "ws_k",
            "ws_rewire_p", "llm_seeds", "llm_calibration_seeds"),
    "mock": ("n_insiders", "rate_limit_per_identity", "horizon_constant", "eps_correct", "recovery_margin", "ba_m",
             "ws_k", "ws_rewire_p", "sanitize_success", "insider_bias", "ta_detectability", "ma_detectability",
             "aitm_rewrite_strength"),
}


def load_settings(path: Path = core.SETTINGS_PATH, profile: str | None = None, overrides=()) -> dict:
    """core.load_settings, with a bad profile or override ending the launch with its message."""
    for item in overrides:
        key, _, text = item.partition("=")
        text = text.strip()
        if "," in text and not text.startswith(("[", "{", '"', "'")) and core.parse_value(text) == text:
            sys.exit(f"--set {key.strip()}={text}: a list needs brackets, e.g. {key.strip()}=[1, 2, 3]")
    try:
        return core.load_settings(Path(path), profile, overrides)
    except ValueError as e:
        sys.exit(str(e))


def missing_runtime(s: Mapping, platform: str) -> List[str]:
    keys = list(PRE_REGISTERED) + list(RUNTIME_BY_PLATFORM[platform])
    missing = [k for k in keys if not core.is_set(s, f"runtime.{k}")]
    if platform == "llm":
        missing += [f"llm_sink_positions.{t}" for t in core.get(s, "llm.topologies")
                    if not core.is_set(s, f"runtime.llm_sink_positions.{t}")]
    return missing
