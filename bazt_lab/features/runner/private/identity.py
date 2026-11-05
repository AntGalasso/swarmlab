"""The identity of the settings behind a block's rows and behind thresholds.json.

A block directory remembers the settings its first launch ran with (`settings.json`); thresholds.json remembers the
settings that decide an attack-free statistic. A later launch under other values of those keys is refused with the keys
named, because it would mix rows (or reuse thresholds) produced by different experiments.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

# Never part of a block's identity: the profile (it names the directory), tracking, dry-run estimates, the analysis,
# performance-only keys, the two pre-registered values (recorded in every manifest) and the live flag the runner sets.
IGNORED = ("profile", "tracking", "estimates", "analysis", "mock.batch_size", "gnn.embedding.batch_size",
           "runtime.ni_margin_accuracy", "runtime.rank_agreement_threshold", "llm.allow_live")
MOCK_ONLY_IGNORED = ("llm", "llm_data", "gnn", "gnn_train", "runtime.llm_seeds", "runtime.llm_calibration_seeds",
                     "runtime.llm_sink_positions", "runtime.random_edge_probability")
LLM_ONLY_IGNORED = ("mock", "runtime.horizon_constant", "runtime.eps_correct")
SCOPED_MOCK = ("epidemiology", "ablations", "adaptive", "presets_robustness", "costs_robustness", "legacy")

# thresholds.json: what decides an attack-free statistic.
THRESHOLD_PREFIXES = ("runtime", "mock.model", "mock.auditor", "mock.tune_seeds", "calibration", "budgets.alpha",
                      "costs", "auditor_presets", "conventional", "bazt", "gnn", "llm.judge", "llm.agents",
                      "llm.rounds", "llm.temperature", "llm.backbones", "llm.seeds_per_cell",
                      "llm.calibration_conversations_per_cell", "llm.cache", "llm.attack_free_tasks")
THRESHOLD_IGNORED = ("runtime.ni_margin_accuracy", "runtime.rank_agreement_threshold", "runtime.recovery_margin",
                     "runtime.llm_seeds", "runtime.llm_sink_positions", "runtime.n_insiders", "runtime.insider_bias",
                     "runtime.ta_detectability", "runtime.ma_detectability", "runtime.aitm_rewrite_strength",
                     "gnn.embedding.batch_size", "llm.allow_live")


def flatten(tree: Mapping, prefix: str = "") -> dict:
    out = {}
    for key, value in tree.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(flatten(value, dotted + "."))
        else:
            out[dotted] = value
    return out


def _under(key: str, prefixes) -> bool:
    return any(key == p or key.startswith(p + ".") for p in prefixes)


def block_view(settings: Mapping, block: str, platform: str) -> dict:
    """The settings that make up a block's identity."""
    ignored = list(IGNORED) + list(MOCK_ONLY_IGNORED if platform == "mock" else LLM_ONLY_IGNORED)
    if platform == "mock":
        ignored += [f"mock.{x}" for x in SCOPED_MOCK if block != f"mock_{x}"]
    else:
        ignored += [] if block == "llm_adaptive_check" else ["llm.adaptive_check"]
    flat = flatten(settings)
    keep = {"analysis.ablation_reference"} if block == "mock_ablations" else set()
    return {k: v for k, v in sorted(flat.items()) if k in keep or not _under(k, ignored)}


def threshold_view(settings: Mapping, platform: str) -> dict:
    """The settings the thresholds of a platform's calibration depend on."""
    other = ("llm", "gnn") if platform == "mock" else ("mock",)
    return {k: v for k, v in sorted(flatten(settings).items())
            if _under(k, THRESHOLD_PREFIXES) and not _under(k, THRESHOLD_IGNORED) and not _under(k, other)}


def differences(saved: Mapping, current: Mapping, limit: int = 6) -> str:
    keys = sorted(k for k in set(saved) | set(current) if saved.get(k, "<unset>") != current.get(k, "<unset>"))
    shown = [f"{k} ({saved.get(k, '<unset>')!r} -> {current.get(k, '<unset>')!r})" for k in keys[:limit]]
    return ", ".join(shown) + (f" and {len(keys) - limit} more" if len(keys) > limit else "")


def read(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")).get("settings") if path.exists() else None
