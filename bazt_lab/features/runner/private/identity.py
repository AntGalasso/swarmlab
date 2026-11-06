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
IGNORED = ("profile", "tracking", "sync", "compute", "estimates", "analysis", "mock.batch_size", "gnn.embedding.batch_size",
           "runtime.ni_margin_accuracy", "runtime.rank_agreement_threshold", "llm.allow_live")
# What only the other platform reads: a platform's identity holds the keys that platform reads and no others.
LLM_RUNTIME_ONLY = ("runtime.llm_seeds", "runtime.llm_calibration_seeds", "runtime.llm_sink_positions",
                    "runtime.random_edge_probability", "conventional.c.judge_score_bins.llm")
# runtime.sanitize_success is read by BAZT on both platforms (it prices sanitize), so it is not listed here.
MOCK_RUNTIME_ONLY = ("runtime.horizon_constant", "runtime.eps_correct", "runtime.insider_bias", "runtime.ta_detectability", "runtime.ma_detectability",
                     "runtime.aitm_rewrite_strength", "conventional.c.judge_score_bins.mock")
MOCK_ONLY_IGNORED = ("llm", "llm_data", "gnn", "gnn_train", *LLM_RUNTIME_ONLY)
LLM_ONLY_IGNORED = ("mock", *MOCK_RUNTIME_ONLY)
SCOPED_MOCK = ("epidemiology", "ablations", "adaptive", "presets_robustness", "costs_robustness", "legacy")

# thresholds.json: what decides an attack-free statistic. Besides the platform's own keys, every section a calibrated
# arm reads while it decides: B (runtime, costs), C (conventional), BAZT and its ablations (bazt), BAZT_upstream
# (bazt_upstream, bazt.prior), the allocation baselines and the GNN stand-in (costs), the thresholded legacy rules
# (mock.legacy structure constants), GNN_recalibrated (gnn, and gnn_train.families, which picks the deployed GNN).
# The LLM side also remembers what produces its replies and items: llm.api (the fake backend, judge and items of a
# rehearsal, or the live ones), llm.providers (the pinned provider of each backbone, also in the request cache key) and
# llm_data (the pinned revisions and vendored digests). Its run-time configuration is not settings (the client timeout
# and retries are constants of llm_harness adapters/live.py) or is ignored (llm.allow_live); llm.ma_judge_model and
# llm.ma_correctness are read by evaluation units only (a calibrating seed run checks no MA answer).
THRESHOLD_PREFIXES = ("runtime", "mock.model", "mock.auditor", "mock.tune_seeds", "calibration", "budgets.alpha",
                      "costs", "auditor_presets", "conventional", "bazt", "bazt_upstream", "mock.legacy", "gnn",
                      "gnn_train.families", "llm.judge", "llm.agents", "llm.rounds", "llm.temperature",
                      "llm.backbones", "llm.seeds_per_cell", "llm.calibration_conversations_per_cell", "llm.cache",
                      "llm.attack_free_tasks", "llm.api", "llm.providers", "llm_data")
THRESHOLD_IGNORED = ("runtime.ni_margin_accuracy", "runtime.rank_agreement_threshold", "runtime.recovery_margin",
                     "runtime.llm_seeds", "runtime.llm_sink_positions", "runtime.n_insiders", "runtime.insider_bias",
                     "runtime.ta_detectability", "runtime.ma_detectability", "runtime.aitm_rewrite_strength",
                     "gnn.embedding.batch_size", "llm.allow_live",
                     "mock.legacy.arms", "mock.legacy.cells",       # the scope of mock_legacy, not a structure constant
                     "mock.legacy.matched_thresholds")              # which cell a matched arm reads: evaluation only


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
    other = MOCK_ONLY_IGNORED if platform == "mock" else LLM_ONLY_IGNORED
    return {k: v for k, v in sorted(flatten(settings).items())
            if _under(k, THRESHOLD_PREFIXES) and not _under(k, THRESHOLD_IGNORED) and not _under(k, other)}


def differences(saved: Mapping, current: Mapping, limit: int = 6) -> str:
    keys = sorted(k for k in set(saved) | set(current) if saved.get(k, "<unset>") != current.get(k, "<unset>"))
    shown = [f"{k} ({saved.get(k, '<unset>')!r} -> {current.get(k, '<unset>')!r})" for k in keys[:limit]]
    return ", ".join(shown) + (f" and {len(keys) - limit} more" if len(keys) > limit else "")


def read(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")).get("settings") if path.exists() else None
