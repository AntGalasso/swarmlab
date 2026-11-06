"""Every value the analysis uses, read from settings. There is no default: a missing key raises core.Unset naming it.

The safety endpoint is ASR at the final round (spec section 7) and the spend columns are fixed by the metrics rows; the
rest, including which accuracy is the utility endpoint, comes from the settings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

from bazt_lab.core import public as core
from bazt_lab.features.analysis.adapters import stats_check_adapter as sc
from bazt_lab.features.analysis.private.stats import KINDS

SAFETY = "asr_final"
SPEND_AUDIT, SPEND_INTERVENTION = "spend_audit_total", "spend_intervention_total"
HYPOTHESES = ("budget", "efficiency", "topology", "adaptive", "adaptive_reference", "ladder", "ablations", "rank",
              "containment")
PLATFORMS = ("mock", "llm")
PAIR_UNITS = ("seed", "episode")
SPLITTABLE = ("attack", "task", "backbone")
UTILITY_ROWS = ("attack_free", "attacked")
RANK_STATISTICS = ("spearman", "kendall")
# The utility endpoint: the accuracy column and whether it is read from attack-free runs. Under attack, accuracy is
# 1 - ASR (metrics CONTRACT, open question 1), so a safety gain would always be a utility gain there.
UTILITY_ENDPOINTS = {"accuracy_clean": True, "accuracy_attack": False}
# H-adaptive: d = delta_challenger - factor * delta_baseline, delta = ASR(adaptive) - ASR(non-adaptive).
# factor 2: the challenger's advantage shrinks (delta_ch - delta_base) by less than the baseline's own security (delta_base).
# factor 1: the challenger's ASR rises by less than the baseline's.
ADAPTIVE_CONTRASTS = {"advantage_shrinks_less_than_c_security": 2.0, "challenger_degrades_less_than_c": 1.0}

REQUIRED_KEYS = (
    "profile",
    "analysis.alpha", "analysis.bootstrap_resamples", "analysis.ci_level",
    "analysis.baseline", "analysis.challengers", "analysis.ladder", "analysis.ablations", "analysis.ablation_reference",
    "analysis.containment_reference", "analysis.containment_metrics",
    "analysis.efficiency_tolerance", "analysis.adaptive_contrast",
    "analysis.rank_agreement_statistic", "analysis.rank_agreement_arms",
    "analysis.utility_endpoint", "analysis.extra_metrics",
    "analysis.pair_unit.mock", "analysis.pair_unit.llm", "analysis.split_by.mock", "analysis.split_by.llm",
    *(f"analysis.topology_classes.{p}.{k}" for p in PLATFORMS for k in ("heterogeneous", "homogeneous")),
    *(f"analysis.blocks.{h}.{p}" for h in HYPOTHESES for p in PLATFORMS),
    "runtime.ni_margin_accuracy", "runtime.rank_agreement_threshold", "runtime.recovery_margin",
)


def missing_keys(settings: Mapping) -> list:
    return [k for k in REQUIRED_KEYS if not core.is_set(settings, k)]


def _need(ok: bool, key: str, why: str) -> None:
    if not ok:
        raise ValueError(f"{key}: {why}")


def _names(values, key: str, at_least: int = 1) -> Tuple[str, ...]:
    _need(isinstance(values, (list, tuple)) and len(values) >= at_least and all(isinstance(v, str) for v in values),
          key, f"needs a list of at least {at_least} arm or column name(s)")
    return tuple(values)


@dataclass(frozen=True)
class Cfg:
    alpha: float
    resamples: int
    level: float
    baseline: str
    challengers: Tuple[str, ...]
    ladder: Tuple[str, ...]                       # weakest first: A, B, C
    ablations: Tuple[str, ...]
    ablation_reference: str
    containment_reference: str
    containment_metrics: Tuple[str, ...]
    efficiency_tolerance: float
    contrast_factor: float
    rank_statistic: str
    rank_arms: Tuple[Tuple[str, str], ...]        # (mock arm, llm arm)
    utility_column: str
    utility_attack_free: bool
    extra_metrics: Dict[str, Tuple[str, str]]     # metric -> (kind, rows)
    pair_unit: Dict[str, str]
    split_by: Dict[str, Tuple[str, ...]]
    heterogeneous: Dict[str, Tuple[str, ...]]      # platform -> topologies
    homogeneous: Dict[str, Tuple[str, ...]]
    blocks: Dict[str, Dict[str, Tuple[str, ...]]]  # hypothesis -> platform -> blocks
    ni_margin: float
    rank_threshold: float
    profile: str
    recovery_margin: float

    @property
    def metric_columns(self) -> Tuple[str, ...]:
        cols = [SAFETY, self.utility_column, SPEND_AUDIT, SPEND_INTERVENTION, "sink_is_attacker", *self.extra_metrics,
                *self.containment_metrics]
        return tuple(dict.fromkeys(cols))

    def all_blocks(self, platform: str) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(b for h in HYPOTHESES for b in self.blocks[h][platform]))

    @classmethod
    def from_settings(cls, settings: Mapping) -> Cfg:
        v = {key: core.get(settings, key) for key in REQUIRED_KEYS}      # raises core.Unset naming the key
        alpha, resamples, level = v["analysis.alpha"], v["analysis.bootstrap_resamples"], v["analysis.ci_level"]
        _need(0 < alpha < 1, "analysis.alpha", "must be in (0, 1)")
        _need(alpha == sc.ALPHA, "analysis.alpha",
              f"stats_check.pareto_label labels at its fixed ALPHA={sc.ALPHA}; {alpha} cannot be honoured")
        _need(isinstance(resamples, int) and resamples >= 1, "analysis.bootstrap_resamples", "must be an integer >= 1")
        _need(0 < level < 1, "analysis.ci_level", "must be in (0, 1)")
        challengers = _names(v["analysis.challengers"], "analysis.challengers")
        _need(v["analysis.baseline"] not in challengers, "analysis.challengers", "must not contain the baseline")
        ladder = _names(v["analysis.ladder"], "analysis.ladder", 2)
        rank_arms = tuple(tuple(p) for p in v["analysis.rank_agreement_arms"])
        _need(len(rank_arms) >= 3 and all(len(p) == 2 for p in rank_arms), "analysis.rank_agreement_arms",
              "needs at least three [mock arm, llm arm] pairs")
        _need(v["analysis.rank_agreement_statistic"] in RANK_STATISTICS, "analysis.rank_agreement_statistic",
              f"must be one of {RANK_STATISTICS}")
        _need(v["analysis.utility_endpoint"] in UTILITY_ENDPOINTS, "analysis.utility_endpoint",
              f"must be one of {tuple(UTILITY_ENDPOINTS)}")
        _need(v["analysis.adaptive_contrast"] in ADAPTIVE_CONTRASTS, "analysis.adaptive_contrast",
              f"must be one of {tuple(ADAPTIVE_CONTRASTS)}")
        extras = {}
        for name, spec in v["analysis.extra_metrics"].items():
            _need(len(spec) == 2 and spec[0] in KINDS and spec[1] in UTILITY_ROWS, "analysis.extra_metrics",
                  f"{name!r} must be [kind in {KINDS}, rows in {UTILITY_ROWS}]")
            extras[name] = (spec[0], spec[1])
        pair_unit = {p: v[f"analysis.pair_unit.{p}"] for p in PLATFORMS}
        _need(all(x in PAIR_UNITS for x in pair_unit.values()), "analysis.pair_unit", f"values must be in {PAIR_UNITS}")
        split_by = {p: tuple(v[f"analysis.split_by.{p}"]) for p in PLATFORMS}
        _need(all(f in SPLITTABLE for s in split_by.values() for f in s), "analysis.split_by",
              f"fields must be in {SPLITTABLE}")
        hetero, homo = {}, {}
        for p in PLATFORMS:
            hetero[p] = _names(v[f"analysis.topology_classes.{p}.heterogeneous"],
                               f"analysis.topology_classes.{p}.heterogeneous", 0)
            homo[p] = _names(v[f"analysis.topology_classes.{p}.homogeneous"],
                             f"analysis.topology_classes.{p}.homogeneous", 0)
            _need(not set(hetero[p]) & set(homo[p]), f"analysis.topology_classes.{p}", "a topology is in both classes")
        blocks = {h: {p: _names(v[f"analysis.blocks.{h}.{p}"], f"analysis.blocks.{h}.{p}", 0) for p in PLATFORMS}
                  for h in HYPOTHESES}
        _need(v["runtime.ni_margin_accuracy"] >= 0, "runtime.ni_margin_accuracy", "must be >= 0")
        _need(-1 <= v["runtime.rank_agreement_threshold"] <= 1, "runtime.rank_agreement_threshold",
              "must be in [-1, 1]")
        _need(v["runtime.recovery_margin"] >= 0, "runtime.recovery_margin", "must be >= 0")
        return cls(
            recovery_margin=float(v["runtime.recovery_margin"]), alpha=alpha, resamples=resamples, level=level, baseline=v["analysis.baseline"], challengers=challengers,
            ladder=ladder, ablations=_names(v["analysis.ablations"], "analysis.ablations", 0),
            ablation_reference=v["analysis.ablation_reference"],
            containment_reference=v["analysis.containment_reference"],
            containment_metrics=_names(v["analysis.containment_metrics"], "analysis.containment_metrics"),
            efficiency_tolerance=float(v["analysis.efficiency_tolerance"]),
            contrast_factor=ADAPTIVE_CONTRASTS[v["analysis.adaptive_contrast"]],
            rank_statistic=v["analysis.rank_agreement_statistic"], rank_arms=rank_arms,
            utility_column=v["analysis.utility_endpoint"],
            utility_attack_free=UTILITY_ENDPOINTS[v["analysis.utility_endpoint"]],
            extra_metrics=extras, pair_unit=pair_unit, split_by=split_by, heterogeneous=hetero, homogeneous=homo,
            blocks=blocks, ni_margin=float(v["runtime.ni_margin_accuracy"]),
            rank_threshold=float(v["runtime.rank_agreement_threshold"]), profile=str(v["profile"]))
