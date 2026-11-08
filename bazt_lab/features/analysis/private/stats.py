"""Paired tests and the seed-cluster bootstrap (docs/EXPERIMENT_PLAN.md section 7).

Continuous metrics: paired Wilcoxon signed-rank (stats_check.wilcoxon_rrb) and a paired bootstrap CI.
Yes/no metrics: McNemar's exact test (the exact two-sided binomial on the discordant pairs).
The bootstrap resamples whole clusters (seeds). Draws come from core.stream keyed by the comparison, pre-drawn as one
[resamples, clusters] block, so a result depends on its data and key only, never on batch size or call order.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np
from scipy import stats

from bazt_lab.core import public as core
from bazt_lab.features.analysis.adapters import stats_check_adapter as sc

KINDS = ("continuous", "binary")


def cluster_sums(values: np.ndarray, cluster: np.ndarray, universe: np.ndarray | None = None):
    """Per-cluster sums and counts. `universe` (sorted cluster ids) aligns several samples on the same clusters."""
    if universe is None:
        universe = np.unique(cluster)
    at = np.searchsorted(universe, cluster)
    size = len(universe)
    return (np.bincount(at, weights=values, minlength=size).astype(np.float64),
            np.bincount(at, minlength=size).astype(np.float64))


def resample_index(key: Sequence, resamples: int, n_clusters: int) -> np.ndarray:
    """int64 [resamples, n_clusters]: which clusters each bootstrap replicate draws, keyed by `key`."""
    u = core.uniforms(core.stream("analysis", "bootstrap", *key), (resamples, n_clusters))
    return (u * n_clusters).long().numpy()


def quantile_ci(samples: np.ndarray, level: float) -> Tuple[float, float]:
    lo, hi = np.quantile(samples, [(1.0 - level) / 2.0, (1.0 + level) / 2.0], axis=0)
    return float(lo), float(hi)


def bootstrap_means(values: np.ndarray, cluster: np.ndarray, key: Sequence, resamples: int) -> np.ndarray:
    """Means of `resamples` cluster-bootstrap replicates (empty if fewer than two clusters)."""
    sums, counts = cluster_sums(values, cluster)
    if len(sums) < 2:
        return np.empty(0)
    idx = resample_index(key, resamples, len(sums))
    return sums[idx].sum(axis=1) / counts[idx].sum(axis=1)


def bootstrap_ci(values: np.ndarray, cluster: np.ndarray, key: Sequence, resamples: int, level: float):
    means = bootstrap_means(values, cluster, key, resamples)
    return quantile_ci(means, level) if len(means) else (math.nan, math.nan)


def collapse(x: np.ndarray, y: np.ndarray, cluster: np.ndarray):
    """One pair per cluster: the mean of x and of y inside each cluster (mock pairs are seeds, spec section 7)."""
    universe = np.unique(cluster)
    sx, n = cluster_sums(x, cluster, universe)
    sy, _ = cluster_sums(y, cluster, universe)
    return sx / n, sy / n, universe


def paired_test(x: np.ndarray, y: np.ndarray, cluster: np.ndarray, kind: str, key: Sequence, resamples: int,
                level: float) -> dict:
    """Test x (challenger) against y (baseline), pair by pair. delta = mean(x - y).

    The p-value treats pairs as independent (spec: conversations on the LLM side). `se_ratio_cluster_vs_naive` > 1
    says the pairs are positively correlated inside a seed, so that p-value is anti-conservative.
    """
    if kind not in KINDS:
        raise ValueError(f"metric kind must be one of {KINDS}, got {kind!r}")
    x, y, cluster = (np.asarray(a) for a in (x, y, cluster))
    keep = ~(np.isnan(x) | np.isnan(y))
    dropped = int((~keep).sum())
    x, y, cluster = x[keep], y[keep], cluster[keep]
    d = x - y
    out = {"test": "wilcoxon" if kind == "continuous" else "mcnemar_exact", "n_pairs": len(d),
           "n_clusters": len(np.unique(cluster)), "n_dropped_nan": dropped,
           "delta": float(d.mean()) if len(d) else math.nan}
    if kind == "binary":
        if not np.isin(np.concatenate([x, y]), (0.0, 1.0)).all():
            raise ValueError("a binary metric must hold only 0 and 1 (NaN for undefined)")
        x_only, y_only = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
        n_disc = x_only + y_only
        out.update(p_raw=float(stats.binomtest(x_only, n_disc, 0.5).pvalue) if n_disc else 1.0,
                   n_discordant=n_disc, n_x_only=x_only, n_y_only=y_only)
    else:
        p, rrb, nonzero = sc.wilcoxon_rrb(x, y) if len(d) else (1.0, 0.0, 0)
        out.update(p_raw=float(p), rrb=float(rrb), n_nonzero=int(nonzero))
    means = bootstrap_means(d, cluster, key, resamples) if len(d) else np.empty(0)
    out["ci"] = list(quantile_ci(means, level)) if len(means) else [math.nan, math.nan]
    naive = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else math.nan
    out["se_ratio_cluster_vs_naive"] = float(means.std(ddof=1) / naive) if len(means) and naive > 0 else math.nan
    return out


def unpaired_test(a: np.ndarray, cluster_a: np.ndarray, b: np.ndarray, cluster_b: np.ndarray, key: Sequence,
                  resamples: int, level: float) -> dict:
    """Two independent samples (e.g. the same hypothesis at two agent counts): delta = mean(a) - mean(b), two-sided
    Mann-Whitney U, and a bootstrap CI that resamples the clusters of each sample on its own."""
    a, cluster_a = np.asarray(a), np.asarray(cluster_a)
    b, cluster_b = np.asarray(b), np.asarray(cluster_b)
    keep_a, keep_b = ~np.isnan(a), ~np.isnan(b)
    dropped = int((~keep_a).sum() + (~keep_b).sum())
    a, cluster_a, b, cluster_b = a[keep_a], cluster_a[keep_a], b[keep_b], cluster_b[keep_b]
    out = {"test": "mann_whitney_u", "n_a": len(a), "n_b": len(b), "n_dropped_nan": dropped,
           "n_clusters_a": len(np.unique(cluster_a)), "n_clusters_b": len(np.unique(cluster_b)),
           "delta": float(a.mean() - b.mean()) if len(a) and len(b) else math.nan,
           "p_raw": float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue) if len(a) and len(b) else math.nan}
    means_a = bootstrap_means(a, cluster_a, (*key, "a"), resamples) if len(a) else np.empty(0)
    means_b = bootstrap_means(b, cluster_b, (*key, "b"), resamples) if len(b) else np.empty(0)
    out["ci"] = list(quantile_ci(means_a - means_b, level)) if len(means_a) and len(means_b) else [math.nan, math.nan]
    return out
