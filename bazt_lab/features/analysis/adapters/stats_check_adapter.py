"""The repo's pre-registered statistics (stats_check.py at the repository root), reused unchanged.

stats_check is not a package of ours and is never edited, so it is reached only from here. What it offers and fits:
  holm            Holm step-down on a dict of p-values
  wilcoxon_rrb    two-sided Wilcoxon signed-rank on a - b, with the signed rank-biserial effect
  pareto_label    the five labels of spec section 7 (significance level fixed at stats_check.ALPHA)
What does not fit: boot_ci (iid resampling, fixed 95%, numpy RNG) is replaced by the seeded cluster bootstrap in
private/stats.py.
"""
from __future__ import annotations

from typing import Sequence

import stats_check

LABEL_STRICT = stats_check.LABEL_STRICT
LABEL_SAFETY = stats_check.LABEL_SAFETY
LABEL_CONTAIN = stats_check.LABEL_CONTAIN
LABEL_UTILITY_LOSS = stats_check.LABEL_UTILITY_LOSS
LABEL_NULL = stats_check.LABEL_NULL
LABELS = (LABEL_STRICT, LABEL_SAFETY, LABEL_CONTAIN, LABEL_UTILITY_LOSS, LABEL_NULL)
WINS = (LABEL_STRICT, LABEL_SAFETY)

ALPHA = stats_check.ALPHA
holm = stats_check.holm
wilcoxon_rrb = stats_check.wilcoxon_rrb


def label(d_safety: float, p_holm_safety: float | None, utility_loss: float, p_holm_utility: float | None,
          ci_utility_loss: Sequence[float], margin: float, alpha: float) -> str:
    """stats_check.pareto_label with the utility endpoint expressed as a LOSS (baseline minus challenger accuracy).

    `d_safety` < 0 means the challenger has the lower ASR. pareto_label compares against stats_check.ALPHA, so a
    different `alpha` is refused instead of being silently ignored.
    """
    if alpha != stats_check.ALPHA:
        raise ValueError(f"analysis.alpha={alpha}: stats_check.pareto_label is fixed at ALPHA={stats_check.ALPHA}")
    return stats_check.pareto_label(d_safety, p_holm_safety, utility_loss, p_holm_utility, ci_utility_loss, margin)
