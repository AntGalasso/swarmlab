"""Rank agreement between the platforms (spec section 4): the pre-registered arms ranked by mean ASR on each platform,
compared with Spearman's rho or Kendall's tau (analysis.rank_agreement_statistic), passed when the statistic reaches
runtime.rank_agreement_threshold.

An arm's score on a platform is its ASR at the final round on its uncapped attacked runs in the cells where every
ranked arm has runs, averaged over episodes within a seed, then over the cells, then over seeds. The interval resamples
the seeds of each platform on its own.
"""
from __future__ import annotations

import warnings
from typing import Dict, Mapping, Sequence

import numpy as np
from scipy import stats as sps

from bazt_lab.features.analysis.private import stats
from bazt_lab.features.analysis.private.config import SAFETY, Cfg
from bazt_lab.features.analysis.private.table import UNCAPPED, Key, Table, sort_key

NAN = float("nan")


def _statistic(name: str, a: Sequence[float], b: Sequence[float]) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                  # constant input warns and returns NaN
        result = sps.spearmanr(a, b) if name == "spearman" else sps.kendalltau(a, b)
    return float(result.statistic)


def _scores(t: Table, cfg: Cfg, platform: str, arms: Sequence[str]):
    """(scores [arms], replicate scores [arms, resamples], cells used), or the reason the platform cannot be ranked."""
    blocks = cfg.blocks["rank"][platform]
    if not blocks:
        return f"{platform}: no block listed in analysis.blocks.rank.{platform}"
    present: Dict[Key, set] = {}
    for key in t.index:
        if key.block in blocks and key.arm in arms and key.audit == UNCAPPED and key.intervention == UNCAPPED \
                and not key.attack_free:
            present.setdefault(key.ctx, set()).add(key.arm)
    usable = [ctx for ctx in sorted(present, key=sort_key) if present[ctx] >= set(arms)]
    if not usable:
        never = sorted(set(arms) - set().union(*present.values())) if present else sorted(arms)
        return f"{platform}: no cell has uncapped attacked runs of every ranked arm (arms with none: {never})"
    asr, seed = t.metrics[SAFETY], t.codes["seed"]
    universe = np.unique(np.concatenate([seed[r] for ctx in usable for r in
                                         t.aligned([ctx.with_arm(a, False) for a in arms])]))
    means = []                                           # [cells][arms] -> mean per seed, NaN where the seed has no run
    for ctx in usable:
        rows = t.aligned([ctx.with_arm(a, False) for a in arms])
        per_arm = []
        for r in rows:
            keep = ~np.isnan(asr[r])
            sums, counts = stats.cluster_sums(asr[r][keep], seed[r][keep], universe)
            per_arm.append(np.divide(sums, counts, out=np.full(len(universe), np.nan), where=counts > 0))
        means.append(per_arm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a seed that no cell has is all-NaN
        by_seed = np.nanmean(np.array(means), axis=0)    # [arms, seeds]
        idx = stats.resample_index(("rank", platform), cfg.resamples, len(universe))
        return np.nanmean(by_seed, axis=1), np.nanmean(by_seed[:, idx], axis=2), len(usable)


def _ranks(scores: np.ndarray) -> np.ndarray:
    return sps.rankdata(scores)                          # 1 = lowest ASR = safest


def rank_agreement(tables: Mapping[str, Table], cfg: Cfg) -> dict:
    pairs = cfg.rank_arms
    arms = {"mock": [p[0] for p in pairs], "llm": [p[1] for p in pairs]}
    scored = {p: _scores(tables[p], cfg, p, arms[p]) for p in ("mock", "llm")}
    problems = [s for s in scored.values() if isinstance(s, str)]
    if problems:
        return {"status": "not_evaluable", "reason": "; ".join(problems)}
    (m, m_rep, m_cells), (ll, ll_rep, l_cells) = scored["mock"], scored["llm"]
    value = _statistic(cfg.rank_statistic, m, ll)
    reps = np.array([_statistic(cfg.rank_statistic, m_rep[:, r], ll_rep[:, r]) for r in range(cfg.resamples)])
    finite = reps[~np.isnan(reps)]
    m_rank, l_rank = _ranks(m), _ranks(ll)
    inversions = [{"pair": [arms["mock"][i], arms["mock"][j]],
                   "mock_better": arms["mock"][i if m[i] < m[j] else j],
                   "llm_better": arms["mock"][i if ll[i] < ll[j] else j]}
                  for i in range(len(pairs)) for j in range(i + 1, len(pairs)) if (m[i] - m[j]) * (ll[i] - ll[j]) < 0]
    return {"status": "ok", "statistic": cfg.rank_statistic, "threshold": cfg.rank_threshold, "value": value,
            "ci": list(stats.quantile_ci(finite, cfg.level)) if len(finite) else [NAN, NAN],
            "supported": bool(value >= cfg.rank_threshold),
            "arms": [{"mock": p[0], "llm": p[1], "mock_asr": float(m[i]), "llm_asr": float(ll[i]),
                      "mock_rank": float(m_rank[i]), "llm_rank": float(l_rank[i])} for i, p in enumerate(pairs)],
            "inversions": inversions, "contexts": {"mock": m_cells, "llm": l_cells}}
