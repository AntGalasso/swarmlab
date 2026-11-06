"""Containment per cost: the security gain of an arm over the paired reference run (A, which has no budget cell and runs
uncapped), divided by what the arm spent. It is a ratio of totals, sum(gain) / sum(spend) over the paired episodes, so
an episode that spends little does not dominate; its interval resamples whole seeds. Undefined (NaN) when nothing was
spent. Every gain metric is a "lower is better" damage metric, so the gain is reference minus arm.
"""
from __future__ import annotations

import numpy as np

from bazt_lab.features.analysis.private import compare, stats
from bazt_lab.features.analysis.private.config import SPEND_AUDIT, SPEND_INTERVENTION, Cfg
from bazt_lab.features.analysis.private.table import UNCAPPED, Table, sort_key

NAN = float("nan")


def containment(t: Table, cfg: Cfg, platform: str) -> dict:
    blocks, status = compare.gate(t, cfg, "containment", platform)
    if status:
        return status
    ref, entries, skipped = cfg.containment_reference, [], []
    for key in sorted((k for k in t.index if k.block in blocks and not k.attack_free and k.arm != ref), key=sort_key):
        rows = None
        for block in (key.block, *blocks):                       # the reference may run in another block (A: headline)
            rows = t.aligned([key, key._replace(arm=ref, audit=UNCAPPED, intervention=UNCAPPED, block=block)])
            if rows is not None and len(rows[0]):
                break
            rows = None
        if rows is None:
            skipped.append({"arm": key.arm, "context": compare.describe(key.ctx),
                            "reason": f"no uncapped {ref} run to pair with"})
            continue
        spend = t.metrics[SPEND_AUDIT][rows[0]] + t.metrics[SPEND_INTERVENTION][rows[0]]
        seed = t.codes["seed"][rows[0]]
        universe = np.unique(seed)
        idx = stats.resample_index(("containment", platform, *map(str, key)), cfg.resamples, len(universe))
        for metric in cfg.containment_metrics:
            gain = t.metrics[metric][rows[1]] - t.metrics[metric][rows[0]]
            ok = ~(np.isnan(gain) | np.isnan(spend))
            if not ok.any():
                continue
            g_sums, _ = stats.cluster_sums(gain[ok], seed[ok], universe)
            s_sums, _ = stats.cluster_sums(spend[ok], seed[ok], universe)
            g_rep, s_rep = g_sums[idx].sum(axis=1), s_sums[idx].sum(axis=1)
            rep = np.divide(g_rep, s_rep, out=np.full(len(idx), NAN), where=s_rep > 0)
            rep = rep[~np.isnan(rep)]
            ratio = gain[ok].sum() / spend[ok].sum() if spend[ok].sum() > 0 else NAN
            entries.append({"arm": key.arm, "context": compare.describe(key.ctx), "metric": metric,
                            "n_pairs": int(ok.sum()), "n_clusters": len(np.unique(seed[ok])),
                            "mean_gain": float(gain[ok].mean()), "mean_spend": float(spend[ok].mean()),
                            "containment_per_cost": float(ratio),
                            "ci": list(stats.quantile_ci(rep, cfg.level)) if len(rep) else [NAN, NAN]})
    return {"status": "ok", "blocks": list(blocks), "reference": ref, "entries": entries, "skipped": skipped}
