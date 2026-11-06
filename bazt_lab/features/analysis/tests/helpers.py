"""Shared helpers of the analysis tests: write synthetic rows, load them, build a config."""
from __future__ import annotations

import numpy as np

from bazt_lab.features.analysis.private import config, table
from bazt_lab.features.analysis.tests import synth


def load(tmp_path, rows, platform="mock", **over):
    """(table, cfg) for synthetic rows written under tmp_path and a settings tree with `over` applied."""
    synth.write_rows(tmp_path, rows)
    cfg = config.Cfg.from_settings(synth.settings(**over))
    return table.load(tmp_path, cfg.all_blocks(platform), cfg.metric_columns), cfg


def independent_holm(p):
    """Holm step-down adjusted p-values, written from the definition (sorted, (m - k) p, running max, cap 1)."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * p[i]))
        adj[i] = running
    return adj
