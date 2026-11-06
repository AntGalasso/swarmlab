"""GNN_standin (spec 2.7, mock only): a detect-and-cut stand-in for the G-Safeguard GNN defence.

The published GNN flags attacked agents each round and cuts all their outgoing edges. Here the detector is the mock's
cheap monitor (free, obs.monitor) read against the calibrated threshold "gnn_standin"; every round (D99, the upstream
behaviour) each active agent above it has all its deliveries of that round denied. Nothing persists. This is NOT the GNN: it is a stand-in and must be reported as one (`stand_in = True`, arm name
`GNN_standin`). The real GNN is the `GNN` arm of the gnn feature, LLM side.
"""
from __future__ import annotations

from bazt_lab.core import public as core
from bazt_lab.features.references.private import common


class GNNStandin(common.Arm):
    name = "GNN_standin"
    stand_in = True
    thresholded = True
    statistic = "gnn_standin"

    def reset(self, ctx: core.EpisodeContext) -> None:
        self.ctx = ctx
        self.c_deny = common.cost_of(ctx, core.ActionKind.DENY)

    def decide(self, obs: core.Observation) -> core.Actions:
        if obs.monitor is None:
            raise ValueError("GNN_standin is driven by the mock's cheap monitor (obs.monitor); it is mock-only")
        score = obs.monitor
        flagged = obs.active & (score > common.threshold_of(self.ctx, self.statistic))
        cost = self.c_deny * obs.attempts.sum(-1).double()
        chosen = common.fit_in_order(common.rank_desc(score, flagged), cost, flagged, obs.remaining_intervention)
        return core.Actions(deny=obs.attempts & chosen[:, :, None], statistics={self.statistic: score})
