"""Public interface of the references feature: the reference arms of spec 2.7 as `ARMS` (core.ArmSpec), and the
helper that turns the rounds BAZT acted at into the schedule the matched-timing arms read. See CONTRACT.md.
"""
from functools import partial

from bazt_lab.core import public as core
from bazt_lab.features.references.private.allocation import BraGreedy, OracleAlloc, RandomAlloc
from bazt_lab.features.references.private.gnn_standin import GNNStandin
from bazt_lab.features.references.private.legacy import (
    CentralityOnly,
    FixedWindow,
    RandomCapped,
    RiskAware,
    RiskAwareTiered,
    ScoreOnly,
)
from bazt_lab.features.references.private.timing import MatchedTiming, schedule_from_rounds

MATCHED_SUFFIX = "_matched"

_LEGACY = {
    "fixed_early": partial(FixedWindow, "early"), "fixed_mid": partial(FixedWindow, "mid"),
    "fixed_late": partial(FixedWindow, "late"), "score_only": ScoreOnly, "centrality_only": CentralityOnly,
    "random_capped": RandomCapped, "risk_aware": RiskAware, "risk_aware_tiered": RiskAwareTiered,
}


def _matched(rule):
    """The rule without its own timing gate, acting only at the rounds of the episode's schedule."""
    return MatchedTiming(rule(own_timing=False))


def _legacy_specs():
    for name, rule in _LEGACY.items():
        calibrated = rule().thresholded
        yield core.ArmSpec(name, rule, platforms=("mock",), calibrated=calibrated)
        yield core.ArmSpec(name + MATCHED_SUFFIX, partial(_matched, rule), platforms=("mock",), calibrated=calibrated)


ARMS = (
    core.ArmSpec("random_alloc", RandomAlloc, calibrated=True),
    core.ArmSpec("oracle_alloc", OracleAlloc, needs_truth=True),
    core.ArmSpec("bra_greedy", BraGreedy, calibrated=True),
    core.ArmSpec("GNN_standin", GNNStandin, platforms=("mock",), calibrated=True),
    *_legacy_specs(),
)

__all__ = ["ARMS", "MATCHED_SUFFIX", "schedule_from_rounds"]
