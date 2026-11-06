"""What the composition root hands the runner. The runner knows no other feature by name."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from bazt_lab.core import public as core


class FeatureMissing(RuntimeError):
    """A feature a block needs is not built, or cannot be imported. `feature` may name several, comma separated."""

    def __init__(self, feature: str, detail: str) -> None:
        super().__init__(f"feature {feature} is not built (not available: {detail})")
        self.feature, self.detail = feature, detail


@dataclass(frozen=True)
class Wiring:
    arm: Callable[[str], core.ArmSpec]                 # raises FeatureMissing, or LookupError for an unknown arm
    platform: Callable[[str], core.Platform]           # "mock" | "llm"; raises FeatureMissing
    tracker: Callable[[Mapping], core.Tracker]         # built only when tracking is enabled; raises FeatureMissing
    adaptive: Optional[Callable[[], Any]] = None       # the adaptive feature's public module (run_search,
                                                       # search_episodes); raises FeatureMissing; only mock_adaptive
    analysis: Optional[Callable[[], Any]] = None       # the analysis feature's public module (analyze, missing_keys)
    sync: Optional[Callable[[], Any]] = None           # the sync feature's public module (start); sync.enabled only
    rebuild: Optional[str] = None                      # "module:function" returning a fresh Wiring, which a spawned worker
                                                       # process calls (a Wiring holds closures and cannot be pickled)
    gnn: Optional[Callable[[], Any]] = None            # the gnn feature's public module (training_finished); raises
                                                       # FeatureMissing; only blocks that run the GNN arm
