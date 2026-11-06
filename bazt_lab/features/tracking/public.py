"""Public interface of the tracking feature: WandbTracker, the core.Tracker contract over Weights & Biases.

See CONTRACT.md. The composition root builds it only when tracking is enabled, so `wandb` is imported only then.
"""
from bazt_lab.features.tracking.adapters.wandb_tracker import WandbTracker
from bazt_lab.features.tracking.private.errors import TrackingError

__all__ = ["TrackingError", "WandbTracker"]
