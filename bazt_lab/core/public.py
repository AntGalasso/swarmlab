"""Public interface of bazt_lab.core: the contracts every feature codes against.

Features import core ONLY through this module (`from bazt_lab.core import public as core`). core never imports a
feature. See docs/ARCHITECTURE_EXPERIMENTS.md for the feature map and the import rules.
"""
from bazt_lab.core.private.checkpoint import CheckpointStore
from bazt_lab.core.private.contracts import (
    ActionKind, Actions, ArmSpec, Defender, EpisodeContext, EpisodeTrace, Judge, NullTracker, Observation,
    Platform, Tensor, Tracker,
)
from bazt_lab.core.private.rng import batch_uniforms, seed_of, stream, uniforms
from bazt_lab.core.private.settings import SETTINGS_PATH, Unset, get, is_set, load_settings, parse_value, put

__all__ = [
    "ActionKind", "Actions", "ArmSpec", "CheckpointStore", "Defender", "EpisodeContext", "EpisodeTrace", "Judge",
    "NullTracker", "Observation", "Platform", "SETTINGS_PATH", "Tensor", "Tracker", "Unset", "batch_uniforms",
    "get", "is_set", "load_settings", "parse_value", "put", "seed_of", "stream", "uniforms",
]
