"""Public interface of bazt_lab.core: the contracts every feature codes against.

Features import core ONLY through this module (`from bazt_lab.core import public as core`). core never imports a
feature. See docs/ARCHITECTURE_EXPERIMENTS.md for the feature map and the import rules.
"""
from bazt_lab.core.private.checkpoint import CheckpointStore, committed_row_files
from bazt_lab.core.private.checkpoint import replace as atomic_replace
from bazt_lab.core.private.compute import as_bool, as_uint8, compiled, inductor_available, tie_jitter_scale
from bazt_lab.core.private.compute import resolve_device as device
from bazt_lab.core.private.compute import resolve_dtype as dtype
from bazt_lab.core.private.contracts import (
    ActionKind,
    Actions,
    ArmSpec,
    Defender,
    EpisodeContext,
    EpisodeTrace,
    Judge,
    NullTracker,
    Observation,
    Platform,
    Tensor,
    Tracker,
)
from bazt_lab.core.private.rng import (
    batch_uniforms,
    episode_keys,
    keyed_normal,
    keyed_permutation_scores,
    keyed_randint,
    keyed_uniform,
    seed_of,
    stream,
    uniforms,
)
from bazt_lab.core.private.settings import SETTINGS_PATH, Unset, get, is_set, load_settings, parse_value, put

__all__ = [
    "ActionKind", "Actions", "ArmSpec", "as_bool", "as_uint8", "atomic_replace", "CheckpointStore", "committed_row_files",
    "compiled", "Defender", "device", "dtype",
    "EpisodeContext", "EpisodeTrace", "inductor_available", "Judge",
    "NullTracker", "Observation", "Platform", "SETTINGS_PATH", "Tensor", "Tracker", "Unset", "batch_uniforms",
    "episode_keys", "get", "is_set", "keyed_normal", "keyed_permutation_scores", "keyed_randint", "keyed_uniform",
    "load_settings", "parse_value", "put", "seed_of", "stream", "tie_jitter_scale", "uniforms",
]
