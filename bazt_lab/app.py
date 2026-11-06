"""Composition root: wires the features to the runner and exposes `main`. It holds no logic of its own.

Every feature but the runner is imported lazily, the first time a block needs it. A feature that is not built (or
cannot be imported) becomes a `FeatureMissing` that the runner reports against the block that needs it, instead of an
ImportError at start-up.
"""
from __future__ import annotations

import importlib

from bazt_lab.features.runner import public as runner

# Features whose public.py exports ARMS, a tuple of core.ArmSpec.
ARM_FEATURES = ("conventional", "references", "bazt_slot", "gnn")
# platform name -> (feature, class exported by its public.py)
PLATFORMS = {"mock": ("mock_engine", "MockPlatform"), "llm": ("llm_harness", "LLMPlatform")}
TRACKER = ("tracking", "WandbTracker")


def _feature(name: str):
    try:
        return importlib.import_module(f"bazt_lab.features.{name}.public")  # arch: allow dynamic-import
    except ImportError as e:
        raise runner.FeatureMissing(name, str(e)) from e


def _exported(feature: str, attribute: str):
    try:
        return getattr(_feature(feature), attribute)
    except AttributeError as e:
        raise runner.FeatureMissing(feature, f"its public.py does not export {attribute}") from e


def build_wiring(first_match: bool = False) -> runner.Wiring:
    """first_match: look an arm up by importing the arm features one at a time and stopping at the first that has it
    (a worker process needs one arm; the parent has already checked the registry for duplicates)."""
    arms: dict = {}
    platforms: dict = {}

    def registry():
        if not arms:
            found, missing = {}, {}
            for feature in ARM_FEATURES:
                try:
                    exported = _exported(feature, "ARMS")
                except runner.FeatureMissing as e:
                    missing[feature] = e.detail
                    continue
                for spec in exported:
                    if spec.name in found:
                        raise ValueError(f"arm {spec.name!r} is exported by both {found[spec.name][0]} and {feature}")
                    found[spec.name] = (feature, spec)
            arms.update(found=found, missing=missing)
        return arms["found"], arms["missing"]

    def first_arm(name):
        for feature in ARM_FEATURES:
            try:
                for spec in _exported(feature, "ARMS"):
                    if spec.name == name:
                        return spec
            except runner.FeatureMissing:
                continue
        raise LookupError(f"no feature exports an arm named {name!r}")

    def arm(name):
        if first_match:
            return first_arm(name)
        found, missing = registry()
        if name in found:
            return found[name][1]
        if missing:
            raise runner.FeatureMissing(", ".join(missing), f"arm {name!r} is not exported by a built feature; "
                                        + "; ".join(f"{f}: {d}" for f, d in missing.items()))
        raise LookupError(f"no feature exports an arm named {name!r}")

    def platform(name):
        if name not in platforms:
            platforms[name] = _exported(*PLATFORMS[name])()
        return platforms[name]

    def tracker(settings):
        return _exported(*TRACKER)(settings)

    return runner.Wiring(arm=arm, platform=platform, tracker=tracker,
                         adaptive=lambda: _feature("adaptive"), gnn=lambda: _feature("gnn"),
                         analysis=lambda: _feature("analysis"), sync=lambda: _feature("sync"),
                         rebuild="bazt_lab.app:build_worker_wiring")


def build_worker_wiring() -> runner.Wiring:
    return build_wiring(first_match=True)


def main(argv=None) -> int:
    return runner.main(argv, wiring=build_wiring())
