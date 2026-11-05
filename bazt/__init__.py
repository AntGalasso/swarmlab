"""bazt -- Budgeted Adaptive Zero-Trust gateway for agent swarms.

PRE-EXISTING SCAFFOLDING (written before the Swarm Alignment Hackathon kickoff). Contains no experimental result;
every number it produces in --mode mock is a pipeline check, not a finding.

Layered on top of the pre-existing simulator: this package imports `swarmlab` and `env_ladder`; neither of
them imports it. Everything attaches to `swarmlab.run_episode` through `EpisodeHooks.gateway_factory`, so a
run without that hook (or with a non-BAZT policy) is bit-identical to the original simulator.

Module map (dependencies point downwards only):

    config -> actions -> ledger -> optimizer -> gateway -> arms -> runner
              evidence, provenance, roles, contagion, audit, metrics, channel  (leaf helpers)
"""
from bazt.config import BaztConfig  # noqa: F401

__all__ = ["BaztConfig"]
