"""bazt/ -- Budget-Aware Zero-Trust Topology defence (BAZT).

Stage-4 extension built on top of the pre-existing `swarmlab.py` simulator.

Dependency direction (non-negotiable): `bazt` imports `swarmlab` and `env_ladder`;
neither `swarmlab.py` nor `env_ladder.py` imports `bazt`.

Every BAZT arm is plugged into `swarmlab.run_episode` through `EpisodeHooks`
(`gateway_factory` / `edge_weight_modifier`), which are default-off and only
consulted for policies in `swarmlab.BAZT_POLICIES`.  All legacy arms and the
`hooks=None` path remain bit-identical.

Mock mode only before kickoff (`SWARMLAB_KICKOFF_UTC = 2026-11-06 20:00 UTC`);
`--mode llm` is refused unconditionally before kickoff and routed through
`llm_client.make_llm_policy` (openrouter / adaption) after kickoff.
"""

from __future__ import annotations

__all__ = [
    "config", "roles", "provenance", "evidence", "contagion",
    "actions", "ledger", "optimizer", "gateway", "audit", "metrics",
    "arms", "runner",
]
