"""swarmlab.run_episode with the no-defense policy: the reference of the statistical fidelity test (the only place
the mock_engine feature imports swarmlab)."""
from __future__ import annotations

from typing import Mapping, Sequence

import swarmlab


def no_defense(topology: str, N: int, T: int, seeds: Sequence[int], model: Mapping) -> dict:
    """Per seed: final ever-infected rate and task MAE of swarmlab's no_defense run (mock agent, no decoys)."""
    cfg = swarmlab.Config(N=N, T=T, **{k: model[k] for k in ("m", "a", "b", "mu_inf", "mu_dec", "sd", "beta", "p_stay",
                                                             "p_decoy", "bias")})
    out = {"infected": [], "mae": []}
    for s in seeds:
        r = swarmlab.run_episode(swarmlab.make_world(topology, s, cfg), "no_defense", cfg,
                                 swarmlab.MockAgentPolicy(cfg), frozenset())
        out["infected"].append(r["final_ever_infected_rate"])
        out["mae"].append(r["task_mae"])
    return out
