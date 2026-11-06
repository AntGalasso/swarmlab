"""Statistical fidelity of the torch engine against swarmlab.run_episode (no defence, N = 12, T = 12).

The two simulators draw different random numbers, so they are compared in distribution: the means over many seeds must
agree within 4.5 standard errors of their difference."""
import math

import pytest
import torch

from bazt_lab.features.mock_engine.adapters import swarmlab_reference
from bazt_lab.features.mock_engine.tests.helpers import make_settings, sim

SEEDS = range(1, 801)


def _se(x):
    x = torch.as_tensor(x, dtype=torch.float64)
    return float(x.var(unbiased=True) / x.numel()) ** 0.5


@pytest.mark.parametrize("topology", ["small_world", "chain"])
def test_final_infection_rate_and_mae_match_swarmlab_in_distribution(topology):
    s = make_settings(mock__model__decoys=0)
    ref = swarmlab_reference.no_defense(topology, 12, 12, SEEDS, s["mock"]["model"])
    w, trace, est = sim("infection", T=12, topology=topology, N=12, settings=s, seeds=tuple(SEEDS))
    mine_inf = trace.infected.any(1).double().mean(1)
    mine_mae = (est[:, -1] - w.mean[:, None]).abs().mean(1)
    for name, a, b in (("infected", mine_inf, ref["infected"]), ("mae", mine_mae, ref["mae"])):
        ref_v = torch.as_tensor(b, dtype=torch.float64)
        se = math.hypot(_se(a), _se(ref_v))
        diff = abs(float(a.mean() - ref_v.mean()))
        assert diff <= 4.5 * se + 1e-9, (topology, name, float(a.mean()), float(b.mean()), se)
