"""Shared builders for the BAZT tests (no test collected from here)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import swarmlab as sl  # noqa: E402
from bazt.config import BaztConfig  # noqa: E402
from bazt.gateway import EnforcementGateway  # noqa: E402

TUNE = tuple(range(9001, 9013))      # tune seeds only: BAZT is never exercised on main/heldout in tests


def chain_adj(n):
    return [sorted(j for j in (i - 1, i + 1) if 0 <= j < n) for i in range(n)]


def make_view(t, n=12, T=12, roll=None, active=None, adj=None):
    """A StepView whose monitor fields are consistent with swarmlab's calibration."""
    cfg = sl.Config(N=n, T=T)
    roll = list(roll) if roll is not None else [0.0] * n
    adj = adj or chain_adj(n)
    deg = tuple(len(a) for a in adj)
    p = tuple(sl.sigmoid(cfg.a * r - cfg.b) for r in roll)
    return sl.StepView(t=t, active=tuple(active if active is not None else range(n)), observed_texts=("",) * n,
                       p_monitor=p, roll_scores=tuple(roll), deg=deg, maxdeg=max(deg))


def make_gateway(bcfg=None, n=12, T=12, topology="chain", seed=9001, adj=None, seed_agents=frozenset(), **sl_kw):
    adj = adj or chain_adj(n)
    return EnforcementGateway("bazt", bcfg or BaztConfig(), sl.Config(N=n, T=T, **sl_kw), topology, seed,
                              adj, adj, seed_agents)


def hot_roll(n=12, hot=(3,), level=2.5):
    return [level if i in hot else 0.0 for i in range(n)]
