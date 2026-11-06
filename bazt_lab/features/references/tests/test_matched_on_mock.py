"""D116 through the real mock platform: a matched-timing control handed a recorded `intervened_by_round` as
`unit["timing_schedule"]` intervenes only at those rounds, never elsewhere, and within its budget. Every setting is a
TEST value."""
import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.mock_engine import public as mock
from bazt_lab.features.references import public as refs
from bazt_lab.features.references.tests.test_legacy import LEGACY

ARMS = {a.name: a for a in refs.ARMS}
MATCHED = sorted(n for n in ARMS if n.endswith(refs.MATCHED_SUFFIX))
THRESHOLDS = {"monitor": 0.5, "risk": 0.05}


def settings():
    s = core.load_settings(profile="sprint")
    test = {
        "runtime.n_insiders": 2, "runtime.rate_limit_per_identity": 1, "runtime.horizon_constant": 3.0,
        "runtime.eps_correct": 2.0, "runtime.ba_m": 2, "runtime.ws_k": 4, "runtime.ws_rewire_p": 0.25,
        "runtime.sanitize_success": 0.9, "runtime.insider_bias": 8.0, "runtime.ta_detectability": 1.5,
        "runtime.ma_detectability": 1.2, "runtime.aitm_rewrite_strength": 1.0, "runtime.recovery_margin": 0.05,
        "mock.model.bias": 8.0, "mock.model.burst_span": 1, "mock.auditor.binormal_method": "shared_threshold",
        "costs.robustness.deny": 1, **{f"mock.legacy.{k}": v for k, v in LEGACY.items()},
    }
    for k, v in test.items():
        core.put(s, k, v)
    return s


def unit(arm, seed, attack="infection", intervention="uncapped", **extra):
    u = dict(block="t", platform="mock", arm=arm, audit="uncapped", intervention=intervention, topology="small_world",
             attack=attack, agents=8, seed=seed, sink=seed % 8, **extra)
    u["id"] = f"{arm}-{attack}-{seed}-{intervention}"
    return u


class _Source:
    """A source arm that denies every delivery at the rounds t with (t + seed) % 3 == 0, standing in for BAZT."""
    name = "src"

    def reset(self, ctx):
        self.shift = torch.tensor([int(str(u).split("-")[2]) for u in ctx.unit_ids])

    def decide(self, obs):
        on = (obs.t + self.shift) % 3 == 0
        return core.Actions(deny=obs.attempts & on[:, None, None])

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


def recorded(seeds, attack):
    """The intervened_by_round column of the source arm's rows: what the runner hands a matched unit (D116)."""
    rows = mock.MockPlatform().run_batch([unit("src", sd, attack) for sd in seeds], core.ArmSpec("src", _Source),
                                         settings(), _Store(), core.NullTracker())
    return {r["seed"]: r["intervened_by_round"] for r in rows}


class _Store:
    def __init__(self):
        self.rows, self.done = {}, set()

    def is_done(self, uid):
        return uid in self.done

    def append_rows(self, uid, rows):
        self.rows.setdefault(uid, []).extend(rows)

    def resume_rows(self, uid):
        return list(self.rows.get(uid, []))

    def mark_done(self, uid):
        self.done.add(uid)


SEEDS = (1, 2, 3, 4)


@pytest.fixture(scope="module")
def schedules():
    attacked, clean = recorded(SEEDS, "infection"), recorded(SEEDS, "none")
    assert any(any(s) for s in attacked.values()) and not all(all(s) for s in attacked.values())
    return attacked, clean


def matched_units(name, schedules, intervention="uncapped"):
    attacked, clean = schedules
    th = dict(THRESHOLDS) if ARMS[name].calibrated else {}
    return [unit(name, sd, intervention=intervention, timing_schedule=attacked[sd], twin_timing_schedule=clean[sd],
                 thresholds=th) for sd in SEEDS]


@pytest.mark.parametrize("name", MATCHED)
@pytest.mark.parametrize("intervention", ["uncapped", 0.2])
def test_a_matched_arm_intervenes_only_at_the_recorded_rounds_and_within_its_budget(name, intervention, schedules):
    us = matched_units(name, schedules, intervention)
    rows = mock.MockPlatform().run_batch(us, ARMS[name], settings(), _Store(), core.NullTracker())
    for u, r in zip(us, rows, strict=True):
        assert r["timing_schedule"] == u["timing_schedule"]
        outside = [t for t, (did, may) in enumerate(zip(r["intervened_by_round"], u["timing_schedule"], strict=True))
                   if did and not may]
        assert outside == [], f"{name} intervened at unscheduled rounds {outside}"
        assert r["violations"] == 0


def test_random_capped_matched_acts_at_every_recorded_round_while_agents_are_left(schedules):
    """Uncapped, it isolates one active agent at each scheduled round: the record it leaves is the schedule it got."""
    us = matched_units("random_capped_matched", schedules)
    rows = mock.MockPlatform().run_batch(us, ARMS["random_capped_matched"], settings(), _Store(), core.NullTracker())
    for u, r in zip(us, rows, strict=True):
        assert sum(u["timing_schedule"]) < r["n_agents"]
        assert r["intervened_by_round"] == u["timing_schedule"]
