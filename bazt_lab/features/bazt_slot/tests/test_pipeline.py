"""The A+B+C pipeline BAZT runs on its own observation: the audits are BAZT's choice and are paid once, the inner arms
spend nothing, and what BAZT takes from them is A's free denies and the reported decision statistics."""
import math

import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.bazt_slot.private import pipeline as pl
from bazt_lab.features.bazt_slot.tests.testkit import NAN, Fake, full_adj, make_ctx

F64 = torch.float64


class Inner:
    """A defender that records the observation it is given and reports a scripted statistic."""

    def __init__(self, name, stat=None, deny=None):
        self.name, self.stat, self.deny, self.seen, self.resets, self.loaded = name, stat, deny, [], 0, None
        self.counter = 0

    def reset(self, ctx):
        self.ctx, self.resets, self.seen, self.counter = ctx, self.resets + 1, [], 0

    def decide(self, obs):
        self.seen.append(obs)
        self.counter += 1
        a = core.Actions(deny=self.deny(obs) if self.deny else torch.zeros_like(obs.attempts))
        if self.stat:
            audited = obs.audit(obs.attempts.any(-1))
            a.statistics = {self.stat: audited[:, :, None].expand_as(obs.attempts).clone()}
        a.events = [{"event": "budget_exhausted", "arm": self.name}]
        return a

    def state_dict(self):
        return {"counter": self.counter}

    def load_state_dict(self, state):
        self.loaded = dict(state)
        self.counter = state["counter"]


def make_pipeline(**inner):
    arms = {"A": Inner("A", deny=lambda o: o.attempts & ~full_adj(*o.attempts.shape[:2])),
            "B": Inner("B", stat="judge"), "C": Inner("C", stat="rl_minus_ts")}
    arms.update(inner)
    return pl.Pipeline({n: (lambda a=a: a) for n, a in arms.items()}), arms


def setup(B=2, N=3, **kw):
    ctx = make_ctx(full_adj(B, N), **kw)
    fake = Fake(ctx)
    return ctx, fake


def test_the_inner_arms_are_reset_with_the_context_of_the_arm():
    pipe, arms = make_pipeline()
    ctx, _ = setup()
    pipe.reset(ctx)
    assert all(a.resets == 1 and a.ctx is ctx for a in arms.values())


def test_inner_arms_are_served_the_audits_bazt_bought_and_nothing_else_is_paid():
    pipe, arms = make_pipeline()
    ctx, fake = setup(audit_cap=10)
    pipe.reset(ctx)
    obs = fake.obs(1)
    scores = torch.tensor([[1.5, NAN, 2.5], [NAN, NAN, NAN]], dtype=F64)
    result = pipe.step(obs, scores)
    seen = arms["B"].seen[0]
    assert fake.audit_calls == [] and (fake.spent_audit == 0).all()          # the real audit was never called
    assert torch.allclose(result.statistics["judge"][0, 0], torch.tensor([1.5] * 3, dtype=F64))
    assert result.statistics["judge"][0, 1].isnan().all() and result.statistics["judge"][1].isnan().all()
    assert torch.equal(seen.audit(torch.tensor([[True, True, False], [False, False, False]]))[0, :2].isnan(),
                       torch.tensor([False, True]))


def test_an_inner_arm_is_never_told_the_intervention_budget_it_does_not_spend():
    pipe, arms = make_pipeline()
    ctx, fake = setup(intervention_cap=5)
    pipe.reset(ctx)
    pipe.step(fake.obs(1), torch.full((2, 3), NAN, dtype=F64))
    assert torch.isinf(arms["B"].seen[0].remaining_intervention).all()
    assert torch.equal(arms["B"].seen[0].remaining_audit, fake.remaining_audit)


def test_a_denies_are_returned_and_the_statistics_of_b_and_c_are_collected_without_their_events():
    pipe, arms = make_pipeline()
    ctx, fake = setup()
    pipe.reset(ctx)
    obs = fake.obs(1, attempts=torch.ones(2, 3, 3, dtype=torch.bool))
    result = pipe.step(obs, torch.zeros(2, 3, dtype=F64))
    assert result.deny_a.shape == (2, 3, 3) and result.deny_a[:, range(3), range(3)].all()    # the self-loops
    assert set(result.statistics) == {"judge", "rl_minus_ts"}
    assert not hasattr(result, "events")


def test_state_round_trips_through_state_dict_and_load_state_dict():
    pipe, arms = make_pipeline()
    ctx, fake = setup()
    pipe.reset(ctx)
    pipe.step(fake.obs(1), torch.zeros(2, 3, dtype=F64))
    saved = pipe.state_dict()
    assert saved == {"A": {"counter": 1}, "B": {"counter": 1}, "C": {"counter": 1}}
    pipe2, arms2 = make_pipeline()
    pipe2.reset(ctx)
    pipe2.load_state_dict(saved)
    assert arms2["C"].counter == 1


def test_a_pipeline_missing_one_of_the_three_arms_refuses_and_says_which():
    with pytest.raises(KeyError, match="'C'"):
        pl.Pipeline({"A": lambda: Inner("A"), "B": lambda: Inner("B")})


def test_the_default_pipeline_is_built_from_the_conventional_features_arms():
    pipe = pl.Pipeline()
    arms = {n: f() for n, f in pipe.factories.items()}
    assert set(arms) == {"A", "B", "C"} and all(a.name == n for n, a in arms.items())


def test_the_real_a_b_c_run_on_a_cached_audit_and_report_their_statistics():
    """Integration with the conventional feature through its public ARMS only."""
    s = core.load_settings(profile="sprint")
    core.put(s, "runtime.rate_limit_per_identity", 2)
    core.put(s, "conventional.c.judge_score_bins", {"mock": [[-5.0, 0.0], [-1.0, 4.0], [math.inf, 12.0]]})
    core.put(s, "conventional.c.presence_window_rounds", 3)
    ctx, fake = setup(settings=s, thresholds={"judge": 0.5, "rl_minus_ts": 100.0})
    pipe = pl.Pipeline()
    pipe.reset(ctx)
    scores = torch.tensor([[2.0, NAN, 0.0], [NAN, 1.0, NAN]], dtype=F64)
    result = pipe.step(fake.obs(1), scores)
    judge = result.statistics["judge"]
    assert judge[0, 0, 1].item() == 2.0 and judge[0, 1, 2].isnan() and judge[1, 1, 0].item() == 1.0
    assert not result.deny_a.any()
    assert fake.audit_calls == []
