"""Arm B: deployed agent zero-trust pipeline (spec 2.2) and the capped behaviour of spec 2.4."""
import io

import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.conventional.private.arm_b import ArmB
from bazt_lab.features.conventional.tests.testkit import COSTS, Fake, adj_of, full_adj, make_ctx, make_settings


def _run(B=1, N=3, *, adj=None, thresholds=None, rate_limit=10, **ctx_kw):
    adj = full_adj(B, N) if adj is None else adj
    ctx = make_ctx(adj, "B", thresholds=thresholds, settings=make_settings(rate_limit=rate_limit), **ctx_kw)
    arm = ArmB()
    arm.reset(ctx)
    return ctx, arm, Fake(ctx)


def test_denies_every_delivery_of_a_sender_whose_judge_score_exceeds_the_threshold():
    ctx, arm, fake = _run(thresholds={"judge": 0.4})
    actions = arm.decide(fake.obs(1, scores=[0.9, 0.1, 0.5]))
    expected = torch.zeros(1, 3, 3, dtype=torch.bool)
    expected[0, 0, [1, 2]] = True
    expected[0, 2, [0, 1]] = True
    assert torch.equal(actions.deny, expected)


def test_score_equal_to_the_threshold_is_not_denied():
    ctx, arm, fake = _run(thresholds={"judge": 0.5})
    assert not arm.decide(fake.obs(1, scores=[0.5, 0.5, 0.5])).deny.any()


def test_reports_the_judge_statistic_per_delivery_with_nan_where_not_evaluated():
    ctx, arm, fake = _run(adj=adj_of(1, [(0, 1), (0, 2), (1, 2)], 3), thresholds={"judge": 0.4})
    attempts = adj_of(1, [(0, 1), (1, 2)], 3)                 # (0, 2) is an edge nobody uses this round
    stat = arm.decide(fake.obs(1, attempts, scores=[0.9, 0.1, 0.5])).statistics["judge"]
    assert stat.dtype == torch.float64 and stat.shape == (1, 3, 3)
    assert stat[0, 0, 1] == 0.9 and stat[0, 1, 2] == 0.1
    assert torch.isnan(stat).sum() == 7


def test_missing_threshold_means_plus_infinity_so_it_reports_without_acting():
    ctx, arm, fake = _run(thresholds={})
    actions = arm.decide(fake.obs(1, scores=[1e9, 1e9, 1e9]))
    assert not actions.deny.any()
    assert actions.statistics["judge"][0, 0, 1] == 1e9


def test_applies_a_first_and_does_not_audit_a_text_a_blocks_entirely():
    adj = adj_of(1, [(0, 1), (1, 2)], 3)
    ctx, arm, fake = _run(adj=adj, thresholds={"judge": 0.4})
    attempts = adj_of(1, [(0, 1), (1, 2), (2, 0)], 3)          # (2, 0) is off the allowlist
    authentic = torch.ones(1, 3, 3, dtype=torch.bool)
    authentic[0, 1, 2] = False                                  # tampered in transit
    actions = arm.decide(fake.obs(1, attempts, scores=[0.1, 0.1, 0.1], authentic=authentic))
    assert torch.equal(actions.deny, adj_of(1, [(1, 2), (2, 0)], 3))
    assert len(fake.audit_calls) == 1
    assert fake.audit_calls[0].tolist() == [[True, False, False]]   # only sender 0 has a deliverable text
    assert torch.isnan(actions.statistics["judge"][0, 1, 2])


def test_audit_requested_once_per_sender_not_once_per_delivery():
    ctx, arm, fake = _run(N=4, thresholds={"judge": 0.4})
    arm.decide(fake.obs(1, scores=[0.1] * 4))
    assert len(fake.audit_calls) == 1 and fake.audit_calls[0].all()
    assert fake.remaining_audit.isinf().all()


def test_unchecked_sender_is_not_denied_by_the_judge():
    ctx, arm, fake = _run(thresholds={"judge": 0.4}, audit_cap=2.0)    # the budget pays for two of three senders
    actions = arm.decide(fake.obs(1, scores=[0.9, 0.9, 0.9]))
    assert actions.deny[0, 0].any() and actions.deny[0, 1].any() and not actions.deny[0, 2].any()
    assert torch.isnan(actions.statistics["judge"][0, 2]).all()


def test_rate_limit_keeps_at_most_r_receivers_per_identity_after_the_judge():
    ctx, arm, fake = _run(N=5, thresholds={"judge": 0.4}, rate_limit=2)
    attempts = fake.obs(1).attempts
    actions = arm.decide(fake.obs(1, attempts, scores=[0.9, 0.1, 0.1, 0.1, 0.1]))
    delivered = attempts & ~actions.deny
    assert delivered[0, 0].sum() == 0                          # judged injected: nothing delivered
    assert (delivered[0, 1:].sum(-1) == 2).all()               # the others reach exactly R = 2 receivers
    assert not (actions.deny & ~attempts).any()


def test_rate_limit_below_the_cap_changes_nothing():
    ctx, arm, fake = _run(N=4, thresholds={"judge": 0.4}, rate_limit=3)
    assert not arm.decide(fake.obs(1, scores=[0.1] * 4)).deny.any()


def test_decisions_depend_only_on_the_current_hop():
    ctx, arm, fake = _run(thresholds={"judge": 0.4})
    arm.decide(fake.obs(1, scores=[0.9, 0.9, 0.9]))
    second = arm.decide(fake.obs(2, scores=[0.1, 0.9, 0.1]))
    ctx2, arm2, fake2 = _run(thresholds={"judge": 0.4})
    alone = arm2.decide(fake2.obs(2, scores=[0.1, 0.9, 0.1]))
    assert torch.equal(second.deny, alone.deny)


def test_batch_invariance_including_the_random_rate_limit_choice():
    B, N = 3, 6
    adj = full_adj(B, N)
    ctx, arm, fake = _run(B=B, N=N, adj=adj, thresholds={"judge": 0.4}, rate_limit=2)
    batch = arm.decide(fake.obs(1, scores=[0.1] * N)).deny
    for b in range(B):
        ctx1, arm1, fake1 = _run(B=1, N=N, adj=adj[b:b + 1], thresholds={"judge": 0.4}, rate_limit=2, ids=[f"u{b}"])
        assert torch.equal(arm1.decide(fake1.obs(1, scores=[0.1] * N)).deny[0], batch[b])


def test_audit_budget_exhaustion_emits_one_event_then_only_a_rules():
    ctx, arm, fake = _run(thresholds={"judge": 0.4}, audit_cap=2.0, T=3)
    first = arm.decide(fake.obs(1, scores=[0.9, 0.1, 0.9]))
    assert [(e["event"], e["arm"], e["b"], e["budget"], e["t"]) for e in first.events] == [
        ("budget_exhausted", "B", 0, "audit", 1)]
    calls = len(fake.audit_calls)
    authentic = torch.ones(1, 3, 3, dtype=torch.bool)
    authentic[0, 0, 1] = False
    second = arm.decide(fake.obs(2, scores=[0.9, 0.9, 0.9], authentic=authentic))
    assert len(fake.audit_calls) == calls                      # no more checks
    assert second.events == [] and not second.statistics["judge"].isfinite().any()
    assert torch.equal(second.deny, adj_of(1, [(0, 1)], 3))    # A's rule only: the tampered delivery
    third = arm.decide(fake.obs(3, scores=[0.9, 0.9, 0.9]))
    assert third.events == [] and not third.deny.any()


def test_intervention_budget_funds_the_most_suspicious_denies_then_falls_back_to_a():
    ctx, arm, fake = _run(thresholds={"judge": 0.5}, intervention_cap=2.0, T=3)
    first = arm.decide(fake.obs(1, scores=[0.9, 0.8, 0.7]))
    assert first.deny.sum() == 2                               # the two deliveries of the most suspicious sender
    assert first.deny[0, 0, 1] and first.deny[0, 0, 2]
    assert [(e["budget"], e["t"]) for e in first.events] == [("intervention", 1)]
    fake.settle(first)
    calls = len(fake.audit_calls)
    second = arm.decide(fake.obs(2, scores=[0.9, 0.9, 0.9]))
    assert not second.deny.any() and second.events == [] and len(fake.audit_calls) == calls


def test_funding_ties_are_broken_by_a_keyed_stream_not_by_index():
    results = set()
    for b in range(8):
        ctx, arm, fake = _run(thresholds={"judge": 0.5}, intervention_cap=1.0, ids=[f"tie{b}"])
        first = arm.decide(fake.obs(1, scores=[0.9, 0.9, 0.9]))
        assert first.deny.sum() == 1
        results.add(tuple(first.deny.nonzero()[0].tolist()))
    assert len(results) > 1                                    # not always the lowest index


def test_episodes_exhaust_independently_and_events_carry_the_episode_index():
    ctx, arm, fake = _run(B=2, thresholds={"judge": 0.4})
    ctx.audit_cap[0] = 0.0                                     # episode 0 starts with no audit budget
    fake.remaining_audit = ctx.audit_cap.clone()
    actions = arm.decide(fake.obs(1, scores=[0.9, 0.9, 0.9]))
    assert [(e["b"], e["budget"]) for e in actions.events] == [(0, "audit")]
    assert not actions.deny[0].any() and actions.deny[1].any()


def test_uncapped_never_emits_an_event():
    ctx, arm, fake = _run(thresholds={"judge": 0.4})
    for t in range(1, 4):
        actions = arm.decide(fake.obs(t, scores=[0.9, 0.9, 0.9]))
        fake.settle(actions)
        assert actions.events == []


def test_settings_and_costs_are_required_not_defaulted():
    adj = full_adj(1, 3)
    with pytest.raises(core.Unset) as e:
        ArmB().reset(make_ctx(adj, "B", settings=make_settings(rate_limit=None)))
    assert e.value.args[0] == "runtime.rate_limit_per_identity"
    with pytest.raises(ValueError):
        ArmB().reset(make_ctx(adj, "B", settings=make_settings(rate_limit=0)))
    with pytest.raises(core.Unset) as e:
        ArmB().reset(make_ctx(adj, "B", settings=make_settings(), costs={k: v for k, v in COSTS.items() if k != "deny"}))
    assert "deny" in e.value.args[0]


def test_state_round_trip_keeps_exhaustion_and_does_not_re_emit():
    ctx, arm, fake = _run(thresholds={"judge": 0.4}, audit_cap=1.0, T=3)
    arm.decide(fake.obs(1, scores=[0.9, 0.9, 0.9]))
    buf = io.BytesIO()
    torch.save(arm.state_dict(), buf)
    buf.seek(0)
    state = torch.load(buf, weights_only=True)
    ctx2, arm2, fake2 = _run(thresholds={"judge": 0.4}, audit_cap=1.0, T=3)
    arm2.load_state_dict(state)
    fake2.remaining_audit = fake.remaining_audit.clone()
    again = arm2.decide(fake2.obs(2, scores=[0.9, 0.9, 0.9]))
    assert again.events == [] and fake2.audit_calls == []


def test_zero_intervention_budget_is_exhausted_from_the_start_but_a_rules_are_free():
    ctx, arm, fake = _run(thresholds={"judge": 0.4}, intervention_cap=0.0)
    authentic = torch.ones(1, 3, 3, dtype=torch.bool)
    authentic[0, 0, 1] = False
    actions = arm.decide(fake.obs(1, scores=[0.9, 0.9, 0.9], authentic=authentic))
    assert [(e["budget"], e["t"]) for e in actions.events] == [("intervention", 1)]
    assert torch.equal(actions.deny, adj_of(1, [(0, 1)], 3)) and fake.audit_calls == []


def test_both_budgets_at_zero_give_one_event_naming_audit():
    ctx, arm, fake = _run(thresholds={"judge": 0.4}, audit_cap=0.0, intervention_cap=0.0)
    first = arm.decide(fake.obs(1, scores=[0.9, 0.9, 0.9]))
    assert [(e["budget"], e["t"]) for e in first.events] == [("audit", 1)]
    assert arm.decide(fake.obs(2)).events == []


def test_is_a_defender():
    assert isinstance(ArmB(), core.Defender) and ArmB().name == "B"


def test_a_sender_whose_text_was_not_checked_is_not_rate_limited():
    ctx, arm, fake = _run(N=5, thresholds={"judge": 100.0}, rate_limit=1, audit_cap=2.0)   # pays for 2 of 5 senders
    actions = arm.decide(fake.obs(1, scores=[0.0] * 5))
    unchecked = actions.statistics["judge"].isnan().all(-1)[0]
    assert unchecked.sum() == 3
    assert not actions.deny[0][unchecked].any()                   # only checked senders are rate limited
    assert (actions.deny[0][~unchecked].sum(-1) == 3).all()
