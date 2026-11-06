"""Arm A: identity-tier zero trust (spec 2.1)."""
import torch

from bazt_lab.core import public as core
from bazt_lab.features.conventional.private.arm_a import ArmA
from bazt_lab.features.conventional.tests.testkit import Fake, adj_of, make_ctx


def _setup(B=2, N=3, edges=((0, 1), (1, 2))):
    ctx = make_ctx(adj_of(B, edges, N), "A")
    arm = ArmA()
    arm.reset(ctx)
    return ctx, arm, Fake(ctx)


def test_denies_deliveries_outside_the_allowlist_and_keeps_edges():
    ctx, arm, fake = _setup()
    attempts = torch.zeros(2, 3, 3, dtype=torch.bool)
    attempts[:, 0, 1] = attempts[:, 1, 2] = True          # on the allowlist
    attempts[:, 2, 0] = True                              # not an edge
    actions = arm.decide(fake.obs(1, attempts))
    expected = torch.zeros(2, 3, 3, dtype=torch.bool)
    expected[:, 2, 0] = True
    assert torch.equal(actions.deny, expected)


def test_denies_unauthenticated_deliveries_only_where_attempted():
    ctx, arm, fake = _setup()
    authentic = torch.ones(2, 3, 3, dtype=torch.bool)
    authentic[0, 0, 1] = False                            # tampered in transit (AiTM), episode 0 only
    authentic[1, 0, 2] = False                            # tampered but never attempted: nothing to deny
    actions = arm.decide(fake.obs(1, authentic=authentic))
    expected = torch.zeros(2, 3, 3, dtype=torch.bool)
    expected[0, 0, 1] = True
    assert torch.equal(actions.deny, expected)
    assert actions.deny.dtype == torch.bool


def test_authentic_none_means_every_delivery_is_authenticated():
    ctx, arm, fake = _setup()
    assert not arm.decide(fake.obs(1, authentic=None)).deny.any()


def test_never_reads_content_never_spends_budgets_never_reports():
    ctx, arm, fake = _setup()
    actions = arm.decide(fake.obs(1))
    assert fake.audit_calls == []                         # no audit, so no audit unit is spent
    assert actions.events == [] and actions.statistics == {}
    for name in ("sanitize", "delay", "edge_quarantine", "rate_limit", "privilege_degradation", "isolation",
                 "reinstatement"):
        assert getattr(actions, name) is None


def test_ignores_the_monitor_and_scores():
    ctx, arm, fake = _setup()
    monitor = torch.ones(2, 3, dtype=torch.float64)
    assert not arm.decide(fake.obs(1, monitor=monitor)).deny.any()


def test_uncapped_even_when_budgets_are_zero():
    ctx = make_ctx(adj_of(1, ((0, 1),), 2), "A", audit_cap=0.0, intervention_cap=0.0)
    arm = ArmA()
    arm.reset(ctx)
    actions = arm.decide(Fake(ctx).obs(1, authentic=torch.zeros(1, 2, 2, dtype=torch.bool)))
    assert actions.deny[0, 0, 1] and actions.events == []   # still applies its rules, emits no event


def test_state_is_empty_and_round_trips():
    ctx, arm, fake = _setup()
    arm.decide(fake.obs(1))
    state = arm.state_dict()
    other = ArmA()
    other.reset(ctx)
    other.load_state_dict(state)
    assert other.state_dict() == state
    assert isinstance(arm, core.Defender) and arm.name == "A"
