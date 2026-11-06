"""Applying a defender's Actions to one round of one conversation (spec section 5)."""
import torch

from bazt_lab.core import public as core
from bazt_lab.features.llm_harness.private import enforce

N = 4
COSTS = {k.value: 1.0 for k in core.ActionKind}
FULL = torch.ones(N, N, dtype=torch.bool) & ~torch.eye(N, dtype=torch.bool)


def edge(*pairs):
    m = torch.zeros(1, N, N, dtype=torch.bool)
    for i, j in pairs:
        m[0, i, j] = True
    return m


def agent(*idx):
    m = torch.zeros(1, N, dtype=torch.bool)
    m[0, list(idx)] = True
    return m


def apply(actions, state=None, remaining=1e9, t=1, T=3, auth=None, limit=2):
    state = state or enforce.State.zeros(N)
    auth = FULL.clone() if auth is None else auth
    return enforce.apply(actions, state, FULL.clone(), auth, FULL, remaining, t, T, COSTS, limit,
                         lambda: torch.rand(N, N, generator=core.stream("t"), dtype=torch.float64))


def test_no_action_delivers_everything_unmodified_and_costs_nothing():
    r = apply(core.Actions())
    assert r.arrive.equal(FULL) and not r.restricted.any() and r.cost == 0 and not r.rejected


def test_deny_drops_the_delivery_and_costs_one_unit_each():
    r = apply(core.Actions(deny=edge((0, 1), (2, 3))))
    assert not r.arrive[0, 1] and not r.arrive[2, 3] and r.restricted[0, 1] and r.cost == 2


def test_denying_a_delivery_that_failed_authentication_is_free():
    auth = FULL.clone()
    auth[0, 1] = False
    r = apply(core.Actions(deny=edge((0, 1))), auth=auth)
    assert r.cost == 0 and not r.arrive[0, 1]


def test_sanitize_arrives_modified_and_delay_arrives_a_round_later():
    r = apply(core.Actions(sanitize=edge((0, 1)), delay=edge((1, 2))))
    assert r.arrive[0, 1] and r.sanitized[0, 1] and not r.arrive[1, 2] and r.delayed[1, 2]
    assert r.restricted[0, 1] and r.restricted[1, 2] and r.cost == 2


def test_edge_quarantine_is_persistent_and_costs_the_remaining_rounds():
    r = apply(core.Actions(edge_quarantine=edge((0, 1))), t=1, T=3)
    assert r.cost == 3 and r.state.quar[0, 1] and not r.arrive[0, 1]
    again = apply(core.Actions(), state=r.state, t=2)
    assert not again.arrive[0, 1] and again.restricted[0, 1] and again.cost == 0


def test_isolation_cuts_sending_and_receiving_and_costs_out_degree_times_remaining_rounds():
    r = apply(core.Actions(isolation=agent(0)), t=2, T=3)
    assert r.cost == 3 * 2 and r.state.iso[0]
    assert not r.arrive[0].any() and not r.arrive[:, 0].any()
    assert not r.restricted.any()                      # isolation losses are not "restricted"


def test_reinstatement_lifts_persistent_actions_and_a_noop_is_free():
    iso = apply(core.Actions(isolation=agent(0)))
    back = apply(core.Actions(reinstatement=agent(0)), state=iso.state, t=2, T=3)
    assert not back.state.iso[0] and back.cost == 3 * (2 - 1) and back.arrive[0].any()
    free = apply(core.Actions(reinstatement=agent(1)))
    assert free.cost == 0


def test_privilege_degradation_sanitizes_every_outgoing_delivery():
    r = apply(core.Actions(privilege_degradation=agent(0)), t=1, T=3)
    assert r.sanitized[0].sum() == 3 and r.state.degr[0] and r.cost == 3 * 3


def test_rate_limit_keeps_at_most_limit_receivers_per_round():
    r = apply(core.Actions(rate_limit=agent(0)), limit=1, t=1, T=3)
    assert r.arrive[0].sum() == 1 and r.restricted[0].sum() == 2 and r.cost == (3 - 1) * 3


def test_actions_over_the_remaining_budget_are_all_rejected_and_counted_once():
    r = apply(core.Actions(deny=edge((0, 1), (2, 3))), remaining=1)
    assert r.rejected and r.cost == 0 and r.arrive.equal(FULL) and not r.restricted.any()
    assert r.counts.sum() == 0


def test_action_counts_follow_action_kind_order():
    r = apply(core.Actions(deny=edge((0, 1)), delay=edge((1, 2))))
    kinds = [k.value for k in core.ActionKind]
    assert r.counts[kinds.index("deny")] == 1 and r.counts[kinds.index("delay")] == 1


def test_a_wrongly_shaped_action_is_refused():
    import pytest
    with pytest.raises(ValueError, match=r"Actions\.deny"):
        apply(core.Actions(deny=torch.zeros(1, N, dtype=torch.bool)))
