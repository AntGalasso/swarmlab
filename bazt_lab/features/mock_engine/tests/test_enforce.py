"""Applying a defender's Actions: contract order, persistent states, per-delivery pricing, budget rejection.

Graph used throughout: N = 4 agents, complete; every delivery is live and authentic unless a test says otherwise."""
import math

import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.mock_engine.private import enforce as E
from bazt_lab.features.mock_engine.tests.helpers import KINDS, agents, make_settings, mask

K = {k: i for i, k in enumerate(KINDS)}
UNIT = dict.fromkeys(KINDS, 1.0)
INF = torch.tensor([math.inf], dtype=torch.float64)


def complete(B=1, N=4):
    return (~torch.eye(N, dtype=torch.bool)).expand(B, N, N).clone()


def run(actions=None, *, state=None, live=None, auth=None, remaining=INF, t=1, T=4, costs=UNIT, limit=1, B=1, N=4):
    adj = complete(B, N)
    live = adj.clone() if live is None else live
    state = E.Persistent.zeros(B, N) if state is None else state
    auth = torch.ones(B, N, N, dtype=torch.bool) if auth is None else auth
    prio = lambda rows: (torch.arange(N, dtype=torch.float64) / N).expand(len(rows), N, N).contiguous()   # noqa: E731
    return E.apply_actions(actions or core.Actions(), state, live, auth, adj, remaining, t, T, costs, limit, prio)


def count(out, kind):
    return out.counts[:, K[kind]].tolist()


def test_no_actions_deliver_everything_for_free():
    out = run()
    assert torch.equal(out.arrive, complete()) and not out.restricted.any() and not out.delayed.any()
    assert not out.sanitized.any() and out.cost.tolist() == [0.0] and not out.rejected.any()
    assert out.counts.sum() == 0 and out.counts.shape == (1, len(KINDS)) and out.counts.dtype == torch.int64


def test_deny_costs_one_unit_per_authentic_delivery_and_restricts_it():
    out = run(core.Actions(deny=mask(1, 4, (0, 0, 1), (0, 0, 2), (0, 3, 1))))
    assert out.cost.tolist() == [3.0] and count(out, "deny") == [3]
    assert not out.arrive[0, 0, 1] and not out.arrive[0, 0, 2] and not out.arrive[0, 3, 1] and out.arrive[0, 1, 0]
    assert torch.equal(out.restricted, mask(1, 4, (0, 0, 1), (0, 0, 2), (0, 3, 1)))


def test_denying_what_the_identity_tier_already_drops_is_free():
    auth = torch.ones(1, 4, 4, dtype=torch.bool)
    auth[0, 0, 1] = False                                                # an AiTM rewrite fails authentication
    out = run(core.Actions(deny=mask(1, 4, (0, 0, 1), (0, 0, 2))), auth=auth)
    assert out.cost.tolist() == [1.0] and count(out, "deny") == [2] and not out.arrive[0, 0, 1]


def test_an_action_on_a_delivery_that_cannot_happen_is_ignored_and_free():
    live = complete()
    live[0, 0, 1] = False                                                # an endpoint is already isolated
    out = run(core.Actions(deny=mask(1, 4, (0, 0, 1)), sanitize=mask(1, 4, (0, 0, 1)), delay=mask(1, 4, (0, 0, 1))),
              live=live)
    assert out.cost.tolist() == [0.0] and not out.restricted[0, 0, 1] and out.counts.sum() == 0


def test_delay_arrives_one_round_later_and_costs_a_unit():
    out = run(core.Actions(delay=mask(1, 4, (0, 1, 2))))
    assert out.delayed[0, 1, 2] and not out.arrive[0, 1, 2] and out.restricted[0, 1, 2]
    assert out.cost.tolist() == [1.0] and count(out, "delay") == [1]


def test_sanitize_delivers_a_modified_copy():
    out = run(core.Actions(sanitize=mask(1, 4, (0, 1, 2))))
    assert out.sanitized[0, 1, 2] and out.arrive[0, 1, 2] and out.restricted[0, 1, 2] and not out.sanitized[0, 2, 1]
    assert out.cost.tolist() == [1.0] and count(out, "sanitize") == [1]


def test_a_delivery_is_charged_once_by_the_first_action_in_contract_order():
    both = mask(1, 4, (0, 1, 2))
    out = run(core.Actions(deny=both, delay=both, sanitize=both))
    assert out.cost.tolist() == [1.0] and not out.delayed.any() and not out.sanitized.any() and not out.arrive[0, 1, 2]
    assert count(out, "deny") == [1] and count(out, "delay") == [0] and count(out, "sanitize") == [0]


def test_delay_and_sanitize_combine_on_one_delivery_and_are_both_charged():
    both = mask(1, 4, (0, 1, 2))
    out = run(core.Actions(delay=both, sanitize=both))
    assert out.cost.tolist() == [2.0] and out.delayed[0, 1, 2] and out.sanitized[0, 1, 2]


def test_isolation_costs_out_degree_times_remaining_rounds_and_persists():
    out = run(core.Actions(isolation=agents(1, 4, (0, 1))), t=2, T=4)             # 3 live out-edges x rounds 2..4
    assert out.cost.tolist() == [9.0] and out.state.iso.tolist() == [[False, True, False, False]]
    assert count(out, "isolation") == [1]
    lost = out.arrive[0].clone()
    assert not lost[1].any() and not lost[:, 1].any() and not out.restricted.any()         # losses, not restrictions
    live = complete()
    live[0, 1, :] = live[0, :, 1] = False
    again = run(core.Actions(isolation=agents(1, 4, (0, 1))), state=out.state, live=live, t=3, T=4)
    assert again.cost.tolist() == [0.0] and count(again, "isolation") == [0]


def test_isolation_pays_for_the_deliveries_it_removes_not_for_the_denies_on_them():
    out = run(core.Actions(isolation=agents(1, 4, (0, 1)), deny=mask(1, 4, (0, 1, 0), (0, 1, 2))), t=2, T=4)
    assert out.cost.tolist() == [9.0] and count(out, "deny") == [0]


def test_isolation_removes_nothing_that_was_not_live():
    live = complete()
    live[0, 1, 3] = live[0, 3, 1] = False                               # agent 3 is already isolated
    out = run(core.Actions(isolation=agents(1, 4, (0, 1))), live=live, t=1, T=4)
    assert out.cost.tolist() == [8.0]                                    # out-degree 2 x 4 rounds


def test_edge_quarantine_is_persistent_and_costs_the_remaining_rounds():
    out = run(core.Actions(edge_quarantine=mask(1, 4, (0, 0, 1))), t=3, T=4)
    assert out.cost.tolist() == [2.0] and out.state.quar[0, 0, 1] and not out.state.quar[0, 1, 0]
    assert out.restricted[0, 0, 1] and not out.arrive[0, 0, 1] and out.arrive[0, 1, 0] and count(out, "edge_quarantine") == [1]
    nxt = run(state=out.state, t=4, T=4)
    assert nxt.cost.tolist() == [0.0] and nxt.restricted[0, 0, 1] and not nxt.arrive[0, 0, 1]


def test_rate_limit_keeps_the_limit_of_receivers_by_priority_and_costs_the_removed_deliveries():
    out = run(core.Actions(rate_limit=agents(1, 4, (0, 0))), limit=1, t=1, T=4)
    assert out.cost.tolist() == [8.0]                                    # (3 - 1) removed per round x 4 rounds
    assert out.arrive[0, 0].tolist() == [False, True, False, False] and out.restricted[0, 0].tolist() == [False, False, True, True]
    assert out.state.rate[0, 0] and count(out, "rate_limit") == [1]
    nxt = run(state=out.state, limit=2, t=2, T=4)
    assert nxt.cost.tolist() == [0.0] and nxt.arrive[0, 0].tolist() == [False, True, True, False]


def test_rate_limit_above_out_degree_removes_nothing():
    out = run(core.Actions(rate_limit=agents(1, 4, (0, 0))), limit=3)
    assert out.cost.tolist() == [0.0] and not out.restricted.any()


def test_privilege_degradation_sanitizes_every_outgoing_delivery_until_reinstatement():
    out = run(core.Actions(privilege_degradation=agents(1, 4, (0, 2))), t=2, T=4)
    assert out.cost.tolist() == [9.0] and out.sanitized[0, 2].tolist() == [True, True, False, True]
    assert out.restricted[0, 2, 0] and out.arrive[0, 2, 0] and not out.sanitized[0, 1].any()
    nxt = run(state=out.state, t=3, T=4)
    assert nxt.cost.tolist() == [0.0] and nxt.sanitized[0, 2].sum() == 3 and count(out, "privilege_degradation") == [1]


def test_reinstatement_lifts_every_persistent_action_on_the_agent_and_costs_the_restored_deliveries():
    st = E.Persistent.zeros(1, 4)
    st.iso[0, 1] = st.degr[0, 1] = st.rate[0, 1] = True
    st.quar[0, 1, 2] = st.quar[0, 3, 1] = st.quar[0, 0, 2] = True
    out = run(core.Actions(reinstatement=agents(1, 4, (0, 1))), state=st, t=2, T=4)
    assert out.cost.tolist() == [6.0] and count(out, "reinstatement") == [1]
    assert not out.state.iso.any() and not out.state.degr.any() and not out.state.rate.any()
    assert out.state.quar[0, 0, 2] and not out.state.quar[0, 1, 2] and not out.state.quar[0, 3, 1]


def test_reinstating_an_unrestricted_agent_is_a_free_no_op():
    out = run(core.Actions(reinstatement=agents(1, 4, (0, 1))))
    assert out.cost.tolist() == [0.0] and count(out, "reinstatement") == [0]


def test_reinstatement_is_applied_before_isolation_in_the_same_round():
    st = E.Persistent.zeros(1, 4)
    st.iso[0, 1] = True
    live = complete()
    live[0, 1, :] = live[0, :, 1] = False
    out = run(core.Actions(reinstatement=agents(1, 4, (0, 1)), isolation=agents(1, 4, (0, 1))), state=st, live=live, t=1, T=4)
    assert out.state.iso[0, 1] and count(out, "reinstatement") == [1] and count(out, "isolation") == [1]


def test_costs_follow_the_cost_condition():
    robust = {**UNIT, "sanitize": 2.0, "isolation": 3.0, "delay": 1.0}
    out = run(core.Actions(sanitize=mask(1, 4, (0, 0, 1), (0, 0, 2)), isolation=agents(1, 4, (0, 3))), costs=robust, t=4, T=4)
    assert out.cost.tolist() == [2 * 2 + 3 * 3 * 1]                      # two sanitizes + 3 out-edges x 1 round x 3


def test_the_real_settings_provide_every_cost_the_engine_charges():
    s = make_settings()
    for condition in ("main", "robustness"):
        costs = E.costs_of(s, condition)
        assert set(costs) == set(KINDS) and all(v >= 0 for v in costs.values())
    assert E.costs_of(s, "robustness")["reinstatement"] == 5 and E.costs_of(s, "main")["reinstatement"] == 1
    with pytest.raises(ValueError, match="cost condition"):
        E.costs_of(s, "cheap")
    del s["costs"]["main"]["deny"]
    with pytest.raises(core.Unset) as e:
        E.costs_of(s, "main")
    assert e.value.args[0] == "costs.main.deny"


def test_actions_beyond_the_remaining_budget_are_all_rejected_and_counted_as_a_violation():
    B = 2
    acts = core.Actions(deny=mask(B, 4, (0, 0, 1), (0, 0, 2), (1, 0, 1), (1, 0, 2)), delay=mask(B, 4, (0, 1, 2), (1, 1, 2)))
    out = run(acts, remaining=torch.tensor([2.0, 3.0], dtype=torch.float64), B=B)
    assert out.rejected.tolist() == [True, False] and out.cost.tolist() == [0.0, 3.0]
    assert torch.equal(out.arrive[0], complete()[0]) and not out.restricted[0].any()
    assert out.counts[0].sum() == 0 and out.counts[1].sum() == 3
    assert not out.arrive[1, 0, 1] and out.delayed[1, 1, 2]


def test_a_rejected_episode_keeps_its_earlier_persistent_actions_but_gains_no_new_ones():
    st = E.Persistent.zeros(1, 4)
    st.quar[0, 0, 1] = True
    out = run(core.Actions(isolation=agents(1, 4, (0, 2))), state=st, remaining=torch.tensor([2.0], dtype=torch.float64),
              t=1, T=4)
    assert out.rejected.tolist() == [True] and not out.state.iso.any() and out.state.quar[0, 0, 1]
    assert out.restricted[0, 0, 1]


def test_an_action_costing_exactly_the_remaining_budget_is_accepted():
    out = run(core.Actions(deny=mask(1, 4, (0, 0, 1), (0, 0, 2))), remaining=torch.tensor([2.0], dtype=torch.float64))
    assert out.cost.tolist() == [2.0] and not out.rejected.any()


def test_free_actions_never_violate_even_with_nothing_left():
    auth = torch.ones(1, 4, 4, dtype=torch.bool)
    auth[0, 0, 1] = False
    out = run(core.Actions(deny=mask(1, 4, (0, 0, 1))), auth=auth, remaining=torch.tensor([0.0], dtype=torch.float64))
    assert not out.rejected.any() and out.cost.tolist() == [0.0]


@pytest.mark.parametrize("bad", [dict(deny=torch.ones(1, 4, 4)), dict(deny=torch.ones(1, 4, 3, dtype=torch.bool)),
                                 dict(isolation=torch.ones(1, 4, 4, dtype=torch.bool)),
                                 dict(isolation=torch.ones(2, 4, dtype=torch.bool))])
def test_malformed_actions_are_refused_with_the_field_name(bad):
    with pytest.raises(ValueError, match=next(iter(bad))):
        run(core.Actions(**bad))


def test_isolation_does_not_pay_again_for_deliveries_an_earlier_quarantine_already_removed():
    first = run(core.Actions(edge_quarantine=mask(1, 4, (0, 0, 1), (0, 0, 2))), t=1, T=6)
    assert first.cost.tolist() == [12.0]
    second = run(core.Actions(isolation=agents(1, 4, (0, 0))), state=first.state, t=2, T=6)
    assert second.cost.tolist() == [5.0]                         # only 0 -> 3 is still live: 1 x 5 remaining rounds


def test_isolation_after_a_rate_limit_pays_only_for_the_deliveries_the_limit_let_through():
    first = run(core.Actions(rate_limit=agents(1, 4, (0, 0))), t=1, T=6)
    assert first.cost.tolist() == [12.0]                          # (3 - 1) x 6
    second = run(core.Actions(isolation=agents(1, 4, (0, 0))), state=first.state, t=2, T=6)
    assert second.cost.tolist() == [5.0]                          # 1 surviving delivery x 5 remaining rounds


def test_reinstating_an_isolated_agent_pays_only_for_the_rounds_it_sends_again():
    st = E.Persistent.zeros(1, 4)
    st.iso[0, 1] = True
    live = complete()
    live[0, 1, :] = live[0, :, 1] = False                         # an isolated agent is not live in the round it is reinstated
    out = run(core.Actions(reinstatement=agents(1, 4, (0, 1))), state=st, live=live, t=3, T=8)
    assert out.cost.tolist() == [15.0]                            # out-degree 3 x rounds 4..8


def test_reinstating_an_agent_that_still_sends_pays_for_the_current_round_too():
    st = E.Persistent.zeros(1, 4)
    st.rate[0, 1] = True
    out = run(core.Actions(reinstatement=agents(1, 4, (0, 1))), state=st, t=3, T=8)
    assert out.cost.tolist() == [18.0]                            # 3 x rounds 3..8
