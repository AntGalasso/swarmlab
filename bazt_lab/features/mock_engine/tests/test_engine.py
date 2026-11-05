"""The round loop: consensus, contagion, monitor, auditor, enforcement and the trace. Scenarios are tiny chains whose
noise is forced (u_adopt = 0 always adopts, u_stay = 1 never persists, ...) so every expected value is derived by hand."""
import math

import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.mock_engine.private import goal
from bazt_lab.features.mock_engine.tests.helpers import (
    KINDS,
    Fn,
    agents,
    chain,
    forced,
    make_settings,
    mask,
    seed_at,
    sim,
)

SPREAD = forced(u_adopt=0.0, u_stay=1.0)          # an exposure always infects; the payload is never kept without one
KEEP = forced(u_adopt=0.0, u_stay=0.0)            # an exposure always infects; an infected agent always keeps it
NO_NOISE = forced(mon_z=0.0, aud_z=0.0)
Actions = core.Actions


def every(attempts):
    return Actions(deny=attempts.clone())


def test_without_an_attack_the_estimates_converge_and_their_mean_is_preserved():
    w, tr, est = sim(T=80)
    assert est.shape == (1, 80, 6) and est.dtype == torch.float64
    assert torch.allclose(est[0, 0], w.x[0]) and torch.allclose(est.mean(2), w.mean[:, None].expand(1, 80), atol=1e-9)
    assert tr.correct[0, -1].all() and not tr.correct[0, 0].all()
    assert not tr.infected.any() and torch.equal(tr.delivered, tr.attempts) and not tr.restricted.any()
    assert tr.attack_free is True and tr.goal_reached.tolist() == [False]


def test_one_consensus_step_is_the_metropolis_update():
    w, _, est = sim(N=3, T=3)
    x = w.x[0]
    expected = torch.stack([x[0] + (x[1] - x[0]) / 3, x[1] + (x[0] - x[1]) / 3 + (x[2] - x[1]) / 3, x[2] + (x[1] - x[2]) / 3])
    assert torch.allclose(est[0, 1], expected, atol=1e-12)


def test_correctness_uses_the_configured_tolerance():
    w, tr, est = sim(T=10, settings=make_settings(runtime__eps_correct=0.5))
    assert torch.equal(tr.correct, (est - w.mean[:, None, None]).abs() <= 0.5)


def test_the_infection_front_moves_one_hop_per_round():
    w, tr, est = sim("infection", T=6, edit=chain(seed_at(0), SPREAD))
    expected = torch.arange(6)[None, :] <= torch.arange(6)[:, None]           # [t, i]: agent i infected at round t
    assert torch.equal(tr.infected[0], expected)
    assert tr.attack_free is False and not tr.attackers.any()


def test_an_exposed_agent_is_infected_with_probability_one_minus_one_beta_to_the_k():
    seeds = tuple(range(1, 2001))
    _, tr, _ = sim("infection", T=2, topology="star_hub", N=4, seeds=seeds, edit=forced(u_stay=1.0))
    assert tr.infected[:, 1, 0].all()                                                  # the hub is the entry
    assert tr.infected[:, 1, 1:].double().mean() == pytest.approx(0.35, abs=0.025)    # one exposure each: beta
    _, tr, _ = sim("infection", T=2, topology="star_hub", N=4, seeds=seeds, edit=chain(forced(u_stay=1.0), seed_at(1, 2)))
    assert tr.infected[:, 1, 0].double().mean() == pytest.approx(1 - 0.65 ** 2, abs=0.04)    # two exposures at the hub


def test_an_infected_agent_keeps_the_payload_with_probability_p_stay():
    seeds = tuple(range(1, 3001))
    cut = Fn(lambda o: every(o.attempts) if o.t == 2 else Actions())               # round 3 brings no new exposure
    _, tr, _ = sim("infection", cut, T=3, N=2, seeds=seeds, edit=chain(seed_at(0), forced(u_adopt=0.0)))
    assert tr.infected[:, 1, 1].all()
    assert tr.infected[:, 2, 1].double().mean() == pytest.approx(0.8, abs=0.03)


def test_a_denied_delivery_leaves_the_senders_own_memory_unchanged():
    w, tr, _ = sim("infection", Fn(lambda o: every(o.attempts) if o.t == 2 else Actions()), T=5, edit=chain(seed_at(0), KEEP))
    assert tr.infected[0, 2, 1]                       # agent 1 kept its payload although every delivery of round 2 was denied
    assert not tr.infected[0, 2, 2]                   # ...and it could not infect agent 2 at round 3
    assert tr.infected[0, 3, 2]                       # round 3's deliveries went through
    assert tr.restricted[0, 1].sum() == 10 and not tr.delivered[0, 1].any()


def test_sanitize_strips_the_payload_but_the_reported_value_keeps_its_bias():
    san = Fn(lambda o: Actions(sanitize=mask(1, 6, (0, 0, 1))) if o.t == 1 else Actions())
    w, tr, est = sim("infection", san, T=3, edit=chain(seed_at(0), SPREAD))
    _, tr_open, est_open = sim("infection", Fn(lambda o: Actions()), T=3, edit=chain(seed_at(0), SPREAD))
    assert not tr.infected[0, 1, 1] and not tr.payload_delivered[0, 0, 0, 1]
    assert tr.restricted[0, 0, 0, 1] and not tr.delivered[0, 0, 0, 1] and tr.delivered[0, 0, 1, 0]
    assert tr_open.infected[0, 1, 1]
    _, _, clean = sim("none", T=3)
    assert est[0, 1, 1].item() == pytest.approx(est_open[0, 1, 1].item())          # the biased number was delivered
    assert est[0, 1, 1].item() != pytest.approx(clean[0, 1, 1].item())             # ...so it is not the clean estimate
    assert w.shift[0] > 0


def test_a_sanitize_that_fails_to_strip_leaves_the_payload():
    san = Fn(lambda o: Actions(sanitize=mask(1, 6, (0, 0, 1))) if o.t == 1 else Actions())
    s = make_settings(runtime__sanitize_success=0.0)
    w, tr, est = sim("infection", san, T=3, settings=s, edit=chain(seed_at(0), SPREAD))
    _, _, clean = sim("none", T=3, settings=s)
    assert tr.infected[0, 1, 1] and tr.payload_delivered[0, 0, 0, 1] and tr.restricted[0, 0, 0, 1]
    assert est[0, 1, 1] - clean[0, 1, 1] == pytest.approx(8.0 / 3)            # w_01 = 1 / (1 + 2), shift = mock.model.bias


def test_a_delayed_delivery_arrives_one_round_later():
    delay = Fn(lambda o: Actions(delay=mask(1, 6, (0, 0, 1))) if o.t == 1 else Actions())
    w, tr, _ = sim("infection", delay, T=6, edit=chain(seed_at(0), SPREAD))
    assert not tr.infected[0, 1, 1] and tr.infected[0, 2, 1]
    assert tr.restricted[0, 0, 0, 1] and not tr.delivered[0, 0, 0, 1] and tr.payload_delivered[0, 0, 0, 1]


def test_a_delay_in_the_last_round_never_arrives():
    delay = Fn(lambda o: Actions(delay=mask(1, 6, (0, 0, 1))) if o.t == 4 else Actions())
    w, tr, _ = sim("infection", delay, T=4, edit=chain(seed_at(0), SPREAD))
    assert tr.restricted[0, 3, 0, 1] and not tr.payload_delivered[0, 3, 0, 1] and not tr.delivered[0, 3, 0, 1]


def test_isolation_cuts_an_agent_off_for_good_and_costs_its_remaining_deliveries():
    iso = Fn(lambda o: Actions(isolation=agents(1, 6, (0, 0))) if o.t == 2 else Actions())
    w, tr, _ = sim("infection", iso, T=6, edit=chain(seed_at(0), SPREAD))
    assert tr.active[0, :, 0].tolist() == [True, False, False, False, False, False]
    assert tr.infected[0, :, 0].tolist() == [True, True, False, False, False, False]
    assert tr.infected[0, 1, 1] and not tr.infected[0, 2, 1]               # round 2's delivery from the isolated agent was cut
    assert tr.infected[0, 2, 2]                                             # agent 1 was infected at round 2 and passed it on
    assert tr.spend_intervention[0].tolist() == [0.0, 5.0, 0, 0, 0, 0]       # out-degree 1 x rounds 2..6
    assert tr.attempts[0, 3, 0, 1] and not tr.delivered[0, 3, 0, 1] and not tr.restricted[0, 3, 0, 1]
    assert tr.action_counts[0, KINDS.index("isolation")] == 1
    seen = iso.seen[2]
    assert not seen.active[0, 0] and not seen.attempts[0, 0].any() and not seen.attempts[0, :, 0].any()


def test_a_budget_cap_is_a_fraction_of_all_edge_deliveries_and_overspending_is_a_violation():
    def plan(o):
        if o.t == 1:
            return Actions(deny=mask(1, 6, (0, 0, 1), (0, 1, 2), (0, 2, 3), (0, 3, 4)))      # 4 > 3 units left
        if o.t == 2:
            return Actions(deny=mask(1, 6, (0, 0, 1), (0, 1, 2), (0, 2, 3)))                 # exactly the 3 left
        return Actions(deny=mask(1, 6, (0, 0, 1))) if o.t == 3 else Actions()                # nothing left
    w, tr, _ = sim("infection", Fn(plan), T=6, edit=chain(seed_at(0), SPREAD), intervention=0.05)   # 0.05 x 10 x 6 = 3
    assert tr.spend_intervention[0].tolist() == [0.0, 3.0, 0, 0, 0, 0] and tr.violations.tolist() == [2]
    assert tr.exhausted_step.tolist() == [2] and tr.delivered[0, 0, 0, 1] and not tr.delivered[0, 1, 0, 1]
    assert tr.action_counts[0, KINDS.index("deny")] == 3


def test_the_audit_budget_pays_per_sent_text_for_a_keyed_random_subset_and_answers_nan_when_unaffordable():
    got = []

    def audit_all(o):
        got.append(o.audit(torch.ones(1, 6, dtype=torch.bool)))
        return Actions()
    w, tr, _ = sim("infection", Fn(audit_all), T=6, edit=chain(seed_at(0), NO_NOISE, forced(u_decoy=1.0)), audit=0.05)
    paid = got[0][0]                                                     # 0.05 x 10 deliveries x 6 rounds = 3 audit units
    granted = ~paid.isnan()
    assert granted.sum() == 3                                            # which three is a keyed draw, never the lowest indices
    assert paid[0].isnan() or paid[0] == pytest.approx(w.d_pay[0].item())
    assert (paid[granted & (torch.arange(6) != 0)] == 0.0).all()
    assert tr.spend_audit[0].tolist() == [3.0, 0, 0, 0, 0, 0] and tr.exhausted_step.tolist() == [1] and got[1].isnan().all()


def test_audit_scores_follow_the_class_of_the_text_and_never_the_arm():
    got = []

    def audit_all(o):
        got.append(o.audit(torch.ones(1, 6, dtype=torch.bool))[0].clone())
        return Actions()
    edit = chain(seed_at(0), NO_NOISE, forced(u_decoy=0.0, u_adopt=1.0))
    w, tr, _ = sim("infection", Fn(audit_all), T=3, edit=edit)
    for t in range(3):
        want = torch.where(tr.infected[0, t], w.d_pay[0], torch.where(w.decoy[0] & w.burst[0, t], w.d_dec[0],
                                                                      torch.zeros((), dtype=torch.float64)))
        assert torch.allclose(got[t], want)
    assert w.d_dec[0] > 0 and (w.decoy & w.burst[:, 0]).any()


def test_audit_requests_for_silent_senders_cost_nothing():
    got = []

    def audit_all(o):
        got.append(o.audit(torch.ones(1, 6, dtype=torch.bool))[0].clone())
        return Actions(isolation=agents(1, 6, (0, 0))) if o.t == 1 else Actions()
    _, tr, _ = sim("none", Fn(audit_all), T=3)
    assert got[1][0].isnan() and not got[1][1].isnan() and tr.spend_audit[0].tolist() == [6.0, 5.0, 5.0]


def test_oracle_is_given_only_to_arms_that_need_truth():
    seen = {}

    def peek(o):
        seen["oracle"] = o.oracle
        seen["truth"] = o.oracle() if o.oracle else None
        return Actions()
    edit = chain(seed_at(0), SPREAD)
    sim("infection", Fn(peek), T=1, edit=edit)
    assert seen["oracle"] is None
    _, tr, _ = sim("infection", Fn(peek), T=1, edit=edit, needs_truth=True)
    assert seen["truth"].tolist() == [[True, False, False, False, False, False]] and seen["truth"].dtype == torch.bool


def test_a_defender_observation_holds_no_ground_truth():
    d = Fn(lambda o: Actions())
    sim("infection", d, T=2, edit=chain(seed_at(0), SPREAD))
    o = d.seen[0]
    assert set(vars(o)) == {"t", "T", "active", "attempts", "monitor", "remaining_audit", "remaining_intervention",
                                   "audit", "authentic", "oracle", "texts", "round_in_conversation"}
    assert o.round_in_conversation is None and o.texts is None                                        # the mock has no texts
    assert o.t == 1 and o.T == 2 and o.active.all() and o.attempts.sum() == 10 and o.authentic.all()
    assert math.isinf(o.remaining_audit[0]) and math.isinf(o.remaining_intervention[0])


def test_the_monitor_scores_the_rolling_text_evidence():
    d = Fn(lambda o: Actions())
    edit = chain(seed_at(0), NO_NOISE, forced(u_adopt=1.0, u_decoy=0.0))
    w, tr, _ = sim("infection", d, T=3, edit=edit)
    p = [o.monitor[0] for o in d.seen]
    sig = lambda z: 1 / (1 + math.exp(-z))                      # noqa: E731
    assert p[0][0] == pytest.approx(sig(2.5 * 1.0 - 2.0)) and p[1][0] == pytest.approx(sig(2.5 * 2.0 - 2.0))
    clean = [i for i in range(1, 6) if not w.decoy[0, i]]
    assert p[0][clean[0]] == pytest.approx(sig(-2.0)) and p[2][clean[0]] == pytest.approx(sig(-2.0))
    dec = next(i for i in w.decoy[0].nonzero().flatten().tolist() if i != 0)           # the edit made agent 0 the seed
    assert p[0][dec] == pytest.approx(sig(2.5 * 1.4 - 2.0))        # decoy header: mu_dec = 2.8, window 2 with a zero pad


def test_aitm_rewrites_the_victims_incoming_deliveries_and_they_fail_authentication():
    d = Fn(lambda o: Actions())
    edit = chain(forced(u_adopt=0.0, u_stay=1.0), lambda w: w.victim.fill_(1))
    w, tr, _ = sim("aitm", d, T=4, edit=edit)
    auth = d.seen[0].authentic[0]
    assert not auth[0, 1] and not auth[2, 1] and auth[1, 0] and auth[1, 2] and auth[3, 4]
    assert tr.payload_delivered[0, 0, 0, 1] and tr.payload_delivered[0, 0, 2, 1] and not tr.payload_delivered[0, 0, 1, 0]
    assert tr.infected[0, 1, 1] and not tr.infected[0, 0].any() and not tr.attackers.any()


def test_the_identity_tier_stops_aitm_for_free():
    a = Fn(lambda o: Actions(deny=o.attempts & ~o.authentic))
    edit = chain(forced(u_adopt=0.0, u_stay=1.0), lambda w: w.victim.fill_(1))
    w, tr, _ = sim("aitm", a, T=4, edit=edit)
    assert not tr.infected.any() and not tr.payload_delivered.any() and tr.spend_intervention.sum() == 0
    assert tr.restricted[0, 0, 0, 1] and tr.restricted[0, 0, 2, 1] and tr.action_counts[0, KINDS.index("deny")] == 8


def test_aitm_with_a_weak_rewrite_hits_a_fraction_of_the_deliveries():
    seeds = tuple(range(1, 1501))
    edit = chain(forced(u_adopt=1.0), lambda w: w.victim.fill_(2))
    _, tr, _ = sim("aitm", None, T=1, seeds=seeds, edit=edit, settings=make_settings(runtime__aitm_rewrite_strength=0.4))
    assert tr.payload_delivered[:, 0, :, 2].double().sum(0)[[1, 3]].div(len(seeds)).tolist() == pytest.approx([0.4, 0.4], abs=0.04)


def test_insiders_report_a_shifted_value_every_round_and_bias_the_consensus():
    w, tr, est = sim("insider_pi", T=12, edit=chain(seed_at(0, 3), forced(u_adopt=1.0)))
    assert tr.attackers[0].nonzero().flatten().tolist() == [0, 3]
    assert tr.infected[0, :, [0, 3]].all() and not tr.infected[0, :, [1, 2, 4, 5]].any()
    assert est[0, -1].mean() > w.mean[0] + 1.0 and not tr.correct[0, -1].all()


def test_the_attack_starts_at_its_scheduled_round():
    _, tr, _ = sim("infection", T=6, edit=chain(seed_at(0), forced(u_adopt=1.0)),
                   attack_params={"start_frac": 0.5, "rate": 1.0})
    assert not tr.infected[0, :3].any() and tr.infected[0, 3:, 0].all()


def test_attack_free_runs_are_flagged_and_have_no_goal():
    _, tr, _ = sim("none")
    assert tr.attack_free is True
    _, tr, _ = sim("infection", edit=seed_at(0))
    assert tr.attack_free is False


def test_the_trace_is_complete_and_consistent_under_arbitrary_actions():
    g = torch.Generator().manual_seed(3)

    def noise(o):
        B, N = o.active.shape
        edge = lambda: torch.rand(B, N, N, generator=g) < 0.1                       # noqa: E731
        node = lambda: torch.rand(B, N, generator=g) < 0.05                         # noqa: E731
        return Actions(deny=edge(), sanitize=edge(), delay=edge(), edge_quarantine=edge() & (torch.rand(B, N, N, generator=g) < 0.2),
                       rate_limit=node(), privilege_degradation=node(), isolation=node() & (torch.rand(B, N, generator=g) < 0.3),
                       reinstatement=node())
    w, tr, est = sim("infection", Fn(noise), T=8, topology="barabasi_albert", N=9, seeds=(1, 2, 3, 4), intervention=0.5)
    B, T, N = 4, 8, 9
    assert tr.infected.shape == tr.active.shape == tr.correct.shape == (B, T, N)
    for name in ("attempts", "delivered", "restricted", "payload_delivered"):
        t = getattr(tr, name)
        assert t.shape == (B, T, N, N) and t.dtype == torch.bool
    assert not (tr.delivered & tr.restricted).any()
    assert not (tr.delivered & ~tr.attempts).any() and not (tr.restricted & ~tr.attempts).any()
    assert not (tr.payload_delivered & ~(tr.delivered | tr.restricted)).any()
    assert torch.equal(tr.attempts, w.adj[:, None].expand(B, T, N, N))
    for name in ("spend_audit", "spend_intervention"):
        assert getattr(tr, name).shape == (B, T) and getattr(tr, name).dtype == torch.float64 and (getattr(tr, name) >= 0).all()
    assert tr.action_counts.shape == (B, len(KINDS)) and tr.action_counts.dtype == torch.int64
    assert tr.violations.dtype == torch.int64 and tr.exhausted_step.dtype == torch.int64 and tr.sink.dtype == torch.int64
    assert tr.spend_intervention.sum(1).le(0.5 * w.adj.sum((1, 2)) * T).all()                  # never above the cap
    assert tr.unit_ids[0]["seed"] == 1 and len(tr.unit_ids) == B and tr.goal_reached.shape == (B,)


def test_statistics_are_handed_to_the_collector_with_the_live_deliveries():
    calls = []
    stat = lambda o: Actions(statistics={"s": torch.full(o.attempts.shape, 0.5, dtype=torch.float64)})    # noqa: E731
    sim("infection", Fn(stat), T=3, edit=seed_at(0), collect=lambda s, live, active: calls.append((s, live, active)))
    assert len(calls) == 3 and set(calls[0][0]) == {"s"} and calls[0][1].sum() == 10 and calls[0][2].all()


def test_a_batch_gives_each_episode_the_result_it_has_alone():
    def policy(o):
        flagged = o.monitor > 0.5
        return Actions(deny=o.attempts & flagged[:, :, None])
    kw = dict(topology="small_world", N=8, T=7)
    both = sim("infection", Fn(policy), seeds=(1, 2, 3), **kw)
    for k, sd in enumerate((1, 2, 3)):
        one = sim("infection", Fn(policy), seeds=(sd,), **kw)
        for a, b in ((both[2][k], one[2][0]), (both[1].infected[k], one[1].infected[0]),
                     (both[1].delivered[k], one[1].delivered[0]), (both[1].spend_intervention[k], one[1].spend_intervention[0])):
            assert torch.equal(a, b)


def test_malformed_actions_name_the_field_and_a_non_actions_return_is_refused():
    with pytest.raises(ValueError, match="deny"):
        sim("none", Fn(lambda o: Actions(deny=torch.ones(1, 6, 6))), T=1)
    with pytest.raises(TypeError, match="Actions"):
        sim("none", Fn(lambda o: None), T=1)


@pytest.mark.parametrize("family,expect", [("insider_pi", True), ("insider_ta", False), ("insider_ma", False),
                                           ("infection", False), ("aitm", False), ("none", False)])
def test_targeted_goal_per_family(family, expect):
    """Sink 2; the estimate of agent 2 is pushed 5 above the mean (> eps 2) but it never carries the payload."""
    B, T, N = 1, 3, 4
    infected = torch.zeros(B, T, N, dtype=torch.bool)
    est = torch.tensor([[50.0, 50.0, 55.0, 50.0]], dtype=torch.float64)
    got = goal.reached(family, infected, est, sink=torch.tensor([2]), victim=torch.tensor([-1]),
                       attackers=torch.zeros(B, N, dtype=torch.bool), mean=torch.tensor([50.0], dtype=torch.float64),
                       shift=torch.tensor([8.0], dtype=torch.float64), eps=2.0)
    assert got.tolist() == [expect] and got.dtype == torch.bool


def test_targeted_goal_when_the_sink_acts_on_or_keeps_the_payload():
    B, T, N = 2, 4, 3
    infected = torch.zeros(B, T, N, dtype=torch.bool)
    infected[0, 1, 1] = True                                           # sink 1 infected once, mid-episode
    infected[1, 3, 1] = True                                           # sink 1 infected at the last round only
    args = dict(est=torch.full((B, N), 50.0, dtype=torch.float64), sink=torch.tensor([1, 1]), victim=torch.tensor([-1, -1]),
                attackers=torch.zeros(B, N, dtype=torch.bool), mean=torch.full((B,), 50.0, dtype=torch.float64),
                shift=torch.full((B,), 8.0, dtype=torch.float64), eps=2.0)
    assert goal.reached("insider_ta", infected, **args).tolist() == [True, True]
    assert goal.reached("infection", infected, **args).tolist() == [True, True]
    assert goal.reached("insider_ma", infected, **args).tolist() == [False, True]
    args["attackers"][:, 1] = True                                      # an attacker sink is not a victim
    assert goal.reached("insider_ta", infected, **args).tolist() == [False, False]


def test_targeted_goal_of_aitm_is_the_victims_final_answer_pushed_the_attackers_way():
    est = torch.tensor([[50.0, 57.0, 50.0], [50.0, 43.0, 50.0], [50.0, 51.0, 50.0]], dtype=torch.float64)
    got = goal.reached("aitm", torch.zeros(3, 2, 3, dtype=torch.bool), est, sink=torch.tensor([0, 0, 0]),
                       victim=torch.tensor([1, 1, 1]), attackers=torch.zeros(3, 3, dtype=torch.bool),
                       mean=torch.full((3,), 50.0, dtype=torch.float64), shift=torch.full((3,), 8.0, dtype=torch.float64), eps=2.0)
    assert got.tolist() == [True, False, False]                          # wrong in the shift's direction only


def test_an_observation_lists_only_the_deliveries_a_defender_can_still_act_on():
    quarantine = Fn(lambda o: Actions(edge_quarantine=mask(1, 6, (0, 0, 1))) if o.t == 1 else Actions())
    w, tr, _ = sim("infection", quarantine, T=4, edit=chain(seed_at(0), SPREAD))
    assert quarantine.seen[0].attempts[0, 0, 1]                                  # round 1: nothing cut yet
    for later in quarantine.seen[1:]:
        assert not later.attempts[0, 0, 1] and later.attempts[0, 1, 0] and later.attempts[0, 1, 2]
    assert tr.attempts[0, 2, 0, 1] and tr.restricted[0, 2, 0, 1] and not tr.delivered[0, 2, 0, 1]   # the trace keeps it


def test_the_trace_hands_the_metrics_the_tampered_deliveries_and_the_flag_matrix():
    seeds = tuple(range(1, 41))
    edit = chain(forced(u_adopt=0.0, u_stay=1.0), lambda w: w.victim.fill_(1))
    stop = Fn(lambda o: Actions(deny=o.attempts & ~o.authentic))
    w, tr, _ = sim("aitm", stop, T=3, edit=edit, seeds=seeds, settings=make_settings(runtime__aitm_rewrite_strength=0.5))
    tampered = tr.meta["tampered"]
    assert tampered.dtype == torch.bool and tampered.shape == tr.attempts.shape
    assert not (tampered & ~tr.attempts).any() and not tampered[:, :, :, [0, 2, 3, 4, 5]].any()     # only into the victim
    assert not tampered[:, :, 1, :].any() and 0.3 < tampered[:, :, [0, 2], 1].double().mean() < 0.7
    assert (tampered <= tr.restricted).all()                                       # the identity tier dropped every one
    assert tr.meta["flagged"].shape == tr.infected.shape
    _, clean, _ = sim("none", T=3)
    assert not clean.meta["tampered"].any()


def test_the_audit_subset_of_an_episode_does_not_depend_on_how_often_the_arm_asks_for_other_episodes():
    """An arm that calls obs.audit a different number of times (here: a second call only when the batch holds a second
    episode) must give episode 0 the same subset, since the draw is keyed by that episode's own calls."""
    def grants(extra_episode):
        seen = []

        def ask(o):
            B, N = o.active.shape
            want = torch.ones(B, N, dtype=torch.bool)
            if extra_episode and o.t == 1:
                o.audit(torch.zeros(B, N, dtype=torch.bool))     # an empty request: not a call for any episode
            seen.append(o.audit(want)[0])
            return Actions()
        sim("none", Fn(ask), T=2, topology="chain", N=8, seeds=(1, 2) if extra_episode else (1,), audit=0.05)
        return ~seen[0].isnan()
    assert torch.equal(grants(False), grants(True))
