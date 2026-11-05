"""Arm C: contextual score-based zero trust (spec 2.3), the additive algorithm of Bradatsch et al. (Eqs. 1 to 3) with
the weights of the reference implementation, and the capped behaviour of spec 2.4.

Hand computation used throughout (full graph of 3 agents, judge-score bins of testkit.BINS, presence window 3):
  constants   authentication 8 + channel authentication 5 + confidentiality 5 + integrity 5 = 23
  presence 3, service usage 3, access rate 5, input behaviour 3, trust history 5     (additive_trust.go 21-31)
  risk        judge bin weight (0, 4, 12 for the scores -6, -3, 0) + network threat weight (0 when nobody is flagged)
The decision statistic is RL - TS.
"""
import io

import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.conventional.private.arm_c import ArmC
from bazt_lab.features.conventional.tests.testkit import (
    BINS,
    COSTS,
    INF,
    NAN,
    Fake,
    adj_of,
    full_adj,
    make_ctx,
    make_settings,
)

STAT = "rl_minus_ts"
SCORES = [-6.0, -3.0, 0.0]          # risk weight 0, 4, 12 under BINS


def _run(B=1, N=3, *, adj=None, thresholds=None, window=3, bins=BINS, **ctx_kw):
    adj = full_adj(B, N) if adj is None else adj
    settings = make_settings(rate_limit=None, bins=bins, window=window)
    ctx = make_ctx(adj, "C", thresholds=thresholds, settings=settings, **ctx_kw)
    arm = ArmC()
    arm.reset(ctx)
    return ctx, arm, Fake(ctx)


def by_sender(stat, b=0):
    """The statistic of each sender, which must be the same on all its evaluated deliveries."""
    out = []
    for i in range(stat.shape[1]):
        values = stat[b, i][~stat[b, i].isnan()].unique()
        assert len(values) == 1, (i, values)
        out.append(values.item())
    return out


# --------------------------------------------------------------------------------------------- arithmetic
def test_trust_score_and_risk_level_follow_the_reference_weights_round_by_round():
    ctx, arm, fake = _run()
    expected = [
        [-26.0, -22.0, -14.0],       # round 1, empty histories: TS = 23 + presence 3 = 26; RL = 0, 4, 12
        [-37.0, -33.0, -25.0],       # round 2: + service 3 + rate 5 + input 3 = 37; history [26, 26] misses 37
        [-42.0, -38.0, -30.0],       # round 3: history [26, 37] contains the partial score 37: + 5 = 42
        [-42.0, -38.0, -30.0],       # round 4: history [26, 42] contains 37: 42 again
    ]
    for t, want in enumerate(expected, start=1):
        stat = arm.decide(fake.obs(t, scores=SCORES)).statistics[STAT]
        assert stat.dtype == torch.float64 and stat.shape == (1, 3, 3)
        assert by_sender(stat) == want
        assert torch.isnan(stat.diagonal(dim1=-2, dim2=-1)).all()        # nobody sends to itself


def test_access_rate_outside_the_senders_range_loses_its_weight():
    ctx, arm, fake = _run()
    arm.decide(fake.obs(1, scores=SCORES))
    attempts = adj_of(1, [(0, 1), (1, 0), (1, 2), (2, 0), (2, 1)], 3)    # sender 0 sends one message, not two
    stat = arm.decide(fake.obs(2, attempts, scores=SCORES)).statistics[STAT]
    assert stat[0, 0, 1] == -32.0        # 23 + 3 + 3 + 0 + 3 = 32; RL 0
    assert stat[0, 1, 0] == -33.0        # the others keep their rate: 37 against RL 4


def test_input_behaviour_outside_the_senders_range_loses_its_weight():
    ctx, arm, fake = _run()
    arm.decide(fake.obs(1, scores=SCORES))
    stat = arm.decide(fake.obs(2, scores=[-3.0, -3.0, 0.0])).statistics[STAT]
    assert stat[0, 0, 1] == -30.0        # score -3 is outside [-6, -6]: TS = 34; RL = 4
    assert stat[0, 1, 0] == -33.0        # sender 1 repeats -3: TS = 37; RL = 4


def test_a_recipient_the_sender_never_wrote_to_loses_the_service_usage_weight():
    ctx, arm, fake = _run()
    arm.decide(fake.obs(1, adj_of(1, [(0, 1), (1, 0), (2, 0)], 3), scores=SCORES))
    stat = arm.decide(fake.obs(2, adj_of(1, [(0, 2), (1, 0), (2, 0)], 3), scores=SCORES)).statistics[STAT]
    assert stat[0, 0, 2] == -34.0        # 23 + 3 + 0 + 5 + 3 = 34; RL 0
    assert stat[0, 1, 0] == -33.0        # sender 1 writes to the same recipient again: 37 against RL 4


@pytest.mark.parametrize("window, expected", [(3, -33.0), (1, -30.0)])
def test_enterprise_presence_counts_the_rounds_since_the_sender_was_last_active(window, expected):
    ctx, arm, fake = _run(window=window)
    arm.decide(fake.obs(1, scores=SCORES))
    silent = adj_of(1, [(0, 1), (0, 2), (2, 0), (2, 1)], 3)             # sender 1 is away in rounds 2 and 3
    away = torch.tensor([[True, False, True]])
    for t in (2, 3):
        arm.decide(fake.obs(t, silent, scores=SCORES, active=away))
    stat = arm.decide(fake.obs(4, scores=SCORES)).statistics[STAT]       # back after 2 absent rounds
    # sender 1: 23 + presence (3 if 2 <= window) + service 3 + rate 5 + input 3; history [26, 26] misses it; RL = 4
    assert stat[0, 1, 0] == expected


def test_a_score_outside_every_bin_adds_no_risk():
    ctx, arm, fake = _run(bins=[[-1.0, 4.0]])
    stat = arm.decide(fake.obs(1, scores=[-2.0, -1.0, 0.0])).statistics[STAT]
    assert by_sender(stat) == [4.0 - 26.0, 4.0 - 26.0, 0.0 - 26.0]       # -1 is inside the bin (upper bound inclusive)


def test_bin_bounds_are_upper_inclusive():
    ctx, arm, fake = _run()
    stat = arm.decide(fake.obs(1, scores=[-5.0, -1.0, -0.999])).statistics[STAT]
    assert by_sender(stat) == [0.0 - 26.0, 4.0 - 26.0, 12.0 - 26.0]


# ---------------------------------------------------------------------------------------- the decision
@pytest.mark.parametrize("threshold, denied", [(-14.0, False), (-14.5, True)])
def test_denies_when_rl_minus_ts_exceeds_the_threshold_and_not_at_equality(threshold, denied):
    ctx, arm, fake = _run(thresholds={STAT: threshold})
    actions = arm.decide(fake.obs(1, scores=SCORES))                      # sender 2: RL 12, TS 26, statistic -14
    assert bool(actions.deny[0, 2].any()) is denied
    assert not actions.deny[0, :2].any()


def test_missing_threshold_reports_the_statistic_and_never_acts():
    ctx, arm, fake = _run(thresholds={})
    for t in (1, 2):
        actions = arm.decide(fake.obs(t, scores=[100.0, 100.0, 100.0]))
        assert not actions.deny.any() and actions.events == []
    assert len(fake.audit_calls) == 2 and set(actions.statistics) == {STAT}


def test_flagged_agents_raise_the_network_threat_level_for_everyone():
    ctx, arm, fake = _run(thresholds={STAT: -20.0})
    first = arm.decide(fake.obs(1, scores=SCORES))
    assert first.deny[0, 2, [0, 1]].all() and not first.deny[0, :2].any()           # only sender 2 is flagged
    stat = arm.decide(fake.obs(2, scores=SCORES)).statistics[STAT]
    # level 1 weighs 2 (additive_trust.go 130): RL = bin + 2; TS = 37 for all three
    assert by_sender(stat) == [2.0 - 37.0, 6.0 - 37.0, 14.0 - 37.0]


def test_two_flagged_agents_weigh_three():
    ctx, arm, fake = _run(thresholds={STAT: -23.0})
    arm.decide(fake.obs(1, scores=SCORES))                                # senders 1 and 2 exceed -23
    stat = arm.decide(fake.obs(2, scores=SCORES)).statistics[STAT]
    assert by_sender(stat) == [3.0 - 37.0, 7.0 - 37.0, 15.0 - 37.0]


@pytest.mark.parametrize("N", [10, 12])
def test_network_threat_weight_saturates_at_the_top_of_the_reference_table(N):
    ctx, arm, fake = _run(N=N, thresholds={STAT: -1000.0})               # everyone is flagged in round 1
    arm.decide(fake.obs(1, scores=[-6.0] * N))
    stat = arm.decide(fake.obs(2, scores=[-6.0] * N)).statistics[STAT]
    assert stat[0, 0, 1] == 25.0 - 37.0                                   # level 10 weighs 25 (additive_trust.go 139)


def test_a_denial_by_a_rule_of_a_does_not_flag_the_sender():
    ctx, arm, fake = _run(thresholds={STAT: 1000.0})
    tampered = torch.ones(1, 3, 3, dtype=torch.bool)
    tampered[0, 0] = False                                                # every delivery of sender 0 altered in transit
    first = arm.decide(fake.obs(1, scores=SCORES, authentic=tampered))
    assert first.deny[0, 0].sum() == 2                                    # A's rule drops them
    stat = arm.decide(fake.obs(2, scores=SCORES)).statistics[STAT]
    assert stat[0, 1, 0] == 4.0 - 37.0                                    # no threat weight: nobody was flagged


def test_a_denied_delivery_still_enters_the_history():
    ctx, arm, fake = _run(thresholds={STAT: -20.0})
    arm.decide(fake.obs(1, scores=SCORES))                                # sender 2 denied, flagged
    stat = arm.decide(fake.obs(2, scores=SCORES)).statistics[STAT]
    # trust history never penalises an earlier flag: sender 2 keeps service 3, rate 5, input 3 (TS 37, as the others)
    assert stat[0, 2, 0] == (12.0 + 2.0) - 37.0


# ---------------------------------------------------------------------------- A first, audits, budgets
def test_applies_a_first_and_audits_only_senders_with_a_deliverable_text():
    adj = adj_of(1, [(0, 1), (1, 2)], 3)
    ctx, arm, fake = _run(adj=adj, thresholds={STAT: 1000.0})
    attempts = adj_of(1, [(0, 1), (1, 2), (2, 0)], 3)                      # (2, 0) is off the allowlist
    authentic = torch.ones(1, 3, 3, dtype=torch.bool)
    authentic[0, 1, 2] = False
    actions = arm.decide(fake.obs(1, attempts, scores=SCORES, authentic=authentic))
    assert torch.equal(actions.deny, adj_of(1, [(1, 2), (2, 0)], 3))
    assert [c.tolist() for c in fake.audit_calls] == [[[True, False, False]]]
    stat = actions.statistics[STAT]
    assert stat[0, 0, 1] == -26.0 and stat.isnan().sum() == 8


def test_an_unchecked_sender_is_neither_denied_nor_remembered():
    ctx, arm, fake = _run(thresholds={STAT: -1000.0}, audit_cap=2.0)      # the budget pays for two of three senders
    actions = arm.decide(fake.obs(1, scores=SCORES))
    assert actions.deny[0, 0, [1, 2]].all() and actions.deny[0, 1, [0, 2]].all() and not actions.deny[0, 2].any()
    assert torch.isnan(actions.statistics[STAT][0, 2]).all()
    state = arm.state_dict()
    assert not state["recipients"][0, 2].any() and state["input_lo"][0, 2] == INF


def test_audit_budget_exhaustion_emits_one_event_then_only_a_rules():
    ctx, arm, fake = _run(thresholds={STAT: -1000.0}, audit_cap=3.0, T=3)
    first = arm.decide(fake.obs(1, scores=SCORES))
    assert [(e["event"], e["arm"], e["b"], e["budget"], e["t"]) for e in first.events] == [
        ("budget_exhausted", "C", 0, "audit", 1)]
    calls = len(fake.audit_calls)
    authentic = torch.ones(1, 3, 3, dtype=torch.bool)
    authentic[0, 0, 1] = False
    second = arm.decide(fake.obs(2, scores=SCORES, authentic=authentic))
    assert len(fake.audit_calls) == calls and second.events == []
    assert torch.equal(second.deny, adj_of(1, [(0, 1)], 3))                # A's rule only
    assert second.statistics[STAT].isnan().all()
    third = arm.decide(fake.obs(3, scores=SCORES))
    assert third.events == [] and not third.deny.any()


def test_intervention_budget_funds_the_highest_statistics_first_then_falls_back_to_a():
    ctx, arm, fake = _run(thresholds={STAT: -30.0}, intervention_cap=1.0, T=3)
    first = arm.decide(fake.obs(1, scores=SCORES))                        # six deliveries exceed -30; one is affordable
    assert first.deny.sum() == 1 and first.deny[0, 2].any()                # sender 2 has the highest statistic (-14)
    assert [(e["budget"], e["t"]) for e in first.events] == [("intervention", 1)]
    fake.settle(first)
    calls = len(fake.audit_calls)
    second = arm.decide(fake.obs(2, scores=SCORES))
    assert not second.deny.any() and second.events == [] and len(fake.audit_calls) == calls


def test_uncapped_never_emits_an_event():
    ctx, arm, fake = _run(thresholds={STAT: -1000.0})
    for t in range(1, 4):
        actions = arm.decide(fake.obs(t, scores=SCORES))
        fake.settle(actions)
        assert actions.events == []


def test_episodes_exhaust_independently():
    ctx, arm, fake = _run(B=2, thresholds={STAT: -1000.0})
    ctx.audit_cap[0] = 0.0
    fake.remaining_audit = ctx.audit_cap.clone()
    actions = arm.decide(fake.obs(1, scores=SCORES))
    assert [(e["b"], e["budget"]) for e in actions.events] == [(0, "audit")]
    assert not actions.deny[0].any() and actions.deny[1].any()


# ------------------------------------------------------------------------------- settings and state
def test_judge_score_bins_and_presence_window_are_required_not_defaulted():
    adj = full_adj(1, 3)
    with pytest.raises(core.Unset) as e:
        ArmC().reset(make_ctx(adj, "C", settings=make_settings(rate_limit=None, bins=None)))
    assert e.value.args[0] == "conventional.c.judge_score_bins.mock"
    with pytest.raises(core.Unset) as e:
        ArmC().reset(make_ctx(adj, "C", settings=make_settings(rate_limit=None, window=None)))
    assert e.value.args[0] == "conventional.c.presence_window_rounds"
    with pytest.raises(core.Unset) as e:
        ArmC().reset(make_ctx(adj, "C", settings=make_settings(rate_limit=None),
                              costs={k: v for k, v in COSTS.items() if k != "deny"}))
    assert "deny" in e.value.args[0]


@pytest.mark.parametrize("bins", [
    [], [[1.0]], [[1.0, 2.0, 3.0]], [[2.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 2.0]], [[1.0, -1.0]],
    [["a", 1.0]], [[NAN, 1.0]], [[1.0, NAN]], 5, "bins", [[True, 1.0]]])
def test_malformed_judge_score_bins_are_refused(bins):
    with pytest.raises(ValueError, match="judge_score_bins"):
        ArmC().reset(make_ctx(full_adj(1, 3), "C", settings=make_settings(rate_limit=None, bins=bins)))


@pytest.mark.parametrize("window", [-1, 1.5, True, "3"])
def test_malformed_presence_window_is_refused(window):
    with pytest.raises(ValueError, match="presence_window_rounds"):
        ArmC().reset(make_ctx(full_adj(1, 3), "C", settings=make_settings(rate_limit=None, window=window)))


def test_state_survives_a_checkpoint_and_continues_exactly_like_an_uninterrupted_run():
    rounds = [[-6.0, -3.0, 0.0], [-6.0, 0.0, 0.0], [-3.0, -3.0, -6.0], [0.0, -6.0, -3.0], [-6.0, -3.0, 0.0]]
    attempts = [None, adj_of(1, [(0, 1), (1, 0), (1, 2), (2, 0)], 3), None, adj_of(1, [(0, 2), (2, 1)], 3), None]
    straight = _run(thresholds={STAT: -32.0})
    resumed = _run(thresholds={STAT: -32.0})
    want, got = [], []
    for t, (s, a) in enumerate(zip(rounds, attempts, strict=False), start=1):
        out = straight[1].decide(straight[2].obs(t, a, scores=s))
        want.append((out.deny.clone(), out.statistics[STAT].clone(), out.events))
    for t, (s, a) in enumerate(zip(rounds[:2], attempts[:2], strict=False), start=1):
        resumed[1].decide(resumed[2].obs(t, a, scores=s))
    buf = io.BytesIO()
    torch.save(resumed[1].state_dict(), buf)
    buf.seek(0)
    state = torch.load(buf, weights_only=True)
    ctx, fresh, fake = _run(thresholds={STAT: -32.0})
    fresh.load_state_dict(state)
    for t, (s, a) in enumerate(zip(rounds[2:], attempts[2:], strict=False), start=3):
        out = fresh.decide(fake.obs(t, a, scores=s))
        got.append((out.deny.clone(), out.statistics[STAT].clone(), out.events))
    for (d1, s1, e1), (d2, s2, e2) in zip(want[2:], got, strict=False):
        assert torch.equal(d1, d2) and torch.equal(s1.nan_to_num(nan=-1e9), s2.nan_to_num(nan=-1e9)) and e1 == e2
    assert any(d.any() for d, _, _ in want[2:])                          # the continuation actually acted


def test_state_round_trip_keeps_exhaustion_and_does_not_re_emit():
    ctx, arm, fake = _run(thresholds={STAT: -1000.0}, audit_cap=1.0, T=3)
    arm.decide(fake.obs(1, scores=SCORES))
    state = arm.state_dict()
    ctx2, arm2, fake2 = _run(thresholds={STAT: -1000.0}, audit_cap=1.0, T=3)
    arm2.load_state_dict(state)
    fake2.remaining_audit = fake.remaining_audit.clone()
    again = arm2.decide(fake2.obs(2, scores=SCORES))
    assert again.events == [] and fake2.audit_calls == []


def test_state_dict_holds_tensors_and_plain_values_only_and_is_a_copy():
    ctx, arm, fake = _run()
    arm.decide(fake.obs(1, scores=SCORES))
    state = arm.state_dict()
    assert all(isinstance(v, (torch.Tensor, int, float, bool)) for v in state.values())
    state["recipients"].zero_()
    assert arm.state_dict()["recipients"].any()                           # mutating the copy leaves the arm alone


# ---------------------------------------------------------------------------------------- invariants
@pytest.mark.parametrize("cap", [3.0, INF])
def test_decisions_do_not_depend_on_the_batch(cap):
    B, N = 3, 5
    g = torch.Generator().manual_seed(11)
    adj = (torch.rand(B, N, N, generator=g) < 0.8) & ~torch.eye(N, dtype=torch.bool)
    rounds = [(adj & (torch.rand(B, N, N, generator=g) < 0.85), torch.randint(-8, 3, (B, N), generator=g).double())
              for _ in range(4)]

    def play(sel, ids):
        ctx, arm, fake = _run(B=len(ids), N=N, adj=adj[sel], thresholds={STAT: -27.0}, intervention_cap=cap, ids=ids)
        out = []
        for t, (att, sc) in enumerate(rounds, start=1):
            a = arm.decide(fake.obs(t, att[sel], scores=sc[sel]))
            fake.settle(a)
            events = [(e["b"], e["event"], e["budget"], e["t"]) for e in a.events]
            out.append((a.deny.clone(), a.statistics[STAT].nan_to_num(nan=-1e9), events))
        return out

    batch = play(slice(0, B), [f"u{b}" for b in range(B)])
    assert any(d.any() for d, _, _ in batch)
    for b in range(B):
        alone = play(slice(b, b + 1), [f"u{b}"])
        for (d1, s1, e1), (d2, s2, e2) in zip(batch, alone, strict=False):
            assert torch.equal(d1[b], d2[0]) and torch.equal(s1[b], s2[0])
            assert [e[1:] for e in e1 if e[0] == b] == [e[1:] for e in e2]


def test_never_reads_the_monitor_or_the_oracle_and_uses_only_deny():
    ctx, arm, fake = _run(thresholds={STAT: -20.0})
    actions = arm.decide(fake.obs(1, scores=SCORES, monitor=torch.ones(1, 3, dtype=torch.float64)))
    for name in ("sanitize", "delay", "edge_quarantine", "rate_limit", "privilege_degradation", "isolation",
                 "reinstatement"):
        assert getattr(actions, name) is None
    assert actions.deny.dtype == torch.bool and actions.deny.shape == (1, 3, 3)


def test_is_a_defender():
    assert isinstance(ArmC(), core.Defender) and ArmC().name == "C"


def test_judge_score_bins_are_read_for_the_contexts_platform():
    other = [[-1.0, 7.0], [INF, 9.0]]
    settings = make_settings(rate_limit=None)
    core.put(settings, "conventional.c.judge_score_bins", {"mock": BINS, "llm": other})
    stats = {}
    for platform in ("mock", "llm"):
        ctx = make_ctx(full_adj(1, 3), "C", platform=platform, settings=settings)
        arm = ArmC()
        arm.reset(ctx)
        stats[platform] = by_sender(arm.decide(Fake(ctx).obs(1, scores=SCORES)).statistics[STAT])
    assert stats["mock"] == [-26.0, -22.0, -14.0]
    assert stats["llm"] == [7.0 - 26.0, 7.0 - 26.0, 9.0 - 26.0]


@pytest.mark.parametrize("platform, missing", [("", "<platform> (ctx.platform is empty)"), ("llm", "llm"), ("gpu", "gpu")])
def test_empty_platform_or_platform_without_bins_refuses_naming_the_key(platform, missing):
    settings = make_settings(rate_limit=None)
    core.put(settings, "conventional.c.judge_score_bins", {"mock": BINS})
    with pytest.raises(core.Unset) as e:
        ArmC().reset(make_ctx(full_adj(1, 3), "C", platform=platform, settings=settings))
    assert e.value.args[0] == f"conventional.c.judge_score_bins.{missing}"


def test_reset_clears_every_trace_of_an_earlier_pass():
    """Closed-loop calibration replays an episode under a new threshold with the same instance: reset must be total."""
    fresh_ctx, fresh, fresh_fake = _run(thresholds={STAT: -20.0}, intervention_cap=2.0)
    ctx, arm, fake = _run(thresholds={STAT: -35.0})
    for t in (1, 2, 3):
        fake.settle(arm.decide(fake.obs(t, scores=SCORES)))                # a first pass that flags everybody
    arm.reset(fresh_ctx)
    got, want = Fake(fresh_ctx), fresh_fake
    for t in (1, 2, 3):
        x, y = arm.decide(got.obs(t, scores=SCORES)), fresh.decide(want.obs(t, scores=SCORES))
        got.settle(x), want.settle(y)
        assert torch.equal(x.deny, y.deny) and torch.equal(x.statistics[STAT].nan_to_num(nan=-1e9),
                                                           y.statistics[STAT].nan_to_num(nan=-1e9)) and x.events == y.events


def test_the_statistic_is_reported_while_the_arm_acts_on_a_given_threshold():
    """A closed-loop calibration needs both at once: the statistic of the run in which C's own flags feed back."""
    ctx, arm, fake = _run(thresholds={STAT: -20.0})
    first = arm.decide(fake.obs(1, scores=SCORES))
    assert first.deny.any() and not first.statistics[STAT].isnan().all()
