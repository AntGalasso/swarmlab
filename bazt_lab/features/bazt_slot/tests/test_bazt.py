"""BAZT as a defender: the three components together on small fake episodes."""
import math

import numpy as np
import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.bazt_slot.private import bazt as bz
from bazt_lab.features.bazt_slot.tests.testkit import (
    COSTS_ROBUST,
    INF,
    Fake,
    FakePipeline,
    adj_of,
    full_adj,
    make_ctx,
    make_settings,
    monitor_of,
    run,
    small_world,
    spectral_radius_np,
    star_hub,
)

F64 = torch.float64
EDGE_MASKS = ("deny", "sanitize", "delay", "edge_quarantine")
AGENT_MASKS = ("rate_limit", "privilege_degradation", "isolation", "reinstatement")
THRESHOLDS = {"bazt_risk": 0.0, "judge": 1.5}          # log-odds 0 = p 0.5; the judge's alpha-point


def make(variant="BAZT", *, B=1, N=5, T=6, adj=None, thresholds=THRESHOLDS, **ctx_kw):
    adj = full_adj(B, N) if adj is None else adj
    ctx = make_ctx(adj, arm=variant, thresholds=thresholds, T=T, **ctx_kw)
    defender = bz.Bazt(variant, pipeline=FakePipeline())
    defender.reset(ctx)
    return defender, Fake(ctx), ctx


def acted(actions):
    return {name: getattr(actions, name) for name in EDGE_MASKS + AGENT_MASKS if getattr(actions, name) is not None
            and getattr(actions, name).any() and name != "deny"}


def targets(actions):
    """The agents the actions of this round are about: senders of edge actions and subjects of agent actions."""
    who = set()
    for mask in acted(actions).values():
        who |= set(mask.nonzero()[:, 1].tolist())
    return who


# --------------------------------------------------------------------------------------------- the basics
def test_reset_refuses_when_a_setting_is_missing_and_names_the_key():
    ctx = make_ctx(full_adj(1, 3), settings={})
    with pytest.raises(core.Unset) as e:
        bz.Bazt("BAZT", pipeline=FakePipeline()).reset(ctx)
    assert e.value.args[0] == "bazt.katz_decay"


def test_an_unknown_variant_is_refused():
    with pytest.raises(ValueError, match="BAZT_nope"):
        bz.Bazt("BAZT_nope")


def test_a_benign_network_is_left_alone_and_pays_for_nothing_it_does_not_need():
    d, fake, _ = make(audit_cap=50, intervention_cap=50)
    run(d, fake, 5, monitor=monitor_of(1, 5))
    assert all(not acted(a) for a in fake.history)
    assert (fake.spent_intervention == 0).all()


def test_a_suspicious_sender_is_acted_on_and_only_it():
    d, fake, _ = make(intervention_cap=100)
    run(d, fake, 3, monitor=monitor_of(1, 5, hot=[1]))
    first = fake.history[0]
    assert acted(first) and targets(first) == {1}
    assert all(targets(a) <= {1} for a in fake.history)


def test_free_denies_are_only_a_rules_deliveries_off_the_allowlist_or_unauthenticated():
    d, fake, ctx = make()
    authentic = torch.ones(1, 5, 5, dtype=torch.bool)
    authentic[0, 3, 4] = False
    actions = d.decide(fake.obs(1, monitor=monitor_of(1, 5), authentic=authentic))
    assert actions.deny.nonzero().tolist() == [[0, 3, 4]]


# ---------------------------------------------------------------------------- audits, verdicts and statistics
def test_while_calibrating_every_sender_is_audited_nothing_is_done_and_statistics_are_reported():
    d, fake, _ = make(thresholds={}, audit_cap=INF, intervention_cap=INF)
    fake.scores = torch.arange(5, dtype=F64)[None]
    actions = d.decide(fake.obs(1, monitor=monitor_of(1, 5, hot=[2])))
    assert fake.audit_calls[0].all()
    assert not acted(actions)
    assert set(actions.statistics) == {"bazt_risk", "judge"}
    assert torch.allclose(actions.statistics["judge"][0, :, 0][1:], fake.scores[0, 1:])


def test_the_risk_statistic_is_the_free_monitor_evidence_per_delivery_and_absent_without_a_monitor():
    d, fake, _ = make(thresholds={})
    actions = d.decide(fake.obs(1, monitor=monitor_of(1, 5, hot=[2])))
    stat = actions.statistics["bazt_risk"]
    assert stat.shape == (1, 5, 5)
    assert stat[0, 2, 0].item() == pytest.approx(math.log(0.97 / 0.03), rel=1e-6)
    assert stat[0, 0, 0].isnan()                                    # no delivery from 0 to itself
    d2, fake2, _ = make(thresholds={})
    assert "bazt_risk" not in d2.decide(fake2.obs(1)).statistics


def test_a_flagged_audit_makes_a_quiet_sender_eligible_through_the_verdict_memory():
    d, fake, _ = make(intervention_cap=INF)
    fake.scores = torch.tensor([[0.0, 0.0, 9.0, 0.0, 0.0]], dtype=F64)            # judge flags agent 2 (> 1.5)
    quiet = monitor_of(1, 5, base=0.2)
    run(d, fake, 3, monitor=quiet)
    assert any(2 in targets(a) for a in fake.history)
    assert all(targets(a) <= {2} for a in fake.history)


def test_an_audit_that_clears_a_loud_sender_calls_off_the_response():
    d, fake, _ = make(intervention_cap=INF)
    fake.scores = torch.zeros(1, 5, dtype=F64)                                       # the judge finds nothing anywhere
    run(d, fake, 3, monitor=monitor_of(1, 5, hot=[1], level=0.8))
    uncleared, fake2, _ = make(intervention_cap=INF, audit_cap=0)
    run(uncleared, fake2, 3, monitor=monitor_of(1, 5, hot=[1], level=0.8))
    assert sum(len(targets(a)) for a in fake.history) < sum(len(targets(a)) for a in fake2.history)


def test_audits_are_bought_by_value_and_the_audit_budget_is_never_exceeded():
    d, fake, _ = make(audit_cap=2, intervention_cap=INF)
    run(d, fake, 4, monitor=monitor_of(1, 5, hot=[1, 3], level=0.6, base=0.3))
    assert fake.spent_audit.item() <= 2
    assert all(c.sum().item() <= 2 for c in fake.audit_calls)


# ---------------------------------------------------------------------------- budgets (Hua et al.) and pacing
def random_episode(seed, B=6, N=6, T=6):
    gen = torch.Generator().manual_seed(seed)
    adj = (torch.rand(B, N, N, generator=gen, dtype=F64) < 0.5) & ~torch.eye(N, dtype=torch.bool)
    caps = torch.randint(0, 40, (B,), generator=gen).to(F64)
    acaps = torch.randint(0, 12, (B,), generator=gen).to(F64)
    mon = torch.rand(T, B, N, generator=gen, dtype=F64) * 0.9 + 0.05
    scores = torch.randn(B, N, generator=gen, dtype=F64) * 2
    return adj, caps, acaps, mon, scores


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("variant", ["BAZT", "BAZT_no_topology", "BAZT_no_graduated"])
def test_the_budgets_are_never_exceeded(variant, seed):
    adj, caps, acaps, mon, scores = random_episode(seed)
    ctx = make_ctx(adj, arm=variant, thresholds=THRESHOLDS, T=6)
    ctx = core.EpisodeContext(**{**ctx.__dict__, "audit_cap": acaps, "intervention_cap": caps})
    d = bz.Bazt(variant, pipeline=FakePipeline())
    d.reset(ctx)
    fake = Fake(ctx)
    run(d, fake, 6, monitor=lambda t: mon[t - 1], scores=scores)        # Fake.settle raises Overspend otherwise
    assert (fake.spent_intervention <= caps).all() and (fake.spent_audit <= acaps).all()


@pytest.mark.parametrize("quarantine_once", [False, True])
def test_the_budget_holds_whichever_way_the_platform_charges_a_quarantine(quarantine_once):
    adj, caps, acaps, mon, scores = random_episode(3)
    ctx = make_ctx(adj, thresholds=THRESHOLDS, T=6)
    ctx = core.EpisodeContext(**{**ctx.__dict__, "audit_cap": acaps, "intervention_cap": caps})
    d = bz.Bazt("BAZT", pipeline=FakePipeline())
    d.reset(ctx)
    fake = Fake(ctx, quarantine_charged_once=quarantine_once)
    run(d, fake, 6, monitor=lambda t: mon[t - 1], scores=scores)
    assert (fake.spent_intervention <= caps).all()


# Only the recurring edge actions (sanitize, delay) are worth buying: the pacing binds on them, since their benign price is
# (1 - p) x cost x the rounds they recur. A persistent action is charged once for every remaining delivery and recurs
# once, so its benign price is below its cost and only the hard cap stops it (with a convergent Katz sum, D115, the
# quarantine is the dense choice on this graph and a cap of 30 goes to five of them in round 1).
RECURRING_ONLY = {f"bazt__utility_loss__{k}": 50.0 for k in ("edge_quarantine", "rate_limit", "privilege_degradation",
                                                                "isolation")}


def test_a_tight_budget_is_spread_over_the_horizon_not_spent_in_the_first_round():
    caps = 20.0
    settings = make_settings(**RECURRING_ONLY)
    d, fake, _ = make(N=6, intervention_cap=caps, audit_cap=0, settings=settings)   # no audit: the pacing alone decides
    mon = monitor_of(1, 6, hot=[1, 2, 3], level=0.7, base=0.03)
    run(d, fake, 1, monitor=mon)
    spent_capped = fake.spent_intervention.item()
    u, fake_u, _ = make(N=6, intervention_cap=INF, audit_cap=0, settings=settings)
    run(u, fake_u, 1, monitor=mon)
    assert 0 < spent_capped < fake_u.spent_intervention.item()
    assert fake.remaining_intervention.item() > 0


def test_the_ablation_without_a_budget_is_refused_on_a_capped_cell_and_runs_on_the_uncapped_one():
    for cap in ({"intervention_cap": 1.0}, {"audit_cap": 3.0}):
        ctx = make_ctx(full_adj(1, 6), arm="BAZT_no_budget", **cap)
        with pytest.raises(ValueError, match=r"BAZT_no_budget.*uncapped"):
            bz.Bazt("BAZT_no_budget", pipeline=FakePipeline()).reset(ctx)
    d, fake, _ = make("BAZT_no_budget", N=6, intervention_cap=INF, audit_cap=INF)
    a = d.decide(fake.obs(1, monitor=monitor_of(1, 6, hot=[1, 2], level=0.97)))
    assert targets(a) == {1, 2}


def test_budget_exhausted_is_raised_once_per_budget_and_episode():
    d, fake, ctx = make(N=5, intervention_cap=6, audit_cap=1)
    events = []
    for t in range(1, 5):
        a = d.decide(fake.obs(t, monitor=monitor_of(1, 5, hot=[1], level=0.97)))
        fake.settle(a)
        events += a.events
    kinds = [(e["budget"], e["event"]) for e in events]
    assert kinds.count(("audit", "budget_exhausted")) <= 1 and kinds.count(("intervention", "budget_exhausted")) <= 1
    assert all(e["arm"] == "BAZT" and e["unit_id"] == "u0" and e["b"] == 0 for e in events)
    assert ("audit", "budget_exhausted") in kinds


def test_no_event_is_raised_while_a_budget_is_uncapped():
    d, fake, _ = make(intervention_cap=INF, audit_cap=INF)
    run(d, fake, 4, monitor=monitor_of(1, 5, hot=[1]))
    assert not [e for a in fake.history for e in a.events]


# -------------------------------------------------------------------------------------------------- variants
def test_the_ablation_without_graduated_actions_only_isolates_and_never_reinstates():
    d, fake, _ = make("BAZT_no_graduated", intervention_cap=INF)
    run(d, fake, 5, monitor=monitor_of(1, 5, hot=[1], level=0.97))
    assert any(a.isolation is not None and a.isolation.any() for a in fake.history)
    for a in fake.history:
        for name in ("sanitize", "delay", "edge_quarantine", "rate_limit", "privilege_degradation", "reinstatement"):
            assert getattr(a, name) is None or not getattr(a, name).any(), name


def test_graduated_bazt_prefers_a_lighter_action_when_the_evidence_is_weak():
    """A suspicion of 0.55 does not pay for isolating an agent that is busy with the task, but a lighter response may."""
    full, fake, _ = make(N=5, intervention_cap=INF, thresholds={"bazt_risk": 0.0, "judge": 99.0})
    run(full, fake, 4, monitor=monitor_of(1, 5, hot=[1], level=0.8))
    iso_only, fake_i, _ = make("BAZT_no_graduated", N=5, intervention_cap=INF, thresholds={"bazt_risk": 0.0, "judge": 99.0})
    run(iso_only, fake_i, 4, monitor=monitor_of(1, 5, hot=[1], level=0.8))
    kinds_full = {n for a in fake.history for n in acted(a)}
    assert kinds_full - {"isolation"} or not kinds_full             # uses something other than isolation, or nothing
    assert fake.spent_intervention.item() <= fake_i.spent_intervention.item() or kinds_full == {"isolation"}


def test_the_hub_of_a_star_is_worth_more_than_a_leaf_only_when_the_topology_is_used():
    """Hub 0 sends to leaves 1..5; leaf 6 sends to leaf 7 only. Both read 0.6 on the monitor."""
    edges = [(0, j) for j in range(1, 6)] + [(6, 7)]
    adj = adj_of(1, edges, 8)
    mon = monitor_of(1, 8, hot=[0, 6], level=0.6, base=0.02)
    with_topology, fake, _ = make(B=1, N=8, adj=adj, T=6, intervention_cap=12, audit_cap=0)
    a = with_topology.decide(fake.obs(1, monitor=mon))
    without, fake2, _ = make("BAZT_no_topology", B=1, N=8, adj=adj, T=6, intervention_cap=12, audit_cap=0)
    b = without.decide(fake2.obs(1, monitor=mon))
    assert targets(a) == {0}                                        # the hub only
    assert targets(b) and targets(b) <= {0, 6}                      # the graph does not tell them apart: a jitter-level tie


def test_nothing_is_bought_in_the_last_round():
    d, fake, _ = make(T=3, intervention_cap=INF)
    a = d.decide(fake.obs(3, monitor=monitor_of(1, 5, hot=[1])))
    assert not acted(a)


# ------------------------------------------------------------------------------ persistent state and reinstatement
ISOLATION_ONLY = {f"bazt__utility_loss__{k}": 50.0 for k in ("sanitize", "delay", "edge_quarantine", "rate_limit",
                                                               "privilege_degradation")}


def test_an_isolated_agent_is_not_acted_on_again_and_its_belief_fades_until_it_is_reinstated():
    """Isolation is the only restriction worth buying here (on a complete graph with a convergent Katz sum the lighter
    quarantine and rate limit win otherwise)."""
    d, fake, _ = make(N=5, T=14, intervention_cap=INF, thresholds={"bazt_risk": 0.0, "judge": 99.0},
                      settings=make_settings(**ISOLATION_ONLY))
    mon = lambda t: monitor_of(1, 5, hot=[1] if t == 1 else [], level=0.97)         # noqa: E731
    run(d, fake, 13, monitor=mon)
    iso_rounds = [t for t, a in enumerate(fake.history, 1) if a.isolation is not None and a.isolation[0, 1]]
    back = [t for t, a in enumerate(fake.history, 1) if a.reinstatement is not None and a.reinstatement[0, 1]]
    assert iso_rounds and len(iso_rounds) == 1
    assert back and back[0] > iso_rounds[0]
    assert not fake.isolated[0, 1] or back[-1] < iso_rounds[0]


def test_a_quarantined_edge_is_not_proposed_again():
    d, fake, _ = make(N=5, intervention_cap=INF)
    run(d, fake, 4, monitor=monitor_of(1, 5, hot=[1], level=0.97))
    seen = set()
    for a in fake.history:
        if a.edge_quarantine is not None:
            now = {tuple(x) for x in a.edge_quarantine.nonzero().tolist()}
            assert not (now & seen)
            seen |= now


# --------------------------------------------------------------------- determinism, batching and checkpoints
def scenario_batch(variant, B, seed):
    adj, caps, acaps, mon, scores = random_episode(seed, B=B)
    ctx = make_ctx(adj, arm=variant, thresholds=THRESHOLDS, T=6)
    ctx = core.EpisodeContext(**{**ctx.__dict__, "audit_cap": acaps, "intervention_cap": caps})
    return ctx, mon, scores


def play(ctx, mon, scores, variant="BAZT", rounds=6):
    d = bz.Bazt(variant, pipeline=FakePipeline())
    d.reset(ctx)
    fake = Fake(ctx)
    run(d, fake, rounds, monitor=lambda t: mon[t - 1], scores=scores)
    return fake.history


def same(a, b):
    for name in EDGE_MASKS + AGENT_MASKS:
        x, y = getattr(a, name), getattr(b, name)
        assert (x is None) == (y is None) or (x is None and not y.any()) or (y is None and not x.any()), name
        if x is not None and y is not None:
            assert torch.equal(x, y), name


def test_the_same_inputs_give_the_same_actions():
    ctx, mon, scores = scenario_batch("BAZT", 5, 21)
    for a, b in zip(play(ctx, mon, scores), play(ctx, mon, scores), strict=True):
        same(a, b)


def sub_ctx(ctx, b):
    return core.EpisodeContext(**{**ctx.__dict__, "B": 1, "adj": ctx.adj[b:b + 1], "sink": ctx.sink[b:b + 1],
                                  "audit_cap": ctx.audit_cap[b:b + 1], "intervention_cap": ctx.intervention_cap[b:b + 1],
                                  "unit_ids": (ctx.unit_ids[b],),
                                  "arm_stream": lambda ns, _b, b=b: ctx.arm_stream(ns, b)})


@pytest.mark.parametrize("variant", ["BAZT", "BAZT_no_topology", "BAZT_no_graduated"])
def test_an_episode_does_not_depend_on_the_batch_it_runs_in(variant):
    ctx, mon, scores = scenario_batch(variant, 6, 33)
    full = play(ctx, mon, scores, variant)
    for b in (0, 3, 5):
        alone = play(sub_ctx(ctx, b), mon[:, b:b + 1], scores[b:b + 1], variant)
        for a_full, a_one in zip(full, alone, strict=True):
            for name in EDGE_MASKS + AGENT_MASKS:
                x, y = getattr(a_full, name), getattr(a_one, name)
                if x is None and y is None:
                    continue
                xs = torch.zeros_like(y) if x is None else x[b:b + 1]
                ys = torch.zeros_like(xs) if y is None else y
                assert torch.equal(xs, ys), (variant, b, name)


def test_a_run_resumed_from_a_saved_state_continues_exactly(tmp_path):
    ctx, mon, scores = scenario_batch("BAZT", 4, 7)
    d = bz.Bazt("BAZT", pipeline=FakePipeline())
    d.reset(ctx)
    fake = Fake(ctx)
    fake.scores = scores.clone()
    for t in range(1, 4):
        fake.settle(d.decide(fake.obs(t, monitor=mon[t - 1])))
    store = core.CheckpointStore(tmp_path)
    store.save_state("u", d.state_dict())
    rest = []
    for t in range(4, 7):
        a = d.decide(fake.obs(t, monitor=mon[t - 1]))
        fake.settle(a)
        rest.append(a)
    d2 = bz.Bazt("BAZT", pipeline=FakePipeline())
    d2.reset(ctx)
    d2.load_state_dict(store.load_state("u"))
    fake2 = Fake(ctx)
    fake2.scores = scores.clone()
    fake2.remaining_audit, fake2.remaining_intervention = fake.remaining_audit.clone(), fake.remaining_intervention.clone()
    fake2.isolated = fake.isolated.clone()
    # rebuild the platform state at round 4 by replaying its first three rounds on a twin
    twin = Fake(ctx)
    twin.scores = scores.clone()
    d3 = bz.Bazt("BAZT", pipeline=FakePipeline())
    d3.reset(ctx)
    for t in range(1, 4):
        twin.settle(d3.decide(twin.obs(t, monitor=mon[t - 1])))
    resumed = []
    for t in range(4, 7):
        a = d2.decide(twin.obs(t, monitor=mon[t - 1]))
        twin.settle(a)
        resumed.append(a)
    for a, b in zip(rest, resumed, strict=True):
        same(a, b)


def test_the_state_holds_tensors_and_plain_values_only():
    d, fake, _ = make(intervention_cap=INF)
    run(d, fake, 2, monitor=monitor_of(1, 5, hot=[1]))

    def check(x):
        if isinstance(x, dict):
            [check(v) for v in x.values()]
        else:
            assert isinstance(x, (torch.Tensor, int, float, bool, str, list, tuple)), type(x)
    check(d.state_dict())


def test_the_robustness_cost_condition_changes_what_is_bought_not_whether_the_budget_holds():
    adj, caps, acaps, mon, scores = random_episode(4)
    ctx = make_ctx(adj, thresholds=THRESHOLDS, T=6, costs=COSTS_ROBUST)
    ctx = core.EpisodeContext(**{**ctx.__dict__, "audit_cap": acaps, "intervention_cap": caps})
    d = bz.Bazt("BAZT", pipeline=FakePipeline())
    d.reset(ctx)
    fake = Fake(ctx)
    run(d, fake, 6, monitor=lambda t: mon[t - 1], scores=scores)
    assert (fake.spent_intervention <= caps).all()


@pytest.mark.parametrize("variant", ["BAZT", "BAZT_no_budget", "BAZT_no_topology", "BAZT_no_graduated"])
def test_every_variant_reports_its_own_risk_statistic_when_calibrating(variant):
    d, fake, _ = make(variant, thresholds={})
    stat = d.decide(fake.obs(1, monitor=monitor_of(1, 5, hot=[2]))).statistics["bazt_risk"]
    assert stat[0, 2, 0].item() == pytest.approx(math.log(0.97 / 0.03), rel=1e-6)


@pytest.mark.parametrize("variant", ["BAZT", "BAZT_no_graduated"])
def test_a_reset_gives_a_second_pass_the_state_of_a_fresh_arm(variant):
    ctx, mon, scores = scenario_batch(variant, 3, 17)
    first = play(ctx, mon, scores, variant)
    d = bz.Bazt(variant, pipeline=FakePipeline())
    d.reset(ctx)
    warm = Fake(ctx)
    run(d, warm, 6, monitor=lambda t: mon[t - 1], scores=scores)
    d.reset(ctx)                                    # a new calibration pass on the same instance
    fake = Fake(ctx)
    again = run(d, fake, 6, monitor=lambda t: mon[t - 1], scores=scores)
    for a, b in zip(first, again, strict=True):
        same(a, b)


# ------------------------------------------------------------------------------------------- tie jitter, edge cases
def test_reported_statistics_are_jittered_so_ties_split_without_reordering():
    d, fake, _ = make(thresholds={})
    fake.scores = torch.full((1, 5), 1.0, dtype=F64)                      # five senders tie exactly
    stat = d.decide(fake.obs(1, monitor=monitor_of(1, 5, hot=[2]))).statistics["judge"]
    per_sender = torch.stack([row[~row.isnan()][0] for row in stat[0]])
    assert per_sender.unique().numel() == 5 and (per_sender - 1.0).abs().max().item() < 1e-6


def test_the_risk_statistic_and_the_eligibility_test_use_the_same_jittered_value():
    d, fake, _ = make(thresholds={"bazt_risk": 0.0, "judge": 99.0})
    mon = monitor_of(1, 5, base=0.5)                                      # log-odds 0 for everybody: all tie at the threshold
    stat = d.decide(fake.obs(1, monitor=mon)).statistics["bazt_risk"]
    assert (stat[0, 0, 1:] == stat[0, 0, 1]).all()                        # one jitter per sender
    above = set()
    for t in range(1, 5):
        above |= set((d.decide(fake.obs(t, monitor=mon)).statistics["bazt_risk"][0, :, 1] > 0.0).nonzero().flatten().tolist())
    assert above and above != {0, 1, 2, 3, 4}                              # the tie at the threshold is split, not all-or-nothing


def test_no_edge_action_touches_an_agent_isolated_in_the_same_round():
    for seed in range(8):
        adj, caps, acaps, mon, scores = random_episode(seed)
        ctx = make_ctx(adj, thresholds=THRESHOLDS, T=6)
        d = bz.Bazt("BAZT", pipeline=FakePipeline())
        d.reset(ctx)
        fake = Fake(ctx)
        fake.scores = scores
        for t in range(1, 7):
            a = d.decide(fake.obs(t, monitor=mon[t - 1]))
            if a.isolation is not None:
                for name in ("sanitize", "delay", "edge_quarantine"):
                    m = getattr(a, name)
                    if m is not None:
                        assert not (m & (a.isolation[:, :, None] | a.isolation[:, None, :])).any(), (seed, t, name)
            fake.settle(a)


def test_the_quarantine_mirror_follows_the_engines_order_of_application():
    """Engine: reinstatement first, then the new quarantines; a quarantine on an edge of an agent isolated in the same
    Actions is not applied; degradation and rate limits on an agent isolated in the same Actions are not applied."""
    d, fake, ctx = make(N=4)
    z = lambda *s: torch.zeros(*s, dtype=torch.bool)                       # noqa: E731
    d.quarantined[0, 1, 2] = True
    d.degraded[0, 1] = d.limited[0, 1] = True
    quarantine, back, iso = z(1, 4, 4), z(1, 4), z(1, 4)
    quarantine[0, 1, 2] = quarantine[0, 3, 0] = quarantine[0, 0, 3] = True
    back[0, 1] = True
    iso[0, 3] = True
    d._remember_actions(core.Actions(edge_quarantine=quarantine, reinstatement=back, isolation=iso,
                                     privilege_degradation=iso.clone(), rate_limit=iso.clone()),
                        active=torch.ones(1, 4, dtype=torch.bool))
    assert d.quarantined[0, 1, 2]                                         # lifted by the reinstatement, set again after it
    assert not d.quarantined[0, 3, 0] and not d.quarantined[0, 0, 3]      # edges of the agent isolated in the same round
    assert not d.degraded[0, 1] and not d.limited[0, 1]                   # reinstated
    assert not d.degraded[0, 3] and not d.limited[0, 3]


# ------------------------------------------------------------- D108: deliveries restricted by a calibrated statistic
def test_threshold_restricted_counts_priced_restrictions_and_never_a_rules_denies_or_a_rate_limit():
    d, fake, ctx = make(N=4)
    z = lambda *s: torch.zeros(*s, dtype=torch.bool)                       # noqa: E731
    d.degraded[0, 2] = d.limited[0, 3] = True
    d.quarantined[0, 0, 1] = True
    san = z(1, 4, 4)
    san[0, 1, 2] = True
    actions = core.Actions(deny=torch.ones(1, 4, 4, dtype=torch.bool), sanitize=san, rate_limit=torch.eye(4, dtype=torch.bool)[None])
    mask = d._threshold_restricted(actions, torch.ones(1, 4, dtype=torch.bool))
    expected = z(1, 4, 4)
    expected[0, 1, 2] = True                    # the sanitize of this round
    expected[0, 0, 1] = True                    # a quarantine kept from an earlier round
    expected[0, 2, :] = True                    # every delivery of a degraded sender
    expected &= ctx.adj
    assert torch.equal(mask, expected)          # A's denies and the rate-limited sender 3 are not in it


def test_an_isolated_agent_counts_on_every_edge_it_has():
    d, fake, ctx = make(N=4)
    iso = torch.zeros(1, 4, dtype=torch.bool)
    iso[0, 1] = True
    mask = d._threshold_restricted(core.Actions(isolation=iso), torch.ones(1, 4, dtype=torch.bool))
    expected = torch.zeros(1, 4, 4, dtype=torch.bool)
    expected[0, 1, :] = expected[0, :, 1] = True
    assert torch.equal(mask, expected & ctx.adj)


def test_decide_reports_the_mask_and_it_is_empty_while_calibrating_and_on_a_benign_network():
    d, fake, _ = make(thresholds={})
    assert not d.decide(fake.obs(1, monitor=monitor_of(1, 5, hot=[2]))).threshold_restricted.any()
    d, fake, _ = make(intervention_cap=INF)
    assert not d.decide(fake.obs(1, monitor=monitor_of(1, 5))).threshold_restricted.any()


def test_a_suspicious_sender_the_arm_restricts_is_reported_as_threshold_restricted():
    d, fake, _ = make(intervention_cap=INF)
    a = d.decide(fake.obs(1, monitor=monitor_of(1, 5, hot=[1])))
    assert acted(a)
    m = a.threshold_restricted
    assert m.dtype == torch.bool and m.shape == (1, 5, 5) and m.any()
    assert m[0, 1].any() or m[0, :, 1].any() or m.nonzero()[:, 1].tolist().count(1) > 0


def test_the_judge_statistic_is_sender_level_so_its_jitter_is_one_draw_per_sender():
    d, fake, _ = make(thresholds={})
    fake.scores = torch.full((1, 5), 1.0, dtype=F64)
    stat = d.decide(fake.obs(1, monitor=monitor_of(1, 5, hot=[2]))).statistics["judge"][0]
    for i in range(5):
        row = stat[i][~stat[i].isnan()]
        assert row.unique().numel() == 1, i                              # one value per sender, like the score it jitters
    assert torch.stack([stat[i][~stat[i].isnan()][0] for i in range(5)]).unique().numel() == 5


def test_a_cleared_sender_stops_being_eligible_when_its_verdict_has_faded():
    """No monitor (LLM side): a single flag makes the sender eligible for two rounds; afterwards nothing acts on it."""
    d, fake, _ = make(intervention_cap=INF, thresholds={"judge": 1.5})
    fake.scores = torch.tensor([[0.0, 9.0, 0.0, 0.0, 0.0]], dtype=F64)
    d.decide(fake.obs(1))
    fake.remaining_audit[:] = 0                                          # nothing is audited afterwards: the verdict only fades
    elig = []
    for t in range(2, 8):
        d.decide(fake.obs(t))
        elig.append(bool(risk_eligible(d)))
    assert elig[0]
    assert not elig[-1]


def risk_eligible(d):
    from bazt_lab.features.bazt_slot.private import risk
    free = risk.free_logodds(None, torch.ones(1, 5, dtype=torch.bool), d.p.prior)
    return risk.eligible(free, d.memory, math.inf, d.p.clear_logodds)[0, 1]


# ------------------------------------------------------------------ D115: one Katz decay per episode, from its own graph
def test_reset_takes_each_episodes_katz_decay_from_its_own_graph():
    graphs = [star_hub(12), small_world(12, seed=1)]
    adj = torch.as_tensor(np.stack(graphs) > 0)
    d, _, _ = make(B=2, N=12, adj=adj, settings=make_settings(bazt__katz_decay=0.9))
    assert d.decay.dtype == F64 and d.decay.device == adj.device
    assert d.decay.tolist() == pytest.approx([0.9 / spectral_radius_np(g) for g in graphs], rel=1e-9)


def test_the_ablation_without_topology_does_not_depend_on_the_katz_decay():
    runs = []
    for katz in (0.05, 0.95):
        ctx, mon, scores = scenario_batch("BAZT_no_topology", 4, 5)
        ctx = core.EpisodeContext(**{**ctx.__dict__, "settings": make_settings(bazt__katz_decay=katz)})
        runs.append(play(ctx, mon, scores, "BAZT_no_topology"))
    for a, b in zip(*runs, strict=True):
        same(a, b)


# --------------------------------------------- D118: reinstatement gives up the structural share, so no isolate/lift flapping
@pytest.mark.parametrize("dtype", [F64, torch.float32])
def test_an_uncapped_bazt_on_a_constant_monitor_isolates_once_and_does_not_lift_while_the_isolation_pays(dtype):
    """Agents 1-3 read 0.6 on the monitor every round, nothing is audited, and the prior is 0.6 too: the free evidence of
    an isolated agent is the prior (`risk.free_logodds`), so their posterior is 0.6 whether isolated or not. With the
    belief constant, the isolation keeps paying, so the complement pricing never lifts it. The pricing before D118 left
    out the structural share and lifted all three in round 7. (With a prior below the monitor reading the belief drops to
    the prior once the agent is isolated; lifting it then is what the evidence says, not a flap of the prices.)"""
    T = 8
    ctx = make_ctx(full_adj(1, 6), arm="BAZT", thresholds=THRESHOLDS, T=T, intervention_cap=INF, audit_cap=0,
                   settings=make_settings(bazt__prior=0.6))
    ctx = core.EpisodeContext(**{**ctx.__dict__, "dtype": dtype})
    d = bz.Bazt("BAZT", pipeline=FakePipeline())
    d.reset(ctx)
    fake = Fake(ctx)
    run(d, fake, T, monitor=monitor_of(1, 6, hot=[1, 2, 3], level=0.6, base=0.03).to(dtype))
    isolated = {i: [t for t, a in enumerate(fake.history, 1) if a.isolation is not None and a.isolation[0, i]]
                for i in range(6)}
    lifted = [t for t, a in enumerate(fake.history, 1) if a.reinstatement is not None and a.reinstatement.any()]
    assert isolated[1] == isolated[2] == isolated[3] == [1]
    assert all(len(rounds) <= 1 for rounds in isolated.values())          # nobody is isolated twice
    assert lifted == []


# ------------------------------------- D119: an isolated agent keeps its last monitor-based risk; only a clearing audit lifts
HOT = (1, 2, 3)


def d119_world(*, dtype=F64, audit_cap=0, B=1, T=14, isolated_channel=False):
    """Prior 0.1, monitor 0.95 on agents 1-3 every round (the flapping trace). Agents are isolated in round 1."""
    ctx = make_ctx(full_adj(B, 6), arm="BAZT", thresholds={"bazt_risk": 0.0, "judge": 1.5}, T=T, intervention_cap=INF,
                   audit_cap=audit_cap, settings=make_settings(**ISOLATION_ONLY))
    ctx = core.EpisodeContext(**{**ctx.__dict__, "dtype": dtype})
    d = bz.Bazt("BAZT", pipeline=FakePipeline())
    d.reset(ctx)
    return d, Fake(ctx, isolated_channel=isolated_channel), ctx


def lifts(fake):
    return [t for t, a in enumerate(fake.history, 1) if a.reinstatement is not None and a.reinstatement.any()]


def isolations(fake, i):
    return [t for t, a in enumerate(fake.history, 1) if a.isolation is not None and a.isolation[0, i]]


@pytest.mark.parametrize("dtype", [F64, torch.float32])
def test_without_a_clearing_audit_an_isolated_agent_is_never_lifted_and_never_isolated_twice(dtype):
    d, fake, ctx = d119_world(dtype=dtype, audit_cap=0)
    run(d, fake, 12, monitor=monitor_of(1, 6, hot=HOT, level=0.95, base=0.03).to(dtype))
    assert lifts(fake) == []
    for i in HOT:
        assert isolations(fake, i) == [1]
    assert fake.isolated[0, list(HOT)].all()


def test_an_isolated_agent_keeps_the_free_evidence_of_its_last_active_round_not_the_prior():
    d, fake, _ = d119_world(audit_cap=0)
    run(d, fake, 3, monitor=monitor_of(1, 6, hot=HOT, level=0.95, base=0.03))
    held = d.state_dict()["held"]
    assert torch.allclose(held[0, 1], torch.tensor(math.log(0.95 / 0.05), dtype=held.dtype))   # the monitor reading of round 1
    assert torch.allclose(held[0, 0], torch.tensor(math.log(0.03 / 0.97), dtype=held.dtype))    # an active agent: this round's


def drive(d, fake, rounds, mon, clean_from=2, start=1):
    """Agent 1 is flagged by the auditor in round 1 and found clean from `clean_from` on."""
    for t in range(start, start + rounds):
        fake.scores = torch.zeros_like(fake.scores)
        fake.scores[:, 1] = 9.0 if t < clean_from else -9.0
        fake.settle(d.decide(fake.obs(t, monitor=mon)))


def test_a_clearing_audit_lifts_an_isolated_agent_and_every_lift_follows_an_audit_of_that_agent():
    d, fake, _ = d119_world(audit_cap=INF)
    drive(d, fake, 14, monitor_of(1, 6, hot=[1], level=0.95, base=0.03))
    assert isolations(fake, 1) and lifts(fake)
    for t in lifts(fake):
        assert fake.audit_calls and any(call[0, 1] for call in fake.audit_calls[max(0, t - 3):]), t
    # the round of each lift: agent 1 was isolated and was audited that very round
    audited_at = [t for t in range(1, 15) if any(c[0, 1] for c in calls_of_round(fake, t))]
    assert all(t in audited_at for t in lifts(fake))


def calls_of_round(fake, t):
    """The audit requests of round t (the Fake logs one entry per obs.audit call, in order)."""
    return [c for c, r in zip(fake.audit_calls, fake.audit_round, strict=True) if r == t]


def test_an_audit_that_flags_the_isolated_agent_does_not_lift_it():
    d, fake, _ = d119_world(audit_cap=INF)
    drive(d, fake, 10, monitor_of(1, 6, hot=[1], level=0.95, base=0.03), clean_from=99)
    assert lifts(fake) == []


def test_a_run_resumed_mid_isolation_continues_exactly(tmp_path):
    mon = monitor_of(1, 6, hot=[1], level=0.95, base=0.03)
    d, fake, ctx = d119_world(audit_cap=INF)
    drive(d, fake, 3, mon)
    assert fake.isolated[0, 1]
    store = core.CheckpointStore(tmp_path)
    store.save_state("u", d.state_dict())
    twin = Fake(ctx)
    twin.isolated, twin.remaining_audit = fake.isolated.clone(), fake.remaining_audit.clone()
    twin.remaining_intervention = fake.remaining_intervention.clone()
    d2 = bz.Bazt("BAZT", pipeline=FakePipeline())
    d2.reset(ctx)
    d2.load_state_dict(store.load_state("u"))
    drive(d, fake, 8, mon, start=4)
    drive(d2, twin, 8, mon, start=4)
    for a, b in zip(fake.history[3:], twin.history, strict=True):
        same(a, b)
    assert torch.equal(d.held, d2.held) and torch.equal(d.memory, d2.memory)


def test_the_isolated_agents_belief_does_not_depend_on_the_batch_it_runs_in():
    mon = torch.stack([monitor_of(1, 6, hot=[1], level=0.95)[0], monitor_of(1, 6, hot=[2], level=0.8)[0],
                       monitor_of(1, 6, hot=[1, 2], level=0.97)[0]])
    d, fake, ctx = d119_world(audit_cap=INF, B=3)
    drive(d, fake, 10, mon)
    for b in range(3):
        d1, f1, _ = d119_world(audit_cap=INF, B=1)
        drive(d1, f1, 10, mon[b:b + 1])
        assert torch.equal(d.held[b:b + 1], d1.held)
        for a_full, a_one in zip(fake.history, f1.history, strict=True):
            for name in AGENT_MASKS:
                x, y = getattr(a_full, name), getattr(a_one, name)
                xs = torch.zeros_like(fake.isolated[b:b + 1]) if x is None else x[b:b + 1]
                ys = torch.zeros_like(xs) if y is None else y
                assert torch.equal(xs, ys), (b, name)


def test_the_audit_requests_for_isolated_agents_never_pay_ahead_of_senders_and_stay_in_budget():
    d, fake, _ = d119_world(audit_cap=3)
    drive(d, fake, 10, monitor_of(1, 6, hot=[1], level=0.95, base=0.03))
    assert (fake.spent_audit <= 3.0).all()


# ------------------------------------- D123: the audit of an isolated agent goes through its own channel
def test_isolated_agents_are_audited_through_the_isolated_channel_when_the_platform_has_one():
    d, fake, _ = d119_world(audit_cap=INF, isolated_channel=True)
    drive(d, fake, 14, monitor_of(1, 6, hot=[1], level=0.95, base=0.03))
    assert isolations(fake, 1) and lifts(fake)                                   # a clearing audit through the channel lifts
    assert fake.iso_calls and any(c[0, 1] for c in fake.iso_calls)
    for t in lifts(fake):
        assert any(c[0, 1] for c, r in zip(fake.iso_calls, fake.iso_round, strict=True) if r == t), t


def test_the_plain_audit_never_asks_for_an_agent_the_platform_has_isolated_when_the_channel_exists():
    d, fake, _ = d119_world(audit_cap=INF, isolated_channel=True)
    seen_isolated = []
    real = fake.audit

    def audit(mask, record=True):
        if record:
            seen_isolated.append(bool((mask & fake.isolated).any()))
        return real(mask, record=record)
    fake.audit = audit
    drive(d, fake, 14, monitor_of(1, 6, hot=[1], level=0.95, base=0.03))
    assert not any(seen_isolated)


def test_a_flagging_audit_through_the_channel_does_not_lift_and_the_spend_is_the_budget_unit_cost():
    d, fake, _ = d119_world(audit_cap=INF, isolated_channel=True)
    drive(d, fake, 10, monitor_of(1, 6, hot=[1], level=0.95, base=0.03), clean_from=99)
    assert lifts(fake) == []
    asked = sum(int(c[0].sum()) for c in fake.audit_calls + fake.iso_calls)
    assert 0 < fake.spent_audit[0] <= asked                     # every granted audit is one unit; none is charged twice


def test_without_the_channel_bazt_asks_the_plain_audit_as_before():
    d, fake, _ = d119_world(audit_cap=INF)
    drive(d, fake, 14, monitor_of(1, 6, hot=[1], level=0.95, base=0.03))
    assert lifts(fake) and fake.iso_calls == []
