"""BAZT_upstream: the adapter drives bazt.optimizer; on a hand-built case it buys exactly what the optimizer plans."""
import torch

from bazt_lab.features.bazt_slot import public as slot
from bazt_lab.features.bazt_slot.adapters.upstream import BaztUpstream, reference_plan
from bazt_lab.features.bazt_slot.tests.testkit import INF, Fake, full_adj, make_ctx, make_settings, monitor_of

N, T = 5, 6


def settings():
    s = make_settings()
    return s


def build(cap=INF, audit_cap=0, thresholds=None):
    ctx = make_ctx(full_adj(1, N), arm="BAZT_upstream", T=T, settings=settings(), intervention_cap=cap,
                   audit_cap=audit_cap, thresholds={"upstream_judge": 1.0} if thresholds is None else thresholds)
    d = BaztUpstream()
    d.reset(ctx)
    return d, Fake(ctx), ctx


def direct_plan(monitor, cap):
    return reference_plan(monitor, cap, N, T, 0.9)


def test_the_adapter_buys_what_the_upstream_optimizer_plans():
    mon = [0.03, 0.97, 0.03, 0.6, 0.03]
    plan = direct_plan(mon, INF)
    d, fake, _ = build()
    a = d.decide(fake.obs(1, monitor=torch.tensor([mon], dtype=torch.float64)))
    got_iso = set(a.isolation.nonzero()[:, 1].tolist()) if a.isolation is not None else set()
    got_san = set(a.sanitize.nonzero()[:, 1].tolist()) if a.sanitize is not None else set()
    got_res = set(a.privilege_degradation.nonzero()[:, 1].tolist()) if a.privilege_degradation is not None else set()
    got_blk = {tuple(x[1:]) for x in a.edge_quarantine.nonzero().tolist()} if a.edge_quarantine is not None else set()
    assert plan.isolate or plan.sanitize or plan.restrict or plan.block
    assert (got_iso, got_san, got_res, got_blk) == (set(plan.isolate), set(plan.sanitize), set(plan.restrict),
                                                    set(plan.block))


def test_a_benign_network_gets_no_action():
    d, fake, _ = build()
    a = d.decide(fake.obs(1, monitor=monitor_of(1, N)))
    assert all(getattr(a, n) is None for n in ("isolation", "sanitize", "edge_quarantine", "privilege_degradation"))


def test_the_platform_budget_is_never_exceeded():
    d, fake, _ = build(cap=7)
    for t in range(1, T + 1):
        fake.settle(d.decide(fake.obs(t, monitor=monitor_of(1, N, hot=[1, 3]))))     # Fake raises on an overspend
    assert fake.spent_intervention.item() <= 7


def test_an_audit_verdict_over_the_threshold_confirms_and_is_paid_from_the_audit_budget():
    d, fake, _ = build(audit_cap=3)
    fake.scores = torch.tensor([[0.0, 5.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    d.decide(fake.obs(1, monitor=monitor_of(1, N, hot=[1], level=0.5)))
    assert fake.spent_audit.item() <= 3 and fake.audit_calls


def test_same_inputs_same_actions_and_state_round_trips():
    def play():
        d, fake, _ = build(cap=40)
        out = [d.decide(fake.obs(t, monitor=monitor_of(1, N, hot=[1]))) for t in (1, 2)]
        return d, out
    (d1, o1), (_, o2) = play(), play()
    for x, y in zip(o1, o2, strict=True):
        for n in ("isolation", "sanitize", "edge_quarantine", "privilege_degradation"):
            assert (getattr(x, n) is None) == (getattr(y, n) is None)
    d3, _, _ = build(cap=40)
    d3.load_state_dict(d1.state_dict())
    assert d3.step == 2 and torch.equal(d3.quarantined, d1.quarantined)


def test_calibrating_audits_every_sender_acts_on_nothing_and_reports_the_judge_statistic():
    d, fake, _ = build(audit_cap=INF, thresholds={})
    fake.scores = torch.arange(N, dtype=torch.float64)[None]
    a = d.decide(fake.obs(1, monitor=monitor_of(1, N, hot=[1])))
    assert fake.audit_calls[0].all()
    assert all(getattr(a, n) is None for n in ("isolation", "sanitize", "edge_quarantine", "privilege_degradation"))
    stat = a.statistics["upstream_judge"]
    assert stat.shape == (1, N, N) and torch.allclose(stat[0, 1:, 0], fake.scores[0, 1:]) and stat[0, 0, 0].isnan()


def test_the_verdict_threshold_is_the_calibrated_statistic():
    flagged = {}
    for th in (1.0, 99.0):
        ctx = make_ctx(full_adj(1, N), arm="BAZT_upstream", T=T, settings=settings(), thresholds={"upstream_judge": th},
                       audit_cap=INF)
        d = BaztUpstream()
        d.reset(ctx)
        fake = Fake(ctx)
        fake.scores = torch.tensor([[0.0, 5.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        d.decide(fake.obs(1, monitor=monitor_of(1, N, hot=[1], level=0.5)))
        flagged[th] = d.verdicts[0].override(1, 1)
    assert flagged[1.0] == 0.98 and flagged[99.0] in (0.02, None)


def test_a_reset_forgets_everything_so_a_second_calibration_pass_starts_clean():
    d, fake, ctx = build(cap=40, audit_cap=INF)
    fake.scores = torch.tensor([[0.0, 5.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    for t in (1, 2):
        fake.settle(d.decide(fake.obs(t, monitor=monitor_of(1, N, hot=[1]))))
    d.reset(ctx)
    again, fresh = Fake(ctx), Fake(ctx)
    again.scores = fresh.scores = fake.scores.clone()
    fresh_d = BaztUpstream()
    fresh_d.reset(ctx)
    x, y = d.decide(again.obs(1, monitor=monitor_of(1, N, hot=[1]))), fresh_d.decide(fresh.obs(1, monitor=monitor_of(1, N, hot=[1])))
    for n in ("isolation", "sanitize", "edge_quarantine", "privilege_degradation"):
        assert (getattr(x, n) is None) == (getattr(y, n) is None)
    assert d.step == 1 and not d.restricted.any()


def test_public_exports_the_two_arms_and_the_three_ablations():
    names = [a.name for a in slot.ARMS]
    assert names == ["BAZT", "BAZT_no_budget", "BAZT_no_topology", "BAZT_no_graduated", "BAZT_upstream"]
    assert all(a.factory().name == a.name for a in slot.ARMS)
    assert all(a.calibrated for a in slot.ARMS)


def test_a_verdict_lives_on_the_episode_clock_not_the_conversation_clock():
    """obs.t counts the whole seed run (D54). The verdict TTL is two rounds: it still holds the round after it was made
    and has lapsed two rounds later, even when the conversation restarts at round 1 in between."""
    d, fake, _ = build(audit_cap=INF, thresholds={"upstream_judge": 1.0})
    fake.scores = torch.tensor([[0.0, 5.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    mon = monitor_of(1, N, hot=[1], level=0.5)
    d.decide(fake.obs(5, monitor=mon))                                   # first round of a second conversation
    assert d.verdicts[0].override(1, 5) == 0.98 and d.verdicts[0].override(1, 6) == 0.98
    assert d.verdicts[0].override(1, 7) is None
