"""BAZT_upstream: the adapter drives bazt.optimizer; on a hand-built case it buys exactly what the optimizer plans."""
import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.bazt_slot import public as slot
from bazt_lab.features.bazt_slot.adapters.upstream import BaztUpstream, reference_plan
from bazt_lab.features.bazt_slot.private import jitter
from bazt_lab.features.bazt_slot.tests.testkit import INF, Fake, full_adj, make_ctx, make_settings, monitor_of

N, T = 5, 6


def settings():
    s = make_settings()
    core.put(s, "bazt_upstream.verdict_probability", 0.98)
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
    assert flagged[1.0] == 0.98 and (flagged[99.0] is None or flagged[99.0] == pytest.approx(0.02))


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


# ------------------------------------------------------------- D108: deliveries restricted by a calibrated statistic
FLAG = torch.tensor([[0.0, 5.0, 0.0, 0.0, 0.0]], dtype=torch.float64)


def restricting(a):
    return any(getattr(a, n) is not None and getattr(a, n).any() for n in ("isolation", "sanitize", "edge_quarantine",
                                                                         "privilege_degradation"))


def test_a_restriction_the_monitor_alone_drives_is_not_threshold_restricted():
    d, fake, _ = build(audit_cap=0)                                        # nothing can be audited: no verdict
    a = d.decide(fake.obs(1, monitor=monitor_of(1, N, hot=[1])))
    assert restricting(a) and not a.threshold_restricted.any()


def test_a_restriction_of_an_agent_the_audit_flagged_is_threshold_restricted_and_stays_so():
    d, fake, ctx = build(audit_cap=INF)
    fake.scores = FLAG.clone()
    a = d.decide(fake.obs(1, monitor=monitor_of(1, N, hot=[1], level=0.5)))
    assert restricting(a) and a.threshold_restricted.any()
    rows = a.threshold_restricted.nonzero()
    assert ((rows[:, 1] == 1) | (rows[:, 2] == 1)).all()                  # only deliveries of the flagged agent
    fake.settle(a)
    b = d.decide(fake.obs(2, monitor=monitor_of(1, N, hot=[1], level=0.5)))
    if (d.quarantined | d.restricted).any() or fake.isolated.any():
        assert b.threshold_restricted.any()                                 # persistent restrictions keep counting


def test_while_calibrating_the_mask_is_empty():
    d, fake, _ = build(audit_cap=INF, thresholds={})
    a = d.decide(fake.obs(1, monitor=monitor_of(1, N, hot=[1])))
    assert a.threshold_restricted is not None and not a.threshold_restricted.any()


def test_the_verdict_probability_is_a_setting():
    got = []
    for prob in (0.98, 0.9):
        s = settings()
        core.put(s, "bazt_upstream.verdict_probability", prob)
        c = make_ctx(full_adj(1, N), arm="BAZT_upstream", T=T, settings=s, thresholds={"upstream_judge": 1.0},
                     audit_cap=INF, intervention_cap=INF)
        d = BaztUpstream()
        d.reset(c)
        fake = Fake(c)
        fake.scores = FLAG.clone()
        d.decide(fake.obs(1, monitor=monitor_of(1, N, hot=[1], level=0.5)))
        got.append(d.verdicts[0].override(1, 1))
    assert got == [0.98, 0.9]


def test_a_missing_or_impossible_verdict_probability_refuses_and_names_the_key():
    s = settings()
    del s["bazt_upstream"]["verdict_probability"]
    with pytest.raises(core.Unset) as e:
        BaztUpstream().reset(make_ctx(full_adj(1, N), settings=s))
    assert e.value.args[0] == "bazt_upstream.verdict_probability"
    s = settings()
    core.put(s, "bazt_upstream.verdict_probability", 0.5)
    with pytest.raises(ValueError, match=r"bazt_upstream\.verdict_probability"):
        BaztUpstream().reset(make_ctx(full_adj(1, N), settings=s))


def test_the_pool_is_not_debited_for_actions_the_adapter_drops():
    """The platform price of an isolation (out-degree x rounds left) exceeds the cap, so the adapter drops it; the upstream
    ledger must end with exactly the actions that were emitted."""
    d, fake, _ = build(cap=7, audit_cap=0)
    a = d.decide(fake.obs(1, monitor=monitor_of(1, N, hot=[1, 3])))
    spent = {k.name: v for k, v in d.ledgers[0].spent_by_kind.items()}
    n = lambda m: 0 if m is None else int(m.sum())                      # noqa: E731
    assert spent["ISOLATE"] == n(a.isolation) and spent["BLOCK_EDGE"] == n(a.edge_quarantine)
    assert spent["RESTRICT"] == n(a.privilege_degradation)
    assert spent["SANITIZE"] == (0 if a.sanitize is None else int(a.sanitize.any(-1).sum()))


# ------------------------------------------- D117 (every run float32): the verdict is the reported statistic over the threshold
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_each_verdict_is_the_reported_statistic_over_the_calibrated_threshold(dtype):
    """The audit verdict and the reported `upstream_judge` use the same jittered value, so the threshold calibrated on the
    statistic controls the verdict rate in float32 as in float64. 300 episodes, sender 1 audited with the same score 6.2 in
    each: a tie, and the threshold sits a quarter of the jitter range above it, so the jitter splits the tie."""
    B = 300
    v = torch.tensor(6.2, dtype=dtype)
    th = float(v + 0.25 * torch.tensor(jitter.top_scale(dtype, 6.2), dtype=dtype))
    ctx = make_ctx(full_adj(B, N), arm="BAZT_upstream", T=T, settings=settings(), thresholds={"upstream_judge": th},
                   audit_cap=INF)
    ctx = core.EpisodeContext(**{**ctx.__dict__, "dtype": dtype})
    d = BaztUpstream()
    d.reset(ctx)
    fake = Fake(ctx)
    fake.scores = torch.zeros(B, N, dtype=dtype)
    fake.scores[:, 1] = v
    a = d.decide(fake.obs(1, monitor=monitor_of(B, N, hot=[1], level=0.5).to(dtype)))
    stat = a.statistics["upstream_judge"]
    assert stat.dtype == dtype
    audited = [b for b in range(B) if d.verdicts[b].override(1, 1) is not None]
    assert len(audited) > B // 2
    verdict = torch.tensor([d.verdicts[b].override(1, 1) == 0.98 for b in audited])
    reported = stat[audited, 1, 0] > th
    assert 0 < int(reported.sum()) < len(audited)                  # the threshold splits the tie
    assert torch.equal(verdict, reported)
