"""The swarmlab decision rules ported to torch: fixed_early/mid/late, score_only, centrality_only, random_capped,
risk_aware, risk_aware_tiered. Every expected value below is computed by hand from swarmlab._decide / risk_score."""
import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.references.private.legacy import (
    CentralityOnly,
    FixedWindow,
    RandomCapped,
    RiskAware,
    RiskAwareTiered,
    ScoreOnly,
)
from bazt_lab.features.references.tests.synth import FakePlatform, chain, make_ctx, star

LEGACY = {"confirm_steps": 2, "lam": 1.0, "gamma": 0.7, "dmax": 3, "tau_san": 0.05, "tau_central": 0.75,
          "tau_confirm": 0.85, "random_steps": 3}
SETTINGS = {"mock": {"legacy": LEGACY}}
MON = {"monitor": 0.5}


def ctx_of(adj, T, **kw):
    return make_ctx(adj, T, settings=kw.pop("settings", SETTINGS), **kw)


def const_monitor(values, B=1):
    m = torch.tensor([values] * B, dtype=torch.float64)
    return lambda t: m


def isolated_at(fp):
    """{round: sorted agents of episode 0 isolated that round}"""
    return {r.t: r.actions.isolation[0].nonzero().flatten().tolist() for r in fp.log
            if r.actions.isolation is not None and r.actions.isolation[0].any()}


# ---------------------------------------------------------------------------------------------- fixed windows
@pytest.mark.parametrize("which,first", [("early", 1), ("mid", 5), ("late", 9)])
def test_fixed_windows_act_only_inside_their_third_of_the_horizon(which, first):
    fp = FakePlatform(ctx_of(star(6), 12, thresholds=MON), monitor=const_monitor([0.1, 0.9, 0.2, 0.3, 0.4, 0.1]))
    fp.run(FixedWindow(which))
    assert isolated_at(fp) == {first: [1]}                  # agent 1 alone is above the threshold; once isolated, gone


def test_the_last_window_takes_the_remainder_of_an_uneven_horizon():
    # T = 10: windows (1, 3), (4, 6), (7, 10) as swarmlab.window_of (T // 3 = 3)
    for which, first in (("early", 1), ("mid", 4), ("late", 7)):
        fp = FakePlatform(ctx_of(star(6), 10, thresholds=MON), monitor=const_monitor([0.1, 0.9, 0.1, 0.1, 0.1, 0.1]))
        fp.run(FixedWindow(which))
        assert list(isolated_at(fp)) == [first]
    late = FakePlatform(ctx_of(star(6), 10, thresholds=MON), monitor=lambda t: torch.tensor(
        [[0.1, 0.1, 0.1, 0.1, 0.1, 0.9 if t == 10 else 0.1]], dtype=torch.float64))
    late.run(FixedWindow("late"))
    assert isolated_at(late) == {10: [5]}


def test_a_snapshot_window_isolates_by_highest_monitor_score_while_the_budget_fits():
    # leaf cost at t=1 = 1 delivery x 12 remaining rounds = 12; budget 12 pays one leaf: the highest score (agent 3)
    fp = FakePlatform(ctx_of(star(6), 12, thresholds=MON, intervention_cap=12),
                      monitor=const_monitor([0.1, 0.7, 0.6, 0.9, 0.2, 0.1]))
    fp.run(FixedWindow("early"))                                       # the platform asserts no violation
    assert isolated_at(fp) == {1: [3]} and fp.rem_int.item() == 0.0


def test_matched_timing_variant_drops_the_window():
    fp = FakePlatform(ctx_of(star(6), 12, thresholds=MON), monitor=const_monitor([0.1, 0.9, 0.1, 0.1, 0.1, 0.1]))
    fp.run(FixedWindow("late", own_timing=False))
    assert isolated_at(fp) == {1: [1]}


def test_threshold_is_strict_and_missing_threshold_collects_the_monitor_without_acting():
    fp = FakePlatform(ctx_of(star(4), 3, thresholds={"monitor": 0.9}), monitor=const_monitor([0.1, 0.9, 0.95, 0.0]))
    fp.run(FixedWindow("early", own_timing=False))
    assert isolated_at(fp) == {1: [2]}
    calib = FakePlatform(ctx_of(star(4), 3), monitor=const_monitor([0.1, 0.9, 0.95, 0.0]))
    calib.run(FixedWindow("early"))
    assert not isolated_at(calib)
    for r in calib.log:
        s = r.actions.statistics["monitor"]
        assert s.dtype == torch.float64 and torch.equal(s, r.obs.monitor)


# -------------------------------------------------------------------------------------------------- score_only
def test_score_only_needs_confirm_steps_consecutive_crossings():
    # agent 1 crosses only at t=1 (a spike), agent 2 at t=1 and t=2, agent 3 at t=2 and t=3
    seq = {1: [0.1, 0.9, 0.9, 0.1], 2: [0.1, 0.1, 0.9, 0.9], 3: [0.1, 0.1, 0.1, 0.9]}
    mon = lambda t: torch.tensor([seq[t]], dtype=torch.float64) if t in seq else torch.full((1, 4), 0.1,  # noqa: E731
                                                                                          dtype=torch.float64)
    fp = FakePlatform(ctx_of(star(4), 3, thresholds=MON), monitor=mon)
    fp.run(ScoreOnly())
    assert isolated_at(fp) == {2: [2], 3: [3]}                         # alarm_run reaches 2 at t=2 (agent 2), t=3 (agent 3)


def test_score_only_run_restarts_after_a_quiet_step():
    seq = {1: 0.9, 2: 0.1, 3: 0.9, 4: 0.9}                              # agent 1: crossing, quiet, crossing, crossing
    mon = lambda t: torch.tensor([[0.1, seq[t], 0.1]], dtype=torch.float64)  # noqa: E731
    fp = FakePlatform(ctx_of(star(3), 4, thresholds=MON), monitor=mon)
    fp.run(ScoreOnly())
    assert isolated_at(fp) == {4: [1]}


def test_score_only_reads_its_confirm_steps_from_settings():
    s = {"mock": {"legacy": {**LEGACY, "confirm_steps": 3}}}
    fp = FakePlatform(ctx_of(star(3), 4, thresholds=MON, settings=s), monitor=const_monitor([0.1, 0.9, 0.1]))
    fp.run(ScoreOnly())
    assert isolated_at(fp) == {3: [1]}


# ---------------------------------------------------------------------------------------------- centrality_only
def test_centrality_only_ranks_by_structural_reach_hub_first():
    # star, t=1, T=4: hub reaches 5 leaves at depth 1 (H = 5); a leaf reaches the hub (1) and 4 leaves at depth 2
    # (4 x 0.7) = 3.8. The hub costs 5 x 4 = 20, a leaf 1 x 4 = 4.
    fp = FakePlatform(ctx_of(star(6), 4, intervention_cap=20))
    fp.run(CentralityOnly())
    assert isolated_at(fp) == {1: [0]} and fp.rem_int.item() == 0.0


def test_centrality_only_skips_what_does_not_fit_and_ties_go_to_the_higher_index():
    fp = FakePlatform(ctx_of(star(6), 4, intervention_cap=8))          # the hub (20) is skipped; two leaves fit
    fp.run(CentralityOnly())
    assert isolated_at(fp) == {1: [4, 5]}                              # equal reach: sorted(reverse=True) takes index 5 first


def test_centrality_only_never_picks_an_agent_that_reaches_nobody():
    # directed chain 0 -> 1 -> 2 -> 3, t=1, T=4, reach: a0 = 1 + 0.7 + 0.49, a1 = 1 + 0.7, a2 = 1, a3 = 0
    fp = FakePlatform(ctx_of(chain(4), 4))
    fp.run(CentralityOnly())
    assert isolated_at(fp) == {1: [0, 1, 2]}


def test_centrality_only_needs_no_monitor_and_no_threshold():
    fp = FakePlatform(ctx_of(star(4), 2))                              # obs.monitor is None
    fp.run(CentralityOnly())
    assert all(not r.actions.statistics for r in fp.log)


# ---------------------------------------------------------------------------------------------- random_capped
def test_random_capped_acts_at_exactly_random_steps_many_rounds_and_one_agent_each():
    fp = FakePlatform(ctx_of(star(6), 8, B=2, unit_ids=("a", "b"), arm="random_capped"))
    fp.run(RandomCapped())
    for b in range(2):
        rounds = [r.t for r in fp.log if r.actions.isolation[b].any()]
        assert len(rounds) == 3                                        # random_steps = 3
        assert all(r.actions.isolation[b].sum() == 1 for r in fp.log if r.actions.isolation[b].any())


def test_random_capped_is_reproducible_and_independent_of_batch_size():
    ids = ("a", "b", "c")

    def run(unit_ids):
        fp = FakePlatform(ctx_of(star(6), 8, B=len(unit_ids), unit_ids=unit_ids, arm="random_capped"))
        fp.run(RandomCapped())
        return torch.stack([r.actions.isolation for r in fp.log], 1)    # [B, T, N]

    full, again, single = run(ids), run(ids), run(("b",))
    assert torch.equal(full, again) and torch.equal(full[1], single[0])
    assert not torch.equal(full[0], full[1])


def test_random_capped_picks_active_agents_uniformly():
    B = 3000
    fp = FakePlatform(ctx_of(star(6), 4, B=B, arm="random_capped"))
    fp.run(RandomCapped(own_timing=False), rounds=1)
    counts = fp.log[0].actions.isolation.sum(0)
    assert counts.sum().item() == B and counts.min().item() > 400 and counts.max().item() < 600   # expected 500


def test_random_capped_without_its_own_timing_acts_every_round_until_nobody_is_left():
    fp = FakePlatform(ctx_of(star(4), 6, arm="random_capped"))
    fp.run(RandomCapped(own_timing=False))
    assert [r.t for r in fp.log if r.actions.isolation[0].any()] == [1, 2, 3, 4]    # four agents, one per round


def test_random_capped_takes_a_cheaper_agent_when_the_random_pick_does_not_fit():
    # budget 4, one round left: a leaf costs 1, the hub 5; the hub is never isolated and the platform never overspends
    fp = FakePlatform(ctx_of(star(6), 1, B=400, intervention_cap=4, arm="random_capped"))
    fp.run(RandomCapped(own_timing=False))
    iso = fp.log[0].actions.isolation
    assert not iso[:, 0].any() and (iso.sum(-1) == 1).all()


# --------------------------------------------------------------------------------------------------- risk_aware
P_STAR4 = [0.9, 0.5, 0.5, 0.5]


def test_risk_score_matches_swarmlab_risk_score():
    # star(4), t=1, T=4, h=3, dmax=3, gamma=0.7, lam=1; weights (1 - p).
    # hub: leaves at depth 1: 3 x 0.5 = 1.5; C = 0.5 x 3/3 + 0.5 x (1 - 1/4) = 0.875
    #      R = 0.9 x 1.5 - 1.0 x 0.1 x 0.875 = 1.2625
    # leaf: hub at depth 1: 0.1; two other leaves at depth 2: 2 x 0.5 x 0.7 = 0.7; H = 0.8
    #       C = 0.5 x 1/3 + 0.375 = 0.54166..; R = 0.5 x 0.8 - 0.5 x 0.54166.. = 0.12916..
    fp = FakePlatform(ctx_of(star(4), 4), monitor=const_monitor(P_STAR4))
    fp.run(RiskAware(), rounds=1)
    risk = fp.log[0].actions.statistics["risk"]
    assert risk.dtype == torch.float64
    assert risk[0, 0].item() == pytest.approx(1.2625) and risk[0, 1].item() == pytest.approx(0.4 - 0.5 * (0.5 / 3 + 0.375))
    assert not fp.log[0].actions.isolation.any()                       # no threshold: collects, never acts


def test_risk_score_horizon_shrinks_with_time_and_ignores_isolated_agents():
    ctx = ctx_of(star(4), 4)
    fp = FakePlatform(ctx, monitor=const_monitor(P_STAR4))
    d = RiskAware()
    d.reset(ctx)
    fp.step(d, 1)
    fp.active[0, 2] = False                                            # leaf 2 isolated by someone else
    r = fp.step(d, 2).actions.statistics["risk"]
    # t=2: h=2. hub: leaves 1 and 3 reachable (2 x 0.5), leaf 2 inactive; C = 0.5 + 0.5 x (1 - 2/4) = 0.75
    assert r[0, 0].item() == pytest.approx(0.9 * 1.0 - 0.1 * 0.75)


def test_risk_aware_isolates_the_riskiest_agents_above_the_threshold_within_the_budget():
    fp = FakePlatform(ctx_of(star(4), 4, thresholds={"risk": 0.1}), monitor=const_monitor(P_STAR4))
    fp.run(RiskAware(), rounds=1)
    assert isolated_at(fp) == {1: [0, 1, 2, 3]}                        # all four are above 0.1
    tight = FakePlatform(ctx_of(star(4), 4, thresholds={"risk": 0.1}, intervention_cap=12 + 4),
                         monitor=const_monitor(P_STAR4))
    tight.run(RiskAware(), rounds=1)                                   # hub first (cost 3 x 4 = 12), then leaf 3 (ties: higher index)
    assert isolated_at(tight) == {1: [0, 3]}


def test_risk_aware_requires_its_settings_and_names_the_missing_key():
    s = {"mock": {"legacy": {k: v for k, v in LEGACY.items() if k != "gamma"}}}
    with pytest.raises(core.Unset, match=r"mock\.legacy\.gamma"):
        RiskAware().reset(ctx_of(star(4), 4, settings=s))


# --------------------------------------------------------------------------------------- risk_aware_tiered
TIER = {"risk": 1.0, "monitor": 0.4}


def test_tiered_quarantines_the_confirmed_risk_and_sanitizes_the_ambiguous_band():
    # R: hub 1.2625 >= tau_quar 1.0 -> isolate. Leaves R = 0.1292 in [tau_san 0.05, 1.0) and p = 0.5 > 0.4 -> sanitize.
    fp = FakePlatform(ctx_of(star(4), 4, thresholds=TIER), monitor=const_monitor(P_STAR4))
    fp.run(RiskAwareTiered(), rounds=1)
    a = fp.log[0].actions
    assert isolated_at(fp) == {1: [0]}
    assert a.sanitize[0, 1:].sum().item() == 3 and torch.equal(a.sanitize[0, 1:], fp.log[0].obs.attempts[0, 1:])
    assert not a.sanitize[0, 0].any()
    assert fp.log[0].charged.item() == 3 * 4 + 3                       # hub: 3 deliveries x 4 rounds; 3 sanitized leaves


def test_tiered_spends_the_budget_on_hard_actions_first_then_on_sanitizing_in_risk_order():
    fp = FakePlatform(ctx_of(star(4), 4, thresholds=TIER, intervention_cap=14), monitor=const_monitor(P_STAR4))
    fp.run(RiskAwareTiered(), rounds=1)
    a = fp.log[0].actions
    assert isolated_at(fp) == {1: [0]}
    assert a.sanitize[0].any(-1).nonzero().flatten().tolist() == [1, 2]        # 12 + 1 + 1 = 14; equal risk: lower index first


def test_tiered_degrades_a_confirmed_risk_it_cannot_afford_to_a_sanitization():
    fp = FakePlatform(ctx_of(star(4), 4, thresholds=TIER, intervention_cap=5), monitor=const_monitor(P_STAR4))
    fp.run(RiskAwareTiered(), rounds=1)
    a = fp.log[0].actions
    assert not a.isolation.any()
    assert a.sanitize[0].any(-1).nonzero().flatten().tolist() == [0, 1, 2]      # hub (3), leaf 1 (1), leaf 2 (1) = 5


def test_tiered_shields_a_central_agent_whose_evidence_is_unconfirmed():
    # p = 0.6 in [tau_p, tau_confirm 0.85) and C = 0.875 >= tau_central 0.75 -> sanitize even outside the risk band
    s = {"mock": {"legacy": {**LEGACY, "tau_san": 10.0}}}              # the ambiguous band is empty
    fp = FakePlatform(ctx_of(star(4), 4, thresholds={"risk": 100.0, "monitor": 0.5}, settings=s),
                      monitor=const_monitor([0.6, 0.5, 0.5, 0.5]))
    fp.run(RiskAwareTiered(), rounds=1)
    a = fp.log[0].actions
    assert a.sanitize[0].any(-1).nonzero().flatten().tolist() == [0] and not a.isolation.any()
    confirmed = FakePlatform(ctx_of(star(4), 4, thresholds={"risk": 100.0, "monitor": 0.5}, settings=s),
                             monitor=const_monitor([0.9, 0.5, 0.5, 0.5]))
    confirmed.run(RiskAwareTiered(), rounds=1)                         # p above tau_confirm: not "unconfirmed"
    assert not confirmed.log[0].actions.sanitize.any()


def test_tiered_without_thresholds_collects_both_statistics_and_acts_on_nothing():
    fp = FakePlatform(ctx_of(star(4), 4), monitor=const_monitor(P_STAR4))
    fp.run(RiskAwareTiered())
    for r in fp.log:
        assert not r.actions.isolation.any() and not r.actions.sanitize.any()
        assert set(r.actions.statistics) == {"risk", "monitor"}


# ----------------------------------------------------------------------------------------------- all legacy arms
def every_arm():
    return [lambda: FixedWindow("early"), lambda: FixedWindow("mid", own_timing=False), ScoreOnly, CentralityOnly,
            RandomCapped, lambda: RandomCapped(own_timing=False), RiskAware, RiskAwareTiered]


@pytest.mark.parametrize("make", every_arm())
def test_every_legacy_arm_stays_inside_the_budget_and_resumes_exactly_from_a_checkpoint(make, tmp_path):
    mon = lambda t: torch.tensor([[0.1, 0.9, 0.2, 0.7, 0.3, 0.1 + 0.15 * t]] * 3, dtype=torch.float64)  # noqa: E731

    def world():
        return ctx_of(star(6), 6, B=3, thresholds={"monitor": 0.5, "risk": 0.2}, intervention_cap=30, arm="x")

    ref = FakePlatform(world(), monitor=mon)
    ref.run(make())                                                    # the platform asserts no budget violation
    part = FakePlatform(world(), monitor=mon)
    d1 = make()
    d1.reset(part.ctx)
    for t in (1, 2, 3):
        part.step(d1, t)
    store = core.CheckpointStore(tmp_path)
    store.save_state("u", d1.state_dict())
    d2 = make()
    d2.reset(part.ctx)
    d2.load_state_dict(store.load_state("u"))
    for t in (4, 5, 6):
        part.step(d2, t)
    for a, b in zip(ref.log, part.log, strict=True):
        for k in ("isolation", "sanitize"):
            x, y = getattr(a.actions, k), getattr(b.actions, k)
            assert (x is None and y is None) or torch.equal(x, y)


@pytest.mark.parametrize("make", [lambda: FixedWindow("early"), ScoreOnly, RiskAware, RiskAwareTiered])
def test_monitor_driven_arms_refuse_a_platform_without_the_monitor(make):
    fp = FakePlatform(ctx_of(star(4), 2, thresholds=MON))
    with pytest.raises(ValueError, match="monitor"):
        fp.run(make())


@pytest.mark.parametrize("make", every_arm())
def test_every_legacy_arm_refuses_a_missing_isolation_cost(make):
    ctx = ctx_of(star(4), 2, costs={"inspect": 1.0})
    with pytest.raises(core.Unset, match="isolation"):
        make().reset(ctx)
