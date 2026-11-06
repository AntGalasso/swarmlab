"""D116: every mock episode records the rounds at which the platform applied an action of the arm other than an audit
(`trace.meta["intervened"]`, bool [B, T], 0-based; the row column `intervened_by_round`). A matched-timing control is
handed BAZT's record of the same unit as `unit["timing_schedule"]`, in the same format."""
import json
from typing import ClassVar

import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.metrics import public as metrics
from bazt_lab.features.mock_engine import public as mock
from bazt_lab.features.mock_engine.private import enforce, engine
from bazt_lab.features.mock_engine.tests.helpers import (
    KINDS,
    FakeStore,
    Fn,
    Scripted,
    agents,
    chain,
    forced,
    make_settings,
    mask,
    seed_at,
    sim,
    spec,
    unit,
)

Actions = core.Actions
SPREAD = forced(u_adopt=0.0, u_stay=1.0)
ACTED = [i for i, k in enumerate(KINDS) if k != core.ActionKind.INSPECT.value]


def rounds_of(trace, b=0):
    return trace.meta["intervened"][b].nonzero().flatten().tolist()


def test_the_trace_records_the_rounds_at_which_an_action_was_applied_0_based():
    plan = {2: Actions(isolation=agents(1, 6, (0, 3))), 4: Actions(deny=mask(1, 6, (0, 0, 1))),
            5: Actions(sanitize=mask(1, 6, (0, 1, 2)), delay=mask(1, 6, (0, 4, 5)))}
    _, tr, _ = sim("infection", Scripted(plan), T=6, edit=chain(seed_at(0), SPREAD))
    got = tr.meta["intervened"]
    assert got.dtype == torch.bool and tuple(got.shape) == (1, 6)
    assert got[0].tolist() == [False, True, False, True, True, False]          # rounds t = 2, 4, 5 (Observation.t)


def test_an_audit_is_not_an_intervention():
    def audit_all(o):
        o.audit(torch.ones(1, 6, dtype=torch.bool))
        return Actions()
    _, tr, _ = sim("infection", Fn(audit_all), T=4, edit=seed_at(0))
    assert tr.action_counts[0, KINDS.index("inspect")] > 0 and not tr.meta["intervened"].any()


def test_a_request_that_changes_nothing_is_not_an_intervention():
    """Already isolated, reinstating an unrestricted agent, denying what an isolation already cut: no action applied."""
    plan = {1: Actions(isolation=agents(1, 6, (0, 2))),
            2: Actions(isolation=agents(1, 6, (0, 2))),                       # already isolated
            3: Actions(reinstatement=agents(1, 6, (0, 4))),                   # agent 4 carries no restriction
            4: Actions(deny=mask(1, 6, (0, 2, 3), (0, 1, 2)))}               # both deliveries touch the isolated agent
    _, tr, _ = sim("none", Scripted(plan), T=5)
    assert rounds_of(tr) == [0]


def test_a_persistent_restriction_counts_only_at_the_round_it_was_applied():
    _, tr, _ = sim("none", Scripted({2: Actions(edge_quarantine=mask(1, 6, (0, 0, 1)))}), T=5)
    assert rounds_of(tr) == [1] and tr.restricted[0, 2:, 0, 1].all()


def test_a_round_rejected_over_budget_is_not_an_intervention():
    plan = {1: Actions(deny=mask(1, 6, (0, 0, 1), (0, 1, 2), (0, 2, 3), (0, 3, 4))),    # 4 > 3 units: all rejected
            2: Actions(deny=mask(1, 6, (0, 0, 1)))}
    _, tr, _ = sim("infection", Scripted(plan), T=6, edit=chain(seed_at(0), SPREAD), intervention=0.05)
    assert tr.violations.tolist() == [1] and rounds_of(tr) == [1]


def test_a_free_deny_of_an_unauthentic_delivery_is_an_applied_action():
    a = Fn(lambda o: Actions(deny=o.attempts & ~o.authentic))
    edit = chain(forced(u_adopt=0.0, u_stay=1.0), lambda w: w.victim.fill_(1))
    _, tr, _ = sim("aitm", a, T=4, edit=edit)
    assert tr.spend_intervention.sum() == 0 and tr.meta["intervened"][0].tolist() == [True] * 4


def test_the_recorded_rounds_are_exactly_the_rounds_with_an_applied_action_under_arbitrary_actions(monkeypatch):
    """The record equals, round by round, the platform's own count of applied actions (enforce.Round.counts), and its
    any() equals a nonzero non-audit entry of trace.action_counts."""
    applied = []
    real = enforce.apply_actions

    def spy(*args, **kw):
        out = real(*args, **kw)
        applied.append(out.counts[:, ACTED].sum(1) > 0)
        return out
    monkeypatch.setattr(engine.enforce, "apply_actions", spy)
    g = torch.Generator().manual_seed(5)

    def noise(o):
        B, N = o.active.shape
        edge = lambda p: torch.rand(B, N, N, generator=g) < p                         # noqa: E731
        node = lambda p: torch.rand(B, N, generator=g) < p                            # noqa: E731
        o.audit(node(0.5))
        return Actions(deny=edge(0.01), sanitize=edge(0.01), delay=edge(0.01), edge_quarantine=edge(0.01),
                       rate_limit=node(0.01), privilege_degradation=node(0.01), isolation=node(0.01),
                       reinstatement=node(0.03))
    _, tr, _ = sim("infection", Fn(noise), T=8, topology="barabasi_albert", N=9, seeds=(1, 2, 3, 4, 5, 6),
                   intervention=0.3, audit=0.2)
    want = torch.stack(applied, 1)
    assert torch.equal(tr.meta["intervened"], want)
    assert want.any() and not want.all()
    assert torch.equal(tr.meta["intervened"].any(1), tr.action_counts[:, ACTED].sum(1) > 0)


def test_the_record_of_an_episode_does_not_depend_on_its_batch():
    def policy(o):
        return Actions(deny=o.attempts & (o.monitor > 0.5)[:, :, None])
    kw = dict(topology="small_world", N=8, T=7)
    both = sim("infection", Fn(policy), seeds=(1, 2, 3), **kw)[1]
    for k, sd in enumerate((1, 2, 3)):
        one = sim("infection", Fn(policy), seeds=(sd,), **kw)[1]
        assert torch.equal(both.meta["intervened"][k], one.meta["intervened"][0])


def deny_hot(o):
    return Actions(deny=o.attempts & (o.monitor > 0.5)[:, :, None])


def test_every_row_carries_intervened_by_round_and_it_does_not_depend_on_the_batch_size():
    units = [unit(arm="hot", topology="small_world", attack="infection", agents=8, seed=sd, sink=sd % 8)
             for sd in range(1, 7)]
    arm = core.ArmSpec("hot", lambda: Fn(deny_hot))
    whole = mock.MockPlatform().run_batch(units, arm, make_settings(mock__batch_size=6), FakeStore(), core.NullTracker())
    singles = mock.MockPlatform().run_batch(units, arm, make_settings(mock__batch_size=1), FakeStore(), core.NullTracker())
    assert [json.dumps(r["intervened_by_round"]) for r in whole] == [json.dumps(r["intervened_by_round"]) for r in singles]
    for r in whole:
        assert isinstance(r["intervened_by_round"], list) and len(r["intervened_by_round"]) == r["n_rounds"]
        assert all(isinstance(x, bool) for x in r["intervened_by_round"])
        assert any(r["intervened_by_round"]) == (sum(r[f"actions_{k}"] for k in KINDS if k != "inspect") > 0)
    assert any(any(r["intervened_by_round"]) for r in whole) and "intervened_by_round" in metrics.COLUMNS
    quiet = mock.MockPlatform().run_batch(units[:2], core.ArmSpec("hot", Scripted), make_settings(), FakeStore(),
                                         core.NullTracker())
    assert all(r["intervened_by_round"] == [False] * r["n_rounds"] for r in quiet)


# ----------------------------------------------------------------------------- the schedule format the engine reads
@pytest.mark.parametrize("bad", [[0, 1] * 7 + [0], [1.0] * 15, ["yes"] * 15])
def test_a_timing_schedule_holds_t_booleans_and_round_numbers_or_numbers_are_refused(bad):
    u = unit(arm="scripted", seed=1, sink=0, timing_schedule=bad)                    # chain of 6: T = 15
    with pytest.raises(ValueError, match=r"timing_schedule.*booleans"):
        mock.MockPlatform().run_batch([u], spec(), make_settings(), FakeStore(), core.NullTracker())


def test_the_recorded_column_is_a_schedule_the_engine_reads_back():
    got = []

    class Peek(Scripted):
        def reset(self, ctx):
            super().reset(ctx)
            got.append(ctx.timing_schedule)
    first = mock.MockPlatform().run_batch([unit(arm="hot", seed=2, sink=1, attack="infection")],
                                          core.ArmSpec("hot", lambda: Fn(deny_hot)), make_settings(), FakeStore(),
                                          core.NullTracker())[0]
    again = unit(arm="scripted", seed=2, sink=1, timing_schedule=first["intervened_by_round"])   # attack-free: no twin
    mock.MockPlatform().run_batch([again], spec(Peek), make_settings(), FakeStore(), core.NullTracker())
    assert got[0][0].tolist() == first["intervened_by_round"]


# ------------------------------------------------------------- the attack-free twin keeps its own schedule (D109)
class Peeks(Scripted):
    seen: ClassVar[list] = []

    def reset(self, ctx):
        super().reset(ctx)
        Peeks.seen.append((ctx.unit_ids, None if ctx.timing_schedule is None else ctx.timing_schedule.tolist()))


def test_the_simulated_twin_of_a_matched_unit_runs_with_the_twins_own_schedule():
    """An insider twin is always simulated (D109); it runs with `twin_timing_schedule`, the schedule of the grid's
    attack-free unit, and its saved row carries that schedule, not the attacked unit's."""
    Peeks.seen = []
    attacked_s, twin_s = [i % 3 == 0 for i in range(15)], [i % 5 == 1 for i in range(15)]
    twin = unit(arm="m", seed=1, sink=0, attack="none")
    u = unit(arm="m", seed=1, sink=0, attack="insider_pi", timing_schedule=attacked_s, twin_id=twin["id"],
             twin_timing_schedule=twin_s)
    store = FakeStore()
    mock.MockPlatform(rows_fn=metrics.rows_from_trace).run_batch([u], core.ArmSpec("m", Peeks), make_settings(), store,
                                                                 core.NullTracker())
    by_ids = {ids[0]: sched for ids, sched in Peeks.seen}
    assert by_ids[u["id"]] == [attacked_s] and by_ids[twin["id"]] == [twin_s]
    saved = store.rows[twin["id"]][0]
    assert saved["timing_schedule"] == twin_s and "twin_timing_schedule" not in saved and saved["attack"] == "none"


def test_a_matched_unit_whose_twin_must_run_without_a_twin_schedule_is_refused():
    twin = unit(arm="m", seed=1, sink=0, attack="none")
    u = unit(arm="m", seed=1, sink=0, attack="insider_pi", timing_schedule=[True] * 15, twin_id=twin["id"])
    with pytest.raises(ValueError, match="twin_timing_schedule"):
        mock.MockPlatform().run_batch([u], core.ArmSpec("m", Scripted), make_settings(), FakeStore(), core.NullTracker())
