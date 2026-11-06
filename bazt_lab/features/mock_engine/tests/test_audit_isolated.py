"""D123 on the mock: `Observation.audit_isolated` audits isolated agents from their carried state (D111), pays from the
audit budget like any audit, and leaves every request that does not use it exactly as it was."""

import pytest
import torch

from bazt_lab.core.public import Actions
from bazt_lab.features.mock_engine.tests.helpers import Fn, agents, chain, forced, make_settings, seed_at, sim
from bazt_lab.features.mock_engine.tests.test_engine import NO_NOISE, SPREAD

ALL = torch.ones(1, 6, dtype=torch.bool)
ISO_0_AND_1 = {1: lambda o: Actions(isolation=agents(1, 6, (0, 0), (0, 1)))}


def plan(per_round):
    def fn(o):
        return per_round.get(o.t, lambda o: Actions())(o)
    return fn


def test_an_isolated_infected_agent_is_scored_from_its_carried_state_and_charged():
    got = {}

    def at_2(o):
        got["plain"] = o.audit(ALL)[0].clone()
        got["iso"] = o.audit_isolated(ALL)[0].clone()
        return Actions()
    w, tr, _ = sim("infection", Fn(plan({1: ISO_0_AND_1[1], 2: at_2})), T=3, edit=chain(seed_at(0, 1), NO_NOISE, forced(u_decoy=1.0)))
    assert got["plain"][:2].isnan().all() and not got["plain"][2:].isnan().any()           # audit() never grants them
    assert got["iso"][2:].isnan().all() and torch.allclose(got["iso"][:2], w.d_pay[0].expand(2).to(got["iso"].dtype))
    assert tr.spend_audit[0].tolist() == [0.0, 4.0 + 2.0, 0.0]                              # 4 senders + 2 isolated, unit cost 1


def test_the_score_of_an_isolated_agent_follows_its_carried_state_not_this_rounds_text():
    got = {}

    def at_2(o):
        got["iso"] = o.audit_isolated(ALL)[0].clone()
        return Actions()
    # agent 0 carries the payload, agent 1 never did: isolated both, with the auditor noise off the scores are d_pay and 0
    w, _, _ = sim("infection", Fn(plan({1: ISO_0_AND_1[1], 2: at_2})), T=3, edit=chain(seed_at(0), NO_NOISE, forced(u_stay=1.0, u_adopt=1.0)))
    assert got["iso"][0] == pytest.approx(w.d_pay[0].item()) and got["iso"][1] == 0.0


def test_an_active_agent_asked_for_through_the_isolated_channel_is_not_granted_or_charged():
    got = {}

    def at_1(o):
        got["iso"] = o.audit_isolated(ALL)
        return Actions()
    _, tr, _ = sim("infection", Fn(plan({1: at_1})), T=2, edit=chain(seed_at(0), NO_NOISE))
    assert got["iso"].isnan().all() and tr.spend_audit[0].tolist() == [0.0, 0.0]


def test_the_hard_audit_budget_is_exact_and_the_subset_is_keyed():
    got = []

    def at_2(o):
        got.append(o.audit_isolated(ALL)[0].clone())
        got.append(o.audit_isolated(ALL)[0].clone())
        return Actions()
    iso4 = {1: lambda o: Actions(isolation=agents(1, 6, *[(0, i) for i in range(4)]))}
    def run():
        return sim("infection", Fn(plan({**iso4, 2: at_2})), T=6, edit=chain(seed_at(0), NO_NOISE), audit=0.05)   # 3 units
    _, tr, _ = run()
    assert (~got[0].isnan()).sum() == 3 and got[1].isnan().all()                         # 3 of 4 isolated, then nothing left
    assert tr.spend_audit[0].sum() == 3.0 and tr.exhausted_step.tolist() == [2]
    granted = (~got[0].isnan()).nonzero().flatten().tolist()
    assert set(granted) <= {0, 1, 2, 3}
    got.clear()
    run()
    assert (~got[0].isnan()).nonzero().flatten().tolist() == granted                      # deterministic


def test_a_shared_budget_is_paid_once_by_audit_and_the_isolated_channel():
    got = {}

    def at_2(o):
        got["a"] = o.audit(ALL)[0].clone()
        got["i"] = o.audit_isolated(ALL)[0].clone()
        return Actions()
    iso1 = {1: lambda o: Actions(isolation=agents(1, 6, (0, 0)))}
    _, tr, _ = sim("infection", Fn(plan({**iso1, 2: at_2})), T=6, edit=chain(seed_at(0), NO_NOISE), audit=0.05)
    assert (~got["a"].isnan()).sum() == 3 and got["i"].isnan().all()                      # 5 active asked, 3 affordable
    assert tr.spend_audit[0].sum() == 3.0


def test_an_empty_isolated_request_is_free_and_changes_nothing():
    def run(extra):
        def fn(o):
            if extra:
                assert o.audit_isolated(torch.zeros(1, 6, dtype=torch.bool)).isnan().all()
            o.audit(ALL)
            return Actions()
        return sim("infection", Fn(fn), T=4, edit=chain(seed_at(0), SPREAD), audit=0.2)[1]
    a, b = run(False), run(True)
    for name in ("spend_audit", "infected", "correct", "delivered", "action_counts"):
        assert torch.equal(getattr(a, name), getattr(b, name))


def test_audit_numbering_of_the_plain_channel_is_unchanged_by_use_of_the_isolated_one():
    """Same rng subset for audit() whether or not audit_isolated is called before it (its own call count)."""
    got = []

    def run(first):
        def fn(o):
            if first and o.t == 2:
                o.audit_isolated(ALL)                                           # grants nobody: all active
            got.append(o.audit(ALL)[0].isnan().clone()) if o.t == 2 else None
            return Actions()
        sim("infection", Fn(fn), T=3, edit=chain(seed_at(0), NO_NOISE), audit=0.1)
    run(False)
    run(True)
    assert torch.equal(got[0], got[1])


# A scenario that never touches the new channel; the digest was taken on the code before D123
def _digest(**kw):
    import hashlib
    d = Fn(lambda o: (o.audit(ALL), Actions(isolation=agents(1, 6, (0, 0))) if o.t == 1 else Actions())[1])
    _, tr, est = sim("infection", d, T=5, edit=chain(seed_at(0)), **kw)
    h = hashlib.sha256()
    for name in ("spend_audit", "infected", "correct", "delivered", "active", "action_counts", "exhausted_step"):
        h.update(getattr(tr, name).numpy().tobytes())
    h.update(est.numpy().tobytes())
    return h.hexdigest()[:16]


GOLDEN = {"capped": "0f05f9c97b60b9d3", "uncapped": "532b79bca7c9b779"}


def test_an_arm_that_never_uses_the_channel_is_bit_identical_to_before():
    assert _digest(audit=0.1) == GOLDEN["capped"]
    assert _digest() == GOLDEN["uncapped"]


def test_float32_scores_follow_the_dtype_and_accounting_stays_exact():
    s = make_settings(compute__dtype="float32")
    got = {}

    def at_2(o):
        got["i"] = o.audit_isolated(ALL)
        return Actions()
    _, tr, _ = sim("infection", Fn(plan({1: ISO_0_AND_1[1], 2: at_2})), T=3, settings=s, edit=chain(seed_at(0, 1), NO_NOISE))
    assert got["i"].dtype == torch.float32 and tr.spend_audit.dtype == torch.float64 and tr.spend_audit[0, 1] == 2.0
