"""random_alloc, oracle_alloc and bra_greedy: the shared audit-then-block pipeline and each targeting rule."""
import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.references.private.allocation import BraGreedy, OracleAlloc, RandomAlloc
from bazt_lab.features.references.tests.synth import FakePlatform, adj_from_edges, chain, make_ctx, star

T0 = {"audit_score": 0.0}          # threshold between the fake auditor's -3 (benign) and +3 (infected)


def complete(n):
    return adj_from_edges(n, [(i, j) for i in range(n) for j in range(n) if i != j])


def infected_const(mask_by_agent, B):
    m = torch.zeros(B, len(mask_by_agent), dtype=torch.bool)
    for i, v in enumerate(mask_by_agent):
        m[:, i] = bool(v)
    return lambda t: m


def audited_per_round(platform):
    return [r.audited.sum(-1).tolist() for r in platform.log]


# ------------------------------------------------------------------------------------------------ random_alloc
def test_random_alloc_paces_the_audit_budget_evenly_and_spends_all_of_it():
    ctx = make_ctx(star(6), 8, B=2, audit_cap=4)
    fp = FakePlatform(ctx)
    fp.run(RandomAlloc())
    # cumulative allowance floor(cap * t / T) = floor(t / 2): one audit at t = 2, 4, 6, 8
    assert audited_per_round(fp) == [[0, 0], [1, 1], [0, 0], [1, 1], [0, 0], [1, 1], [0, 0], [1, 1]]
    assert fp.rem_audit.tolist() == [0.0, 0.0]


def test_random_alloc_with_a_fractional_cap_never_rounds_up():
    ctx = make_ctx(star(6), 10, audit_cap=2.4)
    fp = FakePlatform(ctx)
    fp.run(RandomAlloc())
    assert sum(r.audited.sum().item() for r in fp.log) == 2          # floor(2.4); the platform would refuse a third


def test_allocation_requests_no_more_audits_than_the_remaining_audit_budget_holds():
    ctx = make_ctx(star(6), 2, audit_cap=12)                          # paced allowance 6 per round ...
    fp = FakePlatform(ctx)
    fp.rem_audit = torch.tensor([2.0], dtype=torch.float64)           # ... but only 2 units are left
    fp.run(RandomAlloc())
    assert sum(r.requested.sum().item() for r in fp.log) == 2 and sum(r.audited.sum().item() for r in fp.log) == 2


def test_random_alloc_uncapped_audits_every_active_sender_each_round():
    ctx = make_ctx(star(6), 3)
    fp = FakePlatform(ctx)
    fp.run(RandomAlloc())
    assert all(r.audited.all() for r in fp.log)


def test_random_alloc_samples_agents_uniformly():
    B = 3000
    ctx = make_ctx(star(6), 4, B=B, audit_cap=4)                      # one audit at round 1
    fp = FakePlatform(ctx)
    fp.run(RandomAlloc(), rounds=1)
    counts = fp.log[0].audited.sum(0)
    assert counts.sum().item() == B
    assert counts.min().item() > 400 and counts.max().item() < 600  # expected 500, sd about 20


def test_random_alloc_is_reproducible_and_independent_of_batch_size():
    ids = ("a", "b", "c", "d")
    full = FakePlatform(make_ctx(star(6), 8, audit_cap=3, B=4, unit_ids=ids, arm="random_alloc"))
    full.run(RandomAlloc())
    single = FakePlatform(make_ctx(star(6), 8, audit_cap=3, B=1, unit_ids=("c",), arm="random_alloc"))
    single.run(RandomAlloc())
    again = FakePlatform(make_ctx(star(6), 8, audit_cap=3, B=4, unit_ids=ids, arm="random_alloc"))
    again.run(RandomAlloc())
    for rf, rs, ra in zip(full.log, single.log, again.log, strict=True):
        assert torch.equal(rf.audited[2], rs.audited[0]) and torch.equal(rf.audited, ra.audited)
    assert not all(torch.equal(r.audited[0], r.audited[1]) for r in full.log)      # episodes differ from each other


def test_random_alloc_blocks_only_audit_flagged_senders_and_reports_the_audit_score():
    ctx = make_ctx(star(6), 3, thresholds=T0)
    fp = FakePlatform(ctx, infected=infected_const([0, 0, 1, 0, 0, 0], 1))
    fp.run(RandomAlloc())
    q = fp.log[0].actions.edge_quarantine                              # D98: a flag is a persistent edge quarantine
    assert q.dtype == torch.bool and q.shape == (1, 6, 6)
    assert torch.equal(q[0, 2], ctx.adj[0, 2]) and q[0, [0, 1, 3, 4, 5]].sum() == 0
    s = fp.log[0].actions.statistics["audit_score"]
    assert s.dtype == torch.float64 and s[0, 2].item() == 3.0 and s[0, 1].item() == -3.0


@pytest.mark.parametrize("cls", [RandomAlloc, BraGreedy])
def test_a_flagged_sender_stays_blocked_for_the_rest_of_the_episode_and_is_charged_once(cls):
    ctx = make_ctx(star(6), 4, thresholds=T0)
    fp = FakePlatform(ctx, infected=infected_const([0, 0, 1, 0, 0, 0], 1))
    fp.run(cls())
    flagged_round = next(r for r in fp.log if r.actions.edge_quarantine[0, 2].any())
    assert flagged_round.charged.item() == 1 * (4 - flagged_round.t + 1)    # out-degree 1 x remaining rounds
    assert fp.blocked[0, 2].sum() == 1
    for r in fp.log:
        if r.t > flagged_round.t:
            assert not r.actions.edge_quarantine[0, 2].any() and not r.audited[0, 2]   # never re-audited, never re-charged


def test_without_a_threshold_the_arm_audits_and_reports_but_never_acts():
    ctx = make_ctx(star(6), 3)                                        # empty thresholds = calibration
    fp = FakePlatform(ctx, infected=infected_const([1, 1, 1, 1, 1, 1], 1))
    fp.run(RandomAlloc())
    assert all(not r.actions.edge_quarantine.any() for r in fp.log)
    assert all(not r.actions.statistics["audit_score"].isnan().any() for r in fp.log)


def test_unaudited_senders_have_nan_scores():
    ctx = make_ctx(star(6), 4, audit_cap=4)
    fp = FakePlatform(ctx)
    fp.run(RandomAlloc())
    s = fp.log[1].actions.statistics["audit_score"]                   # round 2: one audit
    assert (~s.isnan()).sum().item() == 1


def test_blocking_stays_inside_the_intervention_budget_and_skips_what_does_not_fit():
    # hub 0 has 5 outgoing deliveries, leaf 1 has 1; both infected; budget 3 cannot pay for the hub
    ctx = make_ctx(star(6), 1, thresholds=T0, intervention_cap=3)
    fp = FakePlatform(ctx, infected=infected_const([1, 1, 0, 0, 0, 0], 1))
    fp.run(RandomAlloc())                                             # the platform asserts no violation
    d = fp.log[0].actions.edge_quarantine[0]
    assert d[0].sum() == 0 and torch.equal(d[1], fp.log[0].obs.attempts[0, 1])
    assert fp.rem_int.item() == 2.0


def test_flagged_senders_are_blocked_highest_audit_score_first():
    ctx = make_ctx(star(6), 1, thresholds=T0, intervention_cap=1)       # one delivery: room for a single leaf
    scores = torch.tensor([[0.0, 1.0, 2.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    fp = FakePlatform(ctx, infected=infected_const([0, 1, 1, 0, 0, 0], 1), audit_override=scores)
    d = fp.run(RandomAlloc())[0].actions.edge_quarantine[0]
    assert d[2].sum() == 1 and d[1].sum() == 0                          # leaf 2 scored higher


def test_allocation_arms_refuse_a_missing_cost():
    ctx = make_ctx(star(4), 2, costs={"inspect": 1.0})
    with pytest.raises(core.Unset, match="edge_quarantine"):
        RandomAlloc().reset(ctx)


# ------------------------------------------------------------------------------------------------ oracle_alloc
def test_oracle_alloc_audits_exactly_the_infected_senders_and_nobody_else():
    inf = infected_const([0, 1, 0, 0, 1, 0], 1)
    ctx = make_ctx(star(6), 3, thresholds=T0)
    fp = FakePlatform(ctx, infected=inf)
    fp.run(OracleAlloc())
    assert torch.equal(fp.log[0].audited, inf(0))                      # round 1: exactly the infected senders
    assert torch.equal(fp.log[0].actions.edge_quarantine.any(-1), inf(0))
    assert not any(r.audited.any() for r in fp.log[1:])                # they are cut: nothing left to audit


def test_oracle_alloc_verdicts_go_through_the_auditor_and_its_threshold():
    # D96: the oracle points the audits, the auditor decides. A low verdict lets an infected sender through,
    # and without a threshold the arm audits and never acts.
    low = torch.full((1, 6), -1.0, dtype=torch.float64)
    ctx = make_ctx(star(6), 2, thresholds=T0)
    fp = FakePlatform(ctx, infected=infected_const([0, 1, 0, 0, 0, 0], 1), audit_override=low)
    fp.run(OracleAlloc())
    assert all(not r.actions.edge_quarantine.any() for r in fp.log) and fp.log[0].audited[0, 1]
    fp = FakePlatform(make_ctx(star(6), 2), infected=infected_const([0, 1, 0, 0, 0, 0], 1))
    fp.run(OracleAlloc())
    assert all(not r.actions.edge_quarantine.any() for r in fp.log) and fp.log[0].audited[0, 1]
    fp = FakePlatform(make_ctx(star(6), 2, thresholds=T0), infected=infected_const([0, 1, 0, 0, 0, 0], 1))
    fp.run(OracleAlloc())
    assert torch.equal(fp.log[0].actions.edge_quarantine[0, 1], fp.log[0].obs.attempts[0, 1])


def test_oracle_alloc_does_not_block_what_the_audit_could_not_pay_for():
    ctx = make_ctx(star(6), 2, audit_cap=0, thresholds=T0)
    fp = FakePlatform(ctx, infected=infected_const([0, 1, 0, 0, 0, 0], 1))
    fp.run(OracleAlloc())
    assert all(not r.actions.edge_quarantine.any() for r in fp.log)


def test_oracle_alloc_spends_no_audit_unit_when_nothing_is_infected():
    ctx = make_ctx(star(6), 4, audit_cap=8)
    fp = FakePlatform(ctx)
    fp.run(OracleAlloc())
    assert fp.rem_audit.item() == 8.0


def test_oracle_alloc_respects_the_audit_pacing():
    ctx = make_ctx(star(6), 4, audit_cap=2)                           # floor(2 t / 4): audits at t = 2 and 4
    fp = FakePlatform(ctx, infected=infected_const([1, 1, 1, 1, 1, 1], 1))
    fp.run(OracleAlloc())
    assert audited_per_round(fp) == [[0], [1], [0], [1]]


def test_oracle_alloc_needs_truth_and_other_arms_do_not():
    assert OracleAlloc.needs_truth is True
    assert RandomAlloc.needs_truth is False and BraGreedy.needs_truth is False


# -------------------------------------------------------------------------------------------------- bra_greedy
def brute_weight(adj, i, r, T, last):
    """Weight of agent i's message at round r by explicit enumeration of the unrolled dependency DAG."""
    n = adj.shape[0]
    nodes = {(i, r)}
    frontier = {(i, r)}
    for s in range(r + 1, T + 1):
        frontier = {(j, s) for (u, _) in frontier for j in range(n) if j == u or adj[u, j]}
        nodes |= frontier
    desc = len(nodes) - 1
    return (r - last) * (1 + desc)


def test_bra_greedy_ranks_by_rounds_since_last_audit_times_downstream_nodes():
    adj = chain(4)                                                    # 0 -> 1 -> 2 -> 3, agent 3 never sends
    ctx = make_ctx(adj, 4, audit_cap=4)                               # exactly one audit per round
    fp = FakePlatform(ctx)
    fp.run(BraGreedy())
    last = {0: 0, 1: 0, 2: 0}
    picked = []
    for r in fp.log:
        w = {i: brute_weight(adj, i, r.t, 4, last[i]) for i in last}
        chosen = r.audited[0].nonzero().flatten().tolist()
        assert chosen == [max(w, key=w.get)]
        last[chosen[0]] = r.t
        picked.append(chosen[0])
    assert picked == [0, 1, 2, 0]                                     # hand computation, see the CONTRACT


def test_bra_greedy_ties_are_broken_by_the_arm_stream():
    B = 600
    ctx = make_ctx(complete(6), 3, B=B, audit_cap=3)                  # symmetric: every weight is equal in round 1
    fp = FakePlatform(ctx)
    fp.run(BraGreedy(), rounds=1)
    counts = fp.log[0].audited.sum(0)
    assert counts.sum().item() == B and counts.min().item() > 60 and counts.max().item() < 140   # expected 100


def test_bra_greedy_blocks_flagged_senders_and_stays_inside_both_budgets():
    ctx = make_ctx(star(6), 6, thresholds=T0, audit_cap=6, intervention_cap=4)
    fp = FakePlatform(ctx, infected=infected_const([0, 1, 1, 0, 0, 0], 1))
    fp.run(BraGreedy())                                               # the platform asserts no budget violation
    assert fp.rem_audit.item() >= 0 and fp.rem_int.item() >= 0
    assert sum(r.audited.sum().item() for r in fp.log) <= 6


# ---------------------------------------------------------------------------------------------- shared behaviour
@pytest.mark.parametrize("cls", [RandomAlloc, OracleAlloc, BraGreedy])
def test_state_roundtrip_through_the_checkpoint_store_resumes_exactly(cls, tmp_path):
    def world():
        return make_ctx(star(6), 6, B=3, audit_cap=5, intervention_cap=20, thresholds=T0, arm=cls.name)

    inf = infected_const([0, 1, 1, 0, 0, 0], 3)
    ref = FakePlatform(world(), infected=inf)
    ref.run(cls())
    part = FakePlatform(world(), infected=inf)
    d1 = cls()
    d1.reset(part.ctx)
    for t in (1, 2, 3):
        part.step(d1, t)
    store = core.CheckpointStore(tmp_path)
    store.save_state("u", d1.state_dict())
    d2 = cls()
    d2.reset(part.ctx)
    d2.load_state_dict(store.load_state("u"))
    for t in (4, 5, 6):
        part.step(d2, t)
    for a, b in zip(ref.log, part.log, strict=True):
        assert torch.equal(a.audited, b.audited)
        assert torch.equal(a.actions.edge_quarantine, b.actions.edge_quarantine)


@pytest.mark.parametrize("cls", [RandomAlloc, OracleAlloc, BraGreedy])
def test_outputs_have_the_contract_dtypes(cls):
    ctx = make_ctx(star(5), 2, thresholds=T0)
    fp = FakePlatform(ctx, infected=infected_const([0, 1, 0, 0, 0], 1))
    fp.run(cls())
    a = fp.log[0].actions
    assert a.edge_quarantine.dtype == torch.bool and a.statistics["audit_score"].dtype == torch.float64
    assert cls().name == cls.name
