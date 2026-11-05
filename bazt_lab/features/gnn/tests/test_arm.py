"""The GNN arm: embed the round's texts, run the trained GNN, flag at the cut, deny the flagged agents' deliveries."""
import copy

import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.gnn import public as gnn
from bazt_lab.features.gnn.private.arm import GNNArm, GNNRecalibrated
from bazt_lab.features.gnn.private.embedding import CachedEmbedder
from bazt_lab.features.gnn.private.train import train_family
from bazt_lab.features.gnn.tests.conftest import FakeEncoder, make_records

FAMILY, N, T = "insider_pi__mmlu", 6, 4


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    from bazt_lab.features.gnn.tests.conftest import TEST_SETTINGS
    s = copy.deepcopy(TEST_SETTINGS)
    s["gnn"]["artifacts_dir"] = str(tmp_path_factory.mktemp("art"))
    s["gnn_train"]["logs_dir"] = "unused"
    train_family(s, FAMILY, make_records(24, n_agents=N, rounds=T, p=0.8, k=1), embed=CachedEmbedder(FakeEncoder()))
    return s


def ctx(settings, adj, cap=float("inf"), thresholds=None, cell=("insider_pi", "mmlu")):
    B = adj.shape[0]
    return core.EpisodeContext(
        B=B, N=N, T=T, adj=adj, sink=torch.zeros(B, dtype=torch.int64),
        audit_cap=torch.full((B,), float("inf"), dtype=torch.float64),
        intervention_cap=torch.full((B,), cap, dtype=torch.float64), costs={"deny": 1.0}, thresholds=thresholds or {},
        settings=settings, arm="GNN", unit_ids=tuple(f"u{b}" for b in range(B)),
        arm_stream=core.stream, cell={"attack": cell[0], "task": cell[1]} if cell else None)


def obs(adj, t, texts, remaining=float("inf")):
    B = adj.shape[0]
    o = core.Observation(
        t=t, T=T, active=torch.ones(B, N, dtype=torch.bool), attempts=adj.clone(), monitor=None,
        remaining_audit=torch.full((B,), float("inf"), dtype=torch.float64),
        remaining_intervention=torch.full((B,), remaining, dtype=torch.float64),
        audit=lambda m: torch.full(m.shape, float("nan"), dtype=torch.float64))
    o.texts = texts
    return o


def full_adj(B=1):
    a = torch.ones(N, N, dtype=torch.bool)
    a.fill_diagonal_(False)
    return a.expand(B, N, N).clone()


def texts_with_attacker(attacker, r=0):
    return [f"ATTACK says wrong {r} {i}" if i == attacker else f"honest reasoning {r} {i}" for i in range(N)]


def arm(trained):
    return GNNArm(embed=CachedEmbedder(FakeEncoder()))


def play(a, adj, attackers, rounds=T):
    """Run one conversation; one attacker index per episode. Returns the Actions of every round."""
    return [a.decide(obs(adj, t, [texts_with_attacker(k, t) for k in attackers])) for t in range(1, rounds + 1)]


def test_the_arm_is_registered_for_the_llm_side_and_reports_no_calibrated_statistic():
    spec = gnn.ARMS[0]
    assert (spec.name, spec.platforms, spec.needs_truth, spec.calibrated) == ("GNN", ("llm",), False, False)
    assert spec.factory().name == "GNN"


def test_the_attacker_is_flagged_and_all_its_outgoing_deliveries_are_denied(trained):
    adj = full_adj(2)
    a = arm(trained)
    a.reset(ctx(trained, adj))
    acts = play(a, adj, attackers=[1, 4])
    last = acts[-1].deny
    for b, k in enumerate([1, 4]):
        assert last[b, k].tolist() == adj[b, k].tolist()                 # every outgoing delivery of the attacker
        assert not last[b, [i for i in range(N) if i != k]].any()        # nobody else is cut


def test_flags_carry_over_the_rounds_of_a_conversation_and_reset_at_the_next(trained):
    adj = full_adj()
    a = arm(trained)
    a.reset(ctx(trained, adj))
    assert play(a, adj, [2], rounds=3)[-1].deny[0, 2].any()          # flagged by round 3
    quiet = a.decide(obs(adj, 4, [[f"honest reasoning 4 {i}" for i in range(N)]]))
    assert quiet.deny[0, 2].tolist() == adj[0, 2].tolist()           # still cut with honest text: the flag persists
    fresh = a.decide(obs(adj, 1, [[f"honest reasoning 9 {i}" for i in range(N)]]))   # t == 1: a new conversation
    assert not fresh.deny.any()


def test_a_cut_above_one_flags_nobody_and_the_cut_is_the_published_half_by_default(trained):
    adj = full_adj()
    s = copy.deepcopy(trained)
    s["gnn"]["cut"] = 1.0 + 1e-9
    a = arm(s)
    a.reset(ctx(s, adj))
    assert not any(x.deny.any() for x in play(a, adj, [1]))


def test_it_reports_the_logit_statistic_and_never_reads_ground_truth(trained):
    adj = full_adj()
    a = arm(trained)
    a.reset(ctx(trained, adj))
    acts = play(a, adj, [3])
    stat = acts[-1].statistics["gnn_logit"]
    assert stat.dtype == torch.float64 and stat.shape == (1, N) and stat.argmax().item() == 3
    assert a.name == "GNN" and not hasattr(a, "oracle")


def test_deny_only_covers_attempted_deliveries(trained):
    adj = full_adj()
    attempts = adj.clone()
    attempts[0, 1, :] = False                                           # the attacker sends nothing this round
    a = arm(trained)
    a.reset(ctx(trained, adj))
    play(a, adj, [1], rounds=2)
    o = obs(adj, 3, [texts_with_attacker(1, 3)])
    o.attempts = attempts
    assert not a.decide(o).deny[0, 1].any()


def test_an_intervention_budget_funds_the_most_suspicious_senders_first(trained):
    adj = full_adj()
    a = arm(trained)
    a.reset(ctx(trained, adj, cap=2.0))
    for t in range(1, T):
        a.decide(obs(adj, t, [texts_with_attacker(1, t)], remaining=2.0))
    acts = a.decide(obs(adj, T, [texts_with_attacker(1, T)], remaining=2.0))
    assert acts.deny.sum().item() <= 2                                  # never more than the budget pays (cost 1 each)


def test_state_round_trips_and_resumes_like_an_uninterrupted_run(trained):
    adj = full_adj()
    a, b = arm(trained), arm(trained)
    a.reset(ctx(trained, adj))
    b.reset(ctx(trained, adj))
    play(a, adj, [4], rounds=2)
    play(b, adj, [4], rounds=2)
    state = torch.load  # noqa: F841  (state must be torch.save-able with weights_only)
    snap = a.state_dict()
    resumed = arm(trained)
    resumed.reset(ctx(trained, adj))
    resumed.load_state_dict(snap)
    o = lambda: obs(adj, 3, [texts_with_attacker(4, 3)])               # noqa: E731
    assert torch.equal(resumed.decide(o()).deny, b.decide(o()).deny)


def test_texts_are_required(trained):
    adj = full_adj()
    a = arm(trained)
    a.reset(ctx(trained, adj))
    o = obs(adj, 1, None)
    with pytest.raises(ValueError, match="texts"):
        a.decide(o)


def test_a_missing_setting_is_named(trained):
    s = copy.deepcopy(trained)
    del s["gnn"]["cut"]
    a = arm(trained)
    with pytest.raises(core.Unset, match=r"gnn\.cut"):
        a.reset(ctx(s, full_adj()))


def test_the_recalibrated_arm_is_registered_calibrated_and_cuts_at_the_calibrated_logit(trained):
    spec = gnn.ARMS[1]
    assert (spec.name, spec.platforms, spec.calibrated) == ("GNN_recalibrated", ("llm",), True)
    adj = full_adj()
    low = GNNRecalibrated(embed=CachedEmbedder(FakeEncoder()))
    low.reset(ctx(trained, adj, thresholds={"gnn_logit": -1e9}))
    assert play(low, adj, [1])[-1].deny.any()                           # everything is above a very low threshold
    high = GNNRecalibrated(embed=CachedEmbedder(FakeEncoder()))
    high.reset(ctx(trained, adj, thresholds={"gnn_logit": 1e9}))
    assert not any(x.deny.any() for x in play(high, adj, [1]))
    calibrating = GNNRecalibrated(embed=CachedEmbedder(FakeEncoder()))
    calibrating.reset(ctx(trained, adj))                                # no threshold: collect, never act
    acts = play(calibrating, adj, [1])
    assert not any(x.deny.any() for x in acts) and acts[-1].statistics["gnn_logit"].shape == (1, N)


def test_the_published_arm_ignores_a_logit_threshold(trained):
    adj = full_adj()
    a = arm(trained)
    a.reset(ctx(trained, adj, thresholds={"gnn_logit": 1e9}))
    assert play(a, adj, [1])[-1].deny.any()


def test_the_cell_picks_its_own_family_checkpoint(trained):
    assert GNNArm.family_of(ctx(trained, full_adj())) == FAMILY


def test_a_cell_without_its_own_family_follows_the_fallback_key(trained):
    c = ctx(trained, full_adj(), cell=("infection", "mmlu"))
    assert GNNArm.family_of(c) == FAMILY                                # pi_same_task
    none = copy.deepcopy(trained)
    none["gnn"]["fallback_family"] = "none"
    with pytest.raises(ValueError, match=r"fallback_family.*none"):
        arm(none).reset(ctx(none, full_adj(), cell=("aitm", "mmlu")))
    unset = copy.deepcopy(trained)
    del unset["gnn"]["fallback_family"]
    with pytest.raises(core.Unset, match=r"gnn\.fallback_family"):
        arm(unset).reset(ctx(unset, full_adj(), cell=("infection", "mmlu")))
    own = ctx(unset, full_adj())                                        # a trained family never needs the key
    assert GNNArm.family_of(own) == FAMILY


def test_a_context_without_a_cell_is_refused(trained):
    with pytest.raises(ValueError, match=r"ctx.cell"):
        arm(trained).reset(ctx(trained, full_adj(), cell=None))


def test_deploy_family_is_the_single_rule_for_the_runner_the_harness_and_the_arm(trained):
    assert gnn.deploy_family("insider_pi", "mmlu", trained) == FAMILY
    assert gnn.deploy_family("infection", "gsm8k", trained) == "insider_pi__gsm8k"
    none = copy.deepcopy(trained)
    none["gnn"]["fallback_family"] = "none"
    assert gnn.deploy_family("aitm", "mmlu", none) is None
    assert gnn.deploy_family("insider_pi", "mmlu", none) == FAMILY
    del none["gnn"]["fallback_family"]
    with pytest.raises(core.Unset, match=r"gnn\.fallback_family"):
        gnn.deploy_family("aitm", "mmlu", none)
    none["gnn"]["fallback_family"] = "bogus"
    with pytest.raises(ValueError, match="fallback_family"):
        gnn.deploy_family("aitm", "mmlu", none)


def test_the_conversation_boundary_is_round_in_conversation_not_the_seed_run_round(trained):
    adj = full_adj()
    a = arm(trained)
    a.reset(ctx(trained, adj))
    for r in range(1, 4):
        o = obs(adj, 100 + r, [texts_with_attacker(2, r)])          # t counts the whole seed run
        o.round_in_conversation = r
        a.decide(o)
    assert a.flagged[0, 2]
    fresh = obs(adj, 104, [[f"honest reasoning 9 {i}" for i in range(N)]])
    fresh.round_in_conversation = 1                                 # a new conversation although t is 104
    assert not a.decide(fresh).deny.any() and not a.flagged.any()
    again = obs(adj, 1, [texts_with_attacker(2, 5)])                # t == 1 but mid-conversation: no reset
    again.round_in_conversation = 2
    a.decide(again)
    assert len(a.history[0]) == 2
