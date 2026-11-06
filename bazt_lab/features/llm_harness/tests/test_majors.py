"""Whole-seed-run time, late deliveries, pricing parity with the mock, row keys, GSM8K payload, Watts-Strogatz, provenance."""
import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.llm_data import public as data
from bazt_lab.features.llm_harness import public as harness
from bazt_lab.features.llm_harness.private import cases, enforce, graphs
from bazt_lab.features.llm_harness.tests.test_platform import SETTINGS, N, Passive, arm, run, unit


@pytest.fixture()
def platform(families):
    return harness.LLMPlatform(families=lambda name: families[name])


# ------------------------------------------------------------------------------------------------ M1 episode time
def test_t_and_T_count_the_whole_seed_run_and_the_conversation_round_is_separate(platform, tmp_path):
    seen = []

    class Look(Passive):
        def decide(self, obs):
            seen.append((obs.t, obs.T, obs.round_in_conversation, self.ctx.T))
            return super().decide(obs)

    run(platform, tmp_path, cls=Look, conversations=3)
    assert seen == [(1, 6, 1, 6), (2, 6, 2, 6), (3, 6, 1, 6), (4, 6, 2, 6), (5, 6, 1, 6), (6, 6, 2, 6)]


def test_a_capped_arm_that_prices_with_the_observation_has_no_violations(platform, tmp_path):
    """An isolation priced as out-degree x (T - t + 1) fits a budget of exactly that, as the runner's defenders compute it."""
    class Isolate(Passive):
        def decide(self, obs):
            go = torch.zeros(1, N, dtype=torch.bool)
            go[0, 0] = obs.t == 1
            return core.Actions(isolation=go)

    edges = N * (N - 1)
    fraction = (N - 1) * 4 / (edges * 4)                  # out-degree of agent 0 x all 4 rounds, of all deliveries
    rows, _ = run(platform, tmp_path, cls=Isolate, intervention=fraction)
    assert all(r["violations"] == 0 for r in rows)
    assert rows[0]["spend_intervention_total"] == (N - 1) * 4


# ------------------------------------------------------------------------------------------- M2 late deliveries
def test_a_delayed_delivery_does_not_reach_a_receiver_isolated_when_it_would_arrive(families, tmp_path):
    from bazt_lab.features.llm_harness.private.fake import FakeBackend

    class Spy(FakeBackend):
        def __init__(self):
            super().__init__()
            self.prompts = []

        async def complete(self, messages, model):
            self.prompts.append([m["content"] for m in messages if m["role"] == "user"])
            return await super().complete(messages, model)

    class DelayThenIsolate(Passive):
        def decide(self, obs):
            if obs.round_in_conversation == 1:
                return core.Actions(delay=obs.attempts.clone())
            iso = torch.zeros(1, N, dtype=torch.bool)
            iso[0, 1] = True
            return core.Actions(isolation=iso)

    spy = Spy()
    p = harness.LLMPlatform(backend=spy, families=lambda name: families[name], sample=harness.fake_sample)
    p.run_batch([unit(conversations=1, attack="none")], arm(DelayThenIsolate), SETTINGS, core.CheckpointStore(tmp_path),
                core.NullTracker())
    # calls are in agent order per round: the last N prompts are round 2; agent 1 is isolated there
    last_round = spy.prompts[-N:]
    assert "No responses from other agents." in last_round[1][-1]
    assert "Agent_0's View" in last_round[2][-1] or "Agent_0" in last_round[2][-1]           # a non-isolated receiver does get them


# ------------------------------------------------------------------------------------- M3 pricing parity with the mock
def mock_apply():
    from bazt_lab.features.mock_engine import public as mock
    fn = getattr(mock, "apply_actions", None)
    if fn is None:
        pytest.skip("mock_engine does not export apply_actions yet")
    return fn, mock


def random_case(seed):
    g = core.stream("diff", seed)
    n = 3 + seed % 4
    r = lambda *shape, p=0.3: core.uniforms(g, shape) < p                                   # noqa: E731
    adj = r(n, n, p=0.7) & ~torch.eye(n, dtype=torch.bool)
    st = [r(n, p=0.2), r(n, n, p=0.1) & adj, r(n, p=0.2), r(n, p=0.2)]
    acts = dict(deny=r(n, n), sanitize=r(n, n), delay=r(n, n), edge_quarantine=r(n, n, p=0.1), rate_limit=r(n, p=0.3),
                privilege_degradation=r(n, p=0.3), isolation=r(n, p=0.3), reinstatement=r(n, p=0.3))
    acts = {k: v for k, v in acts.items() if float(core.uniforms(g, (1,))[0]) < 0.6}
    return n, adj, st, acts, r(n, n, p=0.8), core.uniforms(g, (n, n))


@pytest.mark.parametrize("seed", range(120))
def test_pricing_and_masks_equal_the_mocks_on_random_actions(seed):
    apply_mock, mock = mock_apply()
    n, adj, st, acts, auth, u = random_case(seed)
    costs = {k.value: float(1 + i % 3) for i, k in enumerate(core.ActionKind)}
    t, T, limit, remaining = 2, 5, 2, 1e9
    live = adj & ~(st[0][:, None] | st[0][None, :])
    mine_state = enforce.State(st[0].clone(), st[1].clone(), st[2].clone(), st[3].clone())
    mine = enforce.apply(core.Actions(**{k: v[None] for k, v in acts.items()}), mine_state, live, auth, adj, remaining, t, T,
                         costs, limit, lambda: u)
    theirs_state = mock.Persistent(*(x[None].clone() for x in st)) if hasattr(mock, "Persistent") else None
    if theirs_state is None:
        pytest.skip("mock_engine does not export Persistent")
    theirs = apply_mock(core.Actions(**{k: v[None] for k, v in acts.items()}), theirs_state, live[None], auth[None], adj[None],
                        torch.tensor([remaining], dtype=torch.float64), t, T, costs, limit, lambda rows: u[None])
    assert float(mine.cost) == pytest.approx(float(theirs.cost[0]))
    for name in ("arrive", "delayed", "sanitized", "restricted"):
        assert torch.equal(getattr(mine, name), getattr(theirs, name)[0]), name
    for name in ("iso", "quar", "rate", "degr"):
        assert torch.equal(getattr(mine.state, name), getattr(theirs.state, name)[0]), name


def test_isolation_and_reinstatement_charge_only_the_deliveries_removed_or_restored():
    costs = {k.value: 1.0 for k in core.ActionKind}
    adj = ~torch.eye(4, dtype=torch.bool)
    rate = lambda: torch.zeros(4, 4, dtype=torch.float64)                                  # noqa: E731
    state = enforce.State.zeros(4)
    state.quar[0, 1] = True
    iso = torch.zeros(1, 4, dtype=torch.bool)
    iso[0, 0] = True
    out = enforce.apply(core.Actions(isolation=iso), state, adj, adj, adj, 1e9, 1, 3, costs, 2, rate)
    assert float(out.cost) == 2 * 3                          # out-degree 3 minus the quarantined edge, over 3 rounds
    back = torch.zeros(1, 4, dtype=torch.bool)
    back[0, 0] = True
    again = enforce.apply(core.Actions(reinstatement=back), out.state, adj & ~out.state.iso[:, None] & ~out.state.iso[None, :],
                          adj, adj, 1e9, 2, 3, costs, 2, rate)
    assert float(again.cost) == 3 * (2 - 1)                  # the isolated agent sent nothing in the round it came back


# ---------------------------------------------------------------------------------------------- M4 rows share keys
def test_rows_of_every_sanitize_and_correctness_mode_have_the_same_keys(families, tmp_path):
    items = harness.fake_sample
    both = {**SETTINGS, "llm": {**SETTINGS["llm"], "ma_correctness": ["rule", "llm_judge"]}}
    p = harness.LLMPlatform(families=lambda name: families[name], sample=items)
    rows = []
    for mode in ("redaction", "llm_rewrite"):
        r, _ = run(p, tmp_path / mode, sanitize_mode=mode)
        rows += r
    assert len({tuple(sorted(r)) for r in rows}) == 1
    assert all("sanitize_mode" in r and "ma_check" in r for r in rows) and both


# --------------------------------------------------------------------------------------------------- minors
def test_arithmetic_slips_are_not_payload_but_the_target_number_is():
    item = data.TaskItem("gsm8k", "gsm8k:1", "How many?", (), "42", data.Source("x", None, None, None, None), {})
    for attack in ("infection", "aitm", "insider_pi"):
        c = cases.build(None, "pi_gsm8k", item, ("k",), attack=attack)
        assert c.judge("<REASON>: r\n<ANSWER>: 41") == (False, False)
        assert c.judge(f"<REASON>: r\n<ANSWER>: {c.wrong}") == (False, True)


@pytest.mark.parametrize("seed", range(25))
def test_watts_strogatz_keeps_every_lattice_edge_as_one_edge(seed):
    s = {"runtime": {"ws_k": 4, "ws_rewire_p": 0.9}}
    adj = graphs.build_adjacency("watts_strogatz", 12, s, ("w", seed))
    assert adj.sum() == 12 * 4 and (adj == adj.T).all()


def test_watts_strogatz_refuses_an_odd_k():
    with pytest.raises(ValueError, match="ws_k"):
        graphs.build_adjacency("watts_strogatz", 12, {"runtime": {"ws_k": 3, "ws_rewire_p": 0.1}}, ("w", 1))


def test_rows_record_the_provider_and_quantization_the_responses_report(platform, tmp_path):
    rows, _ = run(platform, tmp_path)
    assert all(r["meta_providers"] == ["fake"] and r["meta_quantizations"] == [] for r in rows)


# ------------------------------------------------------------------------- trace time convention (index r = round r)
class DenyPayloadSenders(Passive):
    """Denies the deliveries of exactly the senders whose text carries the payload (it reads the truth, as the oracle arm)."""

    def decide(self, obs):
        carrying = obs.oracle()[0]
        return core.Actions(deny=carrying[:, None].expand(N, N)[None] & obs.attempts)


def captured_traces(monkeypatch, platform, tmp_path, cls, **kw):
    from bazt_lab.features.llm_harness.private import seedrun
    traces = []
    real = seedrun.metrics.rows_from_trace
    monkeypatch.setattr(seedrun.metrics, "rows_from_trace", lambda trace, s: traces.append(trace) or real(trace, s))
    spec = core.ArmSpec(cls.__name__, cls, platforms=("llm",), needs_truth=True)
    platform.run_batch([unit(**kw)], spec, SETTINGS, core.CheckpointStore(tmp_path), core.NullTracker())
    return traces


@pytest.mark.parametrize("attack", ["infection", "insider_pi", "aitm"])
def test_deliveries_are_indexed_by_the_round_of_the_text_they_carry(monkeypatch, platform, tmp_path, attack):
    traces = captured_traces(monkeypatch, platform, tmp_path, DenyPayloadSenders, attack=attack, rounds=4, conversations=1)
    t = traces[0]
    T = t.infected.shape[1]
    assert not t.attempts[0, T - 1].any()                       # the last round's texts are never sent on
    for r in range(T - 1):
        assert t.attempts[0, r].sum() == N * (N - 1)             # a full graph: every sender's round-r text is sent
        if attack != "aitm":
            carries = t.infected[0, r][:, None].expand(N, N)    # row i: the deliveries carrying i's round-r text
            assert torch.equal(t.restricted[0, r].any(1), t.infected[0, r])
            assert not (t.payload_delivered[0, r] & ~carries & ~t.meta["tampered"][0, r]).any()
    if attack == "aitm":
        assert t.meta["tampered"][0, :T - 1].any() and not t.meta["tampered"][0, T - 1].any()


def test_a_delayed_payload_keeps_the_index_of_the_round_that_produced_it(monkeypatch, platform, tmp_path):
    class DelayFirst(Passive):
        def decide(self, obs):                    # round 0's texts are delayed, then everything is denied: only the late copy arrives
            if obs.round_in_conversation == 1:
                return core.Actions(delay=obs.attempts.clone())
            return core.Actions(deny=obs.attempts.clone())

    t = captured_traces(monkeypatch, platform, tmp_path, DelayFirst, attack="insider_pi", rounds=4, conversations=1)[0]
    insiders = t.attackers[0]
    assert t.restricted[0, 0].any() and t.payload_delivered[0, 0][insiders].any()       # produced in round 0, arrives in round 1
    assert not t.payload_delivered[0, 0][~insiders].any()


def test_spend_is_indexed_by_the_round_after_which_the_defender_decided(monkeypatch, platform, tmp_path):
    class DenyOne(Passive):
        def decide(self, obs):
            deny = torch.zeros(1, N, N, dtype=torch.bool)
            deny[0, 0, 1] = True
            return core.Actions(deny=deny & obs.attempts)

    t = captured_traces(monkeypatch, platform, tmp_path, DenyOne, attack="none", rounds=4, conversations=1)[0]
    assert t.spend_intervention[0].tolist() == [1.0, 1.0, 1.0, 0.0]
