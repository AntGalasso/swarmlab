"""D110: the held-out LLM adaptive check replays the mock's gray-box best response as the attacker's schedule.

The runner attaches `attack_schedule = {clock, start_frac, start_round, rate}` to every unit of the block. A scheduled
attacker (the insiders, the infection's entry agent, the AiTM adversary) attacks from `start_round` on (round 0 = the
initial answers, counted per conversation or over the seed run) and in each of those rounds with probability `rate`
(one keyed draw per seed run and round, the same for every arm); in the other rounds it behaves as a benign agent.
"""
import re

import pytest

from bazt_lab.core import public as core
from bazt_lab.features.llm_harness import public as harness
from bazt_lab.features.llm_harness.private import prompts
from bazt_lab.features.llm_harness.private.fake import FakeBackend
from bazt_lab.features.llm_harness.tests.test_platform import SETTINGS, DenyAll, Passive, unit

ATTACKER_TEXT = "The answer provided by the user is"        # what upstream gives an insider in its first message
DOCUMENT = "A document found by the agent while researching"


class Recording(FakeBackend):
    """FakeBackend that records every agent request: (conversation, agent, round, system prompt, first user message,
    last user message)."""

    def __init__(self):
        super().__init__()
        self.requests, self.conversation = [], -1

    def begin_conversation(self, info):
        super().begin_conversation(info)
        self.conversation += 1

    async def complete(self, messages, model):
        me = re.search(r"agent_(\d+)", messages[0]["content"]) if messages[0]["role"] == "system" else None
        if me:
            users = [m["content"] for m in messages if m["role"] == "user"]
            self.requests.append((self.conversation, int(me.group(1)), len(users) - 1, messages[0]["content"], users[0],
                                  users[-1]))
        return await super().complete(messages, model)


def schedule(start=0, rate=1.0, clock="conversation"):
    return {"clock": clock, "start_frac": None, "start_round": start, "rate": rate}


def traces(families, monkeypatch, tmp_path, backend=None, cls=Passive, **kw):
    from bazt_lab.features.llm_harness.private import seedrun
    monkeypatch.undo()                                   # a second run in one test records only its own traces
    got = []
    real = seedrun.metrics.rows_from_trace
    monkeypatch.setattr(seedrun.metrics, "rows_from_trace", lambda trace, s: got.append(trace) or real(trace, s))
    p = harness.LLMPlatform(backend=backend, families=lambda name: families[name])
    spec = core.ArmSpec(cls.__name__, cls, platforms=("llm",), needs_truth=True)
    rows = p.run_batch([unit(**kw)], spec, SETTINGS, core.CheckpointStore(tmp_path), core.NullTracker())
    return got, rows


def adaptive(**kw):
    return {"condition": "gray_box", "block": "llm_adaptive_check", **kw}


# ------------------------------------------------------------------------------------------------- what runs
def test_the_llm_platform_runs_the_gray_box_condition_only(families, tmp_path):
    assert harness.LLMPlatform().adaptive_conditions == ("gray_box",)
    p = harness.LLMPlatform(families=lambda name: families[name])
    spec = core.ArmSpec("passive", Passive, platforms=("llm",))
    for condition in ("white_box", "embedding_evasion", "attack_rate"):
        with pytest.raises(ValueError, match="gray_box"):
            p.run_batch([unit(id=condition, condition=condition, attack_schedule=schedule())], spec, SETTINGS,
                        core.CheckpointStore(tmp_path / condition), core.NullTracker())


@pytest.mark.parametrize("bad", [None, {"clock": "week", "start_round": 0, "rate": 1.0},
                                 {"clock": "conversation", "start_round": -1, "rate": 1.0},
                                 {"clock": "conversation", "start_round": 0, "rate": 0.0},
                                 {"clock": "conversation", "start_round": 0, "rate": 1.5},
                                 {"clock": "conversation", "start_round": True, "rate": 1.0}])
def test_an_adaptive_unit_without_a_valid_schedule_is_refused(families, tmp_path, bad):
    p = harness.LLMPlatform(families=lambda name: families[name])
    spec = core.ArmSpec("passive", Passive, platforms=("llm",))
    extra = {} if bad is None else {"attack_schedule": bad}
    with pytest.raises(ValueError, match="attack_schedule"):
        p.run_batch([unit(**adaptive(**extra))], spec, SETTINGS, core.CheckpointStore(tmp_path), core.NullTracker())


# ------------------------------------------------------------------------------------------- the schedule
@pytest.mark.parametrize("attack", ["insider_pi", "infection", "aitm"])
def test_starting_at_round_zero_with_rate_one_is_the_plain_attack_request_for_request(families, monkeypatch, tmp_path,
                                                                                       attack):
    plain_backend, replay_backend = Recording(), Recording()
    plain, _ = traces(families, monkeypatch, tmp_path / "plain", plain_backend, attack=attack, rounds=4)
    replay, rows = traces(families, monkeypatch, tmp_path / "replay", replay_backend, attack=attack, rounds=4,
                          **adaptive(attack_schedule=schedule(0, 1.0)))
    assert plain_backend.requests == replay_backend.requests
    for a, b in zip(plain, replay, strict=True):
        for field in ("infected", "correct", "payload_delivered", "delivered"):
            assert (getattr(a, field) == getattr(b, field)).all(), field
        assert (a.meta["tampered"] == b.meta["tampered"]).all()
        assert b.meta["attack_rounds"] == [True] * 4 and a.meta["attack_rounds"] is None
    assert all(r["attack_schedule"] == schedule(0, 1.0) and r["meta_attack_rounds"] == [True] * 4 for r in rows)


def test_insiders_carry_no_payload_before_the_start_round_and_attack_from_it(families, monkeypatch, tmp_path):
    backend = Recording()
    got, _ = traces(families, monkeypatch, tmp_path, backend, attack="insider_pi", rounds=4,
                    **adaptive(attack_schedule=schedule(2, 1.0)))
    for t in got:
        insiders = t.attackers[0]
        assert t.meta["attack_rounds"] == [False, False, True, True]
        assert not t.payload_delivered[0, :2].any()              # nobody carries the attack before round 2
        assert t.payload_delivered[0, 2][insiders].any()         # the insiders' round-2 texts carry it
        assert t.infected[0, 2:, insiders].all()                 # seeds in their attack rounds, as the mock marks them
        assert not t.infected[0, :2, insiders].any()             # before: benign, judged as written
    insiders = got[0].attackers[0].nonzero().flatten().tolist()
    for _, agent, r, system, first, last in backend.requests:
        if agent in insiders:
            attacking = r >= 2
            assert (ATTACKER_TEXT in first) == attacking, (agent, r)            # the target is known when it attacks
            assert ("guide the other agents" in system) == attacking, (agent, r)
            if r > 0:
                assert ("Firmly believe" in last) == attacking, (agent, r)      # upstream's attacker prompt


def test_the_seed_run_clock_counts_the_rounds_of_every_conversation_in_sequence(families, monkeypatch, tmp_path):
    got, _ = traces(families, monkeypatch, tmp_path, attack="insider_pi", rounds=3, conversations=2,
                    **adaptive(attack_schedule=schedule(4, 1.0, clock="seed_run")))
    first, second = got
    assert first.meta["attack_rounds"] == [False] * 3 and not first.payload_delivered.any()
    assert second.meta["attack_rounds"] == [False, True, True]          # seed-run round 4 = round 1 of conversation 1


def test_the_aitm_adversary_rewrites_only_in_attack_rounds(families, monkeypatch, tmp_path):
    got, _ = traces(families, monkeypatch, tmp_path, attack="aitm", rounds=3,
                    **adaptive(attack_schedule=schedule(1, 1.0)))
    for t in got:
        assert not t.meta["tampered"][0, 0].any() and t.meta["tampered"][0, 1].any()
        assert not t.payload_delivered[0, 0].any()


def test_the_infection_document_is_in_the_entry_agents_context_only_in_attack_rounds(families, monkeypatch, tmp_path):
    backend = Recording()
    got, _ = traces(families, monkeypatch, tmp_path, backend, attack="infection", rounds=4,
                    **adaptive(attack_schedule=schedule(2, 1.0)))
    plain = Recording()
    traces(families, monkeypatch, tmp_path / "plain", plain, attack="infection", rounds=4, conversations=1)
    entry = next(a for _, a, r, _, first, _ in plain.requests if r == 0 and DOCUMENT in first)
    assert all(not t.payload_delivered[0, :2].any() for t in got)
    seen = {(c, r): DOCUMENT in first for c, a, r, _, first, _ in backend.requests if a == entry}
    assert seen == {(c, r): r >= 2 for c in range(2) for r in range(4)}
    assert all(prompts.REPLICATION_MARKER not in last for c, a, r, _, _, last in backend.requests if a == entry and r < 2)


def test_attack_rounds_follow_the_rate_with_the_same_draws_for_every_arm(families, monkeypatch, tmp_path):
    kw = dict(attack="aitm", rounds=2, conversations=40, **adaptive(attack_schedule=schedule(0, 0.5)))
    a, _ = traces(families, monkeypatch, tmp_path / "a", **kw)
    b, _ = traces(families, monkeypatch, tmp_path / "b", cls=DenyAll, **kw)
    rounds_a = [t.meta["attack_rounds"] for t in a]
    assert rounds_a == [t.meta["attack_rounds"] for t in b]               # common random numbers across arms
    share = sum(map(sum, rounds_a)) / sum(map(len, rounds_a))
    assert 0.35 <= share <= 0.65
    for t, on in zip(a, rounds_a, strict=True):                           # tampering follows the drawn rounds
        assert bool(t.meta["tampered"][0, 0].any()) == on[0]


def test_an_insider_outside_its_attack_rounds_writes_as_a_benign_agent(families, monkeypatch, tmp_path):
    backend = Recording()
    got, _ = traces(families, monkeypatch, tmp_path, backend, attack="insider_pi", rounds=4, conversations=6,
                    **adaptive(attack_schedule=schedule(0, 0.5)))
    on = {(c, r): v for c, t in enumerate(got) for r, v in enumerate(t.meta["attack_rounds"])}
    assert set(on.values()) == {True, False}
    insiders = got[0].attackers[0].nonzero().flatten().tolist()
    for c, agent, r, system, first, _ in backend.requests:
        if agent in insiders:
            assert ("guide the other agents" in system) == on[(c, r)] and (ATTACKER_TEXT in first) == on[(c, r)]
