"""GSM8K insider payload at every hop, one number normalisation, and the live client closed in its own loop."""
import asyncio
import gc
import logging
import weakref

import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.llm_data import public as data
from bazt_lab.features.llm_harness import public as harness
from bazt_lab.features.llm_harness.private import cases, seedrun
from bazt_lab.features.llm_harness.private.backend import Completion
from bazt_lab.features.llm_harness.private.fake import FakeBackend
from bazt_lab.features.llm_harness.private.seedrun import SeedRun
from bazt_lab.features.llm_harness.tests.test_majors import captured_traces
from bazt_lab.features.llm_harness.tests.test_platform import SETTINGS, Passive, SanitizeAll, unit


def hub_seed(families, tmp_path) -> int:
    """A seed whose hub_star placement leaves the hub (agent 0) to a normal agent: the other normal leaves hear the
    insiders only through the hub, at the second hop."""
    p = harness.LLMPlatform(families=lambda name: families[name], sample=harness.fake_sample)
    env = p._environment(SETTINGS, core.CheckpointStore(tmp_path))
    for seed in range(1, 50):
        r = SeedRun(unit(topology="hub_star", seed=seed, task="gsm8k"), core.ArmSpec("p", Passive, platforms=("llm",)),
                    SETTINGS, env, core.CheckpointStore(tmp_path), calibrating=False)
        if 0 not in r.insiders:
            return seed
    raise AssertionError("no hub_star seed with a normal hub")


def hub_traces(families, monkeypatch, tmp_path, task, backend=None, **kw):
    seed = hub_seed(families, tmp_path / "seed")
    p = harness.LLMPlatform(backend=backend or FakeBackend(), families=lambda name: families[name],
                            sample=harness.fake_sample)
    return captured_traces(monkeypatch, p, tmp_path / task, Passive, attack="insider_pi", task=task, topology="hub_star",
                           seed=seed, rounds=4, conversations=1, **kw)[0]


# ------------------------------------------------------------------------------------------ R3-M1: the second hop
@pytest.mark.parametrize("task", ["gsm8k", "mmlu"])
def test_an_agent_adopting_the_insiders_answer_through_a_normal_neighbour_is_infected(families, monkeypatch, tmp_path,
                                                                                       task):
    t = hub_traces(families, monkeypatch, tmp_path, task)
    victims = ~t.attackers[0]
    assert victims[0]
    leaves = [j for j in range(1, victims.numel()) if victims[j]]
    assert leaves
    # the hub adopts after the insiders' round-0 views (round 1); the normal leaves after the hub's round-1 view (round 2)
    assert not t.correct[0, 1, 0] and all(not t.correct[0, 2, j] for j in leaves)
    for j in leaves:
        assert t.infected[0, 2, j], f"leaf {j} adopted the insiders' answer through the hub but is not infected"
        assert t.payload_delivered[0, 2, j, 0], f"leaf {j}'s round-2 delivery to the hub does not carry the payload"
    wrong = (~t.correct[0]) & victims[None, :]
    assert torch.equal(t.infected[0] & victims[None, :], wrong)                     # wrong == infected at every round


# ------------------------------------------------------------------------------------- one number normalisation
@pytest.mark.parametrize("a, b", [("72.0", "72"), ("18.50", "18.5"), ("1,234.", "1234"), ("$5", "5"), ("-0.0", "0"),
                                  ("100", "1E+2"), ("007", "7")])
def test_equal_numbers_have_one_canonical_form(a, b):
    assert cases.canonical_number(a) == cases.canonical_number(b) is not None
    assert "E" not in cases.canonical_number(a)


def test_what_is_not_a_number_has_no_canonical_form():
    assert cases.canonical_number("abc") is None and cases.canonical_number("") is None
    assert cases.canonical_number("nan") is None and cases.canonical_number("inf") is None


@pytest.mark.parametrize("view, reply", [({"answer": "72.0", "reason": "r"}, "<ANSWER>: 72"),
                                         ("I am sure it is 18.50.", "<ANSWER>: 18.5"),
                                         ({"answer": None, "reason": "it makes 1,234 apples"}, "<ANSWER>: $1234.00")])
def test_an_asserted_number_and_a_replied_number_compare_in_one_form(view, reply):
    assert seedrun.asserted_number(view) == cases.number_of(reply)


def test_a_gsm8k_target_written_with_decimals_is_still_the_target():
    item = data.TaskItem("gsm8k", "gsm8k:1", "How many?", (), "42", data.Source("x", None, None, None, None), {})
    for attack in ("infection", "aitm"):
        c = cases.build(None, "pi_gsm8k", item, ("k",), attack=attack)
        assert c.judge(f"<REASON>: r\n<ANSWER>: {c.wrong}.0") == (False, True)
        assert c.judge("<REASON>: r\n<ANSWER>: 42.0") == (True, False)


class ProseInsiders(FakeBackend):
    """GSM8K insiders answer in prose with `insider_says(gold)`; a normal agent shown a prose number answers
    `victim_says(that number)`."""

    def __init__(self, insider_says, victim_says):
        super().__init__()
        self.insider_says, self.victim_says = insider_says, victim_says

    async def complete(self, messages, model):
        system = messages[0]["content"]
        users = [m["content"] for m in messages if m["role"] == "user"]
        gold = int(self.info["correct"])
        if "disruptor" in system:
            return Completion(f"I am sure the answer is {self.insider_says(gold)}.", 4, 4, provider="fake")
        if len(users) > 1 and "Intercepted" not in system[:40] and "the answer is" in users[-1]:
            return Completion(f"<UPDATED_REASON>: they agree\n<ANSWER>: {self.victim_says(gold)}", 4, 4, provider="fake")
        return await super().complete(messages, model)


def test_a_victim_writing_the_insiders_number_in_another_form_is_infected(families, monkeypatch, tmp_path):
    backend = ProseInsiders(lambda gold: f"{gold + 13}.0", lambda gold: str(gold + 13))
    p = harness.LLMPlatform(backend=backend, families=lambda name: families[name], sample=harness.fake_sample)
    t = captured_traces(monkeypatch, p, tmp_path, Passive, attack="insider_pi", task="gsm8k", rounds=3,
                        conversations=1)[0]
    victims = ~t.attackers[0]
    assert not t.correct[0, -1][victims].any() and t.infected[0, -1][victims].all()


def test_an_insider_asserting_the_gold_in_another_form_is_not_payload(families, monkeypatch, tmp_path):
    backend = ProseInsiders(lambda gold: f"{gold}.00", lambda gold: f"{gold}.00")
    p = harness.LLMPlatform(backend=backend, families=lambda name: families[name], sample=harness.fake_sample)
    t = captured_traces(monkeypatch, p, tmp_path, Passive, attack="insider_pi", task="gsm8k", rounds=3,
                        conversations=1)[0]
    victims = ~t.attackers[0]
    assert t.correct[0, -1][victims].all() and not t.infected[0, 1:][:, victims].any()


def test_a_rewritten_view_that_still_asserts_the_insiders_number_carries_the_payload(families, monkeypatch, tmp_path):
    """llm_rewrite keeps the facts of a sanitized view: an insider's number the upstream prompt never fixed (not the drawn
    wrong value) survives, and the delivery carries the payload; redaction removes it."""
    for mode in ("llm_rewrite", "redaction"):
        backend = ProseInsiders(lambda gold: gold + 13, lambda gold: gold + 13)
        p = harness.LLMPlatform(backend=backend, families=lambda name: families[name], sample=harness.fake_sample)
        t = captured_traces(monkeypatch, p, tmp_path / mode, SanitizeAll, attack="insider_pi", task="gsm8k", rounds=3,
                            conversations=1, sanitize_mode=mode)[-1]
        insiders = t.attackers[0]
        sent = t.payload_delivered[0, 0][insiders]
        attempted = t.attempts[0, 0][insiders]
        assert attempted.any()
        if mode == "llm_rewrite":
            assert torch.equal(sent, attempted)
        else:
            assert not sent.any()


# ------------------------------------------------------------------------------- the live client, closed per loop
ALIVE = []                                           # weak references to every ClosingClient built


class ClosingClient:
    """A client that, like httpx's pool, keeps the loop it first ran in; it records where it is closed."""

    def __init__(self, replies):
        self.replies = replies
        self.loop = asyncio.get_running_loop()
        self.closed_in = []
        self.chat = type("Chat", (), {"completions": self})()
        ALIVE.append(weakref.ref(self))

    async def create(self, **kw):
        assert asyncio.get_running_loop() is self.loop
        text = self.replies.pop(0) if self.replies else "<ANSWER>: A"
        return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": text})()})()],
                              "usage": None, "model_extra": {}})()

    async def close(self):
        loop = asyncio.get_running_loop()
        self.closed_in.append((loop is self.loop, loop.is_closed()))
        await asyncio.sleep(0)                       # closing awaits the transports, as httpx does


def closing_backend(tmp_path, made):
    from bazt_lab.features.llm_harness.adapters import live as adapter

    def factory():
        made.append(ClosingClient([]))
        return made[-1]
    backend = adapter.LiveBackend(factory, adapter.open_cache(tmp_path / "c.sqlite"), {"m": "prov"}, 0.0)
    backend.retry_sleep = 0.0
    return backend


def test_each_client_is_closed_in_its_own_loop_when_the_conversation_ends(tmp_path, caplog):
    made = []
    backend = closing_backend(tmp_path, made)

    async def conversation(c):
        await asyncio.gather(*[backend.complete([{"role": "user", "content": f"c{c}-{i}"}], "m") for i in range(4)])
    with caplog.at_level(logging.DEBUG, logger="asyncio"):
        for c in range(3):
            asyncio.run(conversation(c))
    assert len(made) == 3                                                    # one client per loop
    assert all(client.closed_in == [(True, False)] for client in made)        # once, in its loop, before it closed
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]   # no 'Event loop is closed' and the like


def test_no_client_or_loop_outlives_its_conversation(tmp_path):
    made = []
    backend = closing_backend(tmp_path, made)
    ALIVE.clear()
    loops = []

    async def conversation(c):
        loops.append(weakref.ref(asyncio.get_running_loop()))
        await asyncio.gather(*[backend.complete([{"role": "user", "content": f"c{c}-{i}"}], "m") for i in range(3)])
    for c in range(40):
        asyncio.run(conversation(c))
    made.clear()
    gc.collect()
    assert len(backend._clients) == 0
    assert sum(ref() is not None for ref in ALIVE) == 0
    assert sum(ref() is not None for ref in loops) == 0


def test_a_client_without_close_ends_its_conversation_quietly(tmp_path, caplog):
    from bazt_lab.features.llm_harness.adapters import live as adapter

    class Bare(ClosingClient):
        close = None
    backend = adapter.LiveBackend(lambda: Bare([]), adapter.open_cache(tmp_path / "c.sqlite"), {"m": "prov"}, 0.0)
    with caplog.at_level(logging.DEBUG, logger="asyncio"):
        asyncio.run(backend.complete([{"role": "user", "content": "q"}], "m"))
    assert len(backend._clients) == 0 and not [r for r in caplog.records if r.levelno >= logging.WARNING]
