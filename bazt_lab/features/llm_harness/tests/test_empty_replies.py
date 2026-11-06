"""D114: an LLM reply that stays empty after the retries is an unanswerable reply, scored wrong, counted in the row
(trace.meta['empty_replies']), and the run continues. Other failures keep their behaviour."""
import asyncio
import re

import pytest

from bazt_lab.core import public as core
from bazt_lab.features.llm_harness import public as harness
from bazt_lab.features.llm_harness.private.backend import Completion
from bazt_lab.features.llm_harness.private.fake import FakeBackend
from bazt_lab.features.llm_harness.tests.test_platform import Passive, unit
from bazt_lab.features.llm_harness.tests.test_round3 import live, probe_settings

Q = [{"role": "user", "content": "q"}]
RETRIES = 4


# ------------------------------------------------------------------------------------------------- live backend
def test_a_reply_empty_after_every_retry_is_an_empty_answer_not_an_error(tmp_path):
    backend, made = live([""] * (RETRIES + 1), [], tmp_path)
    done = asyncio.run(backend.complete(Q, "m"))
    assert done.text == "" and done.prompt_tokens == 3
    assert backend.cache.conn.execute("SELECT count(*) FROM cache").fetchone() == (0,)      # never a normal answer


def test_a_relaunch_asks_a_known_empty_request_once_more(tmp_path):
    backend, _ = live([""] * (RETRIES + 1) + ["", "answer"], [], tmp_path)
    asyncio.run(backend.complete(Q, "m"))
    again = asyncio.run(backend.complete(Q, "m"))                 # one more attempt only, still empty
    assert again.text == ""
    third = asyncio.run(backend.complete(Q, "m"))
    assert third.text == "answer"
    assert backend.cache.conn.execute("SELECT text FROM cache").fetchall() == [("answer",)]


def test_a_known_empty_request_that_answers_on_the_relaunch_is_cached_as_usual(tmp_path):
    from bazt_lab.features.llm_harness.adapters import live as adapter
    backend, _ = live([""] * (RETRIES + 1) + ["late answer"], [], tmp_path)
    asyncio.run(backend.complete(Q, "m"))
    assert asyncio.run(backend.complete(Q, "m")).text == "late answer"
    fresh = adapter.LiveBackend(lambda: None, backend.cache, {"m": "prov"}, 0.0)         # a cache hit needs no client call
    assert asyncio.run(fresh.complete(Q, "m")).text == "late answer"


def test_other_failures_still_raise_after_the_retries(tmp_path):
    backend, _ = live([ConnectionError("down")] * (RETRIES + 1), [], tmp_path)
    with pytest.raises(RuntimeError, match="failed after 5 attempts"):
        asyncio.run(backend.complete(Q, "m"))


def test_an_empty_reply_then_a_network_failure_is_a_network_failure(tmp_path):
    backend, _ = live([""] * RETRIES + [ConnectionError("down")], [], tmp_path)
    with pytest.raises(RuntimeError, match="ConnectionError"):
        asyncio.run(backend.complete(Q, "m"))


# ----------------------------------------------------------------------------------------------------- harness
class EmptyFor(FakeBackend):
    """FakeBackend whose agent `agent` replies with an empty text in round `at` (round 0 = the first answer)."""

    def __init__(self, agent: int, at: int) -> None:
        super().__init__()
        self.agent, self.at = agent, at

    async def complete(self, messages, model):
        users = [m for m in messages if m["role"] == "user"]
        me = re.search(r"agent_(\d+)", messages[0]["content"])
        if messages[0]["role"] == "system" and me and int(me.group(1)) == self.agent and len(users) - 1 == self.at:
            return Completion("", 5, 0, provider="fake")
        return await super().complete(messages, model)


def _traces(families, monkeypatch, tmp_path, backend, **kw):
    from bazt_lab.features.llm_harness.private import seedrun
    monkeypatch.undo()                                   # a later run in one test records only its own traces
    traces = []
    real = seedrun.metrics.rows_from_trace
    monkeypatch.setattr(seedrun.metrics, "rows_from_trace", lambda trace, s: traces.append(trace) or real(trace, s))
    p = harness.LLMPlatform(backend=backend, families=lambda name: families[name])
    spec = core.ArmSpec("passive", Passive, platforms=("llm",))
    p.run_batch([unit(**kw)], spec, probe_settings(), core.CheckpointStore(tmp_path), core.NullTracker())
    return traces


def _victim(families, monkeypatch, tmp_path, **kw):
    """A non-insider agent of the unit's placement (from a plain run)."""
    t = _traces(families, monkeypatch, tmp_path / "plain", FakeBackend(), conversations=1, **kw)[0]
    return int((~t.attackers[0]).nonzero()[0])


@pytest.mark.parametrize("attack,task", [("insider_pi", "mmlu"), ("insider_ta", "injecagent"), ("none", "mmlu")])
def test_an_empty_agent_reply_is_scored_wrong_and_counted_and_the_run_continues(families, monkeypatch, tmp_path,
                                                                                attack, task):
    victim = _victim(families, monkeypatch, tmp_path, attack=attack, task=task, rounds=3)
    plain = _traces(families, monkeypatch, tmp_path / "p2", FakeBackend(), attack=attack, task=task, rounds=3,
                    conversations=2)
    traces = _traces(families, monkeypatch, tmp_path / "e", EmptyFor(victim, 1), attack=attack, task=task, rounds=3,
                     conversations=2)
    assert len(traces) == 2                                       # both conversations ran
    for plain_t, t in zip(plain, traces, strict=True):
        assert t.meta["empty_replies"] == 1 and isinstance(t.meta["empty_replies"], int)
        assert plain_t.meta["empty_replies"] == 0
        assert not t.correct[0, 1, victim]                        # unanswerable: wrong, whatever the judge of the task says
        assert not t.infected[0, 1, victim]


def test_the_row_carries_the_empty_reply_count(families, tmp_path):
    from bazt_lab.features.llm_harness.tests.test_platform import SETTINGS, arm, unit
    p = harness.LLMPlatform(backend=EmptyFor(0, 0), families=lambda name: families[name])
    rows = p.run_batch([unit(attack="none", conversations=1)], arm(Passive), SETTINGS, core.CheckpointStore(tmp_path),
                       core.NullTracker())
    count = rows[0].get("empty_replies", rows[0].get("meta_empty_replies"))
    assert count == 1
