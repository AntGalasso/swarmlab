"""D123 on the LLM platform: `Observation.audit_isolated` judges the text an isolated agent still wrote this round, with
the judge `audit` uses, pays from the audit budget, and leaves every request that does not use it exactly as it was."""
import hashlib
import json

import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.llm_harness import public as harness
from bazt_lab.features.llm_harness.private import platform as platform_module
from bazt_lab.features.llm_harness.tests.test_empty_replies import EmptyFor
from bazt_lab.features.llm_harness.tests.test_platform import SETTINGS, N, Passive, arm, run, unit

ISOLATE_0 = torch.zeros(1, N, dtype=torch.bool)
ISOLATE_0[0, 0] = True
EVERY = torch.ones(1, N, dtype=torch.bool)


class Probe(Passive):
    """Isolates agent 0 after round 1, then audits (plain and through the isolated channel) what `script` says."""
    log: list

    def decide(self, obs):
        super().decide(obs)
        entry = {"t": obs.t, "texts": obs.texts[0], "active": obs.active[0].clone()}
        if obs.t >= 2:
            entry["plain"] = obs.audit(EVERY)[0].clone()
            entry["iso"] = obs.audit_isolated(EVERY)[0].clone()
        Probe.log.append(entry)
        return core.Actions(isolation=ISOLATE_0.clone()) if obs.t == 1 else core.Actions()


@pytest.fixture()
def platform(families):
    return harness.LLMPlatform(families=lambda name: families[name])


@pytest.fixture(autouse=True)
def fresh_log():
    Probe.log = []


def test_an_isolated_agent_is_judged_on_the_text_it_wrote_and_charged_like_an_audit(platform, tmp_path):
    rows, _ = run(platform, tmp_path, cls=Probe)
    judge = platform_module._fake_judge()
    for e in Probe.log[1:]:
        assert e["plain"][0].isnan(), "audit() never grants an isolated agent"
        assert not e["iso"][0].isnan()
        assert float(e["iso"][0]) == pytest.approx(float(judge.score([e["texts"][0]])[0]))
        assert e["iso"][1:].isnan().all()                                    # active agents are not for this channel
        assert not e["plain"][1:].isnan().any()
    assert sum(r["spend_audit_total"] for r in rows) == len(Probe.log[1:]) * N          # N - 1 active senders + the isolated one


def test_the_isolated_channel_charges_nothing_for_an_agent_that_wrote_no_text(families, tmp_path):
    p = harness.LLMPlatform(backend=EmptyFor(0, 1), families=lambda name: families[name])     # agent 0 is silent in round 2 of each conversation
    rows = p.run_batch([unit()], arm(Probe), SETTINGS, core.CheckpointStore(tmp_path), core.NullTracker())
    later = Probe.log[1:]
    silent = [e["texts"][0].strip() == "" for e in later]
    assert any(silent) and not all(silent)
    for e, quiet in zip(later, silent, strict=True):
        assert bool(e["iso"][0].isnan()) == quiet
    assert sum(r["spend_audit_total"] for r in rows) == sum((N - 1) + (0 if quiet else 1) for quiet in silent)


def test_the_hard_audit_budget_is_exact_through_both_channels(platform, tmp_path):
    rows, _ = run(platform, tmp_path, cls=Probe, audit=3 / (N * (N - 1) * 2), conversations=1)   # three units
    assert rows[0]["spend_audit_total"] == 3
    assert sum(int((~e["iso"].isnan()).sum() + (~e["plain"].isnan()).sum()) for e in Probe.log[1:]) == 3


def test_the_isolated_subset_on_over_request_is_keyed_and_repeatable(platform, tmp_path):
    class Many(Probe):
        def decide(self, obs):
            Passive.decide(self, obs)
            if obs.t == 1:
                return core.Actions(isolation=torch.tensor([[True, True, True, True, False]]))
            if obs.t == 2:
                Many.got.append(obs.audit_isolated(EVERY)[0].isnan().clone())
            return core.Actions()
    Many.got = []
    run(platform, tmp_path / "a", cls=Many, audit=2 / (N * (N - 1) * 2), conversations=1)
    first = [(~m).nonzero().flatten().tolist() for m in Many.got]
    assert len(first[0]) == 2
    Many.got = []
    run(platform, tmp_path / "b", cls=Many, audit=2 / (N * (N - 1) * 2), conversations=1)
    assert [(~m).nonzero().flatten().tolist() for m in Many.got] == first


def _digest(platform, tmp_path, **kw):
    class IsolateAndAudit(Passive):
        def decide(self, obs):
            super().decide(obs)
            obs.audit(obs.active.clone())
            return core.Actions(isolation=ISOLATE_0.clone()) if obs.t == 1 else core.Actions()
    rows, _ = run(platform, tmp_path, cls=IsolateAndAudit, **kw)
    rows = [{k: v for k, v in r.items() if k != "latency"} for r in rows]          # wall time is not a result
    return hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()[:16]


GOLDEN = {"capped": "bdefd8621caac0ad", "uncapped": "6dd73f6592f76680"}


def test_an_arm_that_never_uses_the_channel_gives_bit_identical_rows_to_before(platform, tmp_path):
    assert _digest(platform, tmp_path / "a", audit=5 / (N * (N - 1) * 2), conversations=1) == GOLDEN["capped"]
    assert _digest(platform, tmp_path / "b", conversations=1) == GOLDEN["uncapped"]


def test_float32_scores_follow_the_dtype_while_accounting_stays_exact(families, tmp_path):
    s = json.loads(json.dumps(SETTINGS))
    s["compute"] = {"dtype": "float32", "device": "cpu"}
    p = harness.LLMPlatform(families=lambda name: families[name])
    rows = p.run_batch([unit()], arm(Probe), s, core.CheckpointStore(tmp_path), core.NullTracker())
    assert all(e["iso"].dtype == torch.float32 for e in Probe.log[1:])
    assert sum(r["spend_audit_total"] for r in rows) == len(Probe.log[1:]) * N

