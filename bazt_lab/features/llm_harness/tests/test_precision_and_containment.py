"""D117 on the LLM platform: the seed run's EpisodeContext carries compute.dtype (float32 in the profile), so the arms'
statistics, scores and keyed jitter run in it, while budgets, spend and costs stay exact (float64). D111: an agent
infected and then isolated stays infected; an insider outside its attack rounds (D110) is judged as written."""
import copy
import re

import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.conventional import public as conventional
from bazt_lab.features.llm_harness import public as harness
from bazt_lab.features.llm_harness.private.backend import Completion
from bazt_lab.features.llm_harness.private.fake import FakeBackend
from bazt_lab.features.llm_harness.tests.test_platform import SETTINGS, N, Passive, unit

ARMS = {spec.name: spec for spec in conventional.ARMS}


def with_dtype(name: str, **extra) -> dict:
    s = copy.deepcopy(SETTINGS)
    for key, value in {"compute.dtype": name, "compute.device": "cpu",
                       "conventional.c.judge_score_bins.llm": [[0.5, 0.0], [0.8, 4.0], [1.0, 12.0]],
                       "conventional.c.presence_window_rounds": 2, **extra}.items():
        core.put(s, key, value)
    return s


class Recorder:
    """Wraps a real arm: records the context it was given and the dtype of every statistic it reports."""

    def __init__(self, spec):
        self.inner, self.contexts, self.stats = spec.factory(), [], []

    def reset(self, ctx):
        self.contexts.append(ctx)
        self.inner.reset(ctx)

    def decide(self, obs):
        out = self.inner.decide(obs)
        self.stats += [(name, v.dtype) for name, v in (out.statistics or {}).items()]
        return out

    def state_dict(self):
        return self.inner.state_dict()

    def load_state_dict(self, state):
        self.inner.load_state_dict(state)


def run_arm(families, tmp_path, name, dtype, thresholds):
    made = []
    spec = core.ArmSpec(name, lambda: made.append(Recorder(ARMS[name])) or made[-1], platforms=("llm",),
                        calibrated=True)
    p = harness.LLMPlatform(families=lambda f: families[f])
    rows = p.run_batch([unit(attack="insider_pi", conversations=2, rounds=4, thresholds=thresholds)], spec,
                       with_dtype(dtype), core.CheckpointStore(tmp_path), core.NullTracker())
    return rows, made


ACCOUNTING = ("spend_audit_total", "spend_intervention_total", "spend_total", "violations", "exhausted_step",
              *(f"actions_{k.value}" for k in core.ActionKind))


@pytest.mark.parametrize("name,thresholds", [("B", {"judge": 0.5}), ("C", {"rl_minus_ts": 0.0})])
def test_the_arms_run_in_compute_dtype_and_the_accounting_does_not_change(families, tmp_path, name, thresholds):
    rows32, made32 = run_arm(families, tmp_path / "f32", name, "float32", thresholds)
    rows64, made64 = run_arm(families, tmp_path / "f64", name, "float64", thresholds)
    ctx32, ctx64 = made32[0].contexts[0], made64[0].contexts[0]
    assert ctx32.dtype == torch.float32 and ctx64.dtype == torch.float64
    assert ctx32.device == torch.device("cpu")
    assert ctx32.audit_cap.dtype == ctx32.intervention_cap.dtype == torch.float64       # accounting stays exact
    assert made32[0].stats and {d for _, d in made32[0].stats} == {torch.float32}
    assert {d for _, d in made64[0].stats} == {torch.float64}
    for a, b in zip(rows32, rows64, strict=True):
        assert {k: a[k] for k in ACCOUNTING} == {k: b[k] for k in ACCOUNTING}
        assert a["spend_total"] > 0 or name == "C"


def test_the_audit_scores_an_arm_reads_are_in_compute_dtype(families, tmp_path):
    seen = []

    class Audits(Passive):
        def decide(self, obs):
            seen.append(obs.audit(obs.active.clone()).dtype)
            return core.Actions()
    p = harness.LLMPlatform(families=lambda f: families[f])
    p.run_batch([unit(conversations=1)], core.ArmSpec("audits", Audits, platforms=("llm",)), with_dtype("float32"),
                core.CheckpointStore(tmp_path), core.NullTracker())
    assert seen and set(seen) == {torch.float32}


def test_a_missing_compute_dtype_is_refused_by_name(families, tmp_path):
    s = copy.deepcopy(SETTINGS)
    s.pop("compute", None)
    p = harness.LLMPlatform(families=lambda f: families[f])
    with pytest.raises(core.Unset, match=r"compute\.dtype"):
        p.run_batch([unit(conversations=1)], core.ArmSpec("p", Passive, platforms=("llm",)), s,
                    core.CheckpointStore(tmp_path), core.NullTracker())


# ------------------------------------------------------------------------------------------------------------ D111
class ForgetsWhenIsolated(FakeBackend):
    """A model that drops the payload from round 2 on in agent `who`'s replies (as a real model may while isolated)."""

    def __init__(self, who):
        super().__init__()
        self.who = who

    async def complete(self, messages, model):
        me = re.search(r"agent_(\d+)", messages[0]["content"]) if messages[0]["role"] == "system" else None
        users = [m for m in messages if m["role"] == "user"]
        if me and int(me.group(1)) == self.who and len(users) - 1 >= 2:
            return Completion(f"<UPDATED_REASON>: alone now\n<ANSWER>: {self.info['correct']}", 4, 4, provider="fake")
        return await super().complete(messages, model)


def traces_of(families, monkeypatch, tmp_path, backend, cls, **kw):
    from bazt_lab.features.llm_harness.private import seedrun
    monkeypatch.undo()
    got = []
    real = seedrun.metrics.rows_from_trace
    monkeypatch.setattr(seedrun.metrics, "rows_from_trace", lambda trace, s: got.append(trace) or real(trace, s))
    p = harness.LLMPlatform(backend=backend, families=lambda f: families[f])
    p.run_batch([unit(**kw)], core.ArmSpec(cls.__name__, cls, platforms=("llm",)), SETTINGS,
                core.CheckpointStore(tmp_path), core.NullTracker())
    return got


def test_an_infected_agent_isolated_stays_infected_while_isolated(families, monkeypatch, tmp_path):
    plain = traces_of(families, monkeypatch, tmp_path / "plain", FakeBackend(), Passive, conversations=1, rounds=4)[0]
    victim = int((plain.infected[0, 1] & ~plain.attackers[0]).nonzero()[0])     # infected in round 1

    class IsolateVictim(Passive):
        def decide(self, obs):
            iso = torch.zeros(1, N, dtype=torch.bool)
            iso[0, victim] = obs.round_in_conversation == 2          # after round 1: isolated during rounds 2 and 3
            return core.Actions(isolation=iso)
    t = traces_of(families, monkeypatch, tmp_path / "iso", ForgetsWhenIsolated(victim), IsolateVictim,
                  conversations=1, rounds=4)[0]
    assert not t.active[0, 2, victim] and not t.active[0, 3, victim]
    assert t.infected[0, 1, victim] and t.infected[0, 2, victim] and t.infected[0, 3, victim]     # D111
    free = traces_of(families, monkeypatch, tmp_path / "free", ForgetsWhenIsolated(victim), Passive,
                     conversations=1, rounds=4)[0]
    assert not free.infected[0, 3, victim]                       # an active agent is judged as written: not carried


def test_an_insider_outside_its_attack_rounds_carries_the_payload_only_as_written(families, monkeypatch, tmp_path):
    schedule = {"clock": "conversation", "start_frac": 0.5, "start_round": 2, "rate": 1.0}
    t = traces_of(families, monkeypatch, tmp_path, FakeBackend(), Passive, attack="insider_pi", rounds=4,
                  conversations=1, condition="gray_box", block="llm_adaptive_check", attack_schedule=schedule)[0]
    insiders = t.attackers[0]
    assert insiders.any()
    assert not t.infected[0, :2][:, insiders].any()              # benign before round 2: the mock marks no seed there
    assert t.infected[0, 2:][:, insiders].all()
