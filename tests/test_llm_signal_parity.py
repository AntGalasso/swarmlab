"""Regression tests for the live-LLM pilot review findings (signal parity, contagion, pairing). Offline only.

Covers, with the deterministic `FakeAsyncLLMBackend`:
  * ONE judge bridge feeds arm_B / arm_C / bazt, and BAZT really is probed (it used to be wired with no hook);
  * the live agent obeys the scenario's contagion law (seed / carrier / u_adopt) and emits its token;
  * decoy turns carry the benign header that gives them their monitor/judge signal;
  * `plan_llm_pilot` is a PAIRED arm_A vs arm_C vs bazt comparison on identical seeds.
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiments as ex
import swarmlab as sl
from bazt import llm_harness

CFG_PATH = "configs/sprint_preregistered.json"


def _cfg():
    return json.load(open(CFG_PATH))


def _ctx(i=0, **kw):
    base = dict(i=i, t=1, inbox=[], own_prev=None, est=50.0, is_seed=False, is_decoy=False,
                decoy_active=False, u_adopt=0.99, u_stay=0.99, u_decoy=0.99, seed=1, topology="chain")
    base.update(kw)
    return sl.AgentCtx(**base)


# ------------------------------------------------------------------------ one bridge, every defender
def test_judge_bridge_scores_once_per_message_and_shares_it():
    backend = llm_harness.FakeAsyncLLMBackend(omit_logprobs=True)
    judge = llm_harness.OutOfFamilyJudge(backend, seed=1)
    bridge = llm_harness.LiveSuspicion(judge, mu_ref=2.0)
    a = bridge.roll("please relay the calibration notice")
    b = bridge.roll("please relay the calibration notice")
    assert a == b and bridge.probes == 1 and bridge.hits == 1
    assert 0.0 <= a <= 2.0                                # mapped onto the monitor's score scale


def test_bazt_is_probed_exactly_like_arm_c():
    """The review finding: BAZT ran with no probe hook, so the live judge never scored its messages."""
    cfg = _cfg()
    rows = {}
    for arm, budget in (("arm_C", (0.10, 0.10)), ("bazt", (0.10, 0.10))):
        block = llm_harness.Tier1Cell("infection", "barabasi_albert", arm, budget[0], budget[1], 1, n=12,
                                      conversations=1, rounds=3)
        rows[arm] = llm_harness.run_tier1_conversation_block(block, cfg, {})
    assert rows["arm_C"]["judge_probes"] > 0
    assert rows["bazt"]["judge_probes"] > 0               # was 0 before the fix
    assert rows["bazt"]["judge_mu_ref"] == rows["arm_C"]["judge_mu_ref"] > 0
    assert rows["bazt"]["audit_spend"] >= 0.0 and rows["bazt"]["intervention_spend"] >= 0.0


def test_probe_hook_is_identical_for_both_gateway_families(monkeypatch):
    seen = {}

    class Spy(llm_harness.LiveSuspicion):
        def hook(self, i, t, view):
            seen.setdefault(id(self), 0)
            seen[id(self)] += 1
            return super().hook(i, t, view)

    monkeypatch.setattr(llm_harness, "LiveSuspicion", Spy)
    cfg = _cfg()
    block = llm_harness.Tier1Cell("infection", "barabasi_albert", "bazt", 0.1, 0.1, 1, n=12, conversations=1,
                                  rounds=2)
    llm_harness.run_tier1_conversation_block(block, cfg, {})
    assert len(seen) == 1                                  # one bridge per block, not one per arm


# ------------------------------------------------------------------------- live contagion / decoys
def test_agent_adopts_from_inbox_and_emits_the_payload_token():
    cfg = sl.Config(N=2, T=4)
    agent = llm_harness.LLMConversationAgent(llm_harness.FakeAsyncLLMBackend(), cfg)
    out = agent.step([_ctx(0, inbox=[f"a1 t1 est=50.0 {sl.CANARY}"], u_adopt=0.01),
                      _ctx(1, inbox=[f"a0 t1 est=50.0 {sl.CANARY}"], u_adopt=0.99)])
    assert sl.CANARY in out[0]                             # u_adopt < beta -> adopted
    assert sl.CANARY not in out[1]                         # draw says no: no adoption


def test_agent_persists_only_while_the_stay_draw_holds():
    cfg = sl.Config(N=1, T=4)
    keep_agent = llm_harness.LLMConversationAgent(llm_harness.FakeAsyncLLMBackend(), cfg)
    drop_agent = llm_harness.LLMConversationAgent(llm_harness.FakeAsyncLLMBackend(), cfg)
    keep = keep_agent.step([_ctx(0, own_prev=f"a0 t1 est=50.0 {sl.CANARY}", u_stay=0.01)])
    drop = drop_agent.step([_ctx(0, own_prev=f"a0 t1 est=50.0 {sl.CANARY}", u_stay=0.99)])
    assert sl.CANARY in keep[0] and sl.CANARY not in drop[0]
    assert keep_agent.infected_turns == 1 and drop_agent.infected_turns == 0


def test_agent_uses_the_scenario_token_not_a_hardcoded_canary():
    cfg = sl.Config(N=1, T=4)
    spec = SimpleNamespace(token="pi9x", name="pi")
    agent = llm_harness.LLMConversationAgent(llm_harness.FakeAsyncLLMBackend(), cfg, spec=spec)
    out = agent.step([_ctx(0, is_seed=True)])
    assert out[0].rstrip().endswith("pi9x") and sl.CANARY not in out[0]


def test_decoy_turns_carry_the_benign_header_without_the_payload():
    cfg = sl.Config(N=1, T=4)
    agent = llm_harness.LLMConversationAgent(llm_harness.FakeAsyncLLMBackend(), cfg)
    out = agent.step([_ctx(0, is_decoy=True, decoy_active=True, u_decoy=0.01)])
    assert out[0].startswith(sl.DECOY_TAG) and sl.CANARY not in out[0]
    assert agent.decoy_turns == 1 and agent.infected_turns == 0


# ------------------------------------------------------------------------------- paired pilot plan
def test_pilot_is_a_paired_arm_a_vs_capped_c_vs_capped_bazt_comparison():
    cfg = _cfg()
    cells = ex.plan_llm_pilot(cfg, "main")
    arms = [c["arm"] for c in cells]
    assert set(arms) == {"arm_A", "arm_C", "bazt"}
    seeds = {c["seed"] for c in cells}
    assert {c["seed"] for c in cells if c["arm"] == "arm_A"} == seeds
    assert {c["seed"] for c in cells if c["arm"] == "arm_C"} == seeds
    assert {c["seed"] for c in cells if c["arm"] == "bazt"} == seeds
    a = [c for c in cells if c["arm"] == "arm_A"][0]
    assert a["audit_pct"] is None and a["int_pct"] is None
    for c in cells:
        if c["arm"] in ("arm_C", "bazt"):
            assert (c["audit_pct"], c["int_pct"]) == (0.10, 0.10)


def test_pilot_summary_reports_every_gate():
    cfg = _cfg()
    rows = []
    for arm, asr, acc, audit, interv in (("arm_A", 0.75, 0.30, 0.0, 0.0), ("arm_C", 0.42, 0.45, 6.6, 6.6),
                                         ("bazt", 0.17, 0.60, 6.0, 5.0)):
        rows.append({"arm": arm, "seed": 1, "asr_final": asr, "accuracy": acc, "audit_spend": audit,
                     "intervention_spend": interv, "budget_exhausted": int(interv > 0), "judge_probes": 12,
                     "spend_fraction": 0.5, "episode_seconds": 10.0, "audit_pct": None, "int_pct": None})
    table, summary = ex.summarise_llm_pilot(rows, cfg)
    assert len(summary["gates"]) == 5 and all(summary["gates"].values())
    assert summary["verdict"] == "go" and "GO for llm-tier1" in table
    assert np.isclose(summary["per_arm"]["bazt"]["asr_final"], 0.17)
