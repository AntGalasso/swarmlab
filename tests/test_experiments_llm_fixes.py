"""Regression tests for the LLM-side review findings (C7, C8, C9, H3, M1, m1, m2, m6). Offline only."""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiments as ex
from bazt import llm_harness

CFG_PATH = "configs/sprint_preregistered.json"


def _small_cfg(tmp_path, **llm_over):
    cfg = json.load(open(CFG_PATH))
    cfg["llm"].update({"pilot_conversations": 1, "tier1_conversations": 2, "rounds_per_conversation": 2,
                       **llm_over})
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return str(p), cfg


def _rows(path):
    return [json.loads(line) for line in open(path) if line.strip()]


class _Boom(llm_harness.FakeAsyncLLMBackend):
    """Fake backend that dies after `limit` completions (simulates a kill mid-run)."""

    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    async def complete(self, system, user, temperature=0.0):
        if self.calls >= self.limit:
            raise KeyboardInterrupt("simulated kill")
        return await super().complete(system, user, temperature)


# ------------------------------------------------------------------------------------------ C7 / C8
def test_llm_rows_carry_split_and_the_planned_stage(tmp_path):
    cfgp, cfg = _small_cfg(tmp_path)
    cell = ex.plan_llm_pilot(cfg, "main")[0]
    out = str(tmp_path / "o")
    ex.execute([cell], cfg, out, False, {}, backend=llm_harness.FakeAsyncLLMBackend(), verbose=False)
    row = _rows(ex._jsonl_for(out, "llm_pilot", "main", "llm"))[0]
    assert row["stage"] == "llm_pilot" and row["split"] == "main"
    assert ex.plan_key(row) == ex.plan_key(cell)                 # C8: no KeyError, C7: same key


def test_llm_tier1_row_stage_and_split_follow_the_plan(tmp_path):
    cfgp, cfg = _small_cfg(tmp_path)
    cell = ex.plan_llm_tier1(cfg, "heldout")[0]
    block = llm_harness.Tier1Cell(cell["scenario"], cell["topology"], cell["arm"], None, None, cell["seed"],
                                  n=12, conversations=1, rounds=1, stage=cell["stage"], split=cell["split"])
    row = llm_harness.run_tier1_conversation_block(block, cfg, {})
    assert ex.plan_key(row) == ex.plan_key(cell)


def test_llm_resume_after_a_kill_skips_finished_blocks_and_never_duplicates(tmp_path):
    cfgp, cfg = _small_cfg(tmp_path)
    cells = ex.plan_llm_pilot(cfg, "main")
    assert len(cells) == 2
    out = str(tmp_path / "o")
    path = ex._jsonl_for(out, "llm_pilot", "main", "llm")
    probe = llm_harness.FakeAsyncLLMBackend()
    ex.execute(cells[:1], cfg, str(tmp_path / "probe"), False, {}, backend=probe, verbose=False)
    per_block = probe.calls
    with pytest.raises(KeyboardInterrupt):                       # dies inside the 2nd block
        ex.execute(cells, cfg, out, True, {}, backend=_Boom(per_block + 3), verbose=False)
    assert len(_rows(path)) == 1
    rep = ex.execute(cells, cfg, out, True, {}, backend=llm_harness.FakeAsyncLLMBackend(), verbose=False)
    assert rep["ran"] == 1 and rep["skipped"] == 1
    rep = ex.execute(cells, cfg, out, True, {}, backend=llm_harness.FakeAsyncLLMBackend(), verbose=False)
    assert rep["ran"] == 0 and rep["skipped"] == 2
    rows = _rows(path)
    assert len(rows) == 2 and len({ex.plan_key(r) for r in rows}) == 2


# ------------------------------------------------------------------------------------------------ C9
def test_pilot_uses_the_preregistered_pilot_conversations():
    cfg = json.load(open(CFG_PATH))
    cells = ex.plan_llm_pilot(cfg, "main")
    assert all(c["conversations"] == cfg["llm"]["pilot_conversations"] == 4 for c in cells)


def test_pilot_run_executes_pilot_conversations(tmp_path):
    cfgp, cfg = _small_cfg(tmp_path, pilot_conversations=3, pilot_seeds=1)
    out = str(tmp_path / "o")
    ex.execute(ex.plan_llm_pilot(cfg, "main"), cfg, out, False, {}, backend=llm_harness.FakeAsyncLLMBackend(),
               verbose=False)
    row = _rows(ex._jsonl_for(out, "llm_pilot", "main", "llm"))[0]
    assert row["conversations"] == 3 and len(row["conversation_rows"]) == 3


def test_tier1_plan_carries_tier1_conversations():
    cfg = json.load(open(CFG_PATH))
    assert all(c["conversations"] == cfg["llm"]["tier1_conversations"] for c in ex.plan_llm_tier1(cfg, "main"))


# ------------------------------------------------------------------------------------------------ H3
@pytest.mark.parametrize("stage", ["llm-pilot", "llm-tier1"])
def test_heldout_lock_applies_to_llm_stages(monkeypatch, tmp_path, stage):
    def unguarded(*a, **k):
        raise AssertionError("episodes would run without --confirm-frozen-analysis")

    monkeypatch.setattr(ex, "execute", unguarded)
    with pytest.raises(SystemExit) as e:
        ex.main([stage, "--platform", "llm", "--fake-backend", "--split", "heldout", "--config", CFG_PATH,
                 "--out", str(tmp_path)])
    assert "confirm-frozen-analysis" in str(e.value)


def test_heldout_llm_pilot_runs_with_confirmation(tmp_path):
    cfgp, _ = _small_cfg(tmp_path, pilot_seeds=1)
    out = str(tmp_path / "o")
    rc = ex.main(["llm-pilot", "--platform", "llm", "--fake-backend", "--split", "heldout",
                  "--confirm-frozen-analysis", "--config", cfgp, "--out", out, "--quiet"])
    assert rc == 0


# ------------------------------------------------------------------------------------------------ M1
def test_tier1_applies_every_frozen_threshold(monkeypatch, tmp_path):
    cfgp, cfg = _small_cfg(tmp_path)
    seen = {}
    orig = llm_harness._PersistentGatewayFactory.__init__

    def spy(self, arm, zcfg, bcfg, *a, **k):
        seen["z"], seen["b"] = zcfg, bcfg
        orig(self, arm, zcfg, bcfg, *a, **k)

    monkeypatch.setattr(llm_harness._PersistentGatewayFactory, "__init__", spy)
    thr = {"arm_B": {"tau_alpha": 0.61}, "arm_C": {"rl_offset": 0.23}, "bazt": {"theta0": 0.77}}
    block = llm_harness.Tier1Cell("clean", "barabasi_albert", "arm_C", None, None, 1, n=12, conversations=1,
                                  rounds=1)
    llm_harness.run_tier1_conversation_block(block, cfg, thr)
    assert seen["z"].tau_alpha == pytest.approx(0.61)
    assert seen["z"].rl_offset == pytest.approx(0.23)
    assert seen["b"].theta0 == pytest.approx(0.77)


# ------------------------------------------------------------------------------------------------ m1
def test_llm_calibration_is_requested_on_three_seeds(monkeypatch, tmp_path):
    from bazt import calibration
    got = {}
    monkeypatch.setattr(calibration, "calibrate_thresholds",
                        lambda config, **kw: got.update(kw) or {"thresholds": {}})
    cfgp, cfg = _small_cfg(tmp_path)

    class A:
        out = str(tmp_path)
        platform, split = "llm", "tune"

    ex.stage_calibrate(A, cfg)
    assert list(got["seeds"]) == [9001, 9002, 9003]


# ------------------------------------------------------------------------------------------------ m2
def test_topology_scale_uses_config_scale_N():
    cfg = json.load(open(CFG_PATH))
    cfg["scale_N"] = [12, 30]
    assert {c["N"] for c in ex.plan_topology_scale(cfg, "main", False)} == {12, 30}


# ------------------------------------------------------------------------------------------------ m6
def test_close_runs_even_when_execute_raises(monkeypatch, tmp_path):
    cfgp, cfg = _small_cfg(tmp_path)
    closed = []

    def boom(*a, **k):
        raise RuntimeError("mid-run failure")

    monkeypatch.setattr(ex, "execute", boom)
    monkeypatch.setattr(ex, "_close", lambda b: closed.append(b))
    args = ex.build_parser().parse_args(["llm-pilot", "--platform", "llm", "--fake-backend", "--config", cfgp,
                                         "--out", str(tmp_path)])
    with pytest.raises(RuntimeError):
        ex.stage_llm_pilot(args, cfg)
    assert len(closed) == 1
