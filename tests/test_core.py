"""Offline tests (no network, no API key). Run: pytest -q"""
import asyncio
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import swarmlab as sl  # noqa: E402
from llm_client import LLMBackend, SQLiteLLMCache, run_connectivity_smoke_test  # noqa: E402

CFG = sl.Config()


def _run(topo, seed, policy, decoys=3, cfg=CFG):
    w = sl.make_world(topo, seed, cfg)
    d = sl.pick_decoys(w, decoys)
    return sl.run_episode(w, policy, cfg, sl.MockAgentPolicy(cfg), d)


def test_determinism():
    assert _run("chain", 5, "risk_aware") == _run("chain", 5, "risk_aware")


@pytest.mark.parametrize("topo", sl.TOPOLOGIES + ["complete"])
def test_cap_and_budget(topo):
    for pol in sl.POLICIES:
        r = _run(topo, 3, pol)
        assert r["quarantines_used"] <= (0 if pol == "no_defense" else CFG.K)


def test_crn_prefix_sharing():
    """Before a window opens, the arm must be identical to no_defense (same shared randomness)."""
    for topo in sl.TOPOLOGIES:
        base = _run(topo, 11, "no_defense")["ever_curve"]
        late = _run(topo, 11, "fixed_late")["ever_curve"]  # window opens at t=9
        assert base[:8] == late[:8]
        mid = _run(topo, 11, "fixed_mid")["ever_curve"]    # window opens at t=5
        assert base[:4] == mid[:4]


def test_clean_and_decoy_share_the_world():
    w = sl.make_world("small_world", 4, CFG)
    w2 = sl.make_world("small_world", 4, CFG)
    assert w.adj == w2.adj and w.p0 == w2.p0 and np.allclose(w.x, w2.x)
    assert sl.pick_decoys(w, 0) == frozenset()
    d = sl.pick_decoys(w, 3)
    assert len(d) == 3 and w.p0 not in d


def test_star_decoys_are_peripheral():
    for s in range(1, 20):
        w = sl.make_world("star_hub", s, CFG)
        assert 0 not in sl.pick_decoys(w, 3)


def test_horizon_rule_no_spend_at_last_step():
    """H=0 when no steps remain, so risk_aware never spends at t=T."""
    w = sl.make_world("chain", 2, CFG)
    aset = set(range(CFG.N))
    assert sl.reach_H(w.adj, 0, 0, CFG, aset, [0.0] * CFG.N) == 0.0
    assert sl.reach_H(w.adj, 0, 5, CFG, aset, [0.0] * CFG.N) > 0.0


def test_no_defense_has_no_quarantine_and_never_contains():
    r = _run("star_hub", 1, "no_defense")
    assert r["quarantines_used"] == 0 and r["time_to_containment"] == CFG.T + 1


def test_oracle_never_wastes_budget():
    for s in range(1, 15):
        r = _run("small_world", s, "offline_oracle")
        assert r["collateral_healthy_quarantined"] == 0


def test_node_removal_changes_readout():
    cfg2 = sl.Config(quarantine_mode="node_removal")
    a = _run("chain", 6, "fixed_early", cfg=CFG)
    b = _run("chain", 6, "fixed_early", cfg=cfg2)
    assert a["quarantines_used"] == b["quarantines_used"]
    assert a["task_mae"] != b["task_mae"]


def test_deltas_are_paired_and_bounded():
    rows, _ = sl.run_grid(CFG, ["chain"], ["decoy"], sl.POLICIES, range(1, 6), lambda c: sl.MockAgentPolicy(c), progress=False)
    sl.add_deltas(rows)
    for r in rows:
        if r["policy"] == "no_defense":
            assert r["delta_abs"] == 0
        if r["policy"] == "offline_oracle" and r["delta_rel"] == r["delta_rel"]:
            assert abs(r["delta_rel"] - 1.0) < 1e-9


def test_budget_estimator():
    e = sl.estimate_llm_budget(["a", "b"], ["clean", "decoy"], sl.POLICIES, 30, price_in=0.1, price_out=0.2)
    assert e["worst_case_calls"] == 2 * 2 * 9 * 30 * 12 * 12
    assert e["worst_case_usd"] > 0


def test_llm_mode_refuses_without_flag():
    with pytest.raises(SystemExit):
        sl.main(["--mode", "llm", "--seeds", "1"])


def test_llm_mode_refuses_before_kickoff(monkeypatch):
    monkeypatch.setenv("SWARMLAB_KICKOFF_UTC", "2999-01-01T00:00:00+00:00")
    with pytest.raises(SystemExit):
        sl.main(["--mode", "llm", "--allow-live-llm", "--seeds", "1"])


def test_tune_refuses_eval_seeds():
    with pytest.raises(SystemExit):
        sl.main(["--tune", "--split", "main"])


# ------------------------------------------------------------------------------- llm_client
class FakeTransport:
    def __init__(self):
        self.calls = 0

    async def __call__(self, model, messages, temperature, max_tokens):
        self.calls += 1
        await asyncio.sleep(0)
        return "READY", 5, 1


def test_cache_hits_and_crn_salt():
    with tempfile.TemporaryDirectory() as d:
        cache = SQLiteLLMCache(os.path.join(d, "c.sqlite"))
        tr = FakeTransport()
        be = LLMBackend("m", cache, tr, temperature=0.0, max_tokens=8)
        req = [("sys", "user", "salt-A")] * 3
        out = be.generate(req)
        assert out == ["READY"] * 3
        # identical key requested concurrently may hit the API more than once; a second pass must be all cache
        before = tr.calls
        be.generate(req)
        assert tr.calls == before
        be.generate([("sys", "user", "salt-B")])
        assert tr.calls == before + 1  # different CRN salt -> different key


def test_smoke_test_with_fake_transport(tmp_path):
    ok = run_connectivity_smoke_test(db_path=str(tmp_path / "s.sqlite"), transport=FakeTransport())
    assert ok


def test_retry_then_fail():
    class Boom:
        async def __call__(self, *a):
            raise RuntimeError("down")

    with tempfile.TemporaryDirectory() as d:
        be = LLMBackend("m", SQLiteLLMCache(os.path.join(d, "c.sqlite")), Boom(), retries=0)
        with pytest.raises(RuntimeError):
            be.generate([("s", "u", "x")])
