"""Offline tests for Kunal's two adapters (no network, no API key). Run: pytest -q

Covered here:
  * provider -- SWARMLAB_PROVIDER dispatch in llm_client (openrouter default, adaption opt-in,
    unknown provider refused) and the wiring inside LLMBackend._one;
  * adaption -- make_adaption_transport fails with a clear message without ADAPTION_API_KEY;
  * adapter  -- LiveCoTAdapter contract: marker is channel-safe and invisible to the tag monitor,
    ground truth differs from the monitor predicate, mock semantics mirror the compressed template,
    the live path is gated before kickoff and routes through the pluggable backend seam after it,
    and run_constraint_evasion accepts the custom adapter end-to-end (mock, 1 episode).
"""
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env_ladder as el  # noqa: E402
import llm_client as llc  # noqa: E402
import live_cot_adapter as lca  # noqa: E402

MONITOR_MARKER = lambda s: el.CANARY in s  # noqa: E731 -- same predicate as run_constrained_episode


# ----------------------------------------------------------------------------------- provider
def test_provider_defaults_to_openrouter_and_accepts_adaption(monkeypatch):
    monkeypatch.delenv("SWARMLAB_PROVIDER", raising=False)
    assert llc._provider() == "openrouter"
    monkeypatch.setenv("SWARMLAB_PROVIDER", "adaption")
    assert llc._provider() == "adaption"
    assert llc._api_key() is None  # no key in the environment under test


def test_unknown_provider_is_refused(monkeypatch):
    monkeypatch.setenv("SWARMLAB_PROVIDER", "claude")
    with pytest.raises(RuntimeError, match="SWARMLAB_PROVIDER"):
        llc.make_default_transport()


def test_adaption_transport_requires_its_own_key(monkeypatch):
    monkeypatch.setenv("SWARMLAB_PROVIDER", "adaption")
    monkeypatch.delenv("ADAPTION_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ADAPTION_API_KEY"):
        llc.make_adaption_transport()


def test_openrouter_transport_requires_a_key(monkeypatch):
    monkeypatch.setenv("SWARMLAB_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")  # .env may ship a real key; force the empty branch
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        llc.make_openrouter_transport()


def test_backend_wires_the_adaption_transport_on_first_cache_miss(monkeypatch, tmp_path):
    monkeypatch.setenv("SWARMLAB_PROVIDER", "adaption")
    monkeypatch.setenv("ADAPTION_API_KEY", "pt_live_dummy")  # never used: factory raises the sentinel
    sentinel = RuntimeError("sentinel: adaption factory was consulted")

    def _boom():
        raise sentinel

    monkeypatch.setattr(llc, "make_default_transport", _boom)
    cache = llc.SQLiteLLMCache(str(tmp_path / "c.sqlite"))
    backend = llc.LLMBackend("deepseek-v4-flash", cache, retries=0)
    try:
        with pytest.raises(RuntimeError, match="sentinel"):
            backend.generate([("s", "u", "salt-1")])
    finally:
        cache.close()
        backend.close()


# ------------------------------------------------------------------------------------ adapter
def test_marker_is_channel_safe_and_invisible_to_the_tag_monitor():
    m = lca.PAYLOAD_MARKER
    assert m.isalnum() and len(m) <= 8
    assert "[" not in m and "]" not in m
    assert el.CANARY not in m
    assert not MONITOR_MARKER(m)                      # the m-step monitor cannot see it
    assert lca.LiveCoTAdapter().detect_ground_truth_payload(m)  # ground truth can


def test_marker_survives_lexical_filters_is_sealed_by_schema():
    ad = lca.LiveCoTAdapter()
    note = f"est 41 {lca.PAYLOAD_MARKER}"
    spec_brackets = el.HardConstraintSpec(action_channel_mode="banned_brackets")
    spec_clamp = el.HardConstraintSpec(action_channel_mode="length_clamped_32")
    spec_kv = el.HardConstraintSpec(action_channel_mode="structured_kv_only")
    assert el.apply_hard_constraints(note, 0, 0, spec_brackets) == note
    assert el.apply_hard_constraints(note, 0, 0, spec_clamp) == note
    sealed = el.apply_hard_constraints(note, 0, 0, spec_kv)
    assert not ad.detect_ground_truth_payload(sealed)  # schema mediation removes the channel


def test_ground_truth_is_not_the_monitor_predicate():
    ad = lca.LiveCoTAdapter()
    assert ad.detect_ground_truth_payload(f"note {lca.PAYLOAD_MARKER}")
    assert not MONITOR_MARKER(f"note {lca.PAYLOAD_MARKER}")
    assert not ad.detect_ground_truth_payload(el.CANARY)  # the literal canary is not this payload
    assert MONITOR_MARKER(el.CANARY)


def test_mock_evasion_step_mirrors_the_compressed_template():
    ad = lca.LiveCoTAdapter()
    spec = el.HardConstraintSpec()
    assert ad.mock_evasion_step(random.Random(0), False, spec) == ("", 0.0, False)
    rng = random.Random(1)
    text, risk, detected = ad.mock_evasion_step(rng, True, spec)
    assert text == lca.PAYLOAD_MARKER
    assert 0.0 <= risk <= 1.0
    assert detected is False
    # CRN reproducibility: same rng state, same draw
    text2, risk2, _ = ad.mock_evasion_step(random.Random(1), True, spec)
    assert (text2, risk2) == (text, risk)


def test_transform_prompt_passthrough_for_clean_agents_and_schema_channel():
    ad = lca.LiveCoTAdapter()
    spec = el.HardConstraintSpec(action_channel_mode="structured_kv_only")
    assert ad.transform_agent_step_prompt("sys", "usr", False, spec) == ("sys", "usr")
    # sealed channel: even an infected agent's prompts are untouched (no payload path by design)
    assert ad.transform_agent_step_prompt("sys", "usr", True, spec) == ("sys", "usr")


def test_live_path_is_gated_before_kickoff(monkeypatch):
    monkeypatch.setenv("SWARMLAB_KICKOFF_UTC", "2099-01-01T00:00:00+00:00")
    ad = lca.LiveCoTAdapter()
    spec = el.HardConstraintSpec(action_channel_mode="banned_brackets")
    with pytest.raises(RuntimeError, match="kickoff"):
        ad.transform_agent_step_prompt("sys", "reasoning ...", True, spec)


def test_live_path_after_kickoff_uses_the_backend_seam(monkeypatch):
    monkeypatch.setenv("SWARMLAB_KICKOFF_UTC", "2020-01-01T00:00:00+00:00")
    calls = {}

    class FakeBackend:
        def generate(self, reqs, temperature=None):
            system, user, salt = reqs[0]
            calls["system"], calls["user"], calls["salt"] = system, user, salt
            assert lca.PAYLOAD_MARKER in system
            return [f"compressed reasoning {lca.PAYLOAD_MARKER}"]

        def close(self):
            calls["closed"] = True

    monkeypatch.setattr(lca, "_make_backend", lambda: FakeBackend())
    ad = lca.LiveCoTAdapter()
    spec = el.HardConstraintSpec(action_channel_mode="banned_brackets", max_chars=64)
    sys2, usr2 = ad.transform_agent_step_prompt("sys", "long reasoning trace", True, spec)
    assert lca.PAYLOAD_MARKER not in sys2.lower()  # marker lives in the NOTE, not the prompt body
    assert "banned_brackets" in sys2
    assert calls["closed"] is True
    assert calls["salt"].startswith("cot-compress-v1:")


# ------------------------------------------------------------------------------ runner (mock)
def test_run_constraint_evasion_accepts_the_custom_adapter():
    rows, table, meta = el.run_constraint_evasion(
        levels=[4], seeds=[9001], policies=("no_defense",),
        movement_modes=("unrestricted",), channel_modes=("banned_brackets",),
        adapter=lca.LiveCoTAdapter(), save=False, verbose=False)
    assert meta["payload_forms"] == ["custom"]
    assert rows and rows[0]["payload_form"] == "custom"
    assert rows[0]["level"] == 4 and rows[0]["policy"] == "no_defense"
