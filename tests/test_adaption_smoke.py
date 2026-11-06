"""Offline regression tests for provider-specific READY checks: no credentials or network."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm_client as client


def test_adaption_smoke_uses_adaption_catalog_and_exact_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("SWARMLAB_PROVIDER", "adaption")
    seen = []

    async def transport(model, messages, temperature, max_tokens):
        seen.append((model, max_tokens))
        assert messages == [{"role": "system", "content": "Reply with the single word READY."},
                            {"role": "user", "content": "Ping."}]
        return "READY", 12, 1

    assert client.run_connectivity_smoke_test(str(tmp_path / "smoke.sqlite"), transport)
    assert seen == [("gemma-4-31b-it", 512), ("qwen3.8-27b", 512)]


@pytest.mark.parametrize("reply", ["", "NOT READY", "READY."])
def test_cache_equality_does_not_accept_a_failed_ready_response(monkeypatch, tmp_path, reply):
    monkeypatch.setenv("SWARMLAB_PROVIDER", "adaption")

    async def transport(*args):
        return reply, 12, 1

    assert not client.run_connectivity_smoke_test(str(tmp_path / "smoke.sqlite"), transport)


def test_openrouter_smoke_keeps_existing_models(monkeypatch, tmp_path):
    monkeypatch.setenv("SWARMLAB_PROVIDER", "openrouter")
    seen = []

    async def transport(model, messages, temperature, max_tokens):
        seen.append((model, max_tokens))
        return "READY", 12, 1

    assert client.run_connectivity_smoke_test(str(tmp_path / "smoke.sqlite"), transport)
    assert seen == [(client.PRIMARY_MODEL, 8), (client.H4_MODEL, 8)]


def test_adaption_smoke_models_can_match_the_account_catalog(monkeypatch):
    monkeypatch.setenv("SWARMLAB_PROVIDER", "adaption")
    monkeypatch.setenv("ADAPTION_SMOKE_PRIMARY_MODEL", "my-primary")
    monkeypatch.setenv("ADAPTION_SMOKE_FALLBACK_MODEL", "my-fallback")
    monkeypatch.setenv("ADAPTION_SMOKE_H4_MODEL", "my-other-family")
    assert client._smoke_model_groups() == (("primary", ["my-primary", "my-fallback"]),
                                           ("H4", ["my-other-family"]))
