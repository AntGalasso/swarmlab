"""Exercise the real client's async/sync contract without any network requests."""
import asyncio
import pytest

from llm_client import LLMBackend, SQLiteLLMCache
from bazt.live_backend_adapter import LiveBackendAdapter
from bazt.llm_harness import LLMConversationAgent, OutOfFamilyJudge, make_live_backend
import swarmlab as sl


def test_real_client_contract_agent_async_judge_and_cleanup(tmp_path):
    seen, closed = [], []

    async def transport(model, messages, temperature, max_tokens):
        seen.append((model, messages))
        return ('{"score": 77}' if 'security judge' in messages[0]['content'] else 'est=50.0'), 20, 5

    async def aclose():
        closed.append(True)

    transport.aclose = aclose
    cache = SQLiteLLMCache(str(tmp_path/'cache.sqlite'))
    backend = LLMBackend('real-client-with-offline-transport', cache, transport=transport, retries=0)
    adapter = LiveBackendAdapter(backend)
    try:
        assert LLMConversationAgent(adapter, sl.Config()).step([]) == []
        assert adapter.generate_sync([('Agent system', 'Estimate')]) == ['est=50.0']
        assert OutOfFamilyJudge(adapter).score_sync('Suspect message') == (77.0, 'structured_fallback')
        assert asyncio.run(adapter.first_token_yes_logprob('question')) is None
        assert len(seen) == 2
    finally:
        adapter.close()
        adapter.close()
    assert closed == [True]
    assert cache.conn is None
    assert backend._loop.is_closed()


def test_live_judge_never_substitutes_a_hash_score(tmp_path):
    async def malformed(*args):
        # `no` is now a documented 0/100 answer; an unparseable reply is what must raise.
        return 'I cannot help with that.', 10, 1
    backend = LLMBackend('offline', SQLiteLLMCache(str(tmp_path/'cache.sqlite')), transport=malformed, retries=0)
    adapter = LiveBackendAdapter(backend)
    try:
        with pytest.raises(ValueError, match='no structured score'):
            OutOfFamilyJudge(adapter).score_sync('Any message')
    finally:
        adapter.close()


def test_factory_requires_a_separate_live_judge(monkeypatch, tmp_path):
    monkeypatch.delenv('SWARMLAB_TIER1_JUDGE_MODEL', raising=False)
    with pytest.raises(ValueError, match='different model'):
        make_live_backend('agent', str(tmp_path/'cache.sqlite'))


def test_factory_builds_both_models_and_closes_both(monkeypatch, tmp_path):
    import llm_client
    monkeypatch.setenv('SWARMLAB_PROVIDER', 'adaption')
    monkeypatch.setenv('SWARMLAB_TIER1_JUDGE_MODEL', 'qwen3.8-27b')
    seen = []
    async def transport(model, messages, temperature, max_tokens):
        seen.append(model)
        return ('{"score": 10}' if model == 'qwen3.8-27b' else 'est=50.0'), 20, 5
    monkeypatch.setattr(llm_client, 'make_default_transport', lambda: transport)
    backend = make_live_backend(cache_path=str(tmp_path/'cache.sqlite'))
    try:
        assert backend.generate_sync([('agent', 'test')]) == ['est=50.0']
        assert OutOfFamilyJudge(backend.judge_backend).score_sync('message') == (10.0, 'structured_fallback')
        assert seen == ['gemma-4-31b-it', 'qwen3.8-27b']
    finally:
        backend.close()
    assert backend.backend.cache.conn is None
    assert backend.judge_backend.backend.cache.conn is None
