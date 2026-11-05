"""CachedEmbedder (dedup + cache by text hash) and the sentence-transformers adapter (mocked; no network)."""
import inspect
from typing import ClassVar

import pytest
import torch

from bazt_lab.features.gnn.adapters import sbert
from bazt_lab.features.gnn.private.embedding import CachedEmbedder
from bazt_lab.features.gnn.tests.conftest import DIM, FakeEncoder, fake_vector


def test_rows_follow_the_input_order_and_equal_texts_share_a_vector():
    enc = FakeEncoder()
    out = CachedEmbedder(enc)(["b", "a", "b"])
    assert out.shape == (3, DIM) and out.dtype == torch.float32
    assert torch.equal(out[0], fake_vector("b")) and torch.equal(out[1], fake_vector("a"))
    assert torch.equal(out[0], out[2])


def test_each_distinct_text_is_encoded_once_in_one_batched_call():
    enc = FakeEncoder()
    CachedEmbedder(enc)(["a", "b", "a", "c", "b"])
    assert enc.calls == [["a", "b", "c"]]                      # first-seen order, one call


def test_cached_texts_are_not_encoded_again():
    enc = FakeEncoder()
    embed = CachedEmbedder(enc)
    first = embed(["a", "b"])
    out = embed(["b", "new", "a"])
    assert enc.calls == [["a", "b"], ["new"]]
    assert torch.equal(out[0], first[1]) and torch.equal(out[2], first[0])


def test_a_fully_cached_call_makes_no_encoder_call():
    enc = FakeEncoder()
    embed = CachedEmbedder(enc)
    embed(["a", "b"])
    embed(["b", "a", "a"])
    assert len(enc.calls) == 1


def test_a_text_keeps_the_vector_of_its_first_batch():
    """The vector of a text never changes once computed, whatever batch asks for it later."""

    class Drifting(FakeEncoder):
        def __call__(self, texts):
            return super().__call__(texts) * (1 + len(self.calls))      # a different scale on every call

    embed = CachedEmbedder(Drifting())
    first = embed(["a"])
    embed(["z"])
    assert torch.equal(embed(["z", "a"])[1], first[0])


def test_result_carries_no_graph():
    out = CachedEmbedder(FakeEncoder())(["a"])
    assert not out.requires_grad


def test_an_encoder_that_returns_the_wrong_number_of_rows_is_refused():
    with pytest.raises(ValueError, match=r"2 texts.*1 row"):
        CachedEmbedder(lambda texts: torch.zeros(1, DIM))(["a", "b"])


def test_no_texts_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        CachedEmbedder(FakeEncoder())([])


def test_non_string_input_is_refused():
    with pytest.raises(TypeError, match="str"):
        CachedEmbedder(FakeEncoder())(["a", None])


# ------------------------------------------------------------------------------------------ the adapter
class FakeSentenceTransformer:
    constructed: ClassVar[list] = []
    encode_calls: ClassVar[list] = []

    def __init__(self, name, **kwargs):
        FakeSentenceTransformer.constructed.append((name, kwargs))

    def encode(self, inputs, **kwargs):
        FakeSentenceTransformer.encode_calls.append((list(inputs), kwargs, torch.is_inference_mode_enabled()))
        return torch.ones(len(inputs), DIM, dtype=torch.float64)


@pytest.fixture
def fake_st(monkeypatch):
    FakeSentenceTransformer.constructed, FakeSentenceTransformer.encode_calls = [], []
    sbert.load_encoder.cache_clear()
    monkeypatch.setattr(sbert, "SentenceTransformer", FakeSentenceTransformer)
    yield FakeSentenceTransformer
    sbert.load_encoder.cache_clear()


def test_the_model_is_loaded_once_at_the_pinned_revision_on_cpu(fake_st):
    sbert.load_encoder("org/model", "abc123", 16)
    sbert.load_encoder("org/model", "abc123", 16)
    assert fake_st.constructed == [("org/model", {"revision": "abc123", "device": "cpu"})]


def test_encode_is_batched_unnormalised_and_runs_in_inference_mode(fake_st):
    out = sbert.load_encoder("org/model", "abc123", 16)(["x", "y"])
    (inputs, kwargs, inference), = fake_st.encode_calls
    assert inputs == ["x", "y"] and inference
    assert kwargs == {"batch_size": 16, "convert_to_tensor": True, "normalize_embeddings": False,
                      "show_progress_bar": False}
    assert out.dtype == torch.float32 and out.shape == (2, DIM)


def test_a_different_revision_is_a_different_model(fake_st):
    sbert.load_encoder("org/model", "rev1", 16)
    sbert.load_encoder("org/model", "rev2", 16)
    assert [kw["revision"] for _, kw in fake_st.constructed] == ["rev1", "rev2"]


def test_the_installed_library_accepts_the_arguments_the_adapter_passes():
    """Guards against API drift in sentence-transformers without downloading a model."""
    real = inspect.signature(sbert.SentenceTransformer.encode).parameters
    assert {"batch_size", "convert_to_tensor", "normalize_embeddings", "show_progress_bar"} <= set(real)
    init = inspect.signature(sbert.SentenceTransformer.__init__).parameters
    assert {"revision", "device"} <= set(init)


# ------------------------------------------------------------------------------------- the shared embedder
def test_the_shared_embedder_is_built_once_per_model_settings_and_reads_them_from_settings(settings, monkeypatch):
    from bazt_lab.features.gnn.private import embedding
    built = []

    def fake_load(model, revision, batch_size):
        built.append((model, revision, batch_size))
        return FakeEncoder()

    monkeypatch.setattr(sbert, "load_encoder", fake_load)
    monkeypatch.setattr(embedding, "_SHARED", {})
    a, b = embedding.shared_embedder(settings), embedding.shared_embedder(settings)
    assert a is b and built == [("fake/minilm", "rev0", 64)]
    other = {**settings, "gnn": {**settings["gnn"], "embedding": {**settings["gnn"]["embedding"], "revision": "rev1"}}}
    assert embedding.shared_embedder(other) is not a and len(built) == 2


def test_the_shared_embedder_names_a_missing_setting(settings, monkeypatch):
    from bazt_lab.core import public as core
    from bazt_lab.features.gnn.private import embedding
    monkeypatch.setattr(embedding, "_SHARED", {})
    del settings["gnn"]["embedding"]["revision"]
    with pytest.raises(core.Unset, match=r"gnn.embedding.revision"):
        embedding.shared_embedder(settings)
