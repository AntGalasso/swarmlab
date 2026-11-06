"""FakeJudge: deterministic, offline, scores by planted markers."""
import math

import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.judge.private.fake import FakeJudge

MARKERS = {"[PLANT-A]": 0.95, "[PLANT-B]": 0.6}


def test_is_a_core_judge_and_returns_float64_log_probabilities():
    j = FakeJudge(MARKERS)
    assert isinstance(j, core.Judge)
    out = j.score(["benign one", "has [PLANT-A] inside", "benign two"])
    assert out.dtype == torch.float64 and out.shape == (3,) and bool((out <= 0).all())


def test_marked_texts_score_their_marker_probability_and_the_highest_marker_wins():
    out = FakeJudge(MARKERS).score(["x [PLANT-A]", "x [PLANT-B]", "[PLANT-B] and [PLANT-A]"])
    torch.testing.assert_close(out, torch.tensor([math.log(0.95), math.log(0.6), math.log(0.95)],
                                                 dtype=torch.float64))


def test_benign_texts_stay_below_benign_max_and_are_spread_out():
    j = FakeJudge(MARKERS, benign_max=0.05)
    out = j.score([f"benign text {i}" for i in range(50)])
    assert bool((out <= math.log(0.05)).all()) and torch.unique(out).numel() > 40


def test_scores_depend_only_on_the_text_not_on_batch_or_instance():
    texts = [f"t{i}" for i in range(10)] + ["z [PLANT-A]"]
    a = FakeJudge(MARKERS).score(texts)
    b = torch.cat([FakeJudge(MARKERS).score([t]) for t in texts])
    c = FakeJudge(MARKERS).score(texts[::-1]).flip(0)
    assert torch.equal(a, b) and torch.equal(a, c)


def test_empty_batch_and_counters():
    j = FakeJudge(MARKERS)
    assert j.score([]).shape == (0,) and j.score([]).dtype == torch.float64
    j.score(["a", "b"])
    j.score(["c"])
    assert (j.calls, j.texts_scored) == (4, 3)


@pytest.mark.parametrize("markers, benign_max", [
    ({"": 0.9}, 0.05), ({"m": 0.0}, 0.05), ({"m": 1.5}, 0.05), ({"m": 0.9}, 0.0), ({"m": 0.9}, 1.0),
])
def test_invalid_parameters_are_refused(markers, benign_max):
    with pytest.raises(ValueError):
        FakeJudge(markers, benign_max=benign_max)


def test_non_string_input_is_a_type_error():
    with pytest.raises(TypeError):
        FakeJudge(MARKERS).score([1])


def test_a_lone_surrogate_is_scored_like_the_live_judge_does():
    out = FakeJudge(MARKERS).score(["bad \ud83d text", "bad \ufffd text"])
    assert bool(torch.isfinite(out).all()) and out[0] == out[1]
