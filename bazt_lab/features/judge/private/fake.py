"""FakeJudge: a Judge that scores texts by planted markers. No network, no files, no secrets.

For tests and the end-to-end fake run. A text that contains a marker gets that marker's P(yes) (the largest when
several are present). Any other text gets a small P(yes) drawn from a stream keyed by the text, so benign scores are
spread out (a calibration at alpha needs a distribution, not a constant) and depend on the text alone. The scores
are log-probabilities, the same scale and sign as the live judge's. Numbers it produces are pipeline checks.
"""
from __future__ import annotations

import math
from typing import List, Mapping

import torch

from bazt_lab.core import public as core
from bazt_lab.features.judge.private.config import _LONE_SURROGATE


class FakeJudge:
    name = "fake"

    def __init__(self, markers: Mapping[str, float], *, benign_max: float = 0.05) -> None:
        """markers: substring planted in attack texts -> P(yes) for texts that contain it, in (0, 1].
        benign_max: benign texts get P(yes) in (0, benign_max]. A test double's constant, not an experiment value."""
        for marker, p in markers.items():
            if not isinstance(marker, str) or not marker:
                raise ValueError(f"a marker must be a non-empty string, got {marker!r}")
            if not 0.0 < p <= 1.0:
                raise ValueError(f"P(yes) of marker {marker!r} must be in (0, 1], got {p!r}")
        if not 0.0 < benign_max < 1.0:
            raise ValueError(f"benign_max must be in (0, 1), got {benign_max!r}")
        self.markers, self.benign_max = dict(markers), benign_max
        self.calls = 0                 # score() calls, empty ones included
        self.texts_scored = 0

    def _p_yes(self, text: str) -> float:
        if not isinstance(text, str):
            raise TypeError(f"the judge scores str, got {type(text).__name__}")
        planted = [p for marker, p in self.markers.items() if marker in text]
        if planted:
            return max(planted)
        u = float(core.uniforms(core.stream("fake_judge", _LONE_SURROGATE.sub("�", text)), (1,))[0])         # [0, 1)
        return self.benign_max * (1.0 - u)                                         # (0, benign_max]

    def score(self, texts: List[str]) -> torch.Tensor:
        texts = list(texts)
        self.calls += 1
        self.texts_scored += len(texts)
        return torch.tensor([math.log(self._p_yes(t)) for t in texts], dtype=torch.float64)
