"""Mock auditor: an auditor preset as a point on an equal-variance binormal ROC model (spec 5, D56).

A text's score is `z + mean`, with z ~ N(0, 1) drawn from a stream keyed without the arm and `mean` set by the text's
class: benign 0, suspicious-looking benign (a decoy header) `d_decoy`, payload `d_payload`. One reference threshold
reproduces the preset exactly: P(score > tau) is FPR_benign, FPR_decoy and TPR for the three classes.

OPEN ITEM of the spec (11): how the binormal parameters follow from the two false-positive classes. Method
`shared_threshold` (implemented): d'(payload vs benign) = z(TPR) - z(FPR_benign), d'(payload vs decoy) =
z(TPR) - z(FPR_decoy), realised on ONE score axis so a single auditor has a single score per text.
Method `benign_fpr_only`: the decoy class is not modelled (decoys score like benign texts).
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Mapping, Optional

from bazt_lab.core import public as core

METHODS = ("shared_threshold", "benign_fpr_only")
_z = NormalDist().inv_cdf


@dataclass(frozen=True)
class AuditorModel:
    tpr: float
    fpr_benign: float
    fpr_decoy: float
    d_payload: float
    d_decoy: float
    reference_threshold: float        # the score threshold at which the preset's rates hold


def preset_rates(settings: Mapping, preset: Optional[str], tpr: Optional[float]):
    """(TPR, FPR benign, FPR decoy) of a preset; 'adaptive' takes its TPR from the attacker and the main FPRs."""
    main = core.get(settings, "auditor_presets.main")
    name = main if preset in (None, "main") else preset
    if name == "adaptive":
        lo, hi = core.get(settings, "auditor_presets.adaptive_tpr")
        if tpr is None:
            raise ValueError("preset 'adaptive' needs the attacker's detection rate: attack_params['auditor_tpr']")
        if not lo <= tpr <= hi:
            raise ValueError(f"auditor_tpr={tpr} is outside auditor_presets.adaptive_tpr = [{lo}, {hi}]")
        return (tpr, *core.get(settings, f"auditor_presets.{main}")[1:])
    if name in ("main", "adaptive_tpr") or not isinstance(core.get(settings, "auditor_presets").get(name), list):
        raise ValueError(f"unknown auditor preset {name!r}")
    return tuple(core.get(settings, f"auditor_presets.{name}"))


def auditor_model(settings: Mapping, preset: Optional[str] = None, tpr: Optional[float] = None) -> AuditorModel:
    method = core.get(settings, "mock.auditor.binormal_method")
    if method not in METHODS:
        raise ValueError(f"mock.auditor.binormal_method={method!r}; implemented: {METHODS}")
    t, fb, fd = preset_rates(settings, preset, tpr)
    if not all(0.0 < r < 1.0 for r in (t, fb, fd)):
        raise ValueError(f"preset {preset!r}: TPR and FPRs must be strictly between 0 and 1, got {(t, fb, fd)}")
    d_decoy = _z(fd) - _z(fb) if method == "shared_threshold" else 0.0
    return AuditorModel(t, fb, fd, _z(t) - _z(fb), d_decoy, -_z(fb))
