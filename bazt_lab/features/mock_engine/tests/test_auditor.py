"""The mock auditor: a preset (TPR, FPR benign, FPR suspicious-looking benign) as a point on an equal-variance
binormal ROC model (spec 5, D56). The derivation is an open item of the spec, so it sits behind a settings key."""
from statistics import NormalDist

import pytest

from bazt_lab.core import public as core
from bazt_lab.features.mock_engine.private import auditor
from bazt_lab.features.mock_engine.tests.helpers import make_settings

N01 = NormalDist()
z = N01.inv_cdf


def tail(mean, threshold):
    """P(score > threshold) for a unit-variance normal score with this mean."""
    return 1.0 - N01.cdf(threshold - mean)


@pytest.mark.parametrize("preset", ["strong_model", "good_classifier", "weak_judge"])
def test_shared_threshold_reproduces_the_preset_point_exactly(preset):
    s = make_settings()
    tpr, fpr_b, fpr_s = core.get(s, f"auditor_presets.{preset}")
    m = auditor.auditor_model(s, preset)
    assert m.d_payload == pytest.approx(z(tpr) - z(fpr_b))                  # d' = z(TPR) - z(FPR), per false-positive class
    assert tail(0.0, m.reference_threshold) == pytest.approx(fpr_b)          # benign texts: N(0, 1)
    assert tail(m.d_decoy, m.reference_threshold) == pytest.approx(fpr_s)    # suspicious-looking benign texts
    assert tail(m.d_payload, m.reference_threshold) == pytest.approx(tpr)    # payload texts
    assert m.d_payload - m.d_decoy == pytest.approx(z(tpr) - z(fpr_s))       # the payload-vs-decoy d' of the second class


def test_the_main_condition_is_the_good_classifier_and_none_means_main():
    s = make_settings()
    assert auditor.auditor_model(s) == auditor.auditor_model(s, "good_classifier") == auditor.auditor_model(s, "main")


def test_benign_fpr_only_ignores_the_decoy_class():
    m = auditor.auditor_model(make_settings(mock__auditor__binormal_method="benign_fpr_only"))
    assert m.d_decoy == 0.0 and m.d_payload == pytest.approx(z(0.95) - z(0.02))


def test_method_is_a_required_key_and_unknown_methods_are_refused():
    s = make_settings()
    del s["mock"]["auditor"]
    with pytest.raises(core.Unset) as e:
        auditor.auditor_model(s)
    assert e.value.args[0] == "mock.auditor.binormal_method"
    with pytest.raises(ValueError, match="binormal_method"):
        auditor.auditor_model(make_settings(mock__auditor__binormal_method="magic"))


def test_adaptive_preset_takes_its_tpr_from_the_attacker_and_fprs_from_the_main_preset():
    s = make_settings()
    m = auditor.auditor_model(s, "adaptive", tpr=0.2)
    assert m.tpr == 0.2 and (m.fpr_benign, m.fpr_decoy) == (0.02, 0.12)
    assert m.d_payload == pytest.approx(z(0.2) - z(0.02))
    with pytest.raises(ValueError, match="auditor_tpr"):
        auditor.auditor_model(s, "adaptive")
    with pytest.raises(ValueError, match="adaptive_tpr"):
        auditor.auditor_model(s, "adaptive", tpr=0.9)


def test_unknown_preset_and_degenerate_rates_are_refused():
    s = make_settings()
    with pytest.raises(ValueError, match="preset"):
        auditor.auditor_model(s, "perfect")
    with pytest.raises(ValueError, match="preset"):
        auditor.auditor_model(s, "adaptive_tpr")
    core.put(s, "auditor_presets.weak_judge", [1.0, 0.2, 0.45])
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        auditor.auditor_model(s, "weak_judge")
