"""Settings are read with core.get (a missing key raises Unset naming it); evidence becomes a posterior by log-odds."""
import math

import pytest
import torch

from bazt_lab.core import public as core
from bazt_lab.features.bazt_slot.private import params as P
from bazt_lab.features.bazt_slot.private import risk as R
from bazt_lab.features.bazt_slot.tests.testkit import TEST_VALUES, make_settings

F64 = torch.float64


def t(x, dtype=F64):
    return torch.as_tensor(x, dtype=dtype)


# ----------------------------------------------------------------------------------------------- settings
def test_params_are_read_from_settings():
    p = P.read(make_settings())
    assert (p.decay, p.iterations, p.sink_weight, p.utility_weight) == (0.5, 300, 1.0, 1.0)
    assert p.rho["isolation"] == 1.0 and p.rho["sanitize"] == 0.15 and set(p.rho) == {
        "sanitize", "delay", "edge_quarantine", "rate_limit", "privilege_degradation", "isolation"}
    assert (p.flag_logodds, p.clear_logodds, p.memory_decay, p.prior, p.q_inspect) == (4.0, 3.0, 0.8, 0.1, 0.8)
    assert (p.sanitize_success, p.rate_limit) == (0.9, 1)


@pytest.mark.parametrize("key", sorted(TEST_VALUES))
def test_a_missing_setting_refuses_and_names_the_key(key):
    s = make_settings()
    node = s
    *path, leaf = key.split(".")
    for part in path:
        node = node[part]
    del node[leaf]
    with pytest.raises(core.Unset) as e:
        P.read(s)
    assert e.value.args[0] == key


@pytest.mark.parametrize("key, bad", [
    ("bazt.katz_decay", 0.0), ("bazt.katz_decay", 1.5), ("bazt.eigen_iterations", 0), ("bazt.eigen_iterations", 2.5),
    ("bazt.sink_weight", 0.5), ("bazt.utility_weight", -1.0), ("bazt.utility_loss.isolation", -0.1),
    ("bazt.flag_logodds", 0.0), ("bazt.clear_logodds", -1.0), ("bazt.memory_decay", 1.0), ("bazt.prior", 0.0),
    ("bazt.prior", 1.0), ("bazt.inspect_conclusive", 0.0), ("bazt.inspect_conclusive", 1.2),
    ("runtime.sanitize_success", 1.2), ("runtime.rate_limit_per_identity", 0),
])
def test_an_out_of_range_setting_refuses_and_names_the_key(key, bad):
    with pytest.raises(ValueError, match=key.replace(".", r"\.")):
        P.read(make_settings(**{key.replace(".", "__"): bad}))


# ------------------------------------------------------------------------------------------------ evidence
def test_logit_and_sigmoid_are_inverse_and_the_logit_is_finite_at_the_ends():
    p = t([0.001, 0.2, 0.5, 0.97])
    assert torch.allclose(R.sigmoid(R.logit(p)), p)
    assert R.logit(t([0.0, 1.0])).isfinite().all()


def test_free_evidence_is_the_monitor_for_active_agents_and_the_prior_otherwise():
    monitor = t([[0.9, 0.2, 0.7]])
    active = t([[1, 1, 0]], torch.bool)
    free = R.free_logodds(monitor, active, prior=0.1)
    assert torch.allclose(R.sigmoid(free), t([[0.9, 0.2, 0.1]]))


def test_free_evidence_without_a_monitor_is_the_prior_everywhere():
    free = R.free_logodds(None, torch.ones(2, 3, dtype=torch.bool), prior=0.25)
    assert torch.allclose(R.sigmoid(free), torch.full((2, 3), 0.25, dtype=F64))


def test_memory_accumulates_a_flag_decays_and_a_clear_verdict_subtracts():
    V = torch.zeros(1, 3, dtype=F64)
    flagged = t([[1, 0, 0]], torch.bool)
    audited = t([[1, 1, 0]], torch.bool)
    V = R.remember(V, flagged, audited, decay=0.5, flag=4.0, clear=3.0)
    assert V.tolist() == [[4.0, -3.0, 0.0]]
    V = R.remember(V, torch.zeros_like(flagged), torch.zeros_like(audited), decay=0.5, flag=4.0, clear=3.0)
    assert V.tolist() == [[2.0, -1.5, 0.0]]


def test_a_flagged_sender_is_never_also_cleared():
    V = R.remember(torch.zeros(1, 1, dtype=F64), t([[1]], torch.bool), t([[1]], torch.bool), 0.5, 4.0, 3.0)
    assert V.item() == 4.0


def test_eligibility_needs_free_evidence_over_the_calibrated_threshold_or_a_positive_verdict():
    free = t([[2.0, 0.0, 2.0, -1.0]])
    V = t([[0.0, 0.0, -3.0, 4.0]])
    assert R.eligible(free, V, threshold=1.0).tolist() == [[True, False, False, True]]


def test_a_missing_threshold_means_only_a_positive_verdict_makes_an_agent_eligible():
    free = t([[50.0, 0.0]])
    V = t([[0.0, 1.0]])
    assert R.eligible(free, V, threshold=math.inf).tolist() == [[False, True]]
