"""The mock model parameters ([mock.model] and the run-time variables the engine reads): required, validated."""
import pytest

from bazt_lab.core import public as core
from bazt_lab.features.mock_engine.private import model as M
from bazt_lab.features.mock_engine.tests.helpers import make_settings


def test_every_parameter_comes_from_the_settings():
    m = M.load_model(make_settings())
    assert (m.m, m.a, m.b, m.mu_inf, m.mu_dec, m.sd) == (2, 2.5, 2.0, 2.0, 2.8, 0.65)
    assert (m.beta, m.p_stay, m.p_decoy, m.decoys, m.burst_span) == (0.35, 0.8, 0.9, 3, 1)
    assert (m.reading_mean, m.reading_sd, m.bias) == (50.0, 10.0, 8.0)
    assert (m.eps_correct, m.sanitize_success, m.rate_limit) == (2.0, 1.0, 1)


@pytest.mark.parametrize("key", ["mock.model.m", "mock.model.a", "mock.model.b", "mock.model.mu_inf", "mock.model.mu_dec",
                                 "mock.model.sd", "mock.model.beta", "mock.model.p_stay", "mock.model.p_decoy",
                                 "mock.model.decoys", "mock.model.reading_mean", "mock.model.reading_sd",
                                 "mock.model.bias", "mock.model.burst_span", "runtime.eps_correct",
                                 "runtime.sanitize_success", "runtime.rate_limit_per_identity"])
def test_a_missing_parameter_refuses_with_its_key_name(key):
    s = make_settings()
    *path, leaf = key.split(".")
    node = s
    for part in path:
        node = node[part]
    del node[leaf]
    with pytest.raises(core.Unset) as e:
        M.load_model(s)
    assert e.value.args[0] == key


@pytest.mark.parametrize("key,value", [("mock.model.beta", 1.5), ("mock.model.p_stay", -0.1), ("mock.model.p_decoy", 2),
                                       ("runtime.sanitize_success", 1.1), ("mock.model.m", 0), ("mock.model.m", 1.5),
                                       ("mock.model.sd", -1), ("mock.model.decoys", -1), ("mock.model.burst_span", 0),
                                       ("mock.model.reading_sd", -1), ("runtime.eps_correct", -0.5),
                                       ("runtime.rate_limit_per_identity", 0), ("runtime.rate_limit_per_identity", True),
                                       ("mock.model.beta", "0.3")])
def test_out_of_range_values_are_refused(key, value):
    with pytest.raises(ValueError, match=key.split(".")[-1]):
        M.load_model(make_settings(**{key.replace(".", "__"): value}))
