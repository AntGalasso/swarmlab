"""The tie-breaking jitter of C's statistic is off in the tests that check exact hand-computed values; the modules
named in JITTERED run the real thing."""
import pytest

from bazt_lab.features.conventional.private.arm_c import ArmC

JITTERED = ("test_arm_c_jitter", "test_arm_c_calibration")


@pytest.fixture(autouse=True)
def _exact_statistics_unless_jittered(request, monkeypatch):
    if request.module.__name__.rpartition(".")[2] not in JITTERED:
        monkeypatch.setattr(ArmC, "jitter", False)
