"""Public interface of the conventional zero-trust arms A, B and C (spec 2.1 to 2.4). See CONTRACT.md."""
from bazt_lab.core import public as core
from bazt_lab.features.conventional.private.arm_a import ArmA
from bazt_lab.features.conventional.private.arm_b import ArmB
from bazt_lab.features.conventional.private.arm_c import ArmC

# A has no tunable score and no capped variant; B and C report a decision statistic that is calibrated at alpha.
ARMS = (
    core.ArmSpec("A", ArmA),
    core.ArmSpec("B", ArmB, calibrated=True),
    core.ArmSpec("C", ArmC, calibrated=True),
)

__all__ = ["ARMS"]
