"""Public interface of the BAZT slot (decision D86). See CONTRACT.md."""
from bazt_lab.core import public as core
from bazt_lab.features.bazt_slot.adapters.upstream import BaztUpstream
from bazt_lab.features.bazt_slot.private.bazt import VARIANTS, Bazt

ARMS = (
    *(core.ArmSpec(name, (lambda name=name: Bazt(name)), calibrated=True) for name in VARIANTS),
    core.ArmSpec("BAZT_upstream", BaztUpstream, calibrated=True),
)

__all__ = ["ARMS"]
