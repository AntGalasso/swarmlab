"""Public interface of bazt_lab.features.mock_engine: the mock platform of the experiments. See CONTRACT.md."""
from bazt_lab.features.mock_engine.private.enforce import Persistent, apply_actions
from bazt_lab.features.mock_engine.private.graphs import TOPOLOGIES
from bazt_lab.features.mock_engine.private.platform import MockPlatform
from bazt_lab.features.mock_engine.private.world import ATTACKS

# apply_actions is the reference pricing and application of an Actions object to one round of deliveries (contract order,
# persistent states, per-delivery costs, budget rejection; see private/enforce.py): other platforms test against it.
__all__ = ["ATTACKS", "MockPlatform", "Persistent", "TOPOLOGIES", "apply_actions"]
