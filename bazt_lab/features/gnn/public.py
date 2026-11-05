"""Public interface of the gnn feature (see CONTRACT.md)."""
from bazt_lab.core import public as core
from bazt_lab.features.gnn.private.arm import GNNArm, GNNRecalibrated, deploy_family
from bazt_lab.features.gnn.private.logs import log_conversations, read_conversations
from bazt_lab.features.gnn.private.train import (
           TrainingDataIncomplete,
           checkpoint_path,
           train_family,
           train_unit,
           training_finished,
)

ARMS = (
    core.ArmSpec("GNN", GNNArm, platforms=("llm",), needs_truth=False, calibrated=False),
    core.ArmSpec("GNN_recalibrated", GNNRecalibrated, platforms=("llm",), needs_truth=False, calibrated=True),
)

__all__ = ["ARMS", "GNNArm", "deploy_family", "GNNRecalibrated", "TrainingDataIncomplete", "checkpoint_path", "log_conversations", "read_conversations",
           "train_family", "train_unit", "training_finished"]
