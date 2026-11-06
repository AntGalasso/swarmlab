"""Public interface of the runner feature: blocks and unit expansion, settings, launch, sizing and the CLI.

The composition root (bazt_lab/app.py) supplies a `Wiring`; the runner imports no other feature. See CONTRACT.md.
"""
from bazt_lab.features.runner.private.blocks import BLOCKS, CALIBRATION_BLOCK, Block
from bazt_lab.features.runner.private.cli import list_blocks, main
from bazt_lab.features.runner.private.launch import FREEZE_MARKER, RESULTS_ROOT, run_block
from bazt_lab.features.runner.private.runtime import PRE_REGISTERED, RUNTIME_BY_PLATFORM, load_settings, missing_runtime
from bazt_lab.features.runner.private.sizing import cache_savings, dry_run, llm_calls
from bazt_lab.features.runner.private.units import (
    GROUP_FIELDS,
    MockGrid,
    UnitList,
    capped_cells,
    cell_key,
    group_key,
    unit_id,
    units_gnn_train,
    units_llm_adaptive_check,
    units_llm_calibration,
    units_llm_capped,
    units_llm_headline,
    units_mock_calibration,
    units_mock_frontier,
    units_mock_heldout,
)
from bazt_lab.features.runner.private.wiring import FeatureMissing, Wiring

__all__ = [
    "BLOCKS", "Block", "CALIBRATION_BLOCK", "FREEZE_MARKER", "FeatureMissing", "GROUP_FIELDS", "MockGrid",
    "PRE_REGISTERED", "RESULTS_ROOT", "RUNTIME_BY_PLATFORM", "UnitList", "Wiring", "cache_savings", "capped_cells",
    "cell_key", "dry_run", "group_key", "list_blocks", "llm_calls", "load_settings", "main", "missing_runtime",
    "run_block", "unit_id", "units_gnn_train", "units_llm_adaptive_check", "units_llm_calibration",
    "units_llm_capped", "units_llm_headline", "units_mock_calibration", "units_mock_frontier", "units_mock_heldout",
]
