"""thresholds.json: the calibrated thresholds of a calibration block, one record per cell (arm, topology, N, backbone,
task). Written one cell at a time, so an interrupted calibration keeps the cells it finished."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping

from bazt_lab.features.runner.private import files

FILE = "thresholds.json"


def load(path: Path) -> Dict[tuple, Dict[str, float]]:
    """{cell: {statistic: threshold}}; empty when nothing is calibrated yet."""
    if not path.exists():
        return {}
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    return {(c["arm"], c["topology"], c["agents"], c["backbone"], c["task"]): c["thresholds"] for c in cells}


def saved_alpha(path: Path) -> float | None:
    """The alpha the saved thresholds were calibrated at; None when nothing is saved yet."""
    return json.loads(path.read_text(encoding="utf-8"))["alpha"] if path.exists() else None


def save_cell(path: Path, alpha: float, cell: tuple, thresholds: Mapping[str, float], view: Mapping,
              diagnostics: Mapping | None = None) -> None:
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"alpha": alpha, "settings": dict(view),
                                                                                "cells": []}
    arm, topology, agents, backbone, task = cell
    record = dict(arm=arm, topology=topology, agents=agents, backbone=backbone, task=task, thresholds=dict(thresholds))
    if diagnostics:
        record["diagnostics"] = dict(diagnostics)
    data["cells"].append(record)
    files.write_json(path, data)


def sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def describe(cell: tuple) -> str:
    return "arm={} topology={} agents={} backbone={} task={}".format(*cell)
