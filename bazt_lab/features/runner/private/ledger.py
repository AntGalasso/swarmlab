"""The last error of each block, kept beside the results so `status` can show it."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from bazt_lab.features.runner.private import files

FILE = "last_errors.json"


def read(root: Path, profile: str) -> Dict[str, str]:
    path = Path(root) / profile / FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def record(root: Path, profile: str, block: str, message: str | None) -> None:
    """Remember the error of `block`, or forget it (message None) once the block ran."""
    errors = read(root, profile)
    if message is None and block not in errors:
        return
    if message is None:
        del errors[block]
    else:
        errors[block] = message
    (Path(root) / profile).mkdir(parents=True, exist_ok=True)
    files.write_json(Path(root) / profile / FILE, errors)


SKIPPED = "skipped_cells.json"


def read_skipped(root: Path, profile: str) -> Dict[str, dict]:
    path = Path(root) / profile / SKIPPED
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def record_skipped(root: Path, profile: str, block: str, units: int, cells: list) -> None:
    """Remember the units a block left out on purpose (a GNN cell without a GNN), or forget them when there are none."""
    skipped = read_skipped(root, profile)
    if not cells and block not in skipped:
        return
    if cells:
        skipped[block] = {"units": units, "cells": cells}
    else:
        del skipped[block]
    (Path(root) / profile).mkdir(parents=True, exist_ok=True)
    files.write_json(Path(root) / profile / SKIPPED, skipped)
