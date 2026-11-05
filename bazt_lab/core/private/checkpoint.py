"""Progressive checkpoints: rows are appended as they are produced, state is saved atomically.

Layout under a block directory:
  units/<unit_id>.jsonl          rows; a unit is done when units/<unit_id>.done exists
  state/<unit_id>.pt             resumable state (torch.save of a dict of tensors and plain values)
A crash leaves at most the rows of an unfinished unit; `resume_rows` returns them so the caller continues after
the last saved row instead of starting again.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Mapping

import torch


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class CheckpointStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        (self.root / "units").mkdir(parents=True, exist_ok=True)
        (self.root / "state").mkdir(parents=True, exist_ok=True)

    def _rows(self, unit_id: str) -> Path:
        return self.root / "units" / f"{unit_id}.jsonl"

    def is_done(self, unit_id: str) -> bool:
        return (self.root / "units" / f"{unit_id}.done").exists()

    def append_rows(self, unit_id: str, rows: Iterable[Mapping]) -> None:
        """Append rows and fsync, so a row written is a row kept."""
        with self._rows(unit_id).open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, default=float) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def resume_rows(self, unit_id: str) -> List[dict]:
        """Rows already saved for an unfinished unit (a torn last line is dropped)."""
        path = self._rows(unit_id)
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break
        return rows

    def mark_done(self, unit_id: str) -> None:
        _atomic_write_bytes(self.root / "units" / f"{unit_id}.done", b"")
        state = self.root / "state" / f"{unit_id}.pt"
        if state.exists():
            state.unlink()

    def save_state(self, unit_id: str, state: Mapping) -> None:
        tmp = self.root / "state" / f"{unit_id}.pt.tmp"
        torch.save(dict(state), tmp)
        os.replace(tmp, self.root / "state" / f"{unit_id}.pt")

    def load_state(self, unit_id: str) -> dict | None:
        path = self.root / "state" / f"{unit_id}.pt"
        return torch.load(path, weights_only=True) if path.exists() else None
