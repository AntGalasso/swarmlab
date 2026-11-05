"""Progressive checkpoints: rows are appended as they are produced, state is saved atomically.

Layout under a block directory:
  units/<unit_id>.jsonl          rows; a unit is done when units/<unit_id>.done exists (per-unit API, LLM side)
  units/batch-<hash>.jsonl       rows of a whole batch (`append_batch`, mock side); <hash> is the blake2b of the unit ids
  units/batch-<hash>.done        JSON list of the unit ids of that batch: the batch is committed when this file exists
  state/<unit_id>.pt             resumable state (torch.save of a dict of tensors and plain values)
A crash leaves at most the rows of an unfinished unit; `resume_rows` returns them so the caller continues after
the last saved row instead of starting again.

Batch commit order: rows file (written under a temporary name, fsynced once, renamed into place), then the done list
(one atomic write). A crash between the two leaves a complete rows file with no done list (an orphan): its units are not
done, and the next `append_batch` removes every orphan before writing, so a rerun never leaves a row twice, whatever
batch composition it uses. One writer per block directory. Readers glob units/*.jsonl; every row carries its unit.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, List, Mapping

import torch


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _drop_torn_tail(path: Path) -> None:
    """Cut a last line left unfinished by a crash, so the next append starts on a fresh line."""
    with path.open("rb+") as f:
        size = f.seek(0, os.SEEK_END)
        if size == 0:
            return
        f.seek(size - 1)
        if f.read(1) == b"\n":
            return
        f.seek(0)
        data = f.read()
        f.seek(0)
        f.truncate(data.rfind(b"\n") + 1)
        f.flush()
        os.fsync(f.fileno())


BATCH = "batch-"


class CheckpointStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        (self.root / "units").mkdir(parents=True, exist_ok=True)
        (self.root / "state").mkdir(parents=True, exist_ok=True)
        self._done: set = set()                 # unit ids with a done marker (per-unit file or a batch done list)
        self._committed: set = set()            # stems of the committed batch files
        self._orphans_cleared = False
        for entry in os.scandir(self.root / "units"):
            if not entry.name.endswith(".done"):
                continue
            stem = entry.name[:-5]
            if entry.stat().st_size == 0:
                self._done.add(stem)
            else:
                self._done.update(json.loads(Path(entry.path).read_text(encoding="utf-8")))
                self._committed.add(stem)

    def _rows(self, unit_id: str) -> Path:
        return self.root / "units" / f"{unit_id}.jsonl"

    def is_done(self, unit_id: str) -> bool:
        return unit_id in self._done

    def append_batch(self, rows_by_unit: Mapping[str, Iterable[Mapping]]) -> None:
        """Write the rows of a whole batch to one file (one fsync) and mark all its units done in one atomic write."""
        ids = list(rows_by_unit)
        name = BATCH + hashlib.blake2b("\n".join(ids).encode(), digest_size=8).hexdigest()
        units = self.root / "units"
        self._clear_orphans()
        lines = [json.dumps(row, default=float) + "\n" for rows in rows_by_unit.values() for row in rows]
        tmp = units / f"{name}.jsonl.tmp"
        with tmp.open("w", encoding="utf-8") as f:
            f.write("".join(lines))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, units / f"{name}.jsonl")
        _atomic_write_bytes(units / f"{name}.done", json.dumps(ids).encode())
        self._committed.add(name)
        self._done.update(ids)
        states = self.root / "state"
        for entry in os.scandir(states):                 # a finished unit keeps no resumable state
            if entry.name.endswith(".pt") and entry.name[:-3] in rows_by_unit:
                os.unlink(entry.path)

    def _clear_orphans(self) -> None:
        """Remove batch row files without a done list (a crash before the commit): their units are recomputed."""
        if self._orphans_cleared:
            return
        for path in (self.root / "units").glob(BATCH + "*.jsonl"):
            if path.stem not in self._committed:
                path.unlink()
        self._orphans_cleared = True

    def append_rows(self, unit_id: str, rows: Iterable[Mapping]) -> None:
        """Append rows and fsync, so a row written is a row kept."""
        path = self._rows(unit_id)
        if path.exists():
            _drop_torn_tail(path)
        with path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, default=float) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def resume_rows(self, unit_id: str) -> List[dict]:
        """Rows already saved for an unfinished unit (a torn last line is dropped)."""
        path = self._rows(unit_id)
        if not path.exists():
            return []
        _drop_torn_tail(path)
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break
        return rows

    def mark_done(self, unit_id: str) -> None:
        _atomic_write_bytes(self.root / "units" / f"{unit_id}.done", b"")
        self._done.add(unit_id)
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
