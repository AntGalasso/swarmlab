"""Progressive checkpoints: rows are appended as they are produced, state is saved atomically.

Layout under a block directory:
  units/<unit_id>.jsonl          rows; a unit is done when units/<unit_id>.done exists (per-unit API, LLM side; a unit
                                 id may not start with "batch-")
  units/batch-<hash>.jsonl       rows of a whole batch (`append_batch`, mock side); <hash> is the blake2b of the unit ids
  units/batch-<hash>.done        JSON list of the unit ids of that batch: the batch is committed when this file exists
  state/<unit_id>.pt             resumable state (torch.save of a dict of tensors and plain values)
A crash leaves at most the rows of an unfinished unit; `resume_rows` returns them so the caller continues after
the last saved row instead of starting again.

Batch commit order: rows file (written under a temporary name, fsynced once, renamed into place), then the done list
(one atomic write). A crash between the two leaves a complete rows file with no done list (an orphan): its units are not
done, and the next `append_batch` removes every orphan before writing, so a rerun never leaves a row twice, whatever
batch composition it uses. One writer per write folder: parallel workers each use their own `shard` (units/<shard>/ and state/<shard>/),
and every store reads the done markers of all shards.

The done rule, for the store and for readers of rows: a rows file <stem>.jsonl in units/ or in a shard is committed when
<stem>.done is in the same folder (a batch done list or a unit's marker). Readers of rows take `committed_row_files`: a
plain recursive glob of units/**/*.jsonl would also read orphans (a store removes them in its own write folder only, so
one left in a shard no later launch writes to stays) and the rows of a unit still running, or abandoned in one shard and
run again in another. Every row carries its unit.

Reading other shards: on Windows, os.replace (MoveFileEx) still holds the file it renamed with DELETE access for a moment
after the new name is visible, and Python's open() does not share delete access, so a done list another worker has
just committed can raise PermissionError (errno 13) when read. `refresh` skips a done list it cannot read or parse yet
and reads it on a later call; its units are not done until then. A done list never changes after its commit, so each
one is read once per store.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping

import torch

REPLACE_RETRIES = 20            # Windows: a reader (uploader, antivirus, editor) holding the target makes a replace
REPLACE_WAIT_SECONDS = 0.05     # fail for a moment with PermissionError; retry instead of stopping the run


def replace(src: Path, dst: Path) -> None:
    """os.replace that waits out a transient Windows sharing violation (PermissionError) before giving up."""
    for attempt in range(REPLACE_RETRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == REPLACE_RETRIES - 1:
                raise
            time.sleep(REPLACE_WAIT_SECONDS * (attempt + 1))


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    replace(tmp, path)


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
ROWS, DONE = ".jsonl", ".done"
_BATCH_DONE = re.compile(BATCH + r"[0-9a-f]{16}\.done")


def _folders(units: str) -> Iterator[tuple[str, list]]:
    """(folder, entry names) of units/ and of every worker shard in it (units/<shard>/; a shard name has no dot). A dotless
    file, or a shard folder removed since the listing, is skipped."""
    listing = os.listdir(units)
    yield units, listing
    for name in listing:
        if "." in name:
            continue
        folder = os.path.join(units, name)
        try:
            names = os.listdir(folder)
        except OSError:
            continue
        yield folder, names


def committed_row_files(root: Path) -> tuple[List[Path], int]:
    """The rows files of a block directory (the `root` of its CheckpointStore) that a store has committed, in units/ and
    in every worker shard units/<shard>/ whatever worker count each launch used, units/ first, then the shards and the
    files in name order; and the number of rows files not committed (not read: they are not results, and their units
    are committed elsewhere or not at all).

    The store's done rule: <stem>.jsonl is committed when <stem>.done is in the same folder, the done list of a batch
    (batch-<16 hex>.done, written after its rows file) or the done marker of a unit (written after its last row). A done
    list counts once it exists, even while another process still holds it (`refresh` only waits to read its ids).
    Raises FileNotFoundError if the block has no units/ folder."""
    files, uncommitted = [], 0
    top, *shards = _folders(str(Path(root) / "units"))
    for folder, names in [top, *sorted(shards)]:
        present = set(names)
        for name in sorted(n for n in names if n.endswith(ROWS)):
            if name[:-len(ROWS)] + DONE in present:
                files.append(Path(folder, name))
            else:
                uncommitted += 1
    return files, uncommitted


def _read_done_list(path: Path) -> list | None:
    """The unit ids in a batch done list, or None when it cannot be read or parsed right now (held by another process,
    removed after the listing, torn by a non-atomic copy)."""
    try:
        ids = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):                       # PermissionError, FileNotFoundError, JSONDecodeError, UnicodeError
        return None
    return ids if isinstance(ids, list) and all(isinstance(i, str) for i in ids) else None


class CheckpointStore:
    """One block directory. With `shard`, this store WRITES only under units/<shard>/ and state/<shard>/ (one shard per
    worker process, so parallel workers never touch each other's files) while it READS the done markers of every
    shard, so a unit finished by any worker, under any earlier worker count, is never run again. Readers of rows take
    `committed_row_files`."""

    def __init__(self, root: Path, shard: str | None = None) -> None:
        self.root = Path(root)
        if shard is not None and (not shard or any(c in shard for c in "/\\.:")):
            raise ValueError(f"shard must be a plain folder name, got {shard!r}")
        self.shard = shard
        self._units = self.root / "units" / shard if shard else self.root / "units"
        self._state = self.root / "state" / shard if shard else self.root / "state"
        self._units.mkdir(parents=True, exist_ok=True)
        self._state.mkdir(parents=True, exist_ok=True)
        self._done: set = set()                 # unit ids with a done marker (per-unit file or a batch done list)
        self._loaded: dict = {}                 # folder -> names of the done markers already read (they never change)
        self._orphans_cleared = False
        self._where: dict = {}                  # unit id -> the batch rows file that holds its row
        self._parsed: dict = {}                 # rows file -> {unit id: row}, the few most recently read
        self.refresh()

    def refresh(self) -> int:
        """Load the done markers that appeared in any shard since the last call (other workers may have finished units
        since). Each marker is read once. A done list that cannot be read or parsed right now (see the module docstring)
        is skipped and read on a later call; its units are not done until then. Returns how many were skipped."""
        skipped = 0
        for folder, names in _folders(str(self.root / "units")):
            loaded = self._loaded.setdefault(folder, set())
            for name in names:
                if not name.endswith(DONE) or name in loaded:
                    continue
                stem = name[:-len(DONE)]
                if _BATCH_DONE.fullmatch(name) is None:
                    self._done.add(stem)                # a per-unit marker: the empty file is the mark
                else:
                    ids = _read_done_list(Path(folder, name))
                    if ids is None:
                        skipped += 1
                        continue
                    self._done.update(ids)
                    self._where.update(dict.fromkeys(ids, Path(folder, stem + ROWS)))
                loaded.add(name)
        return skipped

    def _rows(self, unit_id: str) -> Path:
        return self._units / f"{unit_id}.jsonl"

    @staticmethod
    def _check_per_unit(unit_id: str) -> None:
        if unit_id.startswith(BATCH):
            raise ValueError(f"unit id {unit_id!r}: names starting with {BATCH!r} belong to the batch files (append_batch)")

    def is_done(self, unit_id: str) -> bool:
        return unit_id in self._done

    def done_count(self) -> int:
        """Number of finished units, whether marked one by one (LLM side) or as committed batches (mock side)."""
        return len(self._done)

    def append_batch(self, rows_by_unit: Mapping[str, Iterable[Mapping]]) -> None:
        """Write the rows of a whole batch to one file (one fsync) and mark all its units done in one atomic write."""
        ids = list(rows_by_unit)
        if not ids:                                      # nothing to commit: no file, and no done list rewritten per call
            return
        name = BATCH + hashlib.blake2b("\n".join(ids).encode(), digest_size=8).hexdigest()
        units = self._units
        self._clear_orphans()
        lines = [json.dumps(row, default=float) + "\n" for rows in rows_by_unit.values() for row in rows]
        tmp = units / f"{name}.jsonl.tmp"
        with tmp.open("w", encoding="utf-8") as f:
            f.write("".join(lines))
            f.flush()
            os.fsync(f.fileno())
        replace(tmp, units / f"{name}.jsonl")
        _atomic_write_bytes(units / f"{name}.done", json.dumps(ids).encode())
        self._loaded.setdefault(str(units), set()).add(f"{name}.done")
        self._done.update(ids)
        self._where.update(dict.fromkeys(ids, units / f"{name}.jsonl"))
        states = self._state
        for entry in os.scandir(states):                 # a finished unit keeps no resumable state
            if entry.name.endswith(".pt") and entry.name[:-3] in rows_by_unit:
                os.unlink(entry.path)

    def _clear_orphans(self) -> None:
        """Remove batch row files without a done list (a crash before the commit): their units are recomputed. A rows file
        whose done list exists is committed even if that list could not be read yet."""
        if self._orphans_cleared:
            return
        for path in self._units.glob(BATCH + "*" + ROWS):
            if not path.with_suffix(DONE).exists():
                path.unlink()
        self._orphans_cleared = True

    def append_rows(self, unit_id: str, rows: Iterable[Mapping]) -> None:
        """Append rows and fsync, so a row written is a row kept."""
        self._check_per_unit(unit_id)
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

    def rows_of(self, unit_id: str) -> List[dict]:
        """The saved rows of a finished unit, whether it was saved alone or in a batch file ([] if unknown). Row files are
        parsed once and the last few kept, so reading many units of one batch costs one parse."""
        path = self._where.get(unit_id)
        if path is None:
            return self.resume_rows(unit_id)
        if path not in self._parsed:
            if len(self._parsed) >= 4:
                self._parsed.pop(next(iter(self._parsed)))
            by_id: dict = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                by_id.setdefault(row.get("id"), []).append(row)
            self._parsed[path] = by_id
        return self._parsed[path].get(unit_id, [])

    def mark_done(self, unit_id: str) -> None:
        self._check_per_unit(unit_id)
        _atomic_write_bytes(self._units / f"{unit_id}.done", b"")
        self._done.add(unit_id)
        state = self._state / f"{unit_id}.pt"
        if state.exists():
            state.unlink()

    def save_state(self, unit_id: str, state: Mapping) -> None:
        tmp = self._state / f"{unit_id}.pt.tmp"
        torch.save(dict(state), tmp)
        replace(tmp, self._state / f"{unit_id}.pt")

    def load_state(self, unit_id: str) -> dict | None:
        path = self._state / f"{unit_id}.pt"
        return torch.load(path, weights_only=True) if path.exists() else None
