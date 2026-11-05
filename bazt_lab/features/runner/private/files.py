"""Small-file helpers: atomic writes (a reader sees the old file or the new one, never half of it) and the launch lock."""
from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path

from filelock import FileLock, Timeout


def write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path: Path, data) -> None:
    write_text(path, json.dumps(data, indent=2))


@contextlib.contextmanager
def exclusive(path: Path, refusal: str):
    """Hold an OS file lock for the block, or end the launch with `refusal` when another launch holds it. The OS
    drops the lock when its process dies, so a crashed launch never blocks its own resume."""
    lock = FileLock(path, timeout=0)
    try:
        lock.acquire()
    except Timeout:
        sys.exit(refusal)
    try:
        yield
    finally:
        lock.release()
