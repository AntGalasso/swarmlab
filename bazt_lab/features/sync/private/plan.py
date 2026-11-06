"""Which files go to the Hub. Pure functions over the file system: the mirror layout, the exclusions, and the change
test (a file is changed when its modification time or its size differs from what was last uploaded)."""
from __future__ import annotations

import fnmatch
import posixpath
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

NEVER = "*.tmp"                          # an atomic writer's half-written file; never uploaded


@dataclass(frozen=True)
class Entry:
    repo_path: str                       # <folder named after the path>/<path inside it>
    local: Path
    mtime_ns: int
    size: int                            # the size seen at listing; the upload is capped to it


def _match(rel: str, patterns: Iterable[str]) -> bool:
    name = posixpath.basename(rel)
    return any(fnmatch.fnmatchcase(rel, p) or fnmatch.fnmatchcase(name, p) for p in patterns)


def _keep(rel: str, include: Sequence[str], exclude: Sequence[str]) -> bool:
    if _match(rel, (NEVER,)):
        return False
    return (not include or _match(rel, include)) and not _match(rel, exclude)


def snapshot(paths: Iterable, include: Sequence[str], exclude: Sequence[str]) -> List[Entry]:
    """Every wanted file under `paths` now. A path that does not exist (yet) contributes nothing."""
    out: List[Entry] = []
    for path in map(Path, paths):
        if path.is_file():
            found = [(path, path.name)]
        elif path.is_dir():
            found = [(f, f.relative_to(path).as_posix()) for f in sorted(path.rglob("*")) if f.is_file()]
        else:
            continue
        for file, rel in found:
            if not _keep(rel, include, exclude):
                continue
            try:
                stat = file.stat()
            except OSError:                  # removed between the listing and the stat
                continue
            out.append(Entry(f"{path.name}/{rel}", file, stat.st_mtime_ns, stat.st_size))
    return out


def stamps(entries: Iterable[Entry]) -> Dict[str, Tuple[int, int]]:
    return {e.repo_path: (e.mtime_ns, e.size) for e in entries}


def changed(entries: Iterable[Entry], uploaded: Dict[str, Tuple[int, int]]) -> List[Entry]:
    return [e for e in entries if uploaded.get(e.repo_path) != (e.mtime_ns, e.size)]
