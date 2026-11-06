"""On-disk cache of judge responses (SQLite, WAL), shared by every arm and every run that points at the same file.

The key is a hash of the whole request body: model, pinned provider, the full prompt text and every sampling,
logprob and reasoning field. That is a superset of (model, provider, prompt text), so changing any field of the
request can never return a stale answer. The value is the first-token record (candidates and usage), not the
score, so the parse rule and the floor can change without paying again. First write wins: a second answer to the
same request never replaces the stored one, which keeps reruns identical.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

SCHEMA = 1                           # bump when the stored record changes shape


def cache_key(body: dict) -> str:
    payload = json.dumps([SCHEMA, body], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False, timeout=30.0)
        with self._lock, self._db:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.execute("CREATE TABLE IF NOT EXISTS judge (k TEXT PRIMARY KEY, v TEXT NOT NULL)")

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            row = self._db.execute("SELECT v FROM judge WHERE k = ?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, key: str, record: dict) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT OR IGNORE INTO judge (k, v) VALUES (?, ?)", (key, json.dumps(record)))

    def close(self) -> None:
        with self._lock:
            self._db.close()
