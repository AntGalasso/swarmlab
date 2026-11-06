"""Conversation logs: what the LLM platform writes during `gnn_train` and the trainer reads.

One JSON line per conversation, in `<gnn_train.logs_dir>/<family>/units/<unit id>.jsonl` (a `core.CheckpointStore`
layout, so an append is fsynced and a torn last line is dropped on read). `family` is `<attack>__<task>`. A record:

    {"id", "attack", "task", "backbone", "adj" [N][N] of 0/1 (adj[i][j] = 1: i sends to j),
     "attackers" [indices], "turns" [rounds][N] of the text each agent wrote that round}
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Sequence

from bazt_lab.core import public as core
from bazt_lab.features.gnn.private.graph import check_record, family_key

REQUIRED = ("id", "attack", "task", "backbone", "adj", "attackers", "turns")


def _root(settings: Mapping, family: str) -> Path:
    return Path(core.get(settings, "gnn_train.logs_dir")) / family


def log_conversations(settings: Mapping, unit: Mapping, records: Sequence[dict]) -> None:
    """Append `records` to the log of `unit` (its family and its `id`). Every record is checked first: one bad record
    refuses the whole call and writes nothing."""
    family = family_key(unit["attack"], unit["task"])
    for record in records:
        rid = record.get("id", "<no id>")
        for key in REQUIRED:
            if key not in record:
                raise ValueError(f"record {rid}: missing {key!r}")
        own = family_key(record["attack"], record["task"])
        if own != family:
            raise ValueError(f"record {rid}: belongs to {own}, the unit logs {family}")
        check_record(record)
    core.CheckpointStore(_root(settings, family)).append_rows(unit["id"], records)


def check_logged(settings: Mapping, family: str) -> None:
    """Refuse logs that do not match gnn_train (agents, rounds, backbone), naming the file and the record."""
    want = {k: core.get(settings, f"gnn_train.{k}") for k in ("agents", "rounds", "backbone")}
    root = _root(settings, family)
    if not (root / "units").is_dir():
        return
    store = core.CheckpointStore(root)
    for path in sorted((root / "units").glob("*.jsonl")):
        for record in store.resume_rows(path.stem):
            got = {"agents": len(record["adj"]), "rounds": len(record["turns"]), "backbone": record["backbone"]}
            for what, value in got.items():
                if value != want[what]:
                    raise ValueError(f"{path}: record {record['id']}: {what} is {value}, gnn_train.{what} is {want[what]}")


def read_conversations(settings: Mapping, family: str) -> List[dict]:
    """Every logged conversation of `family`: files in name order, lines in order, a repeated id kept once (first)."""
    root = _root(settings, family)
    if not (root / "units").is_dir():
        return []
    store, seen, out = core.CheckpointStore(root), set(), []
    for path in sorted((root / "units").glob("*.jsonl")):
        for record in store.resume_rows(path.stem):
            if record["id"] not in seen:
                seen.add(record["id"])
                out.append(record)
    return out
