"""Reads back what an offline W&B run recorded, without a network and without a server.

wandb-core writes every record of a run (config, metric definitions, history, artifacts, ...) to
`<run dir>/run-<id>.wandb`, a LevelDB-style log of protobuf `Record`s that `wandb sync` uploads later. This reader
lets a test, or a person before `wandb sync`, see what the run holds. It does not verify the per-record checksums.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import List

BLOCK = 32768                               # a record that does not fit in a block is split into fragments
FILE_HEADER = 7                             # b":W&B", a two byte magic and a version
RECORD_HEADER = 7                           # checksum (4), length (2), kind (1)
FULL, FIRST, MIDDLE, LAST = 1, 2, 3, 4


def records(run_dir) -> List:
    """Every `wandb_internal_pb2.Record` in the run directory's `run-*.wandb`, in the order written."""
    from wandb.proto import wandb_internal_pb2

    data = next(Path(run_dir).glob("run-*.wandb")).read_bytes()
    if data[:4] != b":W&B":
        raise ValueError(f"{run_dir}: not a W&B run log")
    pos, partial, payloads = FILE_HEADER, b"", []
    while pos + RECORD_HEADER <= len(data):
        if BLOCK - pos % BLOCK < RECORD_HEADER:             # too little room for a header: padding to the next block
            pos += BLOCK - pos % BLOCK
            continue
        _, length, kind = struct.unpack("<IHB", data[pos:pos + RECORD_HEADER])
        if kind == 0:                                       # zero padding up to the end of the block
            pos += BLOCK - pos % BLOCK
            continue
        payload = data[pos + RECORD_HEADER:pos + RECORD_HEADER + length]
        pos += RECORD_HEADER + length
        if kind == FULL:
            payloads.append(payload)
        elif kind == FIRST:
            partial = payload
        elif kind == MIDDLE:
            partial += payload
        elif kind == LAST:
            payloads.append(partial + payload)
            partial = b""
    out = []
    for payload in payloads:
        record = wandb_internal_pb2.Record()
        record.ParseFromString(payload)
        out.append(record)
    return out
