#!/usr/bin/env python3
"""Resume the frozen study with one writer, a heartbeat and bounded recovery.

This orchestration file does not change the experiment or its analysis. Runtime
credentials are read from a hidden terminal prompt and retained only in memory.
"""

from __future__ import annotations

import contextlib
import ctypes
import datetime as dt
import fcntl
import getpass
import hashlib
import json
import os
import signal
import sqlite3
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import budget_study as driver
import budget_study_analysis as analysis

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/ricardo/integrated_study_v6"
STATE = {"status": "starting"}
STATE_LOCK = threading.Lock()
WRITE_LOCK = threading.Lock()
STOP = threading.Event()
MAX_ATTEMPTS = 2
PHASE_CREDIT_CAP = 5.0
MAX_REQUEST_RESERVE = ((8192 + 256) * driver.RATES[0] + 8192 * driver.RATES[1]) / 1e6


def utc():
    return dt.datetime.now(dt.UTC).isoformat()


def write_atomic(path, value):
    with WRITE_LOCK:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temp, path)


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def rows(phase):
    p = OUT / phase / "episodes.jsonl"
    return len([s for s in p.read_text().splitlines() if s.strip()]) if p.exists() else 0


def heartbeat():
    with STATE_LOCK:
        value = dict(STATE)
    value.update(
        updated_utc=utc(),
        pid=os.getpid(),
        pid_namespace=os.readlink("/proc/self/ns/pid"),
        main_completed=rows("main"),
        main_total=240,
        heldout_completed=rows("heldout"),
        heldout_total=120,
    )
    write_atomic(OUT / "overnight_heartbeat.json", value)
    write_atomic(OUT / "pipeline_receipt.json", value)


def update(status, **extra):
    with STATE_LOCK:
        STATE.update(status=status, **extra)
    heartbeat()


def watch():
    while not STOP.wait(30):
        heartbeat()


def validate_source():
    frozen = read_json(OUT / "frozen_protocol.json")
    if driver.source_hashes() != frozen["source_sha256"]:
        raise RuntimeError("Frozen study source hash mismatch.")
    specification = read_json(OUT / "analysis_frozen.json")["specification"]
    if analysis.digest(ROOT / "budget_study_analysis.py") != specification["analysis_sha256"]:
        raise RuntimeError("Frozen analysis source hash mismatch.")


class FileLock(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_short),
        ("whence", ctypes.c_short),
        ("start", ctypes.c_longlong),
        ("length", ctypes.c_longlong),
        ("pid", ctypes.c_int),
    ]


def another_cache_connection(phase):
    # SQLite holds this WAL shared-memory lock for the connection lifetime.
    # F_GETLK observes locks even when the writer uses another PID namespace.
    p = OUT / phase / "requests.sqlite-shm"
    if not p.exists():
        return False
    fd = os.open(p, os.O_RDONLY)
    try:
        query = FileLock(fcntl.F_WRLCK, os.SEEK_SET, 128, 1, 0)
        result = FileLock.from_buffer_copy(fcntl.fcntl(fd, fcntl.F_GETLK, bytes(query)))
        return result.type != fcntl.F_UNLCK
    finally:
        os.close(fd)


def checkpoint():
    path = OUT / "overnight_recovery_checkpoint.json"
    if path.exists():
        return read_json(path)
    value = {
        "captured_utc": utc(),
        "reason": "Prior writer interrupted without a final receipt.",
        "original_last_progress_utc": "2026-11-06T23:10:43.300062+00:00",
        "frozen_source_verified": True,
        "phase_credit_cap": PHASE_CREDIT_CAP,
        "max_new_attempts_per_phase": MAX_ATTEMPTS,
        "phases": {},
    }
    backup = OUT / "recovery_before_overnight"
    backup.mkdir(exist_ok=True)
    for phase in ("main", "heldout"):
        p = OUT / phase / "episodes.jsonl"
        data = p.read_bytes() if p.exists() else b""
        entry = {
            "episodes": rows(phase),
            "episode_prefix_bytes": len(data),
            "episode_prefix_sha256": hashlib.sha256(data).hexdigest(),
            "baseline_cache_credits": 0.0,
            "interrupted_inflight_reserve": 0.0,
        }
        cache = OUT / phase / "requests.sqlite"
        if cache.exists():
            connection = sqlite3.connect("file:" + str(cache) + "?mode=ro", uri=True)
            count, pt, ct = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(pt),0), COALESCE(SUM(ct),0) FROM cache"
            ).fetchone()
            destination = sqlite3.connect(backup / (phase + "_requests.sqlite"))
            connection.backup(destination)
            destination.close()
            connection.close()
            entry.update(
                cache_rows=count,
                cached_input_tokens=pt,
                cached_output_tokens=ct,
                baseline_cache_credits=(pt * driver.RATES[0] + ct * driver.RATES[1]) / 1e6,
                interrupted_inflight_reserve=8 * MAX_REQUEST_RESERVE,
            )
        if data:
            (backup / (phase + "_episodes.jsonl")).write_bytes(data)
        value["phases"][phase] = entry
    write_atomic(path, value)
    return value


def allowance(phase, baseline):
    used = baseline["phases"][phase]["baseline_cache_credits"]
    used += baseline["phases"][phase]["interrupted_inflight_reserve"]
    repair = read_json(OUT / "credential_format_repair.json")
    if repair and phase == "main":
        used += repair["interrupted_setup_request_reserve"]
    for p in (OUT / phase).glob("receipt_*.json"):
        receipt = read_json(p)
        if receipt["started_utc"] >= baseline["captured_utc"]:
            used += receipt.get("calculated_credits", 0.0)
            used += receipt.get("requests_without_usage", 0) * MAX_REQUEST_RESERVE
    remaining = PHASE_CREDIT_CAP - used
    if remaining <= MAX_REQUEST_RESERVE:
        raise RuntimeError("Cumulative phase credit allowance exhausted.")
    return remaining


def attempts(phase):
    p = OUT / "overnight_attempts.jsonl"
    values = [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.exists() else []
    return len([r for r in values if r["phase"] == phase and r["event"] == "started"])


def run_phase(phase, baseline):
    original = baseline["phases"][phase]
    p = OUT / phase / "episodes.jsonl"
    prefix = p.read_bytes()[: original["episode_prefix_bytes"]] if p.exists() else b""
    if hashlib.sha256(prefix).hexdigest() != original["episode_prefix_sha256"]:
        raise RuntimeError("Previously completed episode prefix changed.")
    receipt = read_json(OUT / phase / "latest_receipt.json")
    if receipt and receipt["status"] == "completed":
        return
    repair = read_json(OUT / "credential_format_repair.json")
    setup_attempts = repair["setup_attempts_without_new_completed_episodes"] if repair and phase == "main" else 0
    attempt_limit = MAX_ATTEMPTS + setup_attempts
    while attempts(phase) < attempt_limit:
        validate_source()
        remaining = allowance(phase, baseline)
        attempt = attempts(phase) + 1
        driver.append(
            OUT / "overnight_attempts.jsonl",
            {
                "utc": utc(),
                "phase": phase,
                "event": "started",
                "attempt": attempt,
                "episodes_before": rows(phase),
                "remaining_credit_allowance": remaining,
            },
        )
        update("running_" + phase, phase_attempt=attempt)
        try:
            driver.run_phase(
                SimpleNamespace(
                    phase=phase,
                    allow_live_llm=True,
                    prompt_key=True,
                    max_credits=remaining,
                    confirm_frozen_analysis=phase == "heldout",
                ),
                OUT,
            )
            return
        except RuntimeError as exc:
            safe = str(exc)
            transient = any(
                s in safe
                for s in (
                    "HTTP 429",
                    "HTTP 500",
                    "HTTP 502",
                    "HTTP 503",
                    "HTTP 504",
                    "APITimeoutError",
                    "APIConnectionError",
                )
            )
            driver.append(
                OUT / "overnight_attempts.jsonl",
                {
                    "utc": utc(),
                    "phase": phase,
                    "event": "failed",
                    "attempt": attempt,
                    "error": safe,
                    "transient": transient,
                },
            )
            if not transient or attempts(phase) >= attempt_limit:
                raise
            update("waiting_to_retry_" + phase, last_error=safe)
            time.sleep(60)
    raise RuntimeError("Bounded phase recovery exhausted; inspect recorded receipts.")


def main():
    lock = (OUT / "overnight_writer.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("EXISTING_OVERNIGHT_WRITER_ACTIVE", flush=True)
        return
    if any(another_cache_connection(p) for p in ("main", "heldout")):
        print("EXISTING_STUDY_CACHE_CONNECTION_ACTIVE", flush=True)
        return
    validate_source()
    baseline = checkpoint()
    key = getpass.getpass("Adaption runtime key (hidden; memory only): ").strip()
    if not key:
        raise RuntimeError("Missing runtime credential.")
    os.environ.pop("ADAPTION_API_KEY", None)
    driver.getpass.getpass = lambda _: key
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(143)))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(RuntimeError("Overnight runtime limit reached.")))
    signal.alarm(6 * 3600)
    update("ready", started_utc=utc(), model=driver.MODEL, source_frozen=True, credential_storage="process_memory_only")
    threading.Thread(target=watch, daemon=True).start()
    print("OVERNIGHT_WORKER_STARTED; credential hidden; heartbeat every 30 seconds.", flush=True)
    with (
        (OUT / "overnight_execution.log").open("a", buffering=1) as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        try:
            run_phase("main", baseline)
            update("analyzing_main")
            analysis.analyze(OUT, "main")
            run_phase("heldout", baseline)
            update("analyzing_heldout")
            analysis.analyze(OUT, "heldout")
            validate_source()
            update("archiving")
            archive = OUT / "completed_reproduction.zip"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
                for p in sorted(OUT.rglob("*")):
                    if p.is_file() and p != archive and not p.name.endswith((".lock", ".tmp")):
                        z.write(p, p.relative_to(OUT))
                for name in (*driver.SOURCE_FILES, "budget_study_analysis.py", Path(__file__).name):
                    z.write(ROOT / name, "source/" + name)
            update(
                "completed",
                archive=str(archive),
                archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                note="Complete data and frozen analyses await manuscript review; no submission.",
            )
        except BaseException as exc:
            update(
                "blocked",
                error_type=type(exc).__name__,
                error=str(exc) if isinstance(exc, (RuntimeError, ValueError)) else type(exc).__name__,
            )
        finally:
            STOP.set()
            signal.alarm(0)
    print("OVERNIGHT_WORKER_FINISHED", read_json(OUT / "overnight_heartbeat.json")["status"], flush=True)


if __name__ == "__main__":
    main()
