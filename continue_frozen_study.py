#!/usr/bin/env python3
"""Finish the already authorized frozen study; never save its runtime credential."""

import datetime as dt
import getpass
import hashlib
import json
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import budget_study as driver
import budget_study_analysis as analysis

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/ricardo/integrated_study_v6"


def progress(status, **extra):
    data = {"status": status, "updated_utc": dt.datetime.now(dt.UTC).isoformat(), **extra}
    driver.write_json(OUT / "pipeline_receipt.json", data)
    print(json.dumps(data), flush=True)


def completed_rows(phase):
    p = OUT / phase / "episodes.jsonl"
    return len(p.read_text().splitlines()) if p.exists() else 0


def run():
    key = getpass.getpass("Adaption API key (hidden, not saved): ").strip()
    if not key:
        raise RuntimeError("Missing runtime credential.")
    # The value remains in this process. It is never an environment variable or file.
    driver.getpass.getpass = lambda _: key
    started = time.monotonic()
    last = -1
    while True:
        p = OUT / "main/latest_receipt.json"
        if p.exists():
            receipt = json.loads(p.read_text())
            if receipt["status"] in ("completed", "failed"):
                break
        n = completed_rows("main")
        if n != last:
            progress("waiting_for_main", main_completed=n, main_total=240)
            last = n
        if time.monotonic() - started > 14400:
            raise RuntimeError("Main phase did not finish within the watcher allowance.")
        time.sleep(15)
    if receipt["status"] == "failed":
        # One resumption uses only the same frozen source, settings and cached responses.
        progress("resuming_main_once", main_completed=completed_rows("main"))
        driver.run_phase(
            SimpleNamespace(
                phase="main", allow_live_llm=True, prompt_key=True, max_credits=5.0, confirm_frozen_analysis=False
            ),
            OUT,
        )
    progress("analyzing_main", main_completed=completed_rows("main"))
    analysis.analyze(OUT, "main")
    progress("running_heldout", main_completed=completed_rows("main"), heldout_total=120)
    driver.run_phase(
        SimpleNamespace(
            phase="heldout", allow_live_llm=True, prompt_key=True, max_credits=5.0, confirm_frozen_analysis=True
        ),
        OUT,
    )
    analysis.analyze(OUT, "heldout")
    archive = OUT / "completed_reproduction.zip"
    paths = [p for p in OUT.rglob("*") if p.is_file() and p != archive]
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in paths:
            z.write(p, p.relative_to(OUT))
        for p in driver.SOURCE_FILES:
            z.write(ROOT / p, "source/" + p)
        z.write(ROOT / "budget_study_analysis.py", "source/budget_study_analysis.py")
        z.write(Path(__file__), "source/continue_frozen_study.py")
    progress(
        "completed",
        main_completed=completed_rows("main"),
        heldout_completed=completed_rows("heldout"),
        archive=str(archive),
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        note="Statistical outputs await human/assistant review and private manuscript update; no submission made.",
    )


if __name__ == "__main__":
    try:
        run()
    except BaseException as exc:
        progress(
            "blocked",
            error_type=type(exc).__name__,
            error=str(exc) if isinstance(exc, (RuntimeError, ValueError)) else type(exc).__name__,
        )
        raise SystemExit(1) from None
