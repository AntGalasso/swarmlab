"""Worker processes for the batches of a mock block (spawn: the only start method that works on Windows).

The parent feeds `(arm name, units)` batches through a bounded queue; each worker rebuilds the wiring from the
`Wiring.rebuild` entry ("module:function"), opens the block's CheckpointStore with its own shard (it writes only under
units/w<k>/ and state/w<k>/) and runs the batches it takes. A worker failure stops all of them and raises in the parent.
"""
from __future__ import annotations

import importlib
import multiprocessing
import queue
import time
import traceback
from pathlib import Path
from typing import Callable, Iterable, Mapping

from bazt_lab.core import public as core

PROGRESS_SECONDS = 2.0
SETTLE_SECONDS = 2.0            # how long the final count waits for done lists another process still holds
SETTLE_POLL_SECONDS = 0.05


def _load(entry: str):
    module, _, name = entry.partition(":")
    return getattr(importlib.import_module(module), name)


def worker_main(index: int, work, errors, rebuild: str, settings: Mapping, platform_name: str, out: str) -> None:
    try:
        import torch
        torch.set_num_threads(1)
        wiring = _load(rebuild)()
        platform = wiring.platform(platform_name)
        store = core.CheckpointStore(Path(out), shard=f"w{index}")
        tracker = core.NullTracker()
        while True:
            item = work.get()
            if item is None:
                return
            arm, units = item
            store.refresh()
            todo = [u for u in units if not store.is_done(u["id"])]
            if todo:
                platform.run_batch(todo, wiring.arm(arm), settings, store, tracker)
    except BaseException as e:
        errors.put((index, f"{type(e).__name__}: {e}", traceback.format_exc()))
        raise SystemExit(1) from None


class _Pool:
    def __init__(self, workers: int, rebuild: str, settings: Mapping, platform_name: str, out: Path) -> None:
        ctx = multiprocessing.get_context("spawn")
        self.work, self.errors = ctx.Queue(maxsize=2 * workers), ctx.Queue()
        self.procs = [ctx.Process(target=worker_main, name=f"runner-w{k}", daemon=True,
                                  args=(k, self.work, self.errors, rebuild, settings, platform_name, str(out)))
                      for k in range(workers)]
        for p in self.procs:
            p.start()

    def check(self) -> None:
        for k, p in enumerate(self.procs):
            if p.exitcode not in (None, 0):
                try:
                    _, message, trace = self.errors.get(timeout=2)
                except queue.Empty:
                    message, trace = f"exited with code {p.exitcode}", ""
                raise RuntimeError(f"worker {k} failed: {message}\n{trace}".rstrip())

    def put(self, item) -> None:
        while True:
            try:
                self.work.put(item, timeout=0.5)
                return
            except queue.Full:
                self.check()

    def finish(self, settle: Callable[[], None]) -> None:
        for _ in self.procs:
            self.put(None)
        while any(p.is_alive() for p in self.procs):
            self.check()
            settle()
            time.sleep(0.05)
        self.check()

    def stop(self) -> None:
        for p in self.procs:
            if p.is_alive():
                p.terminate()
        for p in self.procs:
            p.join(5)
        for q in (self.work, self.errors):
            q.cancel_join_thread()
            q.close()


def run(batches: Iterable, workers: int, rebuild: str, settings: Mapping, platform_name: str, out: Path, store,
        progress: Callable[[int, int, int], None]) -> int:
    """Run the batches on `workers` processes; returns the units dispatched. `progress(finished, dispatched, batches)`
    is called as units finish. An attacked batch is dispatched only after every attack-free unit before it is done."""
    pool = _Pool(workers, rebuild, settings, platform_name, out)
    base = store.done_count()
    dispatched = nbatches = 0
    last, previous_free = 0.0, False

    def finished(settle_seconds: float = 0.0) -> int:
        """Units done since the launch. With `settle_seconds`: while refresh() reports done lists it could not read yet
        (CheckpointStore.refresh; on Windows the rename that committed one holds it for a moment) and units are still
        missing, read again until they are counted or the time is up."""
        deadline = time.monotonic() + settle_seconds
        while store.refresh() and store.done_count() - base < dispatched and time.monotonic() < deadline:
            time.sleep(SETTLE_POLL_SECONDS)
        return store.done_count() - base

    def settle() -> None:
        nonlocal last
        if time.monotonic() - last >= PROGRESS_SECONDS:
            last = time.monotonic()
            progress(finished(), dispatched, nbatches)

    try:
        for batch in batches:
            attacked = batch[0]["attack"] != "none"
            if attacked and previous_free:          # the twins of the next batches must be done before they start
                while finished() < dispatched:
                    pool.check()
                    time.sleep(0.05)
                previous_free = False
            previous_free = previous_free or not attacked
            pool.put((batch[0]["arm"], batch))
            dispatched, nbatches = dispatched + len(batch), nbatches + 1
            settle()
        pool.finish(settle)
        done = finished(SETTLE_SECONDS)
        if done < dispatched:
            raise RuntimeError(f"the workers stopped with {dispatched - done} of {dispatched} units not done")
        progress(done, dispatched, nbatches)
        return dispatched
    finally:
        pool.stop()
