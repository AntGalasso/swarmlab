"""WandbTracker: the core.Tracker contract over Weights & Biases (wandb 0.31.0, pinned). The only module of this
feature that imports `wandb`, and it does so lazily, after the repo-root `.env` has been loaded.

A failure to set tracking up is an error at start (TrackingError, or core.Unset naming a missing setting). A failure
during the run is printed once per operation on stderr and counted, and the experiment goes on: its results are on
disk in the CheckpointStore, not in W&B. Nothing is ever replaced by a tracker that records nothing.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import functools
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional

from bazt_lab.core import public as core
from bazt_lab.features.tracking.adapters import env, kickoff
from bazt_lab.features.tracking.private import clean
from bazt_lab.features.tracking.private import settings as tracking_settings
from bazt_lab.features.tracking.private.errors import TrackingError
from bazt_lab.features.tracking.private.failures import FailureLog

STEP_KEY = "tracking/step"        # the runner's step, logged with every call and used as the x-axis of every metric


def _key_free(error: Optional[BaseException]) -> bool:
    """True when no exception in the cause/context chain carries the API key: a printed traceback shows them all."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        text = f"{error} {error.args!r}"
        if env.redact(text) != text:
            return False
        error = error.__cause__ or error.__context__
    return True


@functools.cache
def _wandb():
    env.load()                    # wandb reads WANDB_* from the environment, so the .env goes first
    import wandb

    return wandb


@dataclass(frozen=True)
class _Plan:
    settings: tracking_settings.TrackingSettings
    mode: str                     # the mode to use, after the kickoff rule
    notice: Optional[str]


@dataclass
class _Table:
    """One logged table: its columns (fixed by the first rows), the cleaned rows waiting for the next flush and the
    numbered part to log next."""
    columns: List[str]
    pending: List[list] = field(default_factory=list)
    parts: int = 0


class WandbTracker:
    """`WandbTracker()` builds without arguments and reads the settings from the `config` given to `start`.
    `WandbTracker(settings)` checks them at once, so a launch can refuse before it writes anything."""

    def __init__(self, settings: Optional[Mapping] = None) -> None:
        self.mode: Optional[str] = None          # the mode of the current run, after the kickoff rule
        self._run = None
        self._plan_now = None
        self._tables: Dict[str, _Table] = {}
        self._failures = FailureLog(env.redact)
        self._plan = self._prepare(settings) if settings is not None else None

    @property
    def failures(self) -> Dict[str, int]:
        """Operation -> how many times it failed during the current (or last) run."""
        return self._failures.counts

    @staticmethod
    def _prepare(settings: Mapping) -> _Plan:
        ts = tracking_settings.read(settings)
        mode, notice = tracking_settings.effective_mode(ts, kickoff.kickoff_time, dt.datetime.now(dt.UTC))
        if mode == "online":
            if ts.entity is None:
                raise TrackingError("tracking.mode is online but tracking.entity is not set: name the W&B team or "
                                    "user so the runs do not land in a default account")
            env.require_api_key()
        return _Plan(ts, mode, notice)

    def start(self, block: str, config: Mapping, resume_id: Optional[str]) -> str:
        if self._run is not None:
            raise TrackingError("start() was already called on this tracker; call finish() before starting again")
        plan = self._plan or self._prepare(config)
        group = core.get(config, "profile")
        wandb_config = clean.clean_config(config)
        if plan.notice:
            print(plan.notice, file=sys.stderr)
        ts, run = plan.settings, None
        try:
            wb = _wandb()
            if ts.dir:
                Path(ts.dir).mkdir(parents=True, exist_ok=True)
            run = wb.init(project=ts.project, entity=ts.entity, group=group, job_type=block, config=wandb_config,
                          id=resume_id, resume="allow", mode=plan.mode, dir=ts.dir, allow_val_change=True,
                          reinit="create_new", save_code=False, settings={"disable_git": True})
            run.define_metric(STEP_KEY)
            run.define_metric("*", step_metric=STEP_KEY)
        except Exception as error:
            if run is not None:
                with contextlib.suppress(Exception):
                    run.finish()
            text = f"cannot start the {plan.mode} W&B run for block {block!r}: {type(error).__name__}: {error}"
            safe = env.redact(text)
            # The original error stays chained for debugging unless it carries the key: a traceback prints the chain.
            raise TrackingError(safe) from (error if _key_free(error) else None)
        self._run, self.mode, self._plan_now = run, plan.mode, plan
        self._tables, self._failures = {}, FailureLog(env.redact)
        return run.id

    # ------------------------------------------------------------------------- during the run: never raises
    @contextlib.contextmanager
    def _guard(self, operation: str) -> Iterator[None]:
        try:
            yield
        except Exception as error:               # D87: a tracking problem must not stop an experiment
            self._failures.report(operation, error)

    def _need(self):
        if self._run is None:
            raise TrackingError("the tracker is not started: call start() first, and not after finish()")
        return self._run

    def log(self, metrics: Mapping, step: int) -> None:
        with self._guard("log"):
            self._need().log({**metrics, STEP_KEY: int(step)})

    def log_rows(self, table: str, rows: List[Mapping]) -> None:
        """Buffers the rows; every `tracking.flush_rows` of them are logged as one wandb.Table part (a table of one
        row per call costs wandb seconds per call; see CONTRACT.md)."""
        if not rows or self.mode == "disabled":
            return
        with self._guard(f"log_rows[{table}]"):
            self._need()
            state = self._tables.get(table)
            if state is None:
                state = self._tables[table] = _Table(list(dict.fromkeys(key for row in rows for key in row)))
            unknown = {key for row in rows for key in row} - set(state.columns)
            if unknown:
                self._failures.report(f"log_rows[{table}]", ValueError(
                    f"columns {sorted(unknown)} are not in the table and are dropped"))
            state.pending.extend([clean.cell(row.get(column)) for column in state.columns] for row in rows)
            if len(state.pending) >= self._plan_now.settings.flush_rows:
                self._flush(table, state)

    def _flush(self, table: str, state: _Table) -> None:
        """Logs the buffered rows as numbered parts of at most wandb's row limit. Rows go straight into `Table.data`:
        `add_data` re-infers the type of every cell, which is most of the cost. A failed part is reported and its
        rows are dropped from W&B (they are on disk), never retried in a loop."""
        run, wb = self._need(), _wandb()
        pending, state.pending = state.pending, []
        limit = wb.Table.MAX_ARTIFACT_ROWS
        for start in range(0, len(pending), limit):
            key = table if state.parts == 0 else f"{table}_part{state.parts}"
            state.parts += 1
            part = wb.Table(columns=state.columns, allow_mixed_types=True)
            part.data.extend(pending[start:start + limit])
            run.log({key: part})

    def _flush_all(self) -> None:
        for table, state in self._tables.items():
            if state.pending:
                with self._guard(f"log_rows[{table}]"):
                    self._flush(table, state)

    def artifact(self, path: str, name: str, kind: str) -> None:
        if self.mode == "disabled":
            return
        with self._guard("artifact"):
            run, wb = self._need(), _wandb()
            source = Path(path)
            artifact = wb.Artifact(clean.wandb_name(name), type=clean.wandb_name(kind))
            (artifact.add_dir if source.is_dir() else artifact.add_file)(str(source))
            run.log_artifact(artifact)

    def finish(self) -> None:
        run = self._run
        if run is None:
            return
        self._flush_all()
        self._run = None
        with self._guard("finish"):
            run.finish()
        self._tables = {}
        summary = self._failures.summary()
        if summary:
            print(summary, file=sys.stderr)
