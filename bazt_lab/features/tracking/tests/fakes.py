"""A stand-in for the `wandb` module that records what the tracker asks of it, so the tracker's behaviour is tested
without starting wandb-core. The real library is exercised in test_offline_run.py.

`wb.fail` maps a method name ("init", "log", "table_add", "log_artifact", "finish", "define_metric") to an exception
that method raises, or to a callable returning one (or None) per call.
"""
from __future__ import annotations

from pathlib import Path


class FakeTable:
    def __init__(self, columns=None, log_mode="IMMUTABLE", allow_mixed_types=False):
        self.columns = list(columns)
        self.log_mode = log_mode
        self.allow_mixed_types = allow_mixed_types
        self.data = []
        self.logged = 0                    # rows already handed to run.log

    def add_data(self, *row):
        self._wb.maybe_fail("table_add")
        if len(row) != len(self.columns):
            raise ValueError(f"expects {len(self.columns)} columns, found {len(row)}")
        self.data.append(list(row))


class FakeArtifact:
    def __init__(self, name, **kwargs):
        self.name = name
        self.type = kwargs.get("type")
        self.files = []
        self.dirs = []

    def add_file(self, path, name=None):
        if not Path(path).is_file():
            raise ValueError(f"Path is not a file: {path}")
        self.files.append(str(path))

    def add_dir(self, path, name=None):
        if not Path(path).is_dir():
            raise ValueError(f"Path is not a directory: {path}")
        self.dirs.append(str(path))


class FakeRun:
    def __init__(self, wb, kwargs):
        self.wb = wb
        self.kwargs = kwargs
        self.id = kwargs.get("id") or "fresh0001"
        self.metrics = []                  # (name, step_metric) of every define_metric
        self.logs = []                     # one entry per run.log: {key: value, tables snapshotted as ("table", rows)}
        self.artifacts = []
        self.finished = 0

    def define_metric(self, name, step_metric=None, **kw):
        self.wb.maybe_fail("define_metric")
        self.metrics.append((name, step_metric))

    def log(self, data, **kw):
        self.wb.maybe_fail("log")
        entry = {}
        for key, value in data.items():
            if isinstance(value, FakeTable):
                entry[key] = ("table", [list(r) for r in value.data[value.logged:]])
                value.logged = len(value.data)
            else:
                entry[key] = value
        self.logs.append(entry)

    def log_artifact(self, artifact, **kw):
        self.wb.maybe_fail("log_artifact")
        self.artifacts.append(artifact)

    def finish(self, *a, **kw):
        self.finished += 1
        self.wb.maybe_fail("finish")


class FakeWandb:
    """Replaces `wandb`: `init` returns a FakeRun, `Table` and `Artifact` are recorded."""

    def __init__(self, max_rows=200000):
        self.fail = {}
        self.runs = []
        self.inits = []
        self.tables = []
        outer = self

        class Table(FakeTable):
            MAX_ARTIFACT_ROWS = max_rows

            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self._wb = outer
                outer.tables.append(self)

        self.Table = Table
        self.Artifact = FakeArtifact

    def maybe_fail(self, method):
        what = self.fail.get(method)
        error = what() if callable(what) and not isinstance(what, BaseException) else what
        if error is not None:
            raise error

    def init(self, **kwargs):
        self.maybe_fail("init")
        self.inits.append(kwargs)
        run = FakeRun(self, kwargs)
        self.runs.append(run)
        return run
