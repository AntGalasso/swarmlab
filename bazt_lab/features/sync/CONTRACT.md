# sync: contract

Pushes checkpoint folders to a Hugging Face repository from ONE separate child process at low OS priority, every
`sync.every_minutes`, so a crash never loses them and uploading never takes the experiments' CPU. The experiment
process only starts the child and later stops it; it never waits on an upload.

## public.py

```python
from bazt_lab.features.sync import public as sync

handle = sync.start(settings, paths)   # settings: the resolved settings mapping; paths: folders and/or files
...                                    # run the block or the GNN training
handle.stop()                          # final push, then the uploader ends. Call it in a `finally`
```

| name | what |
|---|---|
| `start(settings, paths) -> SyncHandle` | reads `sync.*` from `settings`. `sync.enabled = false` returns a no-op handle (`pid is None`, `alive` False) and reads nothing else. Enabled: refuses **before any process is started** with `core.Unset` (naming `sync.<key>`) for a missing key, `SyncError` for a bad value, no path, two paths with the same name, a missing `HF_TOKEN`, or a child that cannot be launched |
| `SyncHandle.stop()` | writes "stop" to the child, which does a final push, prints one JSON summary and exits. Never raises, idempotent. If the final push takes longer than `sync.final_push_timeout_seconds` it reports that on stderr and returns; the child finishes on its own |
| `SyncHandle.alive` / `.pid` | the child process (on Windows from a venv: the launcher; the real interpreter is its child) |
| `SyncHandle.summary` | `{"commits": n, "failures": {kind: n}}` after a clean stop, else `None` |
| `SyncHandle.reported` | 1 once the death of the child was reported, else 0 |
| `SyncError` | a `RuntimeError`; messages name the setting or variable and never contain a secret |

## Behaviour

- **Mirror**: each path is mirrored under a folder named after it (`results/` becomes `results/...` in the repo; a file
  `thresholds.json` becomes `thresholds.json/thresholds.json`). Two paths with one name are refused.
- **One commit per interval** with every new or changed file (changed = modification time or size differs from the last
  successful upload). Nothing changed means no commit. A failed commit marks nothing uploaded, so the next interval
  retries it. The first push is after the first interval; `stop()` always pushes once more.
- **Never uploaded**: `*.tmp` (an atomic writer's half-written file), whatever `include` says; anything matching
  `sync.exclude`. `sync.include` (empty = everything) limits to matching files. Patterns are `fnmatch` globs matched
  against the path under the folder or the file name (`cache/*`, `.launch.lock`, `*.jsonl`).
- **Files still growing** (rows appended during a push) are uploaded capped to the size seen at listing
  (`huggingface_hub`'s own `PartialFileIO`, the one `CommitScheduler` uses), so a push never tears a file; the growth
  goes in the next push. Not `CommitScheduler` itself: it takes one folder per scheduler (one commit per folder per
  interval) and cannot filter `*.tmp` across several paths in one commit.
- **Priority**: Windows `BELOW_NORMAL_PRIORITY_CLASS` (0x4000) or `IDLE_PRIORITY_CLASS` (0x40) through Popen
  `creationflags`; POSIX `os.nice(10)` or `os.nice(19)` in the child before exec. `sync.priority` is `below_normal`
  or `idle`.
- **Parent death**: the child reads stdin; end of input (a dead or killed parent) makes it do the final push and exit.
- **Failures never stop or alter an experiment**: in the child, the first failure of each kind (`repo`, `commit`) is
  printed on stderr (token redacted), later ones are counted and returned in `summary`. If the child dies, the parent
  prints one line (`sync: uploader process exited with code N; ...`) and carries on.
- **Secrets**: `HF_TOKEN` is read only from the environment or the repo-root `.env` (python-dotenv, `adapters/env.py`,
  no override), inherited by the child through its environment, never put on its command line or in settings, and
  redacted from every line printed.

## Settings keys read

All required once `sync.enabled` is true (no default in code).

| key | meaning |
|---|---|
| `sync.enabled` | bool, owner decision |
| `sync.repo_id` | `user/name`, owner decision |
| `sync.repo_type` | `"dataset"` or `"model"` (configuration: `"dataset"`) |
| `sync.private` | bool, used when the repo is created (configuration: `true`) |
| `sync.every_minutes` | number > 0 (configuration, e.g. `10`) |
| `sync.include`, `sync.exclude` | lists of glob patterns, may be empty (configuration) |
| `sync.priority` | `"below_normal"` or `"idle"` (configuration: `"below_normal"`) |
| `sync.final_push_timeout_seconds` | number > 0, how long `stop()` waits for the final push (configuration, e.g. `600`) |

Environment: `HF_TOKEN` (needs write access; created repos are private when `sync.private`).

## Layout

`public.py`; `private/settings.py` (keys), `private/plan.py` (mirror, exclusions, change test; pure),
`private/loop.py` (the upload loop over an uploader with `ensure_repo()` and `commit(files, message)`),
`private/errors.py`; `adapters/env.py` (token, `.env`, redaction), `adapters/hub.py` (the only importer of
`huggingface_hub`, loaded only in the child), `adapters/child.py` (the child's `main`), `adapters/process.py` (launch,
priority, watcher, stop). `process.UPLOADER` ("module:callable") replaces the Hub uploader in the child; tests use it.

## Tests

`python -m pytest bazt_lab/features/sync -q` with the pinned interpreter. No network and no real token. `test_loop.py`
and `test_plan.py` run in process; `test_process.py` starts real child processes with a fake uploader (a few seconds
each). `test_network.py` is one real round trip, skipped unless `BAZT_NETWORK_TESTS=1` (also needs `HF_TOKEN` and
`BAZT_SYNC_TEST_REPO`, a private scratch dataset repo you own).

## Requests

- runner (owns `profiles.toml` and `app.py`): add the `[sync]` table. Owner decisions, left unset or commented:
  `enabled`, `repo_id`. Configuration: `repo_type = "dataset"`, `private = true`, `every_minutes = 10`,
  `include = []`, `exclude = [".launch.lock"]`, `priority = "below_normal"`, `final_push_timeout_seconds = 600`.
- runner: in `run_block`, after the lock is taken and before units run:
  `handle = sync.start(s, [out, thresholds_dir])` (import the feature in `app.py` and pass it in the wiring, or call it
  through a `Wiring` hook as for the tracker); `handle.stop()` in the same `finally` as `tracker.finish()`. Call it
  only when `sync.enabled` is true or let `start` return the no-op handle; it needs no other guard. A refusal
  (`SyncError`, `core.Unset`) should stop the launch like a tracking refusal.
- gnn: `handle = sync.start(settings, [checkpoint_dir])` around training, `handle.stop()` in `finally`. Write
  checkpoints atomically (write `x.pt.tmp`, then `os.replace`); `*.tmp` is never uploaded.
- quality: `HF_TOKEN` in `.env.example` and the how-to now also covers checkpoint sync (needs write access).

## Known gaps

- The real Hub path is covered only by the gated smoke test and the installed library's source (`create_repo`,
  `create_commit`, `CommitOperationAdd`, `PartialFileIO`); nothing here ran against the Hub.
- A very large first push (many GB) is one commit; the Hub client uploads LFS files in parallel threads inside the
  child, which is I/O bound but not free of CPU (hashing). Priority keeps it behind the experiments.
- The first push happens after the first interval, not at launch.
- On Windows `sys.executable` in a venv is a launcher that starts the real interpreter: `handle.pid` is the launcher.
  Priority is inherited by the real interpreter (verified through the launcher's priority class only).
- Concurrent launches that sync overlapping folders to one repo would write the same paths; the runner's block lock
  prevents two launches of one block.
