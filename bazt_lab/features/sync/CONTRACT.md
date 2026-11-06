# sync: contract

Pushes checkpoint folders to a Hugging Face repository from ONE separate child process at low OS priority, every
`sync.every_minutes`, so a crash never loses them and uploading never takes the experiments' CPU. The experiment
process starts the child and later stops it. While the run goes on it never waits on an upload; `stop()` waits for the
final push, at most `sync.final_push_timeout_seconds`.

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
  `thresholds.json` becomes `thresholds.json/thresholds.json`). Two paths with one name are refused. The runner
  (`runner/private/launch.py`, `_syncing`) passes the profile folder `results/<profile>`, and for `gnn_train` also
  `gnn.artifacts_dir` and `gnn_train.logs_dir`. So every block's `units/` and `state/` folders are mirrored, worker
  shards included (`units/w<k>/`, `state/w<k>/`), and so are the calibration thresholds.
- **No checkpoint rule**: sync uploads every file it finds under the paths (`*.tmp` aside). An orphan batch rows file
  or the rows of a unit still running go up as they are. Readers of the Hub copy apply the done rule of
  `committed_row_files` (`core/private/checkpoint.py`).
- **One push per interval** with every new or changed file (changed = modification time or size differs from the last
  successful upload), as one commit per batch of at most 500 files (`loop.COMMIT_FILES`), so a push never holds
  thousands of handles or gigabytes of staging. Nothing changed means no commit. A failed batch marks nothing uploaded
  (earlier batches stay marked), so the next interval retries it. A file that vanished or cannot be read between
  listing and copy is skipped and retried next interval; it never fails the commit. A listing that fails is reported
  as a `scan` failure and does not end the loop. The first push is after the first interval; `stop()` always pushes
  once more.
- **Never uploaded**: `*.tmp` (an atomic writer's half-written file), whatever `include` says; anything matching
  `sync.exclude`. `sync.include` (empty = everything) limits to matching files. Patterns are `fnmatch` globs matched
  against the path under the folder or the file name (`cache/*`, `.launch.lock`, `*.jsonl`).
- **One consistent version per file**: each file is opened with `FILE_SHARE_READ | FILE_SHARE_WRITE |
  FILE_SHARE_DELETE` (Windows, `CreateFileW` through ctypes; plain `open` elsewhere), copied to a private staging
  directory as the first `fstat`-size bytes of the handle actually opened, and closed at once. The commit uploads the
  staged copies, so NO experiment file is open while the network is used. A file replaced after listing uploads its
  full new version; a file appended to uploads the prefix that was there (the growth goes in the next push); never a
  torn mix. The uploaded stamp is the opened handle's, so nothing is uploaded twice. Not `CommitScheduler`: it takes
  one folder per scheduler and cannot filter `*.tmp` across paths in one commit; and not `PartialFileIO`, which opens
  with Python's `open()` and holds every handle for the whole commit.
- **What the experiment can still hit on Windows**: Windows refuses `os.replace` over a file while ANY handle to it is
  open, whatever sharing flags the reader used (checked: `MoveFileEx` gives "Access is denied" even for a
  `FILE_SHARE_DELETE` handle). Delete and rename are allowed during the copy. So an atomic replace of a file can fail
  only while the child is copying that one file (milliseconds for rows and state, longer for a large checkpoint).
  `core.atomic_replace` (`core/private/checkpoint.py`) waits that out: up to 20 attempts with growing waits, about 9.5 s
  in all. It is used by the CheckpointStore, the runner's `files.write_text` and the gnn writes (`epochs.save_epoch`,
  `checkpoint.save_checkpoint`). The adaptive table's replaces do not use it yet (adaptive CONTRACT.md, Requests).
- **Priority**: Windows `BELOW_NORMAL_PRIORITY_CLASS` (0x4000) or `IDLE_PRIORITY_CLASS` (0x40) through Popen
  `creationflags`; POSIX `os.nice(10)` or `os.nice(19)` in the child before exec. `sync.priority` is `below_normal`
  or `idle`.
- **Parent death**: the child reads stdin; end of input (a dead or killed parent) makes it do the final push and exit.
- **Failures never stop or alter an experiment**: in the child, the first failure of each kind (`repo`, `commit`,
  `scan`) is printed on stderr (token redacted), later ones are counted and returned in `summary`. If the child dies,
  the parent prints one line (`sync: uploader process exited with code N; ...`) and carries on. After `stop()` the
  runner prints the summary's failures and commit count when there are failures (`launch.py`, `_syncing`).
- **Secrets**: `HF_TOKEN` is read only from the environment or the repo-root `.env` (python-dotenv, `adapters/env.py`,
  no override), inherited by the child through its environment, never put on its command line or in settings, and
  redacted from every line printed. `.env.example` and `docs/EXPERIMENTS_HOWTO.md` name it.

## Settings keys read

`sync.enabled` is always required (`core.Unset` when missing). The other keys are required once it is true (no default
in code). The `[sync]` table of `experiments/profiles.toml` sets the configuration keys below.

| key | meaning |
|---|---|
| `sync.enabled` | bool, owner decision |
| `sync.repo_id` | `user/name`, owner decision |
| `sync.repo_type` | `"dataset"` or `"model"` (configuration: `"dataset"`) |
| `sync.private` | bool, used when the repo is created (configuration: `true`) |
| `sync.every_minutes` | finite number > 0 (configuration, e.g. `10`) |
| `sync.include`, `sync.exclude` | lists of glob patterns, may be empty (configuration) |
| `sync.priority` | `"below_normal"` or `"idle"` (configuration: `"below_normal"`) |
| `sync.final_push_timeout_seconds` | finite number > 0, how long `stop()` waits for the final push (configuration, e.g. `600`) |

Environment: `HF_TOKEN` (needs write access; created repos are private when `sync.private`).

## Layout

`public.py`; `private/settings.py` (keys), `private/plan.py` (mirror, exclusions, change test; pure),
`private/loop.py` (the upload loop over an uploader with `ensure_repo()` and `commit(files, message)`),
`private/errors.py`; `adapters/env.py` (token, `.env`, redaction), `adapters/hub.py` (the only importer of
`huggingface_hub`, loaded only in the child; stages and commits one batch), `adapters/shared_file.py` (shared-access
open and the consistent-version copy), `adapters/child.py` (the child's `main`), `adapters/process.py` (launch,
priority, watcher, stop). `process.UPLOADER` ("module:callable") replaces the Hub uploader in the child; tests use it.

## Tests

`python -m pytest bazt_lab/features/sync -q` with the pinned interpreter. No network and no real token. `test_loop.py`
and `test_plan.py` run in process; `test_process.py` starts real child processes with a fake uploader (about 5.5
minutes for the file in one run). `test_network.py` is one real round trip, skipped unless `BAZT_NETWORK_TESTS=1` (also
needs `HF_TOKEN` and `BAZT_SYNC_TEST_REPO`, a private scratch dataset repo you own).

## Requests

- profiles.toml (owner): `sync.enabled` and `sync.repo_id` stay commented, so unset, until the owner decides. The other
  `[sync]` keys are set to the configuration values above.
- adaptive (`private/table.py`, `private/search.py`): the table and progress replaces still call `os.replace` directly.
  They should use `core.atomic_replace` like the other writers (see Behaviour).

## Known gaps

- The real Hub path is covered only by the gated smoke test and the installed library's source (`create_repo`,
  `create_commit`, `CommitOperationAdd`); nothing here ran against the Hub. Tests run the real
  `HubUploader.commit` and real operation objects with only `create_commit` faked.
- A very large first push (many GB) is staged on local disk one batch at a time (up to 500 files; a batch of huge
  checkpoints needs that much free temp space) and is one commit per batch. The Hub client uploads LFS files in
  parallel threads inside the child, which is I/O bound but not free of CPU (hashing). Priority keeps it behind the
  experiments.
- The first push happens after the first interval, not at launch.
- On Windows `sys.executable` in a venv is a launcher that starts the real interpreter: `handle.pid` is the launcher.
  Priority is inherited by the real interpreter (verified through the launcher's priority class only).
- Concurrent launches that sync overlapping folders to one repo would write the same paths. The runner's block lock
  prevents two launches of one block, but every block of a profile mirrors the same profile folder, so two blocks of
  one profile run side by side each upload the other's files too.
