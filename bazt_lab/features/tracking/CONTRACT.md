# tracking: contract

`WandbTracker`, the `core.Tracker` contract over Weights & Biases (`wandb==0.31.0`, D87). One W&B run per block
launch. Offline until the hackathon kickoff. Results never live only in W&B: they are in the `CheckpointStore`.

## public.py

| name | what |
|---|---|
| `WandbTracker(settings=None)` | implements `core.Tracker`. `WandbTracker()` builds without arguments, imports nothing, touches no file and no network. `WandbTracker(settings)` checks the `tracking.*` settings, the kickoff rule and the key at once, so a launch can refuse before it writes anything; `app.py` builds it this way. Property `mode`: the mode of the current run after the kickoff rule (`None` before `start`, kept after `finish()`). Property `failures`: operation name -> how many times it failed during the run (`{}` on a clean run) |
| `TrackingError` | the one error type: tracking cannot be set up as the settings ask. A `RuntimeError`; the message names the setting or the variable and never contains a secret |

`core.Unset` (naming the dotted key) is raised for a missing required setting, as everywhere in `bazt_lab`.

## Methods

| call | behaviour |
|---|---|
| `start(block, config, resume_id) -> run_id` | reads the settings from `config` (the runner passes the resolved settings plus `manifest`), applies the kickoff rule, then `wandb.init(project, entity, group=config["profile"], job_type=block, config=<cleaned config>, id=resume_id, resume="allow", mode=<effective mode>, dir, reinit="create_new", save_code=False, settings={"disable_git": True}, allow_val_change=True)`, then `define_metric("tracking/step")` and `define_metric("*", step_metric="tracking/step")`. Returns the run id; the runner stores it and passes it back next launch. Raises `TrackingError` or `core.Unset` and starts no run when the run cannot be set up (`tracking.dir` is created first when set). Calling it twice without `finish()` raises `TrackingError` |
| `log(metrics, step)` | `run.log({**metrics, "tracking/step": int(step)})`. `tracking/step` is declared the x-axis of every metric (`define_metric("*", step_metric="tracking/step")`), so the runner's step (units done plus skipped) is the axis, not wandb's call counter |
| `log_rows(table, rows)` | buffers the rows per table name (cells cleaned at once) and, every `tracking.flush_rows` buffered rows, logs them as ONE `wandb.Table(allow_mixed_types=True)` part: `<table>`, then `<table>_part1`, `<table>_part2`, ... (a flush larger than `wandb.Table.MAX_ARTIFACT_ROWS`, 200000, is split further). `finish()` flushes what is left before it ends the run, so a table is complete only after `finish()`. Rows are put into `Table.data` directly: `add_data` re-infers every cell's type and was most of the cost. The columns are those of the first call's rows. A row lacking a column gets `None`; a column the table does not have is dropped and reported once. A part that fails to log is reported; the later parts of that flush are not logged, and no row is retried. Cells: `None`, bool, int, float (NaN and infinity kept), str as they are; numpy and torch scalars become Python scalars; lists, dicts and arrays become JSON text |
| `artifact(path, name, kind)` | logs a file or a directory as `wandb.Artifact(name, type=kind)`; `name` and `kind` have every character outside `[A-Za-z0-9_.-]` replaced by `_` |
| `finish()` | ends the run and prints one summary line when any operation failed. Idempotent; a no-op before `start`. After it the tracker can `start` again |

## Modes and the kickoff rule

`tracking.mode` is `online`, `offline` or `disabled`; anything else raises `TrackingError`.

- `offline`: data is written under `<tracking.dir>/wandb/offline-run-<timestamp>-<run id>/` (`./wandb/` in the working
  directory when `tracking.dir` is unset; git-ignored). No key, no network. Upload later with
  `wandb sync <that directory>` (`wandb sync --dry-run` lists what is waiting). Keep the run directories until synced.
- `disabled`: `wandb.init(mode="disabled")`; every call is a no-op, nothing is written, `start` still returns an id (the
  saved one when given). `log_rows` and `artifact` build no table and no artifact.
- `online`: needs `WANDB_API_KEY` and `tracking.entity`; a missing one is a `TrackingError` naming it, raised before
  wandb is touched. **Before the kickoff the run is recorded `offline` instead** (D87) and a notice on stderr says so,
  unless `tracking.allow_online_before_kickoff = true`. The kickoff time is `SWARMLAB_KICKOFF_UTC` when set (ISO time,
  no zone means UTC), else `swarmlab.KICKOFF_UTC`, the same rule as swarmlab's own guard. If neither can be read,
  `TrackingError` (the run is not started online by guess). It is read only when it decides: `offline`, `disabled` and
  allowed `online` never consult it.
- The explicit `mode` passed to `wandb.init` wins over a `WANDB_MODE` in the environment.

## Failures

- At start: every failure (settings, key, kickoff, `wandb.init`, `define_metric`) is a `TrackingError` (or
  `core.Unset`). A half-started run is closed first. `WandbTracker(settings)` runs the same checks (mode, `flush_rows`, entity, key, kickoff) in the constructor, before any file is written. There is no fallback to a tracker that records nothing: the
  runner's `core.NullTracker` is chosen by `tracking.enabled`, never by this feature.
- During the run (`log`, `log_rows`, `artifact`, `finish`): never raises. The first failure of each operation prints
  one line on stderr (`tracking: <operation> failed and is skipped; the results stay on disk ...`), later ones are
  counted in `failures`, and `finish()` prints the totals. Using the tracker before `start` is reported the same way.
- `WANDB_API_KEY` never appears in an error message, a stderr line or a traceback: text is passed through
  `env.redact`, and a start error is raised without chaining when ANY exception in its cause/context chain carries the key.

## What leaves the process

- Config: the resolved settings plus `manifest` as plain JSON. Any key whose last word is `token`, `api_key`, `secret`
  or `password` is removed at every depth, in any spelling: `api_key`, `API_KEY`, `apiKey`, `api-key`, `accessToken`,
  `clientSecret`. A `.` in a key becomes `_` (W&B config keys cannot contain periods); two keys that would merge are
  refused with `TrackingError`. Settings hold no secret (architecture gate); this is a second guard.
- No code and no git state (`save_code=False`, `disable_git`).
- Tables and artifacts: only what the runner and the platforms hand to `log_rows` and `artifact`.

## Settings keys read

Required: `profile` (the W&B group), `tracking.mode`, `tracking.project`, `tracking.flush_rows` (whole number >= 1: rows
buffered per table before one table part is logged; performance only, never changes a result). `profiles.toml` sets
`flush_rows = 3600`. Measured offline with the real library, 360 rows x 70 columns per `log_rows` call: 0.50 s per call
at `flush_rows` 360, 0.36 s at 3600, 0.30 s at 36000. A larger value holds more rows in RAM (about 60 MB per 36000 rows
at 70 columns), and a crash loses the unflushed rows from W&B (not from disk).
Optional: `tracking.entity` (required when the effective mode is `online`), `tracking.dir` (where `wandb/` is
created), `tracking.allow_online_before_kickoff` (bool, default false when unset).
`tracking.enabled` is read by the runner, not here.

Environment (process first, then the repo-root `.env`, never `profiles.toml`): `WANDB_API_KEY` (online only),
`SWARMLAB_KICKOFF_UTC` (optional). Loaded with python-dotenv without override, inside `adapters/env.py` only.
`.env.example` names `WANDB_API_KEY`, and `docs/EXPERIMENTS_HOWTO.md` says how to run `wandb sync` after the kickoff.

## Layout

`adapters/wandb_tracker.py` (the only importer of `wandb`, lazily, after `.env` is loaded), `adapters/env.py` (key,
`.env`, redaction), `adapters/kickoff.py` (the only importer of `swarmlab`), `adapters/offline_log.py` (reads back an
offline run's `run-<id>.wandb` log, used by the tests and usable by a person before `wandb sync`; does not verify
record checksums), `private/settings.py` (keys and the kickoff rule), `private/clean.py` (config, cells, names),
`private/failures.py` (once-per-operation reporting), `private/errors.py`.

## Tests

`python -m pytest bazt_lab/features/tracking -q` with the pinned interpreter. No network, no key, no live call.
`test_tracker.py` runs the tracker against a recording stand-in for `wandb` (`tests/fakes.py`), including every
failure path. `test_offline_run.py` runs the real `wandb` offline in a temp directory (with `WANDB_MODE=disabled` in
the environment, to prove the explicit mode wins) and reads the run back from its own log; it also runs a real
`disabled` run. Each test starts without a key, without a kickoff override and without a real `.env`.

## Invariants

- No experiment value has a default in this feature; a missing required key raises `core.Unset` naming it.
- The key is read only from the environment, never from settings, never printed, logged, put in a config, a URL or a
  traceback.
- Mode `online` before the kickoff never reaches the network unless `allow_online_before_kickoff` is true.
- A tracking failure after `start` never stops an experiment and never goes unreported; a failure at `start` always
  stops the launch.
- Nothing here draws random numbers, reads ground truth or writes outside `tracking.dir` / wandb's own directory.

## Open questions

1. Where should offline runs live? Unset `tracking.dir` leaves them in `./wandb/` of the launch directory
   (git-ignored). A `tracking.dir` under `results/` keeps them next to the block they describe. Owner's call; the
   code works with either. `profiles.toml` has no `tracking.dir` line, so it is unset.
2. `tracking.entity` (the W&B team or user) is commented out in `profiles.toml` as left for the owner. Online runs
   refuse without it, on purpose: a missing entity would send runs to the key owner's default account.

## Requests

- runner/metrics (owner decision): even at about 0.3 s per 360 rows, mock_frontier's about 10,260 calls cost roughly 0.9 h to 1.4 h of tracker time against a 2.1 h block. If that is too much, log per-unit rows to W&B only for a subset or only summary rows; the full rows are in the CheckpointStore anyway.
- callers of `tracker.log_rows` (mock_engine does, `platform.py` `uniform_keys`; llm_harness and metrics do not yet):
  every row of one table name carries the same keys, `None` where a value does not apply. A W&B table cannot gain a
  column after its first rows, so a key that first appears later is dropped from W&B (reported once; the row on disk is
  complete).
- core (optional): `Tracker.finish(exit_code=0)`. `finish()` takes no status, so a block that crashes still ends its
  W&B run as finished (the runner calls it in `finally`); the crash is visible in the runner's output and the missing
  `.done` marks, not on the W&B page.

## Known gaps

- The online path against the real W&B server is not exercised (no network, no key before the kickoff): it is tested
  against the stand-in only. The real library covers `offline` and `disabled`.
- Resuming: a second launch of a block with the saved id creates a second offline directory with the same run id
  (tested). That `wandb sync` of both directories lands in one W&B run is W&B's documented `resume="allow"`
  behaviour and could not be checked offline.
- Not safe for concurrent calls from several threads; the runner calls it from one.
