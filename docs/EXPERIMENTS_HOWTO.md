# Running the BAZT experiments

The spec is `EXPERIMENT_PLAN.md`. The code layout is `docs/ARCHITECTURE_EXPERIMENTS.md`.

## Quick start

```
uv sync
cp .env.example .env          # fill in HF_TOKEN, WANDB_API_KEY, OPENROUTER_API_KEY
uv run python run_experiments.py status
uv run python run_experiments.py run-all --dry-run
uv run python run_experiments.py run-all
uv run python -m bazt_lab.features.analysis
```

Run every command in this document with `uv run` in front, or activate the environment first (`.venv/Scripts/activate` on Windows, `source .venv/bin/activate` elsewhere) and then use plain `python`. On a GPU machine after the CUDA install, a plain `uv run` puts the CPU build of torch back: use the forms in "Running on a GPU".

1. `uv sync` installs Python 3.11 and every dependency at its exact pin, torch from the CPU index.
2. `.env` holds the three keys. A real environment variable of the same name wins over `.env`. The keys are only read by
   the adapters and are never printed.
3. `status` shows every block: how many units are done, what is still missing (unset settings, unbuilt features,
   run-time variables).
4. `run-all --dry-run` prints the size of the whole run: units, LLM calls, dollars, hours. Nothing runs and no call is
   made, so the LLM blocks are sized without `--allow-live-llm` (a real LLM launch still needs the flag, after the
   kickoff). A block that cannot run yet is listed as skipped with its reason (an unset run-time variable or scope
   setting, or a condition its platform does not run), and the other blocks are sized. The plain `--dry-run` (without
   `run-all`) prints one line per block: its sizes, or the unset key that stops it being sized.
5. `run-all` runs the blocks in order and can be stopped at any point. The LLM blocks are skipped until you add
   `--allow-live-llm`, which is allowed only after the kickoff (2026-11-06 20:00 UTC). Run the same command again and it resumes:
   finished units are skipped and an unfinished unit continues after its last saved row.
6. The analysis reads the rows and writes `analysis_report.json` next to the block folders.

Results and checkpoints live in `results/experiments/<profile>/<block>/`: `units/` (rows and done markers) and `state/`,
with `units/w<k>/` and `state/w<k>/` for each parallel worker; `settings.json` (the settings the block's first launch
ran with); and `manifest-*.json`. The calibration blocks keep `thresholds.json` in their own folder (`mock_calibration/`,
`llm_calibration/`). The profile folder holds `analysis_report.json`, the LLM and judge request caches
(`llm_cache.sqlite`, `judge_cache.sqlite`, shared by the blocks of the profile) and `last_errors.json`, the error of each
block whose last launch failed. GNN checkpoints and logs are under `results/experiments/gnn/`. The sync feature uploads
the profile's results folder (and, for `gnn_train`, the GNN folders) to the Hugging Face Hub in the background while a
run is going. It needs `sync.enabled` and an `HF_TOKEN` with WRITE access to the target repository; with a read-only
token the commits fail, the first failure of each kind is printed on stderr, and the run continues. Weights & Biases
runs are written offline to W&B's directory (`wandb/` unless `tracking.dir` is set) before the kickoff; after it, upload
them with `wandb sync <offline run dir>`.

That is all you need to run. The rest of this page is detail.

## Details

### Settings and profiles

`experiments/profiles.toml` holds every value. No code has a default for an experiment value. A commented-out key is
unset: a block that needs it refuses to start and names the key. `status` lists the unset keys each block still needs,
with the options that the comments of `profiles.toml` give.

- Profile `sprint` (active): 60 LLM conversations per cell, capped C and BAZT at budget cell (10, 10). Profile
  `research`: 100 conversations, cells (2, 2), (10, 10), (50, 50).
- The profiles also set the LLM sanitize modes (`llm.sanitize_modes`: redaction in `sprint`; redaction and an LLM rewrite
  in `research`) and the memory-attack correctness modes (`llm.ma_correctness`: the rule in `sprint`; the rule and the LLM
  judge in `research`).
- Switch for one launch with `--profile research`. Switch the default by editing `active_profile`.
- Override one value for one launch with `--set llm.conversations_per_cell=80` (dotted key, TOML value, repeatable).
- The run-time variables of spec section 10 (`[runtime]`) have no value in the repository. Set them in `profiles.toml`
  or with `--set runtime.n_insiders=3`. `ni_margin_accuracy` and `rank_agreement_threshold` are pre-registered: fix them
  before the first result. They are copied into every manifest.

### Running one block

```
python run_experiments.py --list
python run_experiments.py --run --block mock_calibration
python run_experiments.py --run --block mock_frontier
python run_experiments.py --run --block llm_calibration --allow-live-llm      # after the kickoff only
```

`--device`, `--workers N` and `--compile` set `compute.device`, `compute.workers` and `compute.compile` for one
launch, and `--results-root DIR` replaces the `results/experiments` base. Results do not depend on the compute options.

`run-all` runs the GNN training first, then the two calibration blocks, then the evaluation blocks. A block that replays
another block's results runs after it: `llm_adaptive_check` replays `mock_adaptive`, and `mock_legacy` takes its BAZT
rounds from `mock_frontier` (D116). A block whose source is skipped is skipped too.

Calibrate before evaluating: an evaluation block with calibrated arms (for example B, C, BAZT) refuses to run without
the `thresholds.json` of its platform's calibration block. `mock_heldout` refuses until a file named `ANALYSIS_FROZEN`
exists in the results root. A second concurrent launch of one block is refused (file lock; a crashed process never
blocks a resume).

LLM blocks are refused before the kickoff (2026-11-06 20:00 UTC) and need `--allow-live-llm`. Set
`SWARMLAB_KICKOFF_UTC` if the official start differs. Tracking (W&B) is off unless `tracking.enabled = true`.

### Environment check

```
python scripts/check_env.py            # exit 1 on any mismatch
```

It checks Python 3.11, every installed package against its pin (a local label such as torch's `+cpu` or `+cu130` is
ignored), `uv.lock` against `pyproject.toml`, the torch build (a CUDA build from the optional extras is a warning, not a
failure), that new tensors land on the CPU by default, the thread count (`--threads 8` requires an exact count), and
whether the three keys are set (presence only; `--require-secrets` makes a missing one a failure). After editing a pin
run `uv lock`.

### Running on a GPU

The default install is the CPU build of torch. The same torch release is available as a CUDA build through two optional
extras, which replace the CPU build (the CPU group has to be switched off):

```
uv sync --extra cu130 --no-group cpu     # current drivers
uv sync --extra cu126 --no-group cpu     # older drivers
uv sync                                  # back to the CPU build
```

`uv run` syncs the environment before every command, and without options it syncs the default groups, which include the
CPU build: after the CUDA sync, a plain `uv run python ...` uninstalls the CUDA build and installs the CPU one again
(`uv sync --dry-run --extra cu130 --no-group cpu` shows the swap, `torch==2.14.0+cpu` out and `+cu130` in). So after a
CUDA sync run every command in one of these forms (with `cu126` instead of `cu130` if that is the extra you synced):

```
uv run --extra cu130 --no-group cpu python scripts/check_gpu.py --require-cuda   # the selection of the sync
uv run --no-sync python scripts/check_gpu.py --require-cuda                     # the environment as it is
python scripts/check_gpu.py --require-cuda                                      # with the environment activated
```

`UV_NO_SYNC=1` in the shell makes every `uv run` behave as with `--no-sync`. The same rule holds for every other command
on that machine (`run_experiments.py`, the e2e scripts, `check_env.py`, pytest).

Run `scripts/check_gpu.py --require-cuda`, in one of those forms, before anything else. It prints the torch, CUDA and GPU facts, runs a
small real slice of the mock frontier on CPU and on CUDA, in float64 and float32, with `torch.compile` off and on,
compares every combination with the CPU float64 reference, and reports, per combination, the episodes it planned, the
rows it read back and episodes per second. It exits 1 on any mismatch, and on a combination that read fewer rows than
it planned. `--require-cuda` makes it exit 2 at once when no cuda device is usable: a CPU build of torch (what a
re-sync leaves behind), no driver, or `--devices` without cuda. Without the flag a machine without CUDA runs the CPU
part and reports CUDA as unavailable with exit 0, which is right on a laptop and wrong on the GPU machine.
`check_env.py` reports the CUDA build as a warning. A run uses the GPU with `--device cuda` (or
`--set compute.device=cuda`). Every run, mock and LLM, on CPU and on GPU, computes in float32 (D117,
`compute.dtype = "float32"` in `profiles.toml`); budgets, costs and caps stay exact (float64/int64). float64 is the
reference: `check_gpu.py` compares every float32 combination with the CPU float64 run. Float64 combinations must match
every column exactly. Float32 combinations must agree on budget accounting (spend, violations, exhaustion step) in at
least 99% of the episodes, and on every other number within 0.005 in at least 99% of them.
`compute.compile`, `compute.deterministic` and `compute.workers` are in `profiles.toml` too, and results must not
depend on them. The e2e scripts run the profile's dtype, the one the real runs use.

### Checks

```
python -m pytest -q                              # tests/, bazt_lab/, scripts/
python scripts/check_static.py                   # ruff, compileall, import gate (--fix, --only <step>)
python scripts/e2e_mock.py                       # real launcher, tiny mock grid: kill, resume, analysis
python scripts/e2e_mock.py --quick --workers 2   # the same with 2 worker processes (shards units/w0/, units/w1/)
python scripts/e2e_llm_fake.py --kickoff-override 2000-01-01T00:00:00+00:00
```

The e2e scripts use a temporary results directory and clearly labelled test values for the open settings (passed with
`--set`, never written to `profiles.toml`), kill the launcher part-way, resume it, check every row and run the analysis.
`--workers N` (e2e_mock.py, default 1) sets `compute.workers` for the run, so the kill, the resume and the analysis are
checked on the worker shards too; use the count the real run will use. On the LLM side the kill must leave a unit
part-written (some of its rows on disk, no done marker), so the resume of a unit in the middle is exercised; the script
kills again if a kill left none (at most 4 launches) and otherwise fails with `resume path not exercised`. The analysis
check fails unless, for each platform, the rows the analysis read equal the committed rows on disk of the blocks it
reads, and no block the run wrote is in `blocks_missing`. They send no key to the child process. The LLM script uses the fake API (`llm.api = "fake"`) and still meets the
kickoff guard before 2026-11-06 20:00 UTC (exit code 3) unless `--kickoff-override` names a past time. Exit codes: 0
passed, 1 a check failed, 2 could not start, 3 guard refused. `--keep` keeps the temporary results.

The import gate enforces: core imports no feature; features import each other only through `public.py`; third-party
services (wandb, datasets, huggingface_hub, openai, sentence_transformers, torch_geometric, swarmlab, bazt, llm_client)
only in `adapters/`; only `app.py` imports every feature; every feature has `public.py`, `private/`, `adapters/`,
`tests/` and `CONTRACT.md` (plus an optional thin `__main__.py`); no secret literal; settings are read with `core.get`.

### Environment variables

| variable | needed for |
|---|---|
| `OPENROUTER_API_KEY` | agent backbones and the judge (live calls, after the kickoff) |
| `WANDB_API_KEY` | W&B in `online` mode; offline runs need none |
| `HF_TOKEN` | the sync upload (WRITE access to the repository); optional for the public datasets |
| `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1` | rerun from the local dataset cache (same revisions as the run that filled it) |
| `SWARMLAB_KICKOFF_UTC` | override the kickoff time (ISO, UTC) |

First online dataset load: MMLU 51.5 MB, CommonsenseQA 1.6 MB, GSM8K 2.7 MB. Run the dataset smoke test with
`BAZT_NETWORK_TESTS=1 python -m pytest bazt_lab/features/llm_data/tests/test_network_smoke.py -q`.

### Troubleshooting

| symptom | cause and fix |
|---|---|
| `Refusing <block>: run-time variables not set` | set the listed keys under `[runtime]` or with `--set` |
| `setting not set in profiles.toml: <key>` | an open decision of the spec; the owner sets it |
| `feature <name> is not built` | that feature is missing or fails to import; `status` shows which |
| `no calibrated thresholds for N cell(s)` | run the platform's calibration block first |
| `calibrated at alpha=...` | delete `thresholds.json` to recalibrate at the new alpha |
| `another launch is running this block` | a live process holds `.launch.lock` |
| `Refusing: before kickoff` | expected before 2026-11-06 20:00 UTC; mock blocks are unaffected |
| `check_env.py` warns about a CUDA build | expected after the CUDA sync of "Running on a GPU"; `uv sync` restores the CPU build |
| `check_env.py` reports wrong versions | run `uv sync` (or the CUDA sync on a GPU machine); do not `pip install` over the environment |
| `check_gpu.py` reports CUDA unavailable after the CUDA sync | a plain `uv run` re-synced the CPU build; sync again and use `uv run --no-sync` or `uv run --extra cu130 --no-group cpu` |
| import gate violation | fix the import; there is no allowlist |
