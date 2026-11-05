# Running the BAZT experiments

The spec is `EXPERIMENT_PLAN.md`. The code layout is `docs/ARCHITECTURE_EXPERIMENTS.md`.

## Quick start

```
uv sync
cp .env.example .env          # fill in HF_TOKEN, WANDB_API_KEY, OPENROUTER_API_KEY
python run_experiments.py status
python run_experiments.py run-all --dry-run
python run_experiments.py run-all
python -m bazt_lab.features.analysis
```

1. `uv sync` installs Python 3.11 and every dependency at its exact pin, torch from the CPU index.
2. `.env` holds the three keys. A real environment variable of the same name wins over `.env`. The keys are only read by
   the adapters and are never printed.
3. `status` shows every block: how many units are done, what is still missing (unset settings, unbuilt features,
   run-time variables).
4. `run-all --dry-run` prints the size of the whole run: units, LLM calls, dollars, hours. Nothing runs.
5. `run-all` runs the blocks in order and can be stopped at any point. Run the same command again and it resumes:
   finished units are skipped and an unfinished unit continues after its last saved row.
6. The analysis reads the rows and writes `analysis_report.json` next to the block folders.

Results and checkpoints live in `results/experiments/<profile>/<block>/`: `units/` (rows and done markers),
`state/`, `thresholds.json`, `manifest-*.json`. GNN checkpoints and logs are under `results/experiments/gnn/`. The sync
feature uploads these folders to the Hugging Face Hub in the background while a run is going (it needs `sync.enabled`
and an `HF_TOKEN` with WRITE access to the target repository; a read-only token is refused). Weights & Biases runs are written offline to `wandb/` before the kickoff; after it, upload them with
`wandb sync <offline run dir>`.

That is all you need to run. The rest of this page is detail.

## Details

### Settings and profiles

`experiments/profiles.toml` holds every value. No code has a default for an experiment value; a missing one stops the
launch and names the key.

- Profile `sprint` (active): 60 LLM conversations per cell, capped C and BAZT at budget cell (10, 10). Profile
  `research`: 100 conversations, cells (2, 2), (10, 10), (50, 50).
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

Calibrate before evaluating: evaluation blocks of calibrated arms (B, C, BAZT) refuse to run without the
`thresholds.json` of the platform's calibration block. `mock_heldout` refuses until a file named `ANALYSIS_FROZEN`
exists in the results root. A second concurrent launch of one block is refused (file lock; a crashed process never
blocks a resume).

LLM blocks are refused before the kickoff (2026-11-06 20:00 UTC) and need `--allow-live-llm`. Set
`SWARMLAB_KICKOFF_UTC` if the official start differs. Tracking (W&B) is off unless `tracking.enabled = true`.

### Environment check

```
python scripts/check_env.py            # exit 1 on any mismatch
```

It checks Python 3.11, every installed package against its pin, `uv.lock` against `pyproject.toml`, a CPU build of
torch and the thread count (`--threads 8` requires an exact count), and whether the three keys are set (presence only;
`--require-secrets` makes a missing one a failure). After editing a pin run `uv lock`.

### Checks

```
python -m pytest -q                              # tests/, bazt_lab/, scripts/
python scripts/check_static.py                   # ruff, compileall, import gate (--fix, --only <step>)
python scripts/e2e_mock.py                       # real launcher, tiny mock grid: kill, resume, analysis
python scripts/e2e_llm_fake.py --kickoff-override 2000-01-01T00:00:00+00:00
```

The e2e scripts use a temporary results directory and clearly labelled test values for the open settings (passed with
`--set`, never written to `profiles.toml`), kill the launcher part-way, resume it, check every row and run the analysis.
They send no key to the child process. The LLM script uses the fake API (`llm.api = "fake"`) and still meets the
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
| `check_env.py` reports a CUDA build or wrong versions | run `uv sync`; do not `pip install` over the environment |
| import gate violation | fix the import; there is no allowlist |
