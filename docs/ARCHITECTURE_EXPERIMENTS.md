# bazt_lab: architecture of the experiment code

`bazt_lab` implements `docs/EXPERIMENT_PLAN.md`. It is a modular monolith in the style of OhmBackend and lives beside the
architecture track's `bazt/` package and `swarmlab.py`, which it never modifies.

## Layout

```
run_experiments.py                 thin CLI entry: calls bazt_lab.app.main
bazt_lab/
  app.py                           composition root: builds the arm registry, the platforms and the tracker
  core/public.py                   shared contracts (only import path into core)
  core/private/                    settings, rng streams, checkpoints, contracts, compute (device, dtype, compile)
  features/<feature>/
    public.py                      the feature's only importable surface
    private/                       implementation
    adapters/                      code that talks to an external library or service (wandb, HF, OpenRouter,
                                   G-Safeguard, swarmlab, bazt)
    tests/                         the feature's tests (pytest)
    CONTRACT.md                    what public.py offers, its invariants, and the settings keys it reads
  tools/check_architecture.py      import-boundary gate, run by the test suite
experiments/profiles.toml          every setting; profiles sprint and research
```

## Import rules (enforced by bazt_lab/tools/check_architecture.py)

1. `core` imports no feature and no `bazt_lab.app`. Every other module imports core only through `bazt_lab.core.public`, as
   `from bazt_lab.core import public as core`.
2. A feature imports another feature only as `from bazt_lab.features.<other> import public as <alias>` or
   `from bazt_lab.features.<other>.public import <name>`. Never its `private` or `adapters`.
3. Only `adapters/` modules import third-party services: `wandb`, `datasets`, `huggingface_hub`, `openai`,
   `sentence_transformers`, `torch_geometric`, `swarmlab`, `bazt`, `llm_client`. torch and numpy may be used anywhere.
4. Only `bazt_lab/app.py` imports every feature; it wires them and holds no logic of its own.
5. A feature root holds `__init__.py`, `public.py` and the packages `private/`, `adapters/`, `tests/`, plus `CONTRACT.md`. The one
   other module allowed there is a thin `__main__.py` (`python -m bazt_lab.features.<feature>`): it imports `sys` and its
   own `public` and only calls it, with nothing else in it.
6. No module reads an experiment value it does not get from settings; secrets come only from the environment
   (`HF_TOKEN`, `WANDB_API_KEY`, `OPENROUTER_API_KEY`), never from settings or code. `python-dotenv` is imported only in
   `adapters/`, and `load_dotenv` keeps `override=False`, so `.env` never replaces the environment.
7. Settings are read with `core.get`, which raises `Unset` naming a missing key; `settings.get(key, default)` is refused.
   Outside `adapters/`, `importlib.import_module` and `__import__` take a literal module name. The one exception is
   `app.py`, which may load `bazt_lab.features.<name>.public` for each feature name.

## Features and owners (one writer per file)

| feature | owns | provides |
|---|---|---|
| core | `bazt_lab/core/**` | contracts, settings, streams, checkpoints, compute (device, dtype, compile) |
| runner | `run_experiments.py`, `bazt_lab/app.py`, `bazt_lab/features/runner/**`, `tests/test_run_experiments.py`, `experiments/profiles.toml` | blocks, units, dry-run sizing, launch, calibration orchestration, manifests, CLI |
| tracking | `bazt_lab/features/tracking/**` | `WandbTracker` (Tracker contract) |
| sync | `bazt_lab/features/sync/**` | background upload of results, checkpoints and logs to the Hugging Face Hub (its adapters import only `huggingface_hub` and `python-dotenv`); started by the runner through `public.py`, off unless `sync.enabled` |
| mock_engine | `bazt_lab/features/mock_engine/**` | `MockPlatform` (Platform contract): torch engine, graphs, attack models, auditor presets, budget enforcement |
| conventional | `bazt_lab/features/conventional/**` | arms `A`, `B`, `C` (capped by their budget cell) |
| references | `bazt_lab/features/references/**` | arms `GNN_standin` (mock), `random_alloc`, `oracle_alloc`, `bra_greedy`, the legacy arms `fixed_early`, `fixed_mid`, `fixed_late`, `score_only`, `centrality_only`, `random_capped`, `risk_aware`, `risk_aware_tiered` and their `_matched` controls |
| bazt_slot | `bazt_lab/features/bazt_slot/**` | arms `BAZT` (conforming to the spec), `BAZT_upstream` (adapter to `bazt/`), and the ablations `BAZT_no_budget`, `BAZT_no_topology`, `BAZT_no_graduated` |
| metrics | `bazt_lab/features/metrics/**` | EpisodeTrace -> per-unit rows (spec §6) |
| analysis | `bazt_lab/features/analysis/**` | statistics and hypothesis labels (spec §7) |
| llm_data | `bazt_lab/features/llm_data/**` | task items from Hugging Face (pinned revisions) and vendored InjecAgent / PoisonedRAG |
| judge | `bazt_lab/features/judge/**` | PromptArmor judge (Judge contract), logprob("yes"), fake judge |
| llm_harness | `bazt_lab/features/llm_harness/**`, `.gitmodules`, `vendor/G-safeguard` | `LLMPlatform` (Platform contract) over the pinned G-Safeguard submodule and patch set; its adaptive condition is `gray_box` only (D110) |
| gnn | `bazt_lab/features/gnn/**` | G-Safeguard GNN port (PyG) and its training; arms `GNN` and `GNN_recalibrated` (LLM side only) |
| quality | `pyproject.toml`, `uv.lock`, `ruff.toml`, `.gitignore`, `scripts/**`, `bazt_lab/tools/**`, `tests/test_architecture_bazt_lab.py`, `.env.example`, `docs/EXPERIMENTS_HOWTO.md` | pinned dependencies (`uv sync`), environment check, static checks, import gate, e2e smoke scripts, how-to |
| adaptive | `bazt_lab/features/adaptive/**` | adaptive attacker conditions for the mock (exact best response over a finite strategy space, per cell and arm); the LLM check replays its gray-box best responses (D110) |

A feature that needs a new settings key or a contract change does not edit a file it does not own: it lists the
request in its CONTRACT.md under "Requests" and in its report; the coordinator applies it.

## Data flow

runner expands a block into units, groups them by `GROUP_FIELDS` (arm, budget cell, topology, attack, agents, cost
condition, auditor preset, adaptive condition, attack parameters), and calls `Platform.run_batch` with the arm's
`ArmSpec` from the registry. The runner imports no other feature: `bazt_lab/app.py` hands it a `Wiring` that loads the
arms, platforms and tracker on first use. The platform builds an `EpisodeContext`, steps the defender round by round
(`Observation` in, `Actions` out), applies and charges the actions, produces an `EpisodeTrace`, turns it into rows with
the columns of `metrics.COLUMNS` (for example `empty_replies`, the count of LLM replies that stay empty after the
retries, D114), appends the rows progressively to the `CheckpointStore`, and marks units done. Calibration runs the
uncapped arm attack-free and returns thresholds at alpha; the runner stores them per cell and passes them in
`EpisodeContext.thresholds`. analysis reads the rows through `core.committed_row_files`.

A block directory holds `units/<unit_id>.jsonl` with a `<unit_id>.done` marker per unit (LLM side),
`units/batch-<hash>.jsonl` with a `batch-<hash>.done` list of the unit ids of one batch (mock side), and
`state/<unit_id>.pt` for the resumable state of an unfinished unit. With `compute.workers > 1`, each mock worker writes
only under `units/w<k>/` and `state/w<k>/` and reads the done markers of every shard. `core.committed_row_files`
returns the rows files that have their done marker, in `units/` and in every shard; rows files without one are not
read.

Real-valued tensors are computed in `compute.dtype` (float32 in `profiles.toml`, D117; float64 is the reference). Budgets,
costs and caps are accounted exactly in float64 or int64, whatever the compute dtype.
