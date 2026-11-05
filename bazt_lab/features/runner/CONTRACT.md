# runner: contract

Blocks, unit expansion, dry-run sizing, launch (batching, resume, calibration orchestration, threshold propagation),
manifests, tracking and the CLI of `EXPERIMENT_PLAN.md` (§1.1, §2.8, §5, §8, §9.5). Entry point:
`python run_experiments.py` calls `bazt_lab.app.main`; `bazt_lab/app.py` is the composition root and builds the
`Wiring` this feature runs over.

## public.py

| name | what |
|---|---|
| `main(argv, *, wiring)` | the CLI: `--list`, `--dry-run`, `--run`, `--block` (repeatable), `--profile`, `--set key=value`, `--settings`, `--results-root`, `--allow-live-llm` |
| `Wiring(arm, platform, tracker)` | what the composition root hands over: `arm(name) -> ArmSpec` (raises `FeatureMissing`, or `LookupError` for a name no built feature exports), `platform("mock" or "llm") -> Platform` (raises `FeatureMissing`), `tracker(settings) -> Tracker` (called only when tracking is enabled) |
| `FeatureMissing(feature, detail)` | a feature that is not built or cannot be imported; `feature` may name several, comma separated |
| `BLOCKS`, `Block`, `CALIBRATION_BLOCK` | the 14 blocks (`name`, `platform`, `kind` = `calibrate` / `train` / `evaluate`, `units(settings, strict)`, `requires_freeze`); the calibration block whose `thresholds.json` each platform's evaluation blocks read |
| `units_*`, `capped_cells`, `UnitList`, `MockGrid` | unit expansion. An LLM block is a `UnitList` (one unit = one seed run). A mock block is a `MockGrid`: `len()`, `by_agents()`, `arms()`, `cells()` are closed-form, iteration is lazy |
| `unit_id`, `group_key`, `cell_key`, `GROUP_FIELDS` | identity and grouping of units (below) |
| `run_block(settings, name, allow_live_llm, *, wiring, results_root)` | launches one block (below) |
| `dry_run`, `llm_calls`, `cache_savings`, `list_blocks` | sizing and listing |
| `load_settings`, `missing_runtime`, `PRE_REGISTERED`, `RUNTIME_BY_PLATFORM`, `FREEZE_MARKER`, `RESULTS_ROOT` | settings handling and the run-time variables each platform needs |

## Units

An LLM unit carries `block, platform, arm, audit, intervention, backbone, topology, attack, task, seed, conversations,
agents, rounds` (`gnn_train`: `arm="GNN", attack, task, edge_probability, attackers, backbone, conversations, agents,
rounds`; `llm_adaptive_check` adds `condition`). A mock unit is one episode: `block, platform, arm, audit, intervention,
topology, attack, agents, seed, sink` (plus `condition`, `preset` or `cost_condition` for the blocks that vary them).
`audit` and `intervention` are a fraction of all edge deliveries (`0.1` for 10 %) or `"uncapped"`; `sink` is an agent
index or `None`; `attack` is `"none"` for attack-free runs.

Before a unit reaches a platform the runner adds two fields that take no part in its identity:

- `id`: `unit_id(unit)`, `<arm>-<16 hex digits of sha256 of the unit's JSON>`; it names the unit's files in the store.
- `thresholds`: `{statistic: threshold}` of the unit's calibration cell (`cell_key`), `{}` for an arm that is not
  `ArmSpec.calibrated` and for `gnn_train`. A platform copies it into `EpisodeContext.thresholds`.

`group_key(unit)` = `(arm, audit, intervention, topology, attack, agents, cost_condition, preset, condition)`, absent
fields as `None`. `cell_key(unit)` = `(arm, topology, agents, backbone, task)`; the mock has no backbone and no task.

## Launch

`run_block` refuses first and writes nothing until every refusal has passed (each is a `SystemExit` message starting
`Refusing <block>:`):

1. `requires_freeze` block (`mock_heldout`) and no `ANALYSIS_FROZEN` file in the results root.
2. A run-time variable of the platform (spec §10) or one of the two pre-registered values is not set, or a
   `llm_sink_positions.<topology>` entry of the LLM grid is not set.
3. LLM block: `swarmlab._kickoff_guard` (`--allow-live-llm`, after the kickoff time), imported in `adapters/kickoff.py`.
4. A setting the block needs is not set (names the key; also when a platform raises `core.Unset` during the run).
5. A feature is not built: the platform's or an arm's feature is named ("feature conventional is not built"). An arm that
   does not run on the block's platform (`ArmSpec.platforms`) is refused too.
6. Calibration block: an arm that is not `ArmSpec.calibrated`; `thresholds.json` saved at another alpha than the
   current `budgets.alpha`.
7. Evaluation block: a calibrated arm has no thresholds for a cell of the grid; the message names the cell and the
   calibration block to run first.

Then the launch takes the lock `<block dir>/.launch.lock` (an OS file lock, released by the OS when the process dies, so
resuming a crashed launch is never blocked; a second launch of the same block is refused), writes
`manifest-<started_at>.json`, starts the tracking run and runs:

- `evaluate` and `train` blocks: units are walked in grid order, grouped by `group_key`, the units the store marks done
  are skipped, the rest are cut into batches of `mock.batch_size` units (mock) or one unit (LLM), and each batch is one
  `Platform.run_batch(units, arm_spec, settings, store, tracker)` with `arm_spec = wiring.arm(unit["arm"])` and
  `store = core.CheckpointStore(<results root>/<profile>/<block>/)`.
- `calibrate` blocks: one `Platform.calibrate(units, arm_spec, settings, store)` per cell, on that cell's units. The
  result is written to `thresholds.json` before the next cell starts; cells already in the file are skipped. A result
  that is empty or has a threshold that is not finite is refused and nothing is saved for the cell.

Progress is logged to the tracker after every batch (`runner/units_done`, `runner/units_skipped`, `runner/batches`,
`runner/batch_units`; step = units done plus skipped). The block directory ends `units/ state/` (the store),
`thresholds.json` (calibration blocks), `manifest-*.json`, `tracker_run_id.txt` (tracking enabled), `.launch.lock`.

`thresholds.json`: `{"alpha": ..., "cells": [{"arm", "topology", "agents", "backbone", "task", "thresholds": {...}}]}`.
Each manifest carries `block, profile, git_commit, started_at, versions {python, torch, numpy}, threads,
pre_registered {ni_margin_accuracy, rank_agreement_threshold}, units, thresholds_sha256, settings` (the resolved
settings). The tracker receives the settings plus `manifest` (everything but `settings`) as the run config and the
manifest file as an artifact.

## Invariants

- No experiment value has a default here. Every value is `core.get(settings, key)`; a missing key raises `core.Unset`,
  which the CLI turns into a refusal naming the key. Nothing in this feature draws a random number.
- `--list` and `--dry-run` never expand a mock grid: counts are products of the axis lengths (tested by forbidding
  `MockGrid.__iter__`). The mock frontier is 3 693 600 units at the shipped settings.
- A mock grid yields the units of one `group_key` consecutively, so batches are full (tested).
- Unit ids are a function of the unit alone: same unit, same id, independent of `mock.batch_size` and of what was
  already done.
- `Platform.run_batch` is called only with units not yet done, all of one `group_key`, never more than the batch size.
- Resume: a done unit is never handed to a platform again; a unit with saved rows but no done marker is handed over
  again and the platform continues after its last saved row (`CheckpointStore.resume_rows`).
- Secrets are never read here, never written to a manifest or sent to the tracker (the settings hold none).
- Tracking is off (`core.NullTracker`) unless `tracking.enabled` is `true`; the tracker is always finished, also on a
  crash or a refusal during the run.

## Settings keys read

`active_profile`, `profiles.<name>.*` (through `core.load_settings`).
`runtime.*`: the two pre-registered keys and, per platform, those of `RUNTIME_BY_PLATFORM`; `runtime.llm_seeds`,
`runtime.llm_calibration_seeds`; `runtime.llm_sink_positions.<topology>` for each `llm.topologies`.
`budgets.levels_percent`, `budgets.alpha`.
`mock.batch_size`, `mock.topologies`, `mock.families`, `mock.frontier_agents`, `mock.epidemiology_agents`,
`mock.every_sink`, `mock.main_seeds`, `mock.heldout_seeds`, `mock.tune_seeds`, `mock.uncapped_arms`, `mock.capped_arms`,
`mock.allocation_arms`, `mock.calibrated_arms`; per scoped block `mock.<epidemiology|ablations|adaptive|
presets_robustness|costs_robustness|legacy>.arms` and `.cells`, plus `mock.epidemiology.topologies`,
`mock.adaptive.conditions`, `mock.presets_robustness.presets`.
`llm.agents`, `llm.rounds`, `llm.seeds_per_cell`, `llm.conversations_per_cell`,
`llm.calibration_conversations_per_cell`, `llm.n_backbones`, `llm.backbones`, `llm.topologies`, `llm.attack_cells`,
`llm.attack_free_tasks`, `llm.headline_arms`, `llm.capped_arms`, `llm.capped_cells`, `llm.calibrated_arms`,
`llm.judge_arms`, `llm.judge.model`, `llm.adaptive_check.arms`, `.families`, `.condition`.
`gnn_train.families`, `.backbone`, `.conversations_per_setting`, `.agents`, `.rounds`, `.edge_probabilities`,
`.attackers`.
Dry-run only: `estimates.*`.
`tracking.enabled` (optional; unset means off). `tracking.project`, `tracking.entity`, `tracking.mode` are read by the
tracking feature from the run config.

## Open questions

1. Calibration of arms outside `B`, `C`, `BAZT` (spec §11): `BAZT_upstream`, the ablation arms, the matched-timing
   legacy arms, `GNN_standin`, the GNN's published 0.5 cut. `mock.calibrated_arms` and `llm.calibrated_arms` list the
   arms that are calibrated; an arm whose `ArmSpec.calibrated` is true but that is not listed makes its evaluation
   block refuse, naming the cell.
2. The calibration cell has no auditor preset, cost condition or adaptive condition. `mock_presets_robustness`,
   `mock_costs_robustness` and `mock_adaptive` therefore use the thresholds calibrated under the main condition. If a
   preset moves the score distribution (spec §5: thresholds are read off each preset's ROC model at alpha), the
   calibration block needs a preset axis.
3. `mock_calibration` covers `mock.epidemiology_agents` x `mock.topologies`; a frontier agent count outside that list
   is refused as uncalibrated.
4. Unit ids use 16 hex digits (64 bits): the collision bound over the 3.7 million frontier units is about 4e-7 (tested).
   `unit_id`.

## Requests

- tracking: `WandbTracker()` constructible without arguments; `start(block, config, resume_id)` reads `tracking.project`,
  `tracking.entity`, `tracking.mode` from `config` and returns the run id (stored in `tracker_run_id.txt`).
- mock_engine, llm_harness: `MockPlatform()` and `LLMPlatform()` constructible without arguments, cheap and free of side
  effects (no network, no secrets, no file writes), because `--list` builds them to tell whether they exist.
  `Platform.calibrate` must be callable again on units already marked done (a crash between the calibration and the
  write of its thresholds repeats the call).
- conventional, references, bazt_slot, gnn: each `public.py` exports `ARMS`, a tuple of `ArmSpec`; arm names are unique
  across features; `ArmSpec.calibrated` is true exactly for arms that report `Actions.statistics`.
- quality: declare `filelock` in `pyproject.toml` (the launch lock imports it; it is installed today only as a
  dependency of torch).
- core: `CheckpointStore` writes two files per unit into one directory, which is about 7.4 million files for the mock
  frontier. Sharding `units/` by id prefix would keep directory listings and resume scans fast on Windows.

## Added checks

- Evaluation block: an arm with `ArmSpec.calibrated` that is not in `<platform>.calibrated_arms` is refused, naming that key.
- After each `run_batch` every unit of the batch must be marked done, else the launch stops with a RuntimeError naming the
  platform and the first unit ids (unmarked units would be re-run and their rows written twice).
- A tracker that cannot be built or started (`RuntimeError`, `core.Unset`) refuses the launch with its message; the tracker
  starts before the manifest is written, so a failed start leaves no manifest and no `tracker_run_id.txt`.

## Fail-fast preflight, adaptive search, matched-timing refusal

- Before any write, `run_block` resolves the keys the block's arms need (`private/needs.py`): for arms C and BAZT (also
  `BAZT_<ablation>`; not `BAZT_upstream`) `conventional.c.judge_score_bins.<platform>` and
  `conventional.c.presence_window_rounds`; for arms in `llm.judge_arms` (B, C, BAZT and their capped variants) on LLM
  blocks `llm.judge.provider`, `llm.judge.top_logprobs`, `llm.judge.floor_probability`, on mock blocks
  `mock.auditor.binormal_method`. `runtime.rate_limit_per_identity` and `runtime.recovery_margin` were already required
  for every run. A missing key is a refusal naming it.
- `mock_adaptive` (`Block.adaptive`): preflight calls `adaptive.search_episodes(units, settings)` (refuses what the search
  refuses), the launch calls `adaptive.run_search(units, arm_of, platform, settings, <block dir>, thresholds_of, tracker)`
  with the calibrated thresholds of the cell `(arm, topology, agents)`, then hands the platform `table.annotate(unit)`
  (adds `attack_params`; the unit id is computed before and ignores it). `--dry-run` prints `adaptive search: N episodes`
  under the block. The adaptive feature arrives through `Wiring.adaptive()` (optional, raises `FeatureMissing`).
- Any arm named `*_matched` is refused (`EpisodeContext.timing_schedule` is recorded by nothing yet). Lift this in
  `needs.MATCHED_SUFFIX` / `launch._arm_specs` once BAZT schedules are recorded.
- Block names `mock_frontier, mock_epidemiology, mock_adaptive, mock_ablations, llm_headline, llm_capped,
  llm_adaptive_check` match `analysis.blocks` (tested).

## Integrity of a block (settings identity, seeds, probes)

- `settings.json` in each block directory holds the settings view of the first launch (`private/identity.py`: everything
  but the profile, tracking, estimates, analysis, `mock.batch_size`, the two pre-registered values, the other platform's
  keys and the scoped sections of other blocks). A launch whose view differs is refused, naming the keys
  (`key (old -> new)`); use another `--results-root` or `--profile`. `thresholds.json` keeps the view of the keys that
  decide an attack-free statistic (`runtime.*` except evaluation-only ones, `mock.model`, `mock.auditor`, `costs`,
  `auditor_presets`, `conventional`, `bazt`, `gnn`, `llm.judge`, `calibration`, `budgets.alpha`, the tune seeds, ...); an
  evaluation or a calibration under another view is refused, naming the keys.
- Manifest names carry microseconds (`manifest-<utc, no separators>.json`).
- Seeds: `mock.main_seeds`, `heldout_seeds`, `tune_seeds` (and `mock.adaptive.search_seeds` when set) are inclusive ranges
  and pairwise disjoint; `runtime.llm_seeds` and `llm_calibration_seeds` are lists of `llm.seeds_per_cell` distinct
  seeds, disjoint. A mock block whose seeds touch the held-out range needs `ANALYSIS_FROZEN`, whatever its name. A
  `--set` list without brackets is refused.
- Probes: before a mock block writes anything, `Platform.run_batch` is called on a one-episode batch (smallest N, first
  seed, first sink) in a temporary store named `preflight-probe`, once per (arm, audit capped or not, intervention capped
  or not, topology, attack, preset, condition, cost condition). `core.Unset`, `ValueError` or any exception becomes a
  refusal naming the arm and the cell. LLM blocks have no probe: they resolve the explicit key list of `private/needs.py`.
- `mock_presets_robustness`: the `adaptive` preset is one variant per end point of `auditor_presets.adaptive_tpr`, each unit
  carrying `attack_params = {"auditor_tpr": x}`; an unknown preset or a bad range is a refusal. `group_key` includes
  `attack_params`.
- `mock_adaptive`: the search takes the thresholds of the evaluation rule (oracle_alloc: random_alloc's `audit_score`).
- Dry-run mock hours add `estimates.mock_checkpoint_seconds_per_unit` per unit.

## Commands

- `python run_experiments.py run-all [--dry-run] [--allow-live-llm]` (also `--profile`, `--set`, `--results-root`): calibration
  blocks, then `gnn_train`, then the evaluation blocks (registry order), then the analysis. Skipped with a reason line
  (`<block>: skipped: ...`): scope not set (the key), `mock_heldout` before `ANALYSIS_FROZEN`, LLM blocks without
  `--allow-live-llm`, the analysis when `analysis.missing_keys` is not empty or the feature is missing. The first refusal
  prints one `Refusing <block>: ...` line on stderr and returns 1; a crash propagates; a rerun skips done units and
  resumes.
- `python run_experiments.py status`: per block done/total (counted from the `.done` files, the mock grid is never
  expanded) and state, the last error of each block (`<results>/<profile>/last_errors.json`, written by `--run` and
  `run-all`), every owner decision still unset grouped by block with its options (the comments of profiles.toml),
  and whether HF_TOKEN, WANDB_API_KEY, OPENROUTER_API_KEY are present (`adapters/env.py`; never a value).
- Sync: when `sync.enabled` is true, `--run` and `run-all` call `sync.public.start(settings, [<results>/<profile>,
  gnn.artifacts_dir])` once and `handle.stop()` at the end, also on error or Ctrl-C. `Wiring.sync()` and
  `Wiring.analysis()` are optional lazy getters (`FeatureMissing` when not built).

## Requests (added)

- sync (tracking lane): `[sync]` keys other than `enabled` and `repo_id` once delivered; `public.start(settings, paths) -> handle`.
- quality: the secrets file test values / `scripts` need nothing new; `tracking.flush_rows` is set in profiles.toml.
- analysis: `analysis.public.missing_keys(settings)` and `analyze(results_dir, settings)` are what `run-all` calls.

## Calibration diagnostics

After each `Platform.calibrate` the runner reads the platform's `last_calibration` (a mapping: mode, passes, rate, alpha,
converged or within_tolerance, worst_episode_rate when present; absent means nothing is stored) and writes it as
`diagnostics` in the cell record of `thresholds.json` and under `calibration` (keyed by the cell) in the block manifest.
`status` lists, under `calibration warnings`, every cell with `converged` or `within_tolerance` false, or a
`worst_episode_rate` above 5 x alpha.
