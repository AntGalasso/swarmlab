# runner: contract

Blocks, unit expansion, dry-run sizing, launch (batching, resume, calibration orchestration, threshold propagation),
manifests, tracking and the CLI of `EXPERIMENT_PLAN.md` (§1.1, §2.8, §5, §8, §9.5). Entry point:
`python run_experiments.py` calls `bazt_lab.app.main`; `bazt_lab/app.py` is the composition root and builds the
`Wiring` this feature runs over.

## public.py

| name | what |
|---|---|
| `main(argv=None, *, wiring) -> int` | the CLI. Exactly one of `--list`, `--dry-run`, `--run`; `--block NAME` (repeatable, a name of `BLOCKS`; `--run` needs at least one, `--dry-run` without one sizes every block), `--profile`, `--set KEY=VALUE` (repeatable), `--settings` (default `experiments/profiles.toml`), `--results-root` (default `RESULTS_ROOT`), `--allow-live-llm`, `--device {cpu,cuda,cuda:N}`, `--workers N`, `--compile`. The subcommands `run-all` and `status` take `--settings`, `--profile`, `--set`, `--results-root`, `--device`, `--workers`, `--compile`; `run-all` also takes `--dry-run` and `--allow-live-llm` (Commands) |
| `Wiring(arm, platform, tracker, adaptive, analysis, sync, rebuild, gnn)` | what the composition root hands over: `arm(name) -> ArmSpec` (raises `FeatureMissing`, or `LookupError` for a name no built feature exports), `platform("mock" or "llm") -> Platform` (raises `FeatureMissing`), `tracker(settings) -> Tracker` (called only when tracking is enabled; raises `FeatureMissing` or the tracking feature's `RuntimeError`). Optional, default `None` (counts as not built): `adaptive()`, `analysis()`, `sync()`, `gnn()` return that feature's public module and raise `FeatureMissing`; `rebuild` is the `"module:function"` a worker process calls for a fresh `Wiring` |
| `FeatureMissing(feature, detail)` | a `RuntimeError`, `feature <feature> is not built (not available: <detail>)`: a feature that is not built or cannot be imported; `feature` may name several, comma separated |
| `BLOCKS`, `Block`, `CALIBRATION_BLOCK` | the 14 blocks (`name`, `platform`, `kind` = `calibrate` / `train` / `evaluate`, `description`, `units(settings, strict)` where `strict` forbids placeholders, `requires_freeze`, `adaptive`, `open_decision`, `replays`, `timing_source`); `{"mock": "mock_calibration", "llm": "llm_calibration"}`, the calibration block whose `thresholds.json` each platform's evaluation blocks read |
| `units_*`, `capped_cells`, `UnitList`, `MockGrid` | unit expansion: `units_llm_headline`, `units_llm_capped`, `units_llm_calibration`, `units_llm_adaptive_check`, `units_gnn_train`, `units_mock_calibration`, `units_mock_frontier`, `units_mock_heldout` (the scoped mock blocks are built by `private/units.py`, `scoped`). Every LLM block and `mock_calibration` is a `UnitList` (one LLM unit = one seed run). Every other mock block is a `MockGrid`: `len()`, `by_agents()`, `arms()`, `cells()`, `seed_set()`, `conditions()`, `cost_groups()` and `probes()` work from the axes, iteration is lazy. `capped_cells(levels)`: every (audit, intervention) pair of the levels (percent, turned into fractions) and `"uncapped"`, except the fully uncapped one |
| `unit_id`, `group_key`, `cell_key`, `GROUP_FIELDS` | identity and grouping of units (below) |
| `run_block(settings, name, allow_live_llm, *, wiring, results_root=RESULTS_ROOT)` | launches one block (below); returns 0 |
| `dry_run`, `llm_calls`, `cache_savings`, `list_blocks` | sizing and listing; `list_blocks` builds each block's platform and arms through the wiring to say which features are not built |
| `load_settings`, `missing_runtime`, `PRE_REGISTERED`, `RUNTIME_BY_PLATFORM`, `FREEZE_MARKER`, `RESULTS_ROOT` | settings handling (`core.load_settings`; a bad profile or override ends the launch with its message) and the run-time variables each platform needs; `FREEZE_MARKER = "ANALYSIS_FROZEN"`, `RESULTS_ROOT` = `<repository>/results/experiments` |

## Units

An LLM unit carries `block, platform, arm, audit, intervention, backbone, topology, attack, task, seed, conversations,
agents, rounds, sanitize_mode` (`gnn_train`: `block, platform, arm="GNN", attack, task, edge_probability, attackers,
backbone, conversations, agents, rounds, sanitize_mode`; `llm_adaptive_check` adds `condition`; its `task` is one of `llm.adaptive_check.tasks`, one unit per task).
`conversations` is `llm.conversations_per_cell` (calibration: `llm.calibration_conversations_per_cell`) divided by
`llm.seeds_per_cell`; a total that does not divide ends the launch with its message. An arm that sanitizes (BAZT, its
ablations, BAZT_upstream) has one unit per mode of `llm.sanitize_modes` (D101); every other arm, and every calibration
and `gnn_train` unit, carries the first mode. A mock unit is one episode: `block, platform, arm, audit, intervention,
topology, attack, agents, seed, sink` (plus `condition`, `preset` (with `attack_params` for the adaptive preset) or
`cost_condition` for the blocks that vary them). `audit` and `intervention` are a fraction of all edge deliveries
(`0.1` for 10 %) or `"uncapped"`; `sink` is an agent index or `None`; `attack` is `"none"` for attack-free runs.
Without `strict`, an unset `llm.backbones`, `runtime.llm_seeds`, `runtime.llm_calibration_seeds` or `gnn_train.backbone`
becomes a placeholder (`--list`, `--dry-run`, `status`); a launch and `run-all` expand with `strict`.

Before a unit reaches a platform the runner adds fields that take no part in its identity (`ANNOTATIONS` in
`private/units.py`; `unit_id` drops them):

- `id`: `unit_id(unit)`, `<arm>-<16 hex digits of sha256 of the unit's JSON, keys sorted>`; it names the unit in the
  store.
- `thresholds`: `{statistic: threshold}` of the unit's calibration cell (`cell_key`), `{}` for an arm that is not
  `ArmSpec.calibrated` and for `gnn_train`. `oracle_alloc` gets `{"audit_score": ...}` of `random_alloc` in the same cell
  (D96); a calibrated `*_matched` arm under `mock.legacy.matched_thresholds = "base"` gets the cell of the rule it wraps
  (D120). A platform copies it into `EpisodeContext.thresholds`.
- `twin_id`, attacked mock units: the `unit_id` of the same unit with `attack = "none"` (D109, Order, workers and cost
  estimates).
- `llm_adaptive_check` (attacked units): `attack_schedule = {clock, start_frac, start_round, rate}`, the replayed best
  response (D110, below). The block pins it instead.
- `mock_legacy`, units of a `*_matched` arm (D116): `timing_schedule` (BAZT's `intervened_by_round` of the same unit
  scenario) and, for an attacked unit, `twin_timing_schedule` (that of the attack-free scenario). The block pins its
  source instead (below).
- `mock_adaptive`: `attack_params` of the cell's best response (`table.annotate`), added after the id is computed.

`group_key(unit)` = the values of `GROUP_FIELDS` = `(arm, audit, intervention, topology, attack, agents,
cost_condition, preset, condition, attack_params)`, absent fields as `None`, a dict value as its sorted JSON.
`cell_key(unit)` = `(arm, topology, agents, backbone, task)`; the mock has no backbone and no task (`None`).

## Launch

`run_block` works on a copy of the settings in which `llm.allow_live` is `allow_live_llm` for an LLM block and `false`
for a mock block. It refuses first and writes nothing until every refusal has passed. Each refusal is a `SystemExit`
message starting `Refusing <block>:` (the kickoff guard prints its own `Refusing: ...`), in this order:

1. `requires_freeze` block (`mock_heldout`) and no `ANALYSIS_FROZEN` file in the results root.
2. A run-time variable of the platform (spec §10, `RUNTIME_BY_PLATFORM`), one of the two pre-registered values, or a
   `llm_sink_positions.<topology>` entry of the LLM grid is not set (`run-time variables not set in profiles.toml
   [runtime]: ...`).
3. The block's units carry an adaptive condition its platform does not run (Round-3 rules).
4. LLM block: `swarmlab._kickoff_guard` (`--allow-live-llm`, after the kickoff time), imported in `adapters/kickoff.py`.
5. The grid cannot be built (a key, an unknown preset, a bad range), or `mock.batch_size` (mock; a positive integer),
   `compute.workers` (an integer >= 1; more than 1 on a mock evaluation block needs `Wiring.rebuild`), `budgets.alpha`
   (calibration blocks) or the tracker (tracking enabled; `tracking.enabled` must be true or false) is unset or invalid.
   An unset key reads `setting not set in profiles.toml: <key>`, here and below, also when a platform raises
   `core.Unset` during the run.
6. An arm: a `*_matched` arm in an evaluation block without `Block.timing_source` (Matched-timing controls), an arm whose
   feature is not built ("feature conventional is not built") or that no feature exports, an arm that does not run on
   the block's platform (`ArmSpec.platforms`).
7. The keys and inputs of the block's arms, before the platform is built: `private/needs.py` (Fail-fast preflight), the
   held-out seeds and the seed ranges (Integrity), `calibration.mode` (calibration blocks: `closed_loop` or
   `open_loop`), the gnn feature and the GNN checkpoints (GNN section, Round-2 rules), the adaptive search, the replayed
   best responses, the timing source and `mock.legacy.matched_thresholds` (sections below).
8. The platform's feature is not built.
9. Calibration block: an arm that is not `ArmSpec.calibrated`; a `*_matched` arm under `calibration.mode = "closed_loop"`;
   `thresholds.json` saved at another alpha than the current `budgets.alpha`, or under another threshold view (Integrity).
10. Evaluation block: a calibrated arm whose threshold arm is not in `<platform>.calibrated_arms` (Added checks);
    `thresholds.json` saved under another threshold view; `oracle_alloc` without `random_alloc`'s calibrated
    `audit_score` in one of its cells (D96); a calibrated arm with no thresholds for a cell of the grid (the message
    names up to three cells and the calibration block to run first).
11. The block directory's `settings.json` holds another settings view (Integrity).
12. A probe episode is refused or fails (Integrity, Round-2 rules).
13. `sync.enabled` is not true or false, or it is true and the sync feature is not built.

Then the launch opens `core.CheckpointStore(<results root>/<profile>/<block>/)`, takes the lock
`<block dir>/.launch.lock` (an OS file lock through `filelock`, released by the OS when the process dies, so resuming a
crashed launch is never blocked; a second launch of the same block is refused, `files.BusyLaunch`), starts the uploader
when `sync.enabled` is true (Sync paths), starts the tracking run (resuming the id in `tracker_run_id.txt`), pins
`settings.json` and, for the blocks that have them, `best_responses.json` and `timing_schedules.json` when the block
directory has none yet, writes `manifest-<started_at>.json`, writes `tracker_run_id.txt` (tracking enabled), hands the
manifest to the tracker as an artifact and runs:

- `evaluate` and `train` blocks: GNN cells without a GNN are reported and left out (GNN section), and `mock_adaptive`
  runs its search first (Fail-fast preflight). Units are walked in grid order, grouped by `group_key`, the units the
  store marks done are skipped, the rest are annotated (Units) and cut into batches of `mock.batch_size` units (mock) or
  one unit (LLM), and each batch is one `Platform.run_batch(units, arm_spec, settings, store, tracker)` with
  `arm_spec = wiring.arm(unit["arm"])`. With `compute.workers > 1` a mock evaluation block hands its batches to worker
  processes (Order, workers and cost estimates). `gnn_train` then finishes the GNN training (GNN section).
- `calibrate` blocks: one `Platform.calibrate(units, arm_spec, settings, store)` per cell, in grid order, on every unit
  of that cell (also those the store marks done; `thresholds` is `{}`). The result is written to `thresholds.json`
  before the next cell starts; cells already in the file are skipped. A result that is empty or has a threshold that is
  not finite stops the launch (`<block>: calibrate returned ...`) and nothing is saved for the cell.

Progress is logged to the tracker after every batch and after every calibrated cell (`runner/units_done`,
`runner/units_skipped`, `runner/batches`, `runner/batch_units`; step = units done plus skipped); with worker processes
about every 2 s from `done_count()`, with `runner/batch_units` 0. The launch ends with the line
`<block>: N units run, M already done, results in <block dir>`.

The block directory ends `units/ state/` (the store; `units/w<k>/ state/w<k>/` for worker `k`), `settings.json`,
`thresholds.json` (calibration blocks), `best_responses.json` (`llm_adaptive_check`), `timing_schedules.json`
(`mock_legacy` with `*_matched` arms), the adaptive feature's `best_responses/` (`mock_adaptive`), `manifest-*.json`,
`tracker_run_id.txt` (tracking enabled), `.launch.lock`. The profile directory `<results root>/<profile>/` also holds
`last_errors.json` and `skipped_cells.json` (Commands, GNN section). The store (`core.CheckpointStore`):
`units/<unit id>.jsonl` with its marker `units/<unit id>.done` (per unit, LLM side), `units/batch-<hash>.jsonl` with
its done list `units/batch-<hash>.done` (one batch, mock side, `append_batch`), `state/<unit id>.pt`.

`thresholds.json`: `{"alpha": ..., "settings": {threshold view}, "cells": [{"arm", "topology", "agents", "backbone",
"task", "thresholds": {...}, "diagnostics": {...}}]}` (`diagnostics` when the platform reports them).
`settings.json`: `{"settings": {block view}}`. Each manifest carries `block, profile, git_commit, started_at, versions
{python, torch, numpy}, threads, pre_registered {ni_margin_accuracy, rank_agreement_threshold}, units,
thresholds_sha256, settings` (the resolved settings) and, in a calibration block, `calibration` (Calibration
diagnostics). The tracker receives the settings plus `manifest` (everything but `settings`) as the run config and the
manifest file as an artifact.

## Invariants

- No experiment setting has a default here. Every value is `core.get(settings, key)`; a missing key raises `core.Unset`,
  which the launch turns into a refusal naming the key. Optional (unset means off or absent): `tracking.enabled`,
  `sync.enabled`, `mock.adaptive.search_seeds`. Nothing in this feature draws a random number.
- `--list`, `--dry-run` and `status` count a mock grid from its axes (`len`, `by_agents`, `cost_groups`, `cells`); the
  `adaptive search: N episodes` line of `--dry-run` is the adaptive feature's `search_episodes`, which walks the
  `mock_adaptive` units. `status` is tested by forbidding `MockGrid.__iter__`. The mock frontier is 3 693 600 units at
  the shipped settings.
- A mock grid yields the units of one `group_key` consecutively, so batches are full (tested).
- Unit ids are a function of the unit alone: same unit, same id, independent of `mock.batch_size`, of the worker count
  and of what was already done. They use 16 hex digits (64 bits): the collision bound over the 3 693 600 frontier units
  is about 4e-7 (tested).
- `Platform.run_batch` is called only with units not yet done, all of one `group_key`, never more than the batch size.
- Resume: a done unit is never handed to `Platform.run_batch` again; a unit with saved rows but no done marker is handed
  over again and the platform continues after its last saved row (`CheckpointStore.resume_rows`). `Platform.calibrate`
  gets every unit of a cell that is not in `thresholds.json`, done or not.
- Secrets are never written to a manifest or sent to the tracker (the settings hold none); `status` reads only whether
  they are present (`adapters/env.py`).
- Tracking is off (`core.NullTracker`) unless `tracking.enabled` is `true`; once started, the tracker is always finished,
  also on a crash or a refusal during the run.

## Settings keys read

`active_profile`, `profiles.<name>.*` (through `core.load_settings`).
`runtime.*`: the two pre-registered keys and, per platform, those of `RUNTIME_BY_PLATFORM`; `runtime.llm_seeds`,
`runtime.llm_calibration_seeds`; `runtime.llm_sink_positions.<topology>` for each `llm.topologies`.
`budgets.levels_percent`, `budgets.alpha`, `calibration.mode` (calibration blocks).
`compute.workers`; `compute.device` and `compute.compile` are shown by `status` (`--device`, `--compile`, `--workers` set
`compute.*` for one launch).
`mock.batch_size`, `mock.topologies`, `mock.families`, `mock.frontier_agents`, `mock.epidemiology_agents`,
`mock.every_sink`, `mock.main_seeds`, `mock.heldout_seeds`, `mock.tune_seeds`, `mock.adaptive.search_seeds` (when set),
`mock.uncapped_arms`, `mock.capped_arms`, `mock.allocation_arms`, `mock.calibrated_arms`; per scoped block
`mock.<epidemiology|ablations|adaptive|presets_robustness|costs_robustness|legacy>.arms` and `.cells`, plus
`mock.epidemiology.topologies`, `mock.adaptive.conditions`, `mock.presets_robustness.presets`,
`auditor_presets.<preset>`, `auditor_presets.adaptive_tpr`, `analysis.ablation_reference` (`mock_ablations` always runs
it), `mock.legacy.matched_thresholds` (Matched-timing controls).
`llm.agents`, `llm.rounds`, `llm.seeds_per_cell`, `llm.conversations_per_cell`,
`llm.calibration_conversations_per_cell`, `llm.n_backbones`, `llm.backbones`, `llm.topologies`, `llm.attack_cells`,
`llm.attack_free_tasks`, `llm.sanitize_modes`, `llm.headline_arms`, `llm.capped_arms`, `llm.capped_cells`,
`llm.calibrated_arms`, `llm.judge_arms`, `llm.adaptive_check.arms`, `.families`, `.tasks`, `.condition`, `.mock_cell`,
`.mock_topology.<topology>`, `.start_clock`, `.mock_arm.<arm>` (only for an arm `mock.adaptive.arms` lacks).
`gnn_train.families`, `.backbone`, `.conversations_per_setting`, `.agents`, `.rounds`, `.edge_probabilities`,
`.attackers`, `.logs_dir` (sync); `gnn.artifacts_dir` (sync), `gnn.fallback_family` (the skip line).
The keys `private/needs.py` resolves for the block's arms (Fail-fast preflight).
`status` also lists, while unset, `mock.adaptive.<search_seeds|insider_sets|start_fractions|rates|detectabilities|
evasion_cost|max_candidates>` and `mock.adaptive.knowledge.<condition>` for `mock_adaptive` (the adaptive feature reads
them) and the commented keys of profiles.toml.
Dry-run only: `estimates.*`, `llm.judge.model`, `llm.ma_correctness`.
`tracking.enabled` (optional; unset means off). `tracking.project`, `tracking.entity`, `tracking.mode` are read by the
tracking feature (`WandbTracker(settings)`). `sync.enabled` (optional; unset means off); the other `sync.*` keys are read
by the sync feature.

## Open questions

1. Thresholded legacy rules: every legacy arm of references except `centrality_only` and `random_capped` (and their
   `_matched` controls) is `ArmSpec.calibrated`, but `mock.calibrated_arms` lists none of them, so a `mock_legacy`
   evaluation that runs one is refused naming `mock.calibrated_arms` until it is listed there (under
   `mock.legacy.matched_thresholds = "base"` a matched control needs only the rule it wraps, D120). The other arms
   outside B, C, BAZT follow D96, D106 and D107 (`mock.calibrated_arms`, `llm.calibrated_arms`).
2. The calibration cell has no auditor preset, cost condition or adaptive condition. `mock_presets_robustness`,
   `mock_costs_robustness` and `mock_adaptive` therefore use the thresholds calibrated under the main condition. If a
   preset moves the score distribution (spec §5: thresholds are read off each preset's ROC model at alpha), the
   calibration block needs a preset axis.
3. `mock_calibration` covers `mock.epidemiology_agents` x `mock.topologies`; a frontier agent count outside that list
   is refused as uncalibrated.

## Requests

None open. The runner relies on: `WandbTracker(settings)` (also constructible without arguments), whose
`start(block, config, resume_id)` returns the run id stored in `tracker_run_id.txt`; `MockPlatform()` and
`LLMPlatform()` constructible without arguments and free of side effects (`--list` builds them); `Platform.calibrate`
callable again on units already marked done (a crash between the calibration and the write of its thresholds repeats
the call); `ARMS`, a tuple of `ArmSpec`, in the `public.py` of conventional, references, bazt_slot and gnn, with names
unique across features (`bazt_lab/app.py` refuses a duplicate) and `ArmSpec.calibrated` true exactly for arms that
report `Actions.statistics`; `filelock` declared in `pyproject.toml`; the batch API of `core.CheckpointStore`
(`append_batch`: one rows file and one done list per batch) and its worker shards.

## Added checks

- Evaluation block: an arm with `ArmSpec.calibrated` whose threshold arm (the arm itself, or the rule a `*_matched` arm
  wraps under `mock.legacy.matched_thresholds = "base"`) is not in `<platform>.calibrated_arms` is refused, naming that
  key.
- After each `run_batch` every unit of the batch must be marked done, else the launch stops with a RuntimeError naming the
  platform and the first three unit ids (unmarked units would be re-run and their rows written twice); with worker
  processes the count of done units is checked at the end (`the workers stopped with k of n units not done`).
- A tracker that cannot be built or started (`RuntimeError`, `core.Unset`) refuses the launch with its message; the tracker
  starts before `settings.json` and the manifest are written, so a failed start leaves no manifest, no `settings.json` and
  no `tracker_run_id.txt`.

## Fail-fast preflight, adaptive search, matched-timing refusal

- Before any write, `run_block` resolves the keys the block's arms need (`private/needs.py`, `required_keys`): for arms C
  and BAZT (also `BAZT_<ablation>`; not `BAZT_upstream`) `conventional.c.judge_score_bins.<platform>` and
  `conventional.c.presence_window_rounds`; outside `gnn_train`, for arms in `llm.judge_arms` (B, C, BAZT, BAZT_upstream
  in profiles.toml; the BAZT ablations count as BAZT) on LLM blocks `llm.judge.provider`, `llm.judge.top_logprobs`,
  `llm.judge.floor_probability`, on mock blocks `mock.auditor.binormal_method`; on LLM blocks with BAZT (or an ablation)
  or BAZT_upstream `runtime.sanitize_success`; with BAZT_upstream `bazt_upstream.verdict_probability`; outside
  `gnn_train`, with GNN or GNN_recalibrated `gnn.cut`, `gnn.carry_flags`, `gnn.fallback_family`.
  `runtime.rate_limit_per_identity` and `runtime.recovery_margin` are required for every run (`RUNTIME_BY_PLATFORM`).
  A missing key is a refusal naming it.
- `mock_adaptive` (`Block.adaptive`): preflight calls `adaptive.search_episodes(units, settings)` (refuses what the search
  refuses), the launch calls `adaptive.run_search(units, wiring.arm, platform, settings, <block dir>, thresholds_of,
  tracker)`, where `thresholds_of` gives the calibrated thresholds of a unit's cell `(arm, topology, agents)`, then hands
  the platform `table.annotate(unit)` (adds `attack_params`; the unit id is computed before and ignores it). A
  `ValueError` of the search is a refusal. `--dry-run` prints `adaptive search: N episodes` under the block. The
  adaptive feature arrives through `Wiring.adaptive()` (optional, raises `FeatureMissing`).
- `llm_adaptive_check` (`Block.replays = "mock_adaptive"`, D110, `private/replay.py`): the LLM attacker replays the mock's
  gray-box best response; it needs the adaptive feature too (`Wiring.adaptive()`). Each attacked unit maps onto one cell
  of the mock_adaptive grid: the same arm, family (`attack`), N and condition; the budget cell
  `llm.adaptive_check.mock_cell` ([audit, intervention] of `mock.adaptive.cells`; the LLM check runs uncapped), the
  topology `llm.adaptive_check.mock_topology.<llm topology>` (one of `mock.topologies`), and for an LLM arm
  `mock.adaptive.arms` lacks, `llm.adaptive_check.mock_arm.<arm>`. All are owner keys, unset in profiles.toml and refused
  by name while unset; a value outside the mock grid, an N or family it does not search, is refused naming the grid key.
  Preflight, before any write: `adaptive.load_best_responses(<results>/<profile>/mock_adaptive, settings)` (missing:
  refused, "run --block mock_adaptive first"; other settings: the adaptive feature's refusal), then
  `table.choice(cell).attack_params` per mapped cell (cells without an entry: refused, naming up to three and
  "run --block mock_adaptive"). Only `start_frac` (default 0) and `rate` (default 1) transfer; detectability, potency
  and entry ranks do not. `llm.adaptive_check.start_clock` (`seed_run` | `conversation`, owner key, unset) gives the
  clock: T = rounds x conversations or rounds, and `start_round` = the first round of `adaptive.attack_rounds(T,
  start_frac)` (the mock's rule, 0-based trace rounds; T when it never starts). Probes are annotated too (their one
  conversation sets T). `best_responses.json` in the block directory pins `{source, clock, entries: {mock cell:
  {start_frac, rate}}}` on the first launch (with `settings.json`); a later launch whose entries or clock differ is
  refused before anything runs.
- Matched-timing controls (D116, `private/matched.py`): see the section below. A `*_matched` arm in an evaluation block
  without a `Block.timing_source` is refused, naming the block that has one.
- Block names `mock_frontier, mock_epidemiology, mock_adaptive, mock_ablations, llm_headline, llm_capped,
  llm_adaptive_check` are those the analysis reads (`analysis.blocks`) (tested).

## Integrity of a block (settings identity, seeds, probes)

- `settings.json` in each block directory holds the settings view of the first launch (`private/identity.py`,
  `block_view`: everything but `profile`, `tracking`, `sync`, `compute`, `estimates`, `analysis` (`mock_ablations` keeps
  `analysis.ablation_reference`), `mock.batch_size`, `gnn.embedding.batch_size`, the two pre-registered values,
  `llm.allow_live`, the other platform's keys and the scoped sections of other blocks: `mock.<scope>` outside
  `mock_<scope>`, `llm.adaptive_check` outside `llm_adaptive_check`). A launch whose view differs is refused, naming the
  keys (`key (old -> new)`); use another `--results-root` or `--profile`. `thresholds.json` keeps the view of the keys
  that decide an attack-free statistic (`threshold_view`: `runtime.*` except the pre-registered values,
  `recovery_margin`, `llm_seeds`, `llm_sink_positions`, `n_insiders`, `insider_bias`, `ta_detectability`,
  `ma_detectability`, `aitm_rewrite_strength`; `mock.model`, `mock.auditor`, `mock.tune_seeds`, `calibration`,
  `budgets.alpha`, `costs`, `auditor_presets`, `conventional`, `bazt`, `bazt_upstream`, `mock.legacy` (Round-3 rules),
  `gnn` (not `gnn.embedding.batch_size`), `gnn_train.families`, `llm.judge`, `llm.agents`, `llm.rounds`,
  `llm.temperature`, `llm.backbones`, `llm.seeds_per_cell`, `llm.calibration_conversations_per_cell`, `llm.cache`,
  `llm.attack_free_tasks`, `llm.api`, `llm.providers`, `llm_data`; of these, only the platform's own keys); an
  evaluation or a calibration under another view is refused, naming the keys.
- Manifest names carry microseconds: `manifest-<started_at without ':' and '.'>.json`, e.g.
  `manifest-2026-11-06T145201123456+0000.json` (UTC).
- Seeds, mock blocks: `mock.main_seeds`, `heldout_seeds`, `tune_seeds` (and `mock.adaptive.search_seeds` when set) are
  inclusive ranges of integers and pairwise disjoint. LLM blocks except `gnn_train`: `runtime.llm_seeds` and
  `llm_calibration_seeds` are lists of `llm.seeds_per_cell` distinct seeds, disjoint. A mock block whose seeds touch the
  held-out range needs `ANALYSIS_FROZEN`, whatever its name. A `--set` list without brackets is refused.
- Probes: before a mock block writes anything, `Platform.run_batch` is called on one-episode batches in a temporary store
  named `preflight-probe`, once per (arm, audit capped or not, intervention capped or not, topology, attack, variant:
  preset with its `attack_params`, condition, cost condition): from a `MockGrid` the smallest N, the first seed, the first
  sink; from `mock_calibration` the first unit of each. A probe carries `id`, `thresholds` and, in `mock_legacy` and
  `llm_adaptive_check`, the block's annotation. `core.Unset`, `ValueError` or any exception becomes a refusal naming the
  arm and the cell (or the key). LLM calibration and evaluation blocks are probed too (Round-2 rules); `gnn_train` is
  not. Every block also resolves the explicit key list of `private/needs.py`.
- `mock_presets_robustness`: the `adaptive` preset is one variant per end point of `auditor_presets.adaptive_tpr`, each unit
  carrying `attack_params = {"auditor_tpr": x}`; an unknown preset or a bad range is a refusal. `group_key` includes
  `attack_params`.
- `mock_adaptive`: the search takes the thresholds of the evaluation rule (oracle_alloc: random_alloc's `audit_score`).
- Dry-run mock hours add `estimates.mock_checkpoint_seconds_per_unit` per unit.

## Commands

- `python run_experiments.py run-all [--dry-run] [--allow-live-llm]` (also `--settings`, `--profile`, `--set`,
  `--results-root`, `--device`, `--workers`, `--compile`) runs every block of the registry by kind, `train`, then
  `calibrate`, then `evaluate`, each kind in registry order (`private/runall.py`, `KIND_ORDER`), then the analysis:
  `gnn_train`; `llm_calibration`, `mock_calibration`; `llm_headline`, `llm_capped`, `mock_frontier`,
  `mock_epidemiology`, `mock_ablations`, `mock_adaptive`, `llm_adaptive_check` (after `mock_adaptive`, whose best
  responses it replays), `mock_presets_robustness`, `mock_costs_robustness`, `mock_legacy` (after `mock_frontier`, its
  timing source), `mock_heldout`. A block is skipped with a reason line (`<block>: skipped: ...`), checked in this
  order: `mock_heldout` before `ANALYSIS_FROZEN`; LLM blocks without `--allow-live-llm` (except in `--dry-run`, which
  sizes them); a platform none of whose own run-time variables is set (`no <platform> run-time variable is set
  (runtime.<first>)`; the LLM's own are `random_edge_probability`, `llm_seeds`, `llm_calibration_seeds`; a half-set
  platform is not skipped, its launch refuses); scope not set (the key) or invalid (the expansion's message); a block
  whose units carry an adaptive condition its platform does not run; a block whose `Block.replays` or
  `Block.timing_source` block is skipped (`it replays the results of <block>, which is skipped (<reason>)`). The
  analysis: `analysis: skipped: ...` when the feature is missing or `analysis.public.missing_keys(settings)` is not
  empty; otherwise `analyze(<results>/<profile>, settings)`, and an `analyze` error (`core.Unset`, `ValueError`,
  `FileNotFoundError`) prints `Refusing analysis: ...` on stderr and returns 1. `--dry-run` prints the skip lines,
  `would run, in this order: ...`, the sizes of those blocks and `analysis: would read <results>/<profile>`. The first
  refusal of a block prints its message on stderr, is recorded in `last_errors.json` (a busy block is not) and returns
  1; a crash is recorded and propagates; a rerun skips done units and resumes.
- `python run_experiments.py status`: per block done/total (from `CheckpointStore.done_count()`, the store's done
  markers; the mock grid is never expanded) and state (`scope not set`, `done`, `running or stopped`, `not started`,
  `not runnable: <reason>`), the last error of each block (`<results>/<profile>/last_errors.json`, written by `--run`
  and `run-all`), every owner decision still unset grouped by block with its options (the comments of profiles.toml),
  `calibration warnings`, and whether HF_TOKEN, WANDB_API_KEY, OPENROUTER_API_KEY are present (`adapters/env.py`: the
  process environment, then the repository's `.env`; never a value).
- Sync: when `sync.enabled` is true, `--run` and `run-all` start the uploader beside each block and stop it with
  `handle.stop()` when the block ends, also on error or Ctrl-C (Sync paths). `Wiring.sync()` and `Wiring.analysis()` are
  optional lazy getters (`FeatureMissing`, or `None`, when not built).

## Requests (added)

None open: `sync.public.start(settings, paths) -> handle`, `analysis.public.missing_keys(settings)` and
`analysis.public.analyze(results_dir, settings)` are what `--run` and `run-all` call; the `[sync]` configuration keys and
`tracking.flush_rows` are set in profiles.toml (`sync.enabled`, `sync.repo_id`, `tracking.enabled`, `tracking.entity`
are owner keys, unset).

## Calibration diagnostics

After each `Platform.calibrate` the runner reads the platform's `last_calibration` (a mapping: mode, passes, alpha and,
when present, level, rate, simulated, converged, within_tolerance, worst_episode_rate; absent means nothing is stored)
and writes it as `diagnostics` in the cell record of `thresholds.json` and under `calibration` (keyed by the cell's
description) in the block manifest, which is written again. `status` lists, under `calibration warnings`, every cell
with `converged` or `within_tolerance` false, or a `worst_episode_rate` above 5 x alpha.

## Sync paths (supersedes the sync lines above)

The runner starts the uploader beside each block with `[<results>/<profile>]` (plus `Path(gnn.artifacts_dir)` and
`Path(gnn_train.logs_dir)` for `gnn_train`). The sync feature mirrors a path under its own folder name, so the profile
folder lands as `<profile>/<block>/...` and two profiles in one repository do not overwrite each other.
`gnn.artifacts_dir` is passed exactly as the gnn feature reads it (relative to the working directory). `sync.*` is not
part of a block's settings identity. A start that raises `core.Unset` or a `RuntimeError` (the sync feature's
`SyncError`) is a refusal. After `stop()` any `handle.summary["failures"]` is printed as
`<block>: sync reported failures ... (n commits)`.

## GNN: finishing the training, cells without a GNN

- Every `gnn_train` launch ends with `gnn.train_pending(settings)` (also when every unit is done, because a crash inside
  `train_unit` leaves the units done and the training unfinished), then `training_finished(settings, "<attack>__<task>")`
  for each `gnn_train.families` entry; an unfinished family stops the launch with a line naming it. `gnn_train` needs
  the gnn feature (refused in preflight when it is not built or `Wiring.gnn` is `None`).
- An evaluation cell for which `gnn.deploy_family(attack, task, settings)` is `None` (`gnn.fallback_family = none`:
  infection, AiTM) is skipped for the GNN arms, not refused:
  `<block>: skipped: no GNN for this cell (gnn.fallback_family = none): attack=.. task=..`, recorded in
  `<results>/<profile>/skipped_cells.json` (`{block: {units, cells}}`) and shown by `status` (the total excludes those
  units). A `ValueError` of `deploy_family` is a refusal.

## Round-2 rules

- Order of `run-all`: Commands. A calibration or evaluation block that runs a GNN arm checks the GNN checkpoints
  (`gnn.deploy_family`, `training_finished`) in preflight (`run --block gnn_train first`).
- Identity: a platform's block and threshold view hold only the keys that platform reads (the other platform's run-time
  variables, `conventional.c.judge_score_bins.<other>` and its sections are left out); `sync` and `compute` are left out of both.
- `status`: progress from `CheckpointStore.done_count()` (nothing is created), calibration blocks count calibrated cells
  against planned cells; unset decisions come from the commented keys of profiles.toml mapped to blocks by platform, scope
  and arms (`status.relevant`), plus the `analysis` group; the active device is printed.
- LLM probe: before an LLM calibration or evaluation block, one one-conversation unit per (arm, attack, sanitize mode,
  capped or not) runs through `Platform.run_batch` with `llm.api = "fake"` and `llm.allow_live = false` in a temporary store.
- Sync paths for `gnn_train`: `<results>/<profile>`, `gnn.artifacts_dir`, `gnn_train.logs_dir`.
- Small files are replaced with `core.atomic_replace`. `settings.json` is pinned after the tracker started. A busy block
  (`files.BusyLaunch`) is not written to `last_errors.json`. `llm.judge_arms` lists `BAZT_upstream`; a block that runs
  `BAZT_upstream` also resolves `bazt_upstream.verdict_probability`.
- `--device {cpu,cuda,cuda:N}`, `--workers N` and `--compile`, on every command, are shortcuts for `compute.device`,
  `compute.workers` and `compute.compile`.

## Order, workers and cost estimates

- Attack-free first (D109): within a mock block every `attack = "none"` unit comes before any attacked unit (per variant),
  so an attacked unit's clean reference, the attack-free twin's own run, exists when it starts. Every attacked mock unit
  carries `twin_id` = `unit_id` of the same grid unit with attack `none` (an annotation: it is not part of the unit id).
- `compute.workers` (int >= 1; `--workers N`) runs the pending batches of a mock evaluation block on N spawned worker
  processes (`adapters/workers.py`). The parent does the preflight, the probes, calibration, the adaptive search and the
  tracking, feeds `(arm, units)` batches through a bounded queue and polls `done_count()` for progress logging. Each worker
  rebuilds the wiring from `Wiring.rebuild` ("module:function"; `bazt_lab.app:build_worker_wiring` in production, which
  takes an arm from the first feature that exports it; without it `workers > 1` is refused), opens
  `CheckpointStore(block_dir, shard="w<k>")`, calls `store.refresh()` before each batch, uses a NullTracker and one torch
  thread. The first attacked batch after attack-free ones waits until every dispatched unit is done (one barrier per
  variant). A worker error or a crash stops all workers and raises in the parent; a rerun with any worker count continues.
  Rows are the same for any worker count (they depend on the unit alone); `compute.*` is not part of any identity. LLM
  blocks and calibrations always run in the parent.
- Calibration and `gnn_train` units carry `sanitize_mode` (the first mode), like every LLM unit.
- Sizing: `estimates.mock_seconds_per_episode_by_arm."<N>"."<arm>|capped|uncapped"` (CPU seconds of one process, from the
  profiler campaign); an arm without a value takes the slowest arm of the same kind (capped or uncapped) at that N, and
  `n/a` is printed when N has no table; the dry-run prints CPU hours (units x (cost +
  `estimates.mock_checkpoint_seconds_per_unit`)) and wall hours (divided by `compute.workers`).

## Round-3 rules

- Platform conditions (`blocks.unrunnable(block, settings, wiring)`): the adaptive conditions a platform runs are its
  `Platform.adaptive_conditions` (core contract), asked through `Wiring.platform` only for a block whose units carry a
  condition (`UnitList.conditions()`, `MockGrid.conditions()` from the variants), so building a platform to ask it
  imports its feature and nothing else; a platform whose feature is not built is left to the launch refusal. The mock
  runs the adaptive feature's conditions, the LLM platform `gray_box` only (D110: llm_harness refuses any other). A block
  whose units carry a condition their platform does not run (`llm_adaptive_check` with `llm.adaptive_check.condition`
  other than `gray_box`) is skipped by `run-all` before anything runs, shown as `not runnable: <reason>` by `status`, and
  refused by `--run` before the kickoff guard and before any write, all with the same reason and the block's
  `open_decision` (D110: the held-out LLM adaptive check replays the mock's gray-box best response only).
- `status` reads the table headers of profiles.toml with tomllib (quoted parts, spaces around dots, any trailing
  comment); a commented `# [table]` header counts, an active one must be a table of the parsed file.
- `thresholds.json` identity also holds `bazt_upstream`, the `mock.legacy` structure constants (not `.arms`, `.cells`,
  `.matched_thresholds`) and `gnn_train.families`: every section a calibrated arm reads while it decides (tested per arm).
  The LLM view also holds `llm.api`, `llm.providers` and `llm_data.*` (what produces the replies and the items):
  thresholds calibrated in a rehearsal (`llm.api = "fake"`) are refused by a live evaluation block in the same results
  root, and so is an LLM `thresholds.json` whose view lacks these keys (recalibrate in a fresh root).
- `run-all --dry-run` sizes the LLM blocks without `--allow-live-llm` and says that a real run needs the flag.

## Matched-timing controls (D116)

Matched-timing controls act only at the rounds where BAZT intervened in the same unit; BAZT runs first and its rounds are
recorded with its rows (`intervened_by_round`, T booleans, entry r = round r+1; what counts: mock_engine CONTRACT).

- Source: `Block.timing_source`. `mock_legacy` reads `mock_frontier` (the same axes: `mock.topologies`, `mock.families` and
  `none`, `mock.frontier_agents`, `mock.main_seeds`, `mock.every_sink`); no other block runs `*_matched` arms. The source
  unit of a matched unit is the same unit with `arm = "BAZT"` and `block = "mock_frontier"`, every other field equal
  (cell, topology, family, agents, seed, sink, variant fields); it is found by its `unit_id`.
- Preflight, before any write (`matched.load`): every source unit (and, for an attacked unit, its attack-free source
  unit) must be in the source grid (closed form), else `Refusing mock_legacy: ... mock_frontier never runs BAZT at k of
  the n unit(s) ..., e.g. ...`; the source block must hold `settings.json` and `units/` (else `... holds no BAZT rows:
  run --block mock_frontier first`), its `settings.json` equal to the source's view of this launch's settings (else
  refused, naming the keys), and committed BAZT rows of every source unit (`core.committed_row_files`; files whose first
  row is not BAZT or not of a needed group are skipped unread), else `BAZT has no committed row for k of the n unit(s)
  ...: run --block mock_frontier first`; a row without a valid `intervened_by_round` (produced before D116) or two
  committed rows of one unit with different rounds are refused.
- Handover: `timing_schedule` (and `twin_timing_schedule` for attacked units) is added to every matched unit, also to the
  preflight probes; other arms of the block get nothing. With `compute.workers > 1` the annotated units travel in the batch.
- Pin: `timing_schedules.json` in the block directory, written on the first launch after the tracker started:
  `{source, arm, units, settings (the source block's settings.json view), rounds_sha256 (of every source unit's rounds)}`.
  A later launch that reads other settings, another source or other rounds is refused ("rows of two schedules would mix").
- Order: `run-all` runs `mock_frontier` before `mock_legacy` (registry order) and skips `mock_legacy` when its source is
  skipped, as for `Block.replays`.
- Thresholds of a calibrated matched arm (`mock.legacy.matched_thresholds`, set to `"base"` in profiles.toml by D120; read
  while a calibrated `*_matched` arm is in an evaluation block, refused by name when unset and when it is neither value):
  `"base"` takes the thresholds calibrated for the legacy arm it wraps in the same cell (that arm must be in
  `mock.calibrated_arms`); `"own"` takes its own cell (it must be in `mock.calibrated_arms`). A matched arm in
  `mock_calibration` with `calibration.mode = "closed_loop"` is refused: the closed loop runs it acting, and the
  attack-free tune split has no BAZT rounds. The key is not part of the thresholds identity (it chooses a cell, it
  decides no statistic); it is part of `mock_legacy`'s `settings.json` view.
