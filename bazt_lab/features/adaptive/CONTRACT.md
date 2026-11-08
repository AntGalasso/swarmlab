# adaptive: contract

The mock adaptive attacker of `docs/EXPERIMENT_PLAN.md` section 3. For every cell of the `mock_adaptive` block it finds, by
exact search over a finite strategy space, the `attack_params` that maximise ASR at the final round against the cell's
arm, and stores them so the block's units carry them. Its four conditions are D39; the one held-out LLM check (D38,
D110) replays the gray-box best response. It reads no external service; `adapters/` is an empty package.

## public.py

| name | what |
|---|---|
| `run_search(units, arm_of, platform, settings, out_dir, thresholds_of=None, tracker=None) -> BestResponses` | searches each cell of the evaluation units (attack-free units skipped), saves the table under `<out_dir>/best_responses/`, resumes after a crash, returns the table. It runs in the launching process, before the block's units. `arm_of(name) -> ArmSpec`, `thresholds_of(cell) -> {statistic: threshold}` (calibrated thresholds of the cell) |
| `load_best_responses(out_dir, settings) -> BestResponses` | the saved table; refused if `settings` differ from those it was searched with |
| `TABLE_DIR` | `"best_responses"`, the table's folder under the output directory |
| `BestResponses` | `params_for(unit)`, `annotate(unit)` (unit plus `attack_params`), `choice(unit) -> Choice` (the entry at the finest level stored: (seed, sink), then seed, then the cell), `missing(units)`, `len()` |
| `Choice` | `attack_params`, `asr` (of the chosen strategy on its search episodes), `baseline_asr` (default attack), `candidates`, `episodes` |
| `search_episodes(units, settings) -> int` | episodes the search simulates (dry-run sizing); refuses what `run_search` refuses |
| `load_space`, `Space`, `candidates`, `candidate_count`, `CONDITIONS` | the strategy space |
| `attack_rounds(T, start_frac, rate) -> bool[T]` | reference schedule, used by mock_engine for `start_frac` and `rate` |

## Strategy space (the best-response procedure)

Per (family, N, condition) a finite list, candidate 0 always `{}` (the family's default attack). Each cell (arm, budget
cell, topology, family, N, condition) is searched on its own over this list. N is the unit's `agents` (D14, D15). Others:

- `entry_ranks`: degree ranks (0 = highest out-degree) taking the attacker role: the entry agent (infection), the victim
  (aitm), or `runtime.n_insiders` insiders (contiguous windows of ranks, or all subsets, `mock.adaptive.insider_sets`).
- `start_frac`: the attack begins at round `1 + floor(start_frac * T)`.
- `gray_box`, `white_box`: entry x start.
- `embedding_evasion`: entry x start x `detectability` d (scales the monitor and auditor score of payload texts), with
  `potency = 1 - evasion_cost * (1 - d)` (scales the reported shift).
- `attack_rate`: entry x start x `rate` (share of rounds with an attack, evenly spaced, `attack_rounds`).

Every candidate is simulated on identical episodes (common random numbers: an episode's draws depend on topology, N and
seed only, `mock_engine/private/world.py`). The candidate with the highest mean `asr_final` wins; ties go to the earlier
candidate, so the default attack is beaten only strictly. The objective is the final-round ASR, the safety endpoint of
the confirmatory hypotheses (D63); NRP is a secondary summary (D6, D88) and is not the attacker's objective.
Knowledge level (`mock.adaptive.knowledge.<condition>`): `cell` = one strategy per cell chosen on the search seeds (never
evaluated; gray-box must be `cell`); `seed` or `episode` = chosen per evaluated seed or per (seed, sink) episode
(white-box must be one of these).

## Invariants

- No experiment value has a default: every grid and switch is `core.get(settings, key)`; a missing key raises `core.Unset`
  naming it, an invalid value `ValueError`. Refusals (too many candidates, unknown condition, overlapping seeds, bad
  batch size) happen before anything is written.
- Search seeds (`mock.adaptive.search_seeds`, read only when a condition has knowledge `cell`) must not overlap
  `mock.main_seeds`, `mock.heldout_seeds` or `mock.tune_seeds`; a cell-level search also refuses evaluation seeds inside
  the search seeds.
- Results do not depend on `mock.batch_size` (tested). The search draws no random number itself.
- Progressive checkpoints: one progress line per chunk (fsync) in `<out_dir>/progress/<key>.jsonl`, removed once the
  cell is written; one atomic file per finished cell in `best_responses/`. A rerun continues after the last saved chunk,
  drops a torn line, refuses progress that does not fit the plan. The search keeps its own files: it uses neither
  `CheckpointStore` nor worker shards (`w<k>`).
- A table is refused when settings that change episodes differ (fingerprint) or when a finished cell was searched against
  other thresholds. Defenders see no ground truth; the search only reads `asr_final` rows from the platform.

- Common random numbers: every candidate of one search episode runs under one episode id, so arm-keyed streams
  (random allocation, tie-breaks) are identical across candidates. Searches on evaluation episodes (white-box `seed` or
  `episode`) use the evaluation unit's `id` (else `runner.unit_id`), so `Choice.asr` is exactly the evaluated ASR. A
  platform batch needs distinct ids, so a chunk holds at most one candidate per episode (chunk size <= min(batch size,
  episodes per candidate)).
- Ties are decided on exact integer totals of wrong agents (ASR x victims) when every ASR is such a count, as the
  mock's is, so equal totals tie and the earlier candidate wins; otherwise on the exactly rounded sum. `asr_final` is a
  float64 ratio of bool counts (`metrics/private/kernel.py`), so this arithmetic does not depend on the compute dtype
  (D117). A complete progress file with no table file finishes without simulating. A white-box cell finished for fewer
  evaluation episodes than now asked is searched again; its older entries are kept.
- The thresholds are read as calibrated. The search does not read the calibration diagnostics, `within_tolerance`
  included (D112).
- `Platform.adaptive_conditions`: `MockPlatform` lists `CONDITIONS`; the LLM platform lists `("gray_box",)` (D110), so
  the runner refuses a block whose platform does not run its conditions before anything runs.
- Consumers: `mock_engine` applies `entry_ranks`, `start_frac`, `rate`, `detectability` and `potency` (its `PARAM_KEYS`,
  `world.py`), and `check_homogeneous` does not require equal `attack_params`. The runner's `llm_adaptive_check` replays
  `start_frac` and `rate` only, through `load_best_responses` and `attack_rounds` (`runner/private/replay.py`, D110).
- Runner: `launch.py` calls `search_episodes` in preflight, `run_search` before the block's units (`_search`), and
  hands each batch `table.annotate(unit)` (`_pending`).
- Refusals (`space.Refusal`) are `core.Unset` and `ValueError` at once; the runner prints their message after
  `Refusing <block>:` (`launch.py`).
- The platform is given a `NullTracker`: search episodes never reach the `mock_units` table; progress goes to the real
  tracker as `adaptive/episodes_*`.

- Progress files keep only complete lines; a torn tail (even valid JSON without its newline) is dropped by an atomic
  rewrite. `asr_final` of a platform row must be a finite number in [0, 1], not a bool. A stale cell (searched against
  other thresholds) refuses the whole run before any search. The fingerprint ignores everything the runner's
  mock_adaptive block identity ignores (`compute`, `sync`, `tracking`, `estimates`, batch sizes, the LLM side including
  `conventional.c.judge_score_bins.llm`, the other blocks' sections), so a relaunch the runner accepts is accepted
  here, plus what each cell is checked against on its own (seed splits, block coverage, calibration inputs). It keeps
  every other key the mock reads, arm parameters included (`bazt`, `bazt_upstream`, `conventional`). Like the runner's
  identity it hashes the set leaves (dotted keys), so an ignored key set inside a table the profile does not have yet
  (`conventional.c.judge_score_bins.llm` alone) leaves the fingerprint unchanged; tables written before this rule
  (fingerprint version 1) are refused once. `tests/test_adaptive_fingerprint_identity.py` (repo-root `tests/`) checks
  the hand-kept lists against the runner's on the real profiles.toml.
- Memory: planning keeps one id string per distinct (cell, seed, sink) evaluation episode, not the units.

## Settings keys read

`runtime.n_insiders`, `mock.every_sink`, `mock.batch_size`, `mock.main_seeds`, `mock.adaptive.conditions`,
`.insider_sets` (`contiguous` | `all`), `.start_fractions` (in [0, 1)), `.rates` (in (0, 1], contains 1.0),
`.detectabilities` (in (0, 1], contains 1.0), `.evasion_cost` (in [0, 1]), `.max_candidates`, `.knowledge.<condition>`
(`cell` | `seed` | `episode`, for each condition in `.conditions`). Read only when a condition has knowledge `cell`:
`mock.adaptive.search_seeds`, `mock.heldout_seeds`, `mock.tune_seeds`.

## Open questions

1. The spec lists the best-response procedure as not specified (docs/EXPERIMENT_PLAN.md, section 3). Implemented: exact
   search over the grids above. The grid values, `evasion_cost`, `insider_sets`, the knowledge level of white-box (`seed` or
   `episode`) and the search seeds are the owner's values. Each is a commented key in profiles.toml, so unset.
2. How "embedding-space evasion" maps onto the mock: modelled as lower detectability with a potency cost; the cost
   function is an assumption. `evasion_cost` is the owner's value (commented, unset).
3. White-box over `seed` averages over sinks of one seed; `episode` assumes the attacker also knows the sink. Which of
   the two is the owner's decision.

## Requests

1. profiles.toml (owner): set the `[mock.adaptive]` keys. Only `conditions` is set; `search_seeds`, `insider_sets`,
   `start_fractions`, `rates`, `detectabilities`, `evasion_cost`, `max_candidates` and `knowledge` are commented
   placeholders. Until they are set, `run_search` refuses with `core.Unset` naming the first unset key.
2. adaptive code: `private/table.py` (`_write_json`) and `private/search.py` (`_load_progress`) replace files with
   `os.replace`. Use `core.atomic_replace`, which waits out the Windows sharing violation that a sync copy can cause
   (sync CONTRACT.md, Behaviour).
3. mock_engine: `run_batch` keys rows and the store by unit id, so units sharing an id cannot share a batch; supporting
   it would let the search batch all candidates of an episode together (today a chunk is at most one candidate per
   episode).
4. mock_engine: a search batch carries no `twin_id`, so the reuse of an attack-free twin's run (D109) does not apply:
   `run_batch` simulates an attack-free reference for the batch (in `_simulate_chunk`) that the search never reads,
   since it uses only `asr_final`. Skipping that reference for units without a `twin_id` would roughly halve the search's
   simulation work (an estimate, not measured).
