# adaptive: contract

The mock adaptive attacker of `EXPERIMENT_PLAN.md` section 3. For every cell of the `mock_adaptive` block it finds, by
exact search over a finite strategy space, the `attack_params` that maximise ASR at the final round against the cell's
arm, and stores them so the block's units carry them. It reads no external service; `adapters/` is an empty package.

## public.py

| name | what |
|---|---|
| `run_search(units, arm_of, platform, settings, out_dir, thresholds_of=None, tracker=None) -> BestResponses` | searches each cell of the evaluation units (attack-free units skipped), saves the table under `<out_dir>/best_responses/`, resumes after a crash, returns the table. `arm_of(name) -> ArmSpec`, `thresholds_of(cell) -> {statistic: threshold}` (calibrated thresholds of the cell) |
| `load_best_responses(out_dir, settings) -> BestResponses` | the saved table; refused if `settings` differ from those it was searched with |
| `BestResponses` | `params_for(unit)`, `annotate(unit)` (unit plus `attack_params`), `choice(unit) -> Choice`, `missing(units)`, `len()` |
| `Choice` | `attack_params`, `asr` (of the chosen strategy on its search episodes), `baseline_asr` (default attack), `candidates`, `episodes` |
| `search_episodes(units, settings) -> int` | episodes the search simulates (dry-run sizing); refuses what `run_search` refuses |
| `load_space`, `Space`, `candidates`, `candidate_count`, `CONDITIONS` | the strategy space |
| `attack_rounds(T, start_frac, rate) -> bool[T]` | reference schedule, used by mock_engine for `start_frac` and `rate` |

## Strategy space (the best-response procedure)

Per (family, N, condition) a finite list, candidate 0 always `{}` (the family's default attack). Others:

- `entry_ranks`: degree ranks (0 = highest out-degree) taking the attacker role: the entry agent (infection), the victim
  (aitm), or `runtime.n_insiders` insiders (contiguous windows of ranks, or all subsets, `mock.adaptive.insider_sets`).
- `start_frac`: the attack begins at round `1 + floor(start_frac * T)`.
- `gray_box`, `white_box`: entry x start.
- `embedding_evasion`: adds `detectability` d (scales the monitor and auditor score of payload texts) with
  `potency = 1 - evasion_cost * (1 - d)` (scales the reported shift).
- `attack_rate`: adds `rate` (share of rounds with an attack, evenly spaced, `attack_rounds`).

Every candidate is simulated on identical episodes (common random numbers: episodes depend on topology, N, seed, sink
only). The candidate with the highest mean `asr_final` wins; ties go to the earlier candidate, so the default attack is
beaten only strictly. Knowledge level (`mock.adaptive.knowledge.<condition>`): `cell` = one strategy per cell chosen on
the search seeds (never evaluated; gray-box must be `cell`); `seed` or `episode` = chosen per evaluated seed or per
(seed, sink) episode (white-box must be one of these).

## Invariants

- No experiment value has a default: every grid and switch is `core.get(settings, key)`; a missing key raises `core.Unset`
  naming it, an invalid value `ValueError`. Refusals (too many candidates, unknown condition, overlapping seeds, bad
  batch size) happen before anything is written.
- Search seeds must not overlap `mock.main_seeds`, `mock.heldout_seeds` or `mock.tune_seeds` (`mock.tune_seeds` is read too); a cell-level search refuses evaluation seeds
  inside the search seeds.
- Results do not depend on `mock.batch_size` (tested). The search draws no random number itself.
- Progressive checkpoints: one progress line per chunk (fsync), one atomic file per finished cell; a rerun continues after
  the last saved chunk, drops a torn line, refuses progress that does not fit the plan.
- A table is refused when settings that change episodes differ (fingerprint) or when a finished cell was searched against
  other thresholds. Defenders see no ground truth; the search only reads `asr_final` rows from the platform.

- Common random numbers: every candidate of one search episode runs under one episode id, so arm-keyed streams
  (random allocation, tie-breaks) are identical across candidates. Searches on evaluation episodes (white-box `seed` or
  `episode`) use the evaluation unit's `id` (else `runner.unit_id`), so `Choice.asr` is exactly the evaluated ASR. A
  platform batch needs distinct ids, so a chunk holds at most one candidate per episode (chunk size <= min(batch size,
  episodes per candidate)).
- Ties are decided on exact integer totals of wrong agents (ASR x victims), so equal totals tie and the earlier
  candidate wins. A complete progress file with no table file finishes without simulating. A white-box cell finished for
  fewer evaluation episodes than now asked is searched again; its older entries are kept.
- Refusals (`space.Refusal`) are `core.Unset` and `ValueError` at once, so the runner preflight prints `Refusing`.
- The platform is given a `NullTracker`: search episodes never reach the `mock_units` table; progress goes to the real
  tracker as `adaptive/episodes_*`.

## Settings keys read

`runtime.n_insiders`, `mock.every_sink`, `mock.batch_size`, `mock.main_seeds`, `mock.heldout_seeds`, `mock.tune_seeds`,
`mock.adaptive.conditions`, `.search_seeds`, `.insider_sets` (`contiguous` | `all`), `.start_fractions` (in [0, 1)),
`.rates` (in (0, 1], contains 1.0), `.detectabilities` (in (0, 1], contains 1.0), `.evasion_cost` (in [0, 1]),
`.max_candidates`, `.knowledge.<condition>` (`cell` | `seed` | `episode`).

## Open questions

1. The spec lists the best-response procedure as not specified. Implemented: exact search over the grids above. The grid
   values, `evasion_cost`, `insider_sets`, the knowledge level of white-box (`seed` or `episode`) and the search seeds are
   the owner's decisions; none is set in profiles.toml.
2. How "embedding-space evasion" maps onto the mock: modelled as lower detectability with a potency cost; the cost
   function is an assumption.
3. Whether the platform's final-round ASR of the attack run should be the objective with a paired clean run (as now) or
   NRP.
4. White-box over `seed` averages over sinks of one seed; `episode` assumes the attacker also knows the sink.

## Requests

1. profiles.toml (runner): add the `[mock.adaptive]` keys above with owner-chosen values, left unset until decided. Today
   only `conditions` exists, so `run_search` refuses with `core.Unset` naming the first missing key.
2. runner: for the `mock_adaptive` block call `run_search` on the evaluation units first, then hand each batch
   `table.annotate(unit)`; use `search_episodes` for dry-run sizing. Group batches by `attack_params` is not needed
   (platform reads it per unit).
3. mock_engine: `check_homogeneous` must not require equal `attack_params` (it currently does not); keep `PARAM_KEYS`
   in sync with the keys above.
4. runner preflight: it prints `Refusing` only for `core.Unset` (the message is shown after `setting not set in
   profiles.toml:`); catching `ValueError` as well would give an exact wording.
5. mock_engine: `run_batch` keys rows and the store by unit id, so units sharing an id cannot share a batch; supporting
   it would let the search batch all candidates of an episode together (today a chunk is at most one candidate per
   episode).
