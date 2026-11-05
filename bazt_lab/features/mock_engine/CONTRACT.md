# mock_engine

`MockPlatform` (core `Platform` contract): a torch engine implementing swarmlab's model plus the changes of
EXPERIMENT_PLAN.md sections 3, 4.2, 5 and 6. Batch dimension B = episodes; the only Python loop is over rounds.

## public.py

- `MockPlatform(rows_fn=None)`: `run_batch(units, arm, settings, store, tracker)` and
  `calibrate(units, arm, settings, store)`. `rows_fn` defaults to `metrics.public.rows_from_trace` (tests inject a fake).
- `apply_actions(actions, state, live, authentic, adj, remaining, t, T, costs, rate_limit, rate_u) -> Round`: the reference pricing
  and application of one round of Actions (see private/enforce.py); `Persistent.zeros(B, N)` (also exported) is the initial
  persistent state.
- `TOPOLOGIES` (`star_hub`, `star_leaf`, `chain`, `small_world` = Watts-Strogatz, `barabasi_albert`), `ATTACKS`
  (`none`, `insider_pi`, `insider_ta`, `insider_ma`, `infection`, `aitm`).

## Unit fields

Required: `id, arm, audit, intervention, topology, attack, agents, seed`. `audit` / `intervention` are a fraction in (0, 1]
of all edge deliveries of the episode (|E| x T, directed edges) or `"uncapped"`. Optional: `sink` (int, else a keyed draw),
`cost_condition` (`main` default, `robustness`), `preset` (auditor preset, default `auditor_presets.main`), `thresholds`,
`attack_params` (keys: n_insiders, insider_bias, mu_inf, auditor_tpr, entry_ranks, start_frac, rate, detectability, potency).
A batch shares arm, budget cell, topology, attack, N, cost condition, preset and thresholds.

## Rules and invariants

- Horizon: `T = ceil(runtime.horizon_constant x effective diameter)`, min 1; the diameter is the longest finite shortest
  path of each episode's graph. Episodes of one chunk are grouped by T.
- Graph, readings, placements, noise and uniforms come from streams keyed (topology, N, seed), never the arm, attack,
  budget cell, sink, batch size or chunking (common random numbers); all rounds pre-drawn, float64. Results are identical for
  B = 1 and B = 256 (tested). Rate-limit priorities are drawn lazily from a stream keyed by (topology, N, seed, round).
- D10: a denied, sanitized, delayed or cut delivery never changes the sender's memory (the carry/stay state uses its own raw output).
- AiTM: the victim's incoming deliveries are rewritten with probability `aitm_rewrite_strength`, carry the payload and have
  `obs.authentic` False; A (deny of unauthentic deliveries) is free.
- Auditor: equal-variance binormal; scores `z + mean(class)`, z keyed without the arm. Method `mock.auditor.binormal_method`
  (`shared_threshold` | `benign_fpr_only`).
- Budgets: audit debits `costs.<cond>.inspect` per checked sent text (broadcast checked once), NaN when unaffordable;
  actions debit per affected delivery (agent-level: live out-degree x remaining rounds); an episode whose round actions exceed
  its remaining intervention budget has all of them rejected and a violation counted; `exhausted_step` recorded.
  Action order as in `core.Actions`. Persistent states last until reinstatement.
- `obs.oracle` only when `ArmSpec.needs_truth`. Progressive checkpointing per chunk; done units skipped; a unit with saved
  rows but no done marker is not written twice.
- `calibrate`: attack-free uncapped runs, mode from the owner key `calibration.mode` (unset: `core.Unset`; other value:
  ValueError). The first pass runs the arm acting on nothing and pools its `Actions.statistics` over live deliveries (an
  agent-level statistic counts once per outgoing delivery; NaN is ignored).
  - `open_loop`: threshold = exact (1 - alpha) quantile (kthvalue) of each pooled statistic.
  - `closed_loop`: stateful arms (C's network threat level, BAZT's verdict memory) deny more once they act, so the open-loop
    threshold over-denies. The arm is run acting with candidate thresholds `quantile_level(statistic)` at one common level
    in [0, 1]; the level is bisected (first probe 1 - alpha, at most 40 passes) until the realised deny rate
    `(attempts & ~delivered) / attempts` over the calibration episodes is within `CLOSED_LOOP_TOLERANCE` = 10% of alpha
    (relative). Deterministic and independent of batch size (counts are summed). Every pass is saved with
    `store.save_state("calibration-<arm>-<hash>", ...)` as (level -> rate, thresholds); a resumed calibration reuses a saved
    pass only if its thresholds equal the recomputed ones. Limit: when the rate moves in steps (discrete statistics such
    as C's weight sums, or a feedback cascade) alpha can fall inside a step; the level with the highest rate not above
    alpha(1 + tolerance) is returned (never over-denying) and `last_calibration["within_tolerance"]` is False.
  - Limit of closed loop: it makes the deny rate alpha on the calibration episodes only. A stateful arm at the edge of a
    feedback cliff (C: once its threat level rises it denies ~85% of an episode) can cascade on episodes the calibration
    sample did not contain, so a held-out rate can exceed alpha (C, small_world N=12: 0.14 held-out vs 0.02 on the
    calibration seeds, still below the open loop's 0.20). `worst_episode_rate` reports the largest single-episode rate seen.
  - `MockPlatform.last_calibration` = {mode, passes, simulated, level, rate, within_tolerance, worst_episode_rate}.
- `trace.meta["events"]` holds every `Actions.events` entry the defender returned (e.g. budget_exhausted).
- Torch runs on one thread inside the platform.
- `EpisodeContext.platform` is `"mock"`; `ctx.timing_schedule` is a bool [B, T] stacked from `unit["timing_schedule"]`
  (T bools, 0-based rounds) when every unit of the batch has one, else None.
- `obs.audit` over-request: an episode asking for more texts than it can afford gets a random subset, drawn from a stream
  keyed (topology, N, seed, round, call number) without the arm or the batch (deterministic, batch-size independent).
- `run_batch` calls `tracker.log_rows("mock_units", rows)` per chunk; all rows of the call carry the same keys (None where absent).
- Targeted ASR (`goal_reached`) definitions per family: see `private/goal.py`.

## Settings keys read

`mock.model.{m,a,b,mu_inf,mu_dec,sd,beta,p_stay,p_decoy,decoys,burst_span,reading_mean,reading_sd,bias}`, `mock.batch_size`,
`mock.auditor.binormal_method`, `runtime.{n_insiders,rate_limit_per_identity,horizon_constant,eps_correct,ba_m,ws_k,
ws_rewire_p,sanitize_success,insider_bias,ta_detectability,ma_detectability,aitm_rewrite_strength}`, `budgets.alpha`, `calibration.mode`,
`costs.<main|robustness>.{inspect,sanitize,delay,edge_quarantine,rate_limit,privilege_degradation,isolation,reinstatement,deny}`,
`auditor_presets.*`.

## Open questions

1. Binormal derivation from two FPR classes (spec section 11): implemented behind `mock.auditor.binormal_method`.
2. Targeted ASR for the mock families (not defined by the spec): `private/goal.py`.
3. Clean reference for `accuracy_clean` / recovery: the arm itself without the attack (`clean_reference_defender`); alternative is no defence.
4. Edge `deny` cost: charged per authentic delivery denied.

## Requests (not applied by this lane)

- `mock.auditor.binormal_method` is present in profiles.toml but unset: the owner chooses the method (open question 1).
- `costs.robustness.deny` is unset (owner); robustness-condition units refuse with `Unset("costs.robustness.deny")`.
- `calibration.mode` is an owner key: the runner adds the placeholder (commented) in profiles.toml.
- Arms with discrete statistics (C, BAZT) could add a keyed tie-breaking jitter so alpha is reachable inside a step (conventional / bazt_slot).
- The `[runtime]` keys stay unset (run-time variables) as the spec requires.
