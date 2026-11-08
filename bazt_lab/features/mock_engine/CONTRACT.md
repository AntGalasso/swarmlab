# mock_engine

`MockPlatform` (core `Platform` contract): a torch engine implementing swarmlab's model plus the changes of
docs/EXPERIMENT_PLAN.md sections 3, 4.2, 5 and 6. Batch dimension B = episodes. The round loop is the only loop over time;
graph construction and unit checks loop once per batch over agents or units.

## public.py

- `MockPlatform(rows_fn=None)`: `run_batch(units, arm, settings, store, tracker)` and
  `calibrate(units, arm, settings, store)`. `rows_fn` defaults to `metrics.public.rows_from_trace` (tests inject a fake).
  `adaptive_conditions` (core `Platform`): the adaptive feature's `CONDITIONS`, whose best responses reach the engine as
  `attack_params`.
- `apply_actions(actions, state, live, authentic, adj, remaining, t, T, costs, rate_limit, rate_u) -> Round`: the reference pricing
  and application of one round of Actions (see private/enforce.py); `Persistent.zeros(B, N)` (also exported) is the initial
  persistent state.
- `TOPOLOGIES` (`star_hub`, `star_leaf`, `chain`, `small_world` = Watts-Strogatz, `barabasi_albert`), `ATTACKS`
  (`none`, `insider_pi`, `insider_ta`, `insider_ma`, `infection`, `aitm`).
- `adapters/swarmlab_reference.py` is the only non-test import of swarmlab (the no-defense reference of `tests/test_fidelity.py`).

## Unit fields

Required: `id, arm, audit, intervention, topology, attack, agents, seed`. `audit` / `intervention` are a fraction in (0, 1]
of all edge deliveries of the episode (|E| x T, directed edges) or `"uncapped"`. Optional: `sink` (int, else a keyed draw),
`cost_condition` (`main` default, `robustness`), `preset` (auditor preset, default `auditor_presets.main`), `thresholds`,
`attack_params` (keys: n_insiders, insider_bias, mu_inf, auditor_tpr, entry_ranks, start_frac, rate, detectability, potency).
A batch shares arm, budget cell, topology, attack, N, cost condition, preset and thresholds (`check_units`); `sink` and
`attack_params` may differ per episode. The `attack_params` keys: `n_insiders` (the insider count of an insider family,
default `runtime.n_insiders`); `entry_ranks` (degree ranks of the entry agents, replacing the drawn ones: the infection seed,
the AiTM victim or the insiders); `insider_bias` and `potency` (the reported shift of the insiders, potency scaling it);
`mu_inf` (monitor mean of a payload-carrying message); `detectability` (scales the payload's monitor mean and its auditor
mean); `auditor_tpr` (the detection rate of the adaptive auditor preset, the only key an attack-free run reads); `start_frac`
and `rate` (the attack rounds, through `adaptive.attack_rounds`; without either the attack acts in every round).

## Rules and invariants

- Horizon: `T = ceil(runtime.horizon_constant x effective diameter)`, min 1; the diameter is the longest finite shortest
  path of each episode's graph. Episodes of one chunk are grouped by T.
- Graph, readings, placements, noise and uniforms come from streams keyed (topology, N, seed), never the arm, attack,
  budget cell, sink, batch size or chunking (common random numbers); all rounds pre-drawn, float64. Results are identical for
  B = 1 and B = 256 (`tests/test_platform.py`, `test_results_do_not_depend_on_the_batch_size_or_the_chunking` and
  `test_a_large_batch_gives_each_unit_the_row_it_has_alone`). Rate-limit priorities are drawn lazily from a stream keyed by
  (topology, N, seed, round).
- D10: a denied, sanitized, delayed or cut delivery never changes the sender's memory (the carry/stay state uses its own raw output).
- D111: isolation is containment, not cure. An isolated agent writes, sends and receives nothing and keeps its last own
  output, so one that carried the payload stays infected in `trace.infected` (and in `obs.oracle`, as on the LLM
  platform) for every round it is isolated: protected-sink harm 1, counted as reached, its attempted deliveries
  malicious. A reinstated agent comes back with that state and keeps it with probability `p_stay` like any active
  agent; the spread, the monitor and the auditor see only the texts of active agents, so they are unchanged.
- AiTM: the victim's incoming deliveries are rewritten with probability `aitm_rewrite_strength`, carry the payload and have
  `obs.authentic` False; A (deny of unauthentic deliveries) is free.
- Auditor: equal-variance binormal; scores `z + mean(class)`, z keyed without the arm. Method `mock.auditor.binormal_method`
  (`shared_threshold` | `benign_fpr_only`). `shared_threshold` is the model of D91 (one score threshold; benign and
  suspicious-looking benign texts get separate means, so both false-positive rates hold at it), and profiles.toml sets it.
- Budgets: audit debits `costs.<cond>.inspect` per checked sent text (broadcast checked once). `obs.audit` grants only senders
  that are active and have a live delivery; a request beyond the affordable number is granted as a keyed subset, and an
  ungranted entry is NaN. Action prices (`costs.<cond>.<kind>`, `enforce.py`), each on the deliveries the action affects
  and not on deliveries an earlier action of the round or a persistent state already removed:
  deny (authentic deliveries only; a deny of an unauthentic delivery is free), delay and sanitize (per delivery of the round,
  a sanitize of a degraded sender's or a delayed delivery is free), edge quarantine (per new quarantined edge, over the
  remaining rounds `T - t + 1`), isolation (the deliveries it removes, over the remaining rounds; clamped to the limit when the
  sender is rate-limited), privilege degradation (the sender's surviving deliveries, over the remaining rounds), rate limit
  (the deliveries above the limit, over the remaining rounds), reinstatement (the out-degree after the lift, over the remaining
  rounds, one round fewer when the agent was isolated). An episode whose round actions exceed its remaining intervention
  budget has all of them rejected and a violation counted; `exhausted_step` records the first round a budget reaches zero.
  Action order as in `core.Actions`. Persistent states last until reinstatement.
- `Observation.attempts` (what a defender can act on) excludes quarantined edges and the edges of isolated agents.
  `trace.attempts[r]` is the whole base graph in every round, quarantined and isolated edges included.
- `obs.oracle` only when `ArmSpec.needs_truth`; it is the round's `trace.infected` (D111 state). Progressive checkpointing per chunk; done units skipped; a unit with saved
  rows but no done marker is not written twice.
- `calibrate`: attack-free uncapped runs, mode from the owner key `calibration.mode` (unset: `core.Unset`; other value:
  ValueError). The first pass runs the arm acting on nothing and pools its `Actions.statistics` over live deliveries (an
  agent-level statistic counts once per outgoing delivery; NaN is ignored). Statistics are pooled at every round, whatever the
  arm's own timing, and the realised rate counts the attempts of all rounds.
  - `open_loop`: threshold = exact (1 - alpha) quantile (kthvalue) of each pooled statistic.
  - `closed_loop`: stateful arms (C's network threat level, BAZT's verdict memory) deny more once they act, so the open-loop
    threshold over-denies. The arm is run acting with candidate thresholds `quantile_level(statistic)` at one common level
    in [0, 1]; the level is bisected (first probe 1 - alpha, at most `calibration.max_passes` passes) until the realised deny rate
    `(threshold_restricted & attempts & ~delivered) / attempts` when the arm reports `Actions.threshold_restricted` (D108: only deliveries restricted because a statistic crossed its threshold count; an isolation loss counts when the arm reports it), else `(attempts & ~delivered) / attempts` over the calibration episodes is within `calibration.tolerance` x alpha (relative). Deterministic and independent of batch size (counts are summed). Every pass is saved with
    `store.save_state("calibration2-<arm>-<hash>", ...)` as (level -> rate, worst single-episode rate, thresholds); a resumed calibration reuses a saved
    pass only if its thresholds equal the recomputed ones. Limit: when the rate moves in steps (discrete statistics such
    as C's weight sums, or a feedback cascade) alpha can fall inside a step; D112, the same rule as the LLM platform (`llm_harness/private/platform.py`; `pick_pass` here): the pass with the highest rate
    not above alpha (1 + tolerance) is returned (the first on a tie), never one that denies more, so the arm may then act
    on almost nothing in that cell; if every pass is above it, the lowest rate. `last_calibration["converged"]` (alias
    `within_tolerance`) is then False.
  - Limit of closed loop: it makes the deny rate alpha on the calibration episodes only. A stateful arm at the edge of a
    feedback cliff (C: once its threat level rises it denies ~85% of an episode) can cascade on episodes the calibration
    sample did not contain, so a held-out rate can exceed alpha (C, small_world N=12: 0.14 held-out vs 0.02 on the
    calibration seeds, still below the open loop's 0.20). `worst_episode_rate` reports the largest single-episode rate seen.
  - `MockPlatform.last_calibration`: open loop {mode, passes, simulated, alpha, converged}; closed loop {mode, alpha, level,
    rate, passes, simulated, converged, within_tolerance, worst_episode_rate}.
- Rows: the mock has no replies, so `empty_replies` (D114) is 0 (no `meta["empty_replies"]`).
- Time convention (core EpisodeTrace): `active[r]` is the state DURING round r; a decision taken after round r shows in `active[r+1]` and in `meta["flagged"][r]` (rate-limited, degraded or isolated), so a last-round isolation is still visible.
- `trace.meta["events"]` holds every `Actions.events` entry the defender returned (e.g. budget_exhausted).
- `Actions.threshold_restricted` is consumed by the closed loop only (`MockPlatform.realised_rate`); it is not part of the trace or its rows.
- Torch runs on one thread inside the platform.
- `EpisodeContext.platform` is `"mock"`; `ctx.timing_schedule` is a bool [B, T] stacked from `unit["timing_schedule"]`
  (a list of T Python booleans, entry r = round r+1, the format of `intervened_by_round`) when every unit of the batch
  has one, else None. A schedule of numbers (round numbers, 0/1) or of another length is a ValueError, never cast.
- D116, intervention rounds: `trace.meta["intervened"]` is a bool [B, T]; entry (b, r) is True when the platform applied
  at least one action of the arm other than an audit in round r+1 of episode b, after the budget rule. Applied means
  counted in `action_counts` (enforce.Round.counts): a new isolation, edge quarantine, rate limit or privilege
  degradation (on an agent or edge that did not already carry it), a reinstatement of an agent that had a restriction,
  or a deny, delay or sanitize of a delivery that survived the earlier actions of the round, including a free deny of an
  unauthentic delivery (AiTM). Not an intervention: an audit (`inspect`), a request that changes nothing (state already
  carried, a delivery an earlier action removed, a reinstatement of an unrestricted agent, a sanitize of a degraded
  sender's delivery), every action of a round rejected over budget, and the continued effect of a persistent restriction
  in later rounds. Every arm's trace has it; metrics writes it to the row as `intervened_by_round` (T booleans). It is per
  episode, so it does not depend on the batch, the chunking or the worker. The matched-timing controls get BAZT's value
  of the same unit as `unit["timing_schedule"]` (runner).
- The clean (attack-free) run of a unit that carries `timing_schedule` (D109 twin, or the clean copy of a unit without
  `twin_id`) acts at the rounds of the grid's attack-free unit: `unit["twin_timing_schedule"]` (the runner adds it, the
  attack-free BAZT row's record), which becomes the twin's `timing_schedule` (and the saved twin row's). A unit with
  `timing_schedule` whose clean run must be simulated without `twin_timing_schedule` is a ValueError; a twin read from
  the store needs none.
- `obs.audit_isolated(request [B,N] bool)` (D123): the auditor scores an isolated agent (`~obs.active`) from the payload state
  it carries (D111): `aud_z + d_pay` if it carries it, else `aud_z`, the same auditor model and noise as `obs.audit`.
  Same unit cost, same audit budget (hard cap, exact), NaN where not requested, not isolated or unaffordable; its
  over-request subset is keyed on its own stream (`audit_isolated`) with its own per-episode call count, so `obs.audit`
  numbers its calls and draws its subsets exactly as without it (an arm that never calls it gets bit-identical rows).
- `obs.audit` over-request: an episode asking for more texts than it can afford gets a random subset, drawn from a stream
  keyed (topology, N, seed, round, call number) without the arm or the batch (deterministic, batch-size independent).
- `run_batch` calls `tracker.log_rows("mock_units", rows)` once per simulated horizon group of a chunk; within one call every row carries the same keys (None where absent).
- Targeted ASR (`goal_reached`): the mock's goal per family is in `private/goal.py`; see Open questions.

## Settings keys read

`mock.model.{m,a,b,mu_inf,mu_dec,sd,beta,p_stay,p_decoy,decoys,burst_span,reading_mean,reading_sd,bias}`, `mock.batch_size`,
`mock.auditor.binormal_method`, `runtime.{n_insiders,rate_limit_per_identity,horizon_constant,eps_correct,ba_m,ws_k,
ws_rewire_p,sanitize_success,insider_bias,ta_detectability,ma_detectability,aitm_rewrite_strength}`, `budgets.alpha`,
`calibration.{mode,tolerance,max_passes}`, `compute.{device,dtype,compile}`,
`costs.<main|robustness>.{inspect,sanitize,delay,edge_quarantine,rate_limit,privilege_degradation,isolation,reinstatement,deny}`,
`auditor_presets.*` (including `auditor_presets.adaptive_tpr`).

## Open questions

1. Targeted ASR. `private/goal.py` defines the mock's goal per family (insider_pi: the sink's final answer is pushed toward
   the insiders' target; insider_ta and infection: the sink's output carries the payload in some round; insider_ma: the
   sink's output carries it in the last round; aitm: the victim's final answer is shifted; an insider sink is never a victim).
   D122 records per-family goals for both platforms and is not implemented yet, so this definition stays open until it is.

Settled by decision: the binormal model (D91), the clean reference (D89), the price of one affected delivery (D9, D22a).

## Requests (not applied by this lane)

- `costs.robustness.deny` is unset (owner); robustness-condition units refuse with `Unset("costs.robustness.deny")`.
- `calibration.mode` is an owner key: profiles.toml carries it as a commented placeholder marked OPEN.
- The `[runtime]` keys stay unset (run-time variables) as the spec requires.
- profiles.toml (runner): the `calibration.max_passes` comment still says "before the closest pass is returned"; under
  D112 it is the highest pass not above alpha (1 + tolerance).

## Clean reference (D109), device and precision

- Clean reference: an attacked unit may carry `twin_id`, the id of its attack-free twin unit (same unit, attack `none`,
  `attack_params` cut to the keys an attack-free run reads, `world.ATTACK_FREE_PARAMS` = `auditor_tpr`: the adaptive
  auditor preset of a robustness variant stays, the attacker's strategy goes, so the twin is the grid's own attack-free
  unit; every other key changes nothing in a run without an attack, tests/test_twin.py). The
  paired clean accuracy is the twin's OWN run: when the twin is done in the store (`store.rows_of`) its `accuracy_clean` is
  read (non-insider families); otherwise the twin is simulated with the twin's id (so the arm's keyed draws equal the twin
  unit's own) and its row is saved as the twin's result, so it is never simulated twice. Insider families need the clean
  accuracy over the victims (non-insiders), which a twin row does not hold, so they take it from the twin-keyed run. A
  unit without `twin_id` keeps the old behaviour (clean copy keyed by its own id). The clean accuracy is a float64 ratio.
- Device and precision (`compute.device`, `compute.dtype`, `compute.compile`): the world draws are made once on the CPU in
  float64, then moved to `ctx.device` and converted to `ctx.dtype`; every engine tensor lives on `ctx.device`, real-valued
  state (estimates, monitor, scores) in `ctx.dtype`; budgets, costs and spend stay float64/int64. The returned trace and
  estimates are on the CPU, estimates in float64. The audit, audit_isolated and rate-limit priorities come from
  `core.keyed_uniform` (streams `audit` and `audit_isolated`, each with a per-episode call counter, and stream `rate`),
  identical on every device and batch. The two numeric kernels of a round (`_consensus`, `_write`) go through `core.compiled`.
- float32 against float64 (`compute.dtype` = `float32` in profiles.toml, D117) is checked in two places:
  - `tests/test_compute.py::test_float32_agrees_with_float64_on_the_outcome_metrics` (in this lane) runs 96 episodes
    (small_world, N = 10, T = 8, a defender that denies on monitor above 0.6, budgets 20%). It requires the infected and
    correct entries to agree on at least 99.9%, the final estimates to differ by less than 1e-3 (maximum absolute
    difference), the final-round accuracy means to differ by less than 0.005, and the audit and intervention spend to be
    identical for at least 99% of the episodes.
  - `scripts/check_gpu.py` (run on a GPU machine) calibrates and then evaluates a small frontier slice on each device and
    dtype, and compares the whole evaluation rows against the CPU float64 reference. Budget accounting (`spend_audit_total`, `spend_intervention_total`, `spend_total`, `violations`,
    `exhausted_step`) must be identical for at least 99% of the episodes, and every other row number within 0.005 (absolute
    up to 1, relative above 1) for at least 99% of the episodes.
- Masks are combined as uint8 views (`private/bits.py`): identical results, 30% less engine time on CPU. `run_batch` makes
  no per-unit file system call (`store.is_done` is a set lookup, rows go through `store.append_batch`).
