# conventional: arms A, B and C (spec 2.1 to 2.4)

The three published forms of conventional zero trust, each a `core.Defender`, working on both platforms (`obs.monitor`
is `None` on the LLM side and no arm reads it). Shapes follow `core`: B episodes, N agents, float64 scores, bool masks,
int64 indices. All three act only through `Actions.deny`; `events` and `statistics` carry the budget warning and the
decision statistics.

## public.py

| name | what |
|---|---|
| `ARMS` | `(ArmSpec("A"), ArmSpec("B", calibrated=True), ArmSpec("C", calibrated=True))`; platforms `("mock", "llm")`, `needs_truth=False` |

## Arms

| arm | what it does each round | reads | statistic reported | threshold read |
|---|---|---|---|---|
| `A` | denies every attempted delivery that is off `ctx.adj` or has `obs.authentic` False. One trust tier, never reads content, never audits, never emits events, state is empty. Free and uncapped | `attempts`, `authentic`, `ctx.adj` | none | none |
| `B` | A, then one `obs.audit` request covering every sender with a delivery A passed (one audit unit per sent text), then denies every delivery of a sender whose score exceeds the threshold, then keeps at most `runtime.rate_limit_per_identity` receivers per sender and round among the deliveries of checked senders that passed the judge (a sender whose text was not checked gets A's rules only). Stateless per hop: nothing carries across rounds except the budget latch | A's inputs, `audit`, `remaining_*`, `ctx.costs` | `judge` `[B, N, N]`: the sender's judge score on each evaluated delivery | `judge` |
| `C` | A, the same audit request as B, then the additive trust score against the risk level (below) | B's inputs, `active` | `rl_minus_ts` `[B, N, N]` | `rl_minus_ts` |

A statistic is NaN where nothing was evaluated: a delivery A dropped, a sender whose text was not checked (the platform
could not pay), an episode that is exhausted. Higher means more suspicious. An arm denies when `statistic >
ctx.thresholds[name]` (strict, as `core.Actions` states); a threshold missing from `ctx.thresholds` is +inf, so a
calibration run collects the statistic and acts only through A's rules.

### B's rate limit

The excess deliveries of a sender (those beyond `rate_limit_per_identity` among the ones that passed the judge) are
denied for the round. Which receivers are kept is a keyed random draw (stream `B|rate_limit`, counter `step`, per episode):
receiver order carries no meaning, an index rule would favour low indices. Each denied delivery costs one `deny` unit.

### C: trust score against risk level (Bradatsch et al. 2023, Eqs. 1 to 3)

For every delivery i -> j that A passed and whose sender's text was checked:

```
TS(i, j) = partial(i, j) + 5 * [partial(i, j) in [min, max] of i's past TS]                          (Eq. 1)
partial  = 23 + 3 * [absent_i <= W] + 3 * [j in i's past recipients] + 5 * [rate_i in [min, max] of i's past rates]
              + 3 * [score_i in [min, max] of i's past scores]
RL(i, j) = THREAT[min(flagged, 10)] + bin_weight(score_i)                                            (Eq. 2)
statistic = RL - TS;  deny iff statistic > threshold                                                  (Eq. 3 with the calibrated threshold)
```

`THREAT = [0, 2, 3, 4, 5, 7, 9, 12, 15, 20, 25]`; `flagged` is the number of agents C has flagged so far in the episode;
`W` is `conventional.c.presence_window_rounds`; `bin_weight` is `conventional.c.judge_score_bins[ctx.platform]`. An empty history
contains nothing (weight 0), as the reference's `MinMax` error branch.

Attribute mapping (spec 2.3). Weights and rules are those of `additive_trust.go` at commit
`46e099bf95518ff4f887f0abdedb0f6e8baf073a` of `zintpavowj/ZT_Score-based_Network_Level_AC_PDP_TA`:

| spec attribute | type | reference attribute (line) | weight | what C evaluates for i -> j |
|---|---|---|---|---|
| authentication factors | trust | `device_cert_auth` (34) | 8 | constant: every delivery C evaluates passed A's authentication (`user_faceid_auth`, 19, weighs the same) |
| channel authentication, confidentiality, integrity | trust | `channel_authentication` SHA256-RSA (68), `channel_confidentiality` AES (71), `channel_integrity` SHA256-RSA (74) | 5 + 5 + 5 | constant: the strongest value of each |
| service usage | trust | `user_service_usage` (24; rule 339-346) | 3 | j is among i's past recipients |
| access rate | trust | `user_access_rate` (22; rule 383-397) | 5 | number of deliveries i attempts this round is within [min, max] of its past counts |
| enterprise presence | trust | `user_enterprise_presence_time` (27; rule 404-407) | 3 | consecutive rounds i was inactive before this one is at most `W` |
| input behaviour | trust | `user_input_behavior` (21; rule 323-336) | 3 | the judge score of i's text is within [min, max] of its past scores |
| trust history | trust | `user_trust_history` (31; rule 428-441, called last at 175) | 5 | the partial TS is within [min, max] of the TS of i's past evaluated deliveries; unbounded window, no decay; a flag never lowers it |
| network threat level | risk | `risk_system_network_threat_levels` (130-139; lookup 747-756) | table above | number of agents flagged so far |
| judge score | risk | none (spec section 11) | `conventional.c.judge_score_bins.<platform>` | the sender's judge score, binned |

Constant attributes shift TS by the same amount for every delivery, so with a calibrated threshold they change nothing;
only the relative weights act.

Histories learn from every evaluated delivery, denied or not (a denied delivery still enters i's past recipients, rates,
scores and TS), after the round's decisions. An agent is flagged in the round C wants to deny one of its deliveries,
before the intervention budget funds the denies; it stays flagged for the episode. A's drops do not flag.

Departures from the reference code, forced by the spec or by the experiments (each is an open question below):

- Decision (D92). The paper grants iff TS > RL (Eq. 3, D80); `additive.go:157` grants on `>=`. C denies when `RL - TS >
  threshold` (core contract), so an exact tie at the threshold is granted, as in the reference code.
- Presence. The reference also adds 2 when the user is "expected" (lines 28, 410); the spec's table has no such
  attribute. Its 7-day window (405) becomes `W` rounds.
- Threat level. The reference weighs levels 1 to 10 and gives 0 beyond (747-756); C saturates at 10 so that more flagged
  agents never lower the risk. The reference also counts the weight twice in its dynamic risk score (lines 277 and 283
  both read `risk_system_network_threat_levels[NetworkThreatLevel]`); the spec lists one attribute, so C counts it once.

## Capped behaviour (spec 2.4)

B and C are exhausted for an episode when its remaining audit budget cannot pay one audit unit or its remaining
intervention budget cannot pay one `deny` unit. With whole-number caps and unit costs this is exactly "reached zero". The
round whose charges exhaust a budget (or the first round, if a cap is zero) emits once per episode

```
{"event": "budget_exhausted", "arm": "B" | "C", "b": <episode index>, "unit_id": <ctx.unit_ids[b]>,
 "budget": "audit" | "intervention", "t": <obs.t>}
```

(`audit` when both are exhausted in the same round). From the next round the episode gets A's rules only: no audit
request, all statistics NaN, C's histories frozen. Other episodes of the batch are unaffected. Uncapped episodes (inf)
never emit. A never emits.

When the intervention budget cannot pay all wanted denies, the arm keeps the highest-statistic deliveries first (D94), ties
broken by a keyed random draw (stream `<arm>|fund`, counter `step`); the platform charges them and never sees an
unaffordable deny.

## Invariants

- Ground truth is never read: only `attempts`, `active`, `authentic`, the budgets and `obs.audit`. `monitor` and `oracle`
  are ignored.
- Every B or C denial is of an evaluated delivery (attempted, on `ctx.adj`, authentic, sender checked). Only A's rules
  drop anything else, and they are free.
- Every random draw comes from `core.keyed_uniform` keyed by the episode id (see Device and precision); `step` counts `decide` calls since `reset`, so
  it keeps growing across the conversations of an LLM seed run. Decisions of an episode depend on its own inputs only:
  batch size and chunking change nothing.
- `state_dict()` holds tensors and plain values and copies them. B: `step`, `exhausted`. C adds `recipients`, `rate_lo`,
  `rate_hi`, `input_lo`, `input_hi`, `trust_lo`, `trust_hi`, `absent`, `flagged`. `load_state_dict` is called after
  `reset(ctx)` and continues exactly like an uninterrupted run.
- `reset(ctx)` re-reads the settings and clears the state: one call per episode (an LLM seed run), not per conversation.
- Scores, statistics and budgets are float64; masks bool; indices int64.

## Settings read (`core.get`; a missing key raises `core.Unset` naming it)

| key | arm | meaning |
|---|---|---|
| `runtime.rate_limit_per_identity` | B | integer >= 1: receivers per sender and round (spec section 10) |
| `conventional.c.judge_score_bins.<platform>` (`mock`, `llm`; platform = `ctx.platform`) | C | `[[upper, weight], ...]` per platform: a score weighs the weight of the first bin with `score <= upper` (bins are `(previous upper, upper]`), strictly increasing uppers, finite weights >= 0; above the last upper it weighs 0. TOML example: `[[-5.0, 0.0], [-1.0, 4.0], [inf, 12.0]]`. Empty `ctx.platform` or a missing platform entry: C refuses to start with `core.Unset` naming e.g. `conventional.c.judge_score_bins.mock` |
| `conventional.c.presence_window_rounds` | C | integer >= 0: the presence attribute holds while the sender has been inactive for at most this many consecutive rounds |

Costs from `ctx.costs`: `inspect` and `deny` (B, C). A missing cost raises `core.Unset("costs.<condition>.<kind>")`.

## Open questions

1. Judge-score bins and weights (spec section 11) are the owner's: `conventional.c.judge_score_bins` is unset in
   `experiments/profiles.toml`. The units differ by platform: the LLM judge returns log P("yes") (at most 0), the mock
   auditor a binormal score on a z scale (reference threshold near 2.05 at FPR 0.02), so one list cannot serve both. C
   reads the list of `ctx.platform`.
2. Presence window `W` (`conventional.c.presence_window_rounds`, unset in `experiments/profiles.toml`, owner's value). The
   spec gives "rounds since i was last active" without a window; the reference's is 7 days. A, B and C never isolate an
   agent, so `active` is always true for them and the attribute is constant.

## Requests (for the coordinator; this feature edits no other file)

1. `experiments/profiles.toml`: `[costs.robustness]` has no `deny` value. `deny` is priced at 1 in `[costs.main]` (D22a) and
   `mock_engine.enforce.costs_of` reads `costs.<condition>.deny`. CAGE-4 has no deny duration (D23), so the robustness
   value is the owner's.

## Calibration of C (closed loop)

C is stateful: its own denials raise the network threat level and its histories change TS, so the statistic of an
attack-free run depends on the thresholds C acts on. C supports a closed-loop calibration: it reports `rl_minus_ts` and
acts on `ctx.thresholds` in the same pass, and `reset(ctx)` clears every trace of an earlier pass (tested), so a
platform can replay the same episodes with a candidate threshold on one instance or on fresh ones. The search for the
threshold at deny rate alpha belongs to the platform (`calibration.mode`, `calibration.tolerance`,
`calibration.max_passes`; when no pass lands within the tolerance, the pass with the highest rate not above alpha (1 +
tolerance) is used, D112).

### Tie-breaking jitter of C's statistic

`rl_minus_ts` is a sum of weights that are multiples of one unit `u` (the gcd of the reference weights, the threat table
and `conventional.c.judge_score_bins[platform]`; 1 for integer bins), so many deliveries tie exactly and one threshold
cannot reach alpha inside a step. C therefore adds `(U - 0.5) * 2 * a` to every evaluated delivery's statistic, with
`a = 64 * core.tie_jitter_scale(ctx.dtype, bound)`, where `bound` = 42 + 25 + the largest bin weight is an upper bound on
|RL - TS| (42 is the sum of the trust weights, 25 the largest threat weight). With a largest bin weight of 12 (`bound` 79),
`a` is about 9 x 10^-12 for float64 and 4.8 x 10^-3 for float32, so it survives the dtype's rounding. `U` is uniform in [0, 1) from the keyed stream
`"C|jitter"` at counter `step`. Delivery number r of an episode in row-major (sender, receiver) order, among the
evaluated deliveries of the round, takes element r, so the draw depends on the episode id, the round and r only (never on
the batch, its order or the device) and costs K draws per episode instead of N*N. `reset` refuses when `2a >= u`: the
jitter must stay below half the gap, so distinct values keep their order; ties are split at the threshold, as in a
randomized Neyman-Pearson test. The value reported in `Actions.statistics` is the value compared with `ctx.thresholds`,
so calibration and evaluation use the same statistic. Funding priority uses it too. `ArmC.JITTER` (class attribute,
default True) is copied to the instance field `jitter` in `reset(ctx)`, so one run uses one value; the tests with
hand-computed values set it False. B's `judge` is not jittered: a judge score is continuous (and one score covers all
deliveries of a sender), so it has no gap to stay inside.

### threshold_restricted (D108)

Alpha counts only deliveries restricted because a calibrated statistic crossed its threshold. `Actions.threshold_restricted`
(bool `[B, N, N]`) is set by B to its funded judge denies (`judge > threshold`) and by C to its funded threshold denies
(`rl_minus_ts > threshold`). B's rate-limit denies and A's rules are never in it. It is all False while a threshold is
missing (calibration pass) and for episodes that are exhausted. With B at the open-loop 98% quantile when no feedback is
present, the closed-loop judge threshold equals it (1.945 small_world N=12, 2.035 barabasi_albert N=30 in the repro).

## Device and precision

Every tensor an arm creates lives on `ctx.device`; scores, statistics and the C histories use `ctx.dtype` (the platform
supplies `obs.audit` scores in that dtype). Budget and cost accounting stays float64 / int64 whatever the dtype. Random
draws (B's rate limit, the funding tie break, C's jitter) come from `core.keyed_uniform` keyed by
`core.episode_keys(ctx.unit_ids)` with streams `"<arm>|<namespace>"` and counter = `step`: one batched call, no
per-episode generator. Funding and rate-limit scores are always float64, so the order never depends on the dtype. Mask
algebra of the per-round path goes through `core.as_uint8`/`as_bool` (identical results, faster on CPU). `torch.compile` is
not wrapped around any kernel of this lane (the measured gain would be small next to the engine; open). float32 agrees with
float64 on the number of denies within 2%, and each statistic agrees with float64 to within one lattice unit after
rounding, while the accounting is identical (tests/test_compute.py).
