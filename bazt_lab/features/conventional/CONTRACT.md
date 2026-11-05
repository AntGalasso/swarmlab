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
denied for the round. Which receivers are kept is a keyed random draw (`rate_limit|<step>`, per episode): receiver order
carries no meaning, an index rule would favour low indices. Each denied delivery costs one `deny` unit.

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

When the intervention budget cannot pay all wanted denies, the arm keeps the highest-statistic deliveries first, ties
broken by a keyed random draw (`fund|<step>`); the platform charges them and never sees an unaffordable deny.

## Invariants

- Ground truth is never read: only `attempts`, `active`, `authentic`, the budgets and `obs.audit`. `monitor` and `oracle`
  are ignored.
- Every B or C denial is of an evaluated delivery (attempted, on `ctx.adj`, authentic, sender checked). Only A's rules
  drop anything else, and they are free.
- Every random draw comes from `ctx.arm_stream(f"{namespace}|{step}", b)`; `step` counts `decide` calls since `reset`, so
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

1. Judge-score bins and weight (spec section 11). Their values are the owner's. The units differ by platform: the LLM
   judge returns log P("yes") (at most 0), the mock auditor a binormal score on a z scale (reference threshold near 2.05
   at FPR 0.02), so one list cannot serve both. Resolved: C reads the entry of `ctx.platform`. The placeholder in profiles.toml shows `{ mock = [<float>, ...], llm = [...] }`; C expects each entry as `[[upper, weight], ...]` pairs (a flat float list carries no weights), so the placeholder text should say so.
2. Presence window `W`. The spec gives "rounds since i was last active" without a window; the reference's is 7 days.
   Until an isolating arm runs, `active` is always true for A, B and C and the attribute is constant.
3. Decided (D92: C grants at an exact tie, TS >= RL). Tie convention. D80 fixes the strict `TS > RL`; the core contract fixes `deny iff statistic > threshold`. They agree
   everywhere except at an exact tie with the offset, where C grants (as `additive.go:157`). The statistic is a lattice
   (integer weights), so a tie at a calibrated threshold is common; the platform's calibration must use the same `>`.
4. Decided (D93: the judge score is the input attribute on both platforms). Input behaviour feature. The spec says "message features within i's usual range". The only content feature present
   on both platforms is the judge score, so C uses it, and the judge score also enters RL through the bins. The cheap
   monitor (mock only, free) would make C platform-dependent. One line in `arm_c.py` (`feature = audited`) selects it.
5. Decided (D95: the number of agents C has wanted to deny, never decaying, saturating at 10, counted once). Network threat level. "Number of agents flagged so far" is read as the count of agents C has wanted to deny, never
   decaying, saturating at level 10 (so for N = 12 or 100 it saturates early), counted once.
6. Calibration of a stateful arm. A calibration run collects `rl_minus_ts` without acting, so nobody is flagged and the
   threat level stays 0 and the histories are cold; in an evaluation run flags raise RL, so the deny rate of an open-loop
   calibrated C on attack-free runs is far above alpha (17-100% observed). The fix is a closed-loop calibration in the
   platforms (section "Calibration of C"); the attack-free false-positive rate (spec 6) measures the result.
7. Decided (D94: the most suspicious are funded first). Funding order. Under a cap, the denies funded are the most suspicious first (a generous rule for the baselines).
   First-come order is the alternative.
8. Rate limit. B's limit is a per-round, per-sender cap realised as `deny` (stateless per hop). The persistent
   `RATE_LIMIT` action belongs to BAZT.
9. Exhaustion. "Reaches zero" is implemented as "cannot pay one more unit", which differs only for fractional caps (a
   remainder that buys nothing would otherwise never raise the warning).

## Requests (for the coordinator; this feature edits no other file)

1. `experiments/profiles.toml`: add `[conventional.c]` with `judge_score_bins` and `presence_window_rounds`, unset like the
   `[runtime]` keys (values are the owner's, open questions 1 and 2). The runner's required-key check must include both
   for every block that runs C or BAZT (BAZT builds A, B and C), and `runtime.rate_limit_per_identity` for every block
   that runs B or BAZT.
2. `experiments/profiles.toml`: add `deny` to `[costs.main]` and `[costs.robustness]`. `ActionKind.DENY` is an action the
   engine prices (`mock_engine.enforce.costs_of` reads `costs.<condition>.deny`), and B, C and every allocation arm read
   it. D22a gives 1 for the main condition; CAGE-4 has no deny duration, so the robustness value is the owner's.
3. mock_engine and llm_harness `calibrate()`: ignore NaN entries of a statistic. `judge` and `rl_minus_ts` are per
   delivery `[B, N, N]`; alpha is the share of deliveries with a finite statistic that exceed the threshold. Calibrate on
   the uncapped arm with no thresholds set.
4. mock_engine and llm_harness `obs.audit`: when a request exceeds the affordable number, grant a keyed random subset
   instead of the lowest sender indices (`mock_engine/private/engine.py` grants `want.cumsum(1) <= floor(remaining /
   cost)`). Low indices are the hubs of Barabasi-Albert and `star_hub` graphs, so index order audits the most central
   senders first. B and C request every deliverable sender and leave the choice to the platform (BAZT's pipeline relies
   on this), so the fix belongs there.
5. core `EpisodeContext`: add `platform: str` (`"mock"` or `"llm"`), or have the runner resolve
   `conventional.c.judge_score_bins` per platform before it builds the settings (open question 1).
6. LLM platform: supply `Observation.authentic` (False for a delivery altered in transit, as in AiTM) and keep `active`
   False only for isolated agents; `None` means every delivery is authenticated.

## Calibration of C (closed loop)

C is stateful: its own denials raise the network threat level and its histories change TS, so the statistic of an
attack-free run depends on the thresholds C acts on. C supports a closed-loop calibration: it reports `rl_minus_ts` and
acts on `ctx.thresholds` in the same pass, and `reset(ctx)` clears every trace of an earlier pass (tested), so a
platform can replay the same episodes with a candidate threshold on one instance or on fresh ones. The search for the
threshold at deny rate alpha belongs to the platform (`calibration.mode`).

### Tie-breaking jitter of C's statistic

`rl_minus_ts` is a sum of weights that are multiples of one unit `u` (the gcd of the reference weights, the threat table
and `conventional.c.judge_score_bins[platform]`; 1 for integer bins), so many deliveries tie exactly and one threshold
cannot reach alpha inside a step. C therefore adds `(U - 0.5) * 0.9 * u` to every evaluated delivery's statistic, with
`U` uniform in [0, 1) from `ctx.arm_stream(f"jitter|{step}", b)` (one `[N, N]` draw per episode and round, so the value
of delivery (i, j) is keyed by round, sender, receiver and episode, never by the batch). The jitter is below `u / 2`, so
distinct values keep their order; ties are split at the threshold, as in a randomized Neyman-Pearson test. The value
reported in `Actions.statistics` is the value compared with `ctx.thresholds`, so calibration and evaluation use the same
statistic. Funding priority uses it too. `ArmC.jitter = False` removes it (the tests with hand-computed values do).
B's `judge` is not jittered: a judge score is continuous (and one score covers all deliveries of a sender), so it has no
gap to stay inside.
