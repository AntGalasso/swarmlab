# bazt_slot: the BAZT arms (decision D86, spec 2.5 and 2.6)

## public.py

`ARMS` = `BAZT`, `BAZT_no_budget`, `BAZT_no_topology`, `BAZT_no_graduated` (one `private.bazt.Bazt` per entry of `VARIANTS`)
and `BAZT_upstream` (`adapters.upstream.BaztUpstream`). All five are `calibrated=True` (D107: each is calibrated at alpha on its
own uncapped attack-free run, the ablations included) and run on platforms mock and llm (the `ArmSpec` default). `needs_truth`
is False everywhere: no arm reads ground truth except through `obs.audit`.

## BAZT (spec 2.5), `private/`

A round: (1) free evidence (the monitor, or the prior `bazt.prior` on the LLM side; an isolated agent keeps its last, D119)
plus a decaying memory of paid verdicts give each agent a posterior r; (2) audits of senders are bought by value of information
(`pricing.inspect_value`, the form of `bazt.actions.score_inspect`), then (2b) audits of isolated agents from what the senders
left (D119, see Documented behaviour); (3) the A+B+C pipeline (`pipeline.py`, built from the conventional feature's public
`ARMS`) sees the bought scores through a cached `obs.audit` that debits nothing and an unlimited intervention budget, and
reports its statistics; BAZT keeps A's denies and B's and C's statistics, and a flagged or cleared sender updates the memory;
(4) the posterior is recomputed and the seven actions other than INSPECT are priced and bought within both remaining budgets.
Calibration (no thresholds): every sender is audited, nothing is bought (A's free denies still apply), statistics `judge`,
`rl_minus_ts` (from B and C) and `bazt_risk` (free monitor log-odds, per delivery; absent without a monitor) are reported.

| component | module | method |
|---|---|---|
| enforcement budget | `select.py` | Hua et al. 2507.15886 section 3.1-3.2: a purchase's benign price (1 - P(warranted)) x cost x rounds it recurs (`benign_price`); purchases ordered by likelihood ratio per benign price (`np_order`); the cut is where the cumulative benign price reaches the remaining budget (`choose`); then value-density greedy under the hard cap (`fill`, stable argsort + cumsum, equal to the sequential skip-and-continue greedy) |
| topology-aware contagion | `contagion.py`, `pricing.py` | truncated Katz by matvec recursion (`order - 1` recursion steps of bmm, never matrix powers) with the decay `bazt.katz_decay / rho(ctx.adj)` of each episode (D115, `contagion.relative_decay`); NetShield eigen-drop of a node and NetMelt/Kimura edge value by batched power iteration on A + I (`contagion.leading_pair`, warm-started across rounds, DAG detected by the residual); remaining horizon `T - t` hops |
| graduated selection | `pricing.py`, `bazt.py` | net = expected loss reduction - utility_weight x (1 - r) x rho x centrality share (Poolsappasit 2012 section 5, Lee 2000 cost classes); best kind per target by density (`Bazt._best`), ties to the lighter action |

Ablations: `no_budget` plans as if both budgets were infinite, without the Neyman-Pearson cut (meant for the uncapped cell);
`no_topology` values every agent as one unit at stake and ignores the graph; `no_graduated` considers isolation only (no
reinstatement, so no audit of isolated agents).

### Differences between upstream `bazt` and the spec variant
- Reach: upstream `swarmlab.reach_H` (BFS, path decay gamma, at most `dmax` hops, 0/1 reach); here Katz walks with the decay
  `bazt.katz_decay / rho` (D115), over the whole remaining horizon, with susceptibility `(1 - r) w` and the sink weighted by
  `bazt.sink_weight`.
- Structure: upstream has none; here the eigen-drop share (node and edge) of the spread that does not start at the target.
- Budget: upstream pools one cost per action kind (`cost_inspect` ... `cost_isolate`) with a pacing threshold
  `theta0 exp(kappa z)`; here costs are per delivery (spec section 5) and pacing is the Neyman-Pearson cut on benign price.
- Posterior: upstream fuses provenance taint and oracle verdicts that are conclusive with probability `q_inspect`; here no
  provenance (the spec stack has none), a verdict is the A+B+C pipeline's flag on a bought score, and
  `bazt.inspect_conclusive` only scales the value of an audit (`pricing.inspect_value` and the audit of an isolated agent).
- Actions: eight actions (delay, rate limit, reinstatement are new; edges are charged per remaining delivery).
- Utility loss `rho` per action is a setting; upstream has `rho_san`, `rho_res` and `C/deg`.

## BAZT_upstream, `adapters/upstream.py`

Drives `bazt.optimizer.BudgetedOptimizer.plan` with a `Situation` built per episode (Python loop over b), `bazt.ledger.BudgetLedger`
(pooled, persistent across rounds so pacing works) and `Verdicts`, a `bazt.evidence.InspectionState` whose conclusive
probability is `bazt_upstream.verdict_probability` instead of the module constants. Mapping: INSPECT -> `obs.audit` on one
sender (verdict = the score with its keyed jitter, `jitter.at_own_scale`, over `ctx.thresholds['upstream_judge']`, +inf if
absent; a positive verdict sets p' = `verdict_probability`, a negative one 1 - p'; an ungranted audit (NaN) gives the planner
no verdict), SANITIZE -> sanitize on the node's attempted deliveries, RESTRICT -> privilege_degradation, BLOCK_EDGE ->
edge_quarantine, ISOLATE -> isolation. Its `deny` is A's rule: every attempt off `ctx.adj & obs.authentic`. Upstream defaults
are kept for everything not listed (`gamma`, `dmax` of `swarmlab.Config`; `lam`, `rho_san`, `rho_res`, `theta0`, `kappa`,
`q_inspect`, `inspect_ttl` of `BaztConfig`); `q_inspect` only enters the value of an INSPECT candidate, since a granted audit
always gives a verdict. Upstream costs are the unit's `ctx.costs` (main or robustness condition) rounded to integers >= 1
(`_units`); the ledger capacity is the intervention cap in those units (`unlimited` when uncapped), and each planned action is
then re-priced with the platform's rule and dropped if the round would exceed the remaining budget (so the platform never sees
an overspend). Difference: RESTRICT is persistent (upstream expires after `ttl_restrict` steps) and no privileged-role or
provenance input exists (`privileged` is empty). Calibration (no thresholds): every attempting sender is audited, nothing is
done and `upstream_judge` [B, N, N] is reported (the jittered audit score of the sender on each of its attempted deliveries,
NaN elsewhere); in an evaluation round it is reported only when some audit was granted.

## State and calibration passes
`reset(ctx)` rebuilds every piece of state (memory, the held free evidence of D119, persistent masks, eigenvector warm start,
exhaustion latch, the inner A/B/C arms, upstream ledgers, verdicts and verdict-driven masks), so a second calibration pass on
the same instance equals a fresh arm (tested). Each BAZT variant reports `bazt_risk` (with a monitor) and B's `judge`, C's
`rl_minus_ts`; upstream reports `upstream_judge`.

## Tie-breaking jitter (`private/jitter.py`)
Every reported statistic (`judge`, `rl_minus_ts`, `bazt_risk`, `upstream_judge`) carries a keyed jitter so a discrete statistic
(C's integer-weight sums) splits its ties at the calibrated threshold instead of landing far below alpha (a randomized
Neyman-Pearson test). The jitter is uniform in (-scale/2, scale/2) with `scale = min(top, gap/4)` per episode and call, where
`gap` is the smallest gap between distinct finite values of that episode's statistic (for a sender-level statistic, of its
per-sender values) and `top = max(2^-24, core.tie_jitter_scale(dtype, M))`, M the largest |value| (at least 1): the dtype term
keeps the jitter above eight ulp in float32. It is strictly below half the smallest gap, so the order of distinct values never
changes. Draws: one batched `core.keyed_uniform` over all episodes, keyed by `core.episode_keys(ctx.unit_ids)`, stream
`bazt|<arm>|jitter|<statistic>`, counter = the round; one uniform per (sender, receiver), per sender for the sender-level
statistics (`judge`, `bazt_risk`, `upstream_judge`). A draw depends on the episode key, arm, statistic, round and element only:
not on the batch, the device or the order. BAZT compares the jittered values with the thresholds (flags, eligibility).

`upstream_judge` is the exception to the gap guard (`jitter.at_own_scale`): its verdict is decided audit by audit, before the
round's other scores exist, so the scale is `top_scale(dtype, |score|)` of the score itself (2^-24 in float64, eight ulp in
float32), with the same keyed draw (one per sender and round, in `ctx.dtype`). The verdict compares exactly the value reported,
bit for bit (elementwise IEEE operations), so the threshold calibrated on the statistic controls the verdict rate in float32 as
in float64 (D117: every run is float32). Without a gap guard two distinct scores closer than that scale may swap; ties keep
identical scales.
Resolution in float32: a tied value takes about 13 distinct jittered values (eight ulp of M over the float32 grid), so a
threshold inside a tie of mass m moves the rate in steps of about m / 13; on the tie-heavy test statistic the fresh rate stays
within 0.004 of alpha over alpha in [0.01, 0.15] in both dtypes (sampling noise of that test is about 0.003).

## Compute: device, dtype
Tensors are created on `ctx.device`; scores, statistics, posteriors, net values and Katz/eigen values follow `ctx.dtype`;
budgets, costs, spend and the upstream ledgers stay float64 / exact (tested in float32: accounting dtype and no overspend).
float32 agrees with float64 within a documented tolerance: the number of actions of an episode differs by at most 25% (min 3)
on the random scenarios of the tests; the actions themselves can differ at near-ties. CUDA is exercised by a test that is
skipped when no device is present. BAZT_upstream keeps its scalar Python planner (the upstream package); its own tensors (the
keyed jitter draw, the per-audit request buffer, its masks) live on `ctx.device`. `core.compiled` is not used: no kernel here
was worth the compile risk. `_best` keeps a running maximum over the action kinds instead of a [K, B, N, N] stack and argmax.
The spectral radius behind the Katz decay is computed in float64 from the bool base graph (once per episode), then the decay
is cast to `ctx.dtype`, so float32 and float64 runs use the same decay up to rounding. Batch independence is bitwise as long
as CPU bmm is: with MKL it is at N = 12 (tested), but from about N = 30 a batched bmm differs from a single-episode one in the
last ulp, for every bmm of the lane (Katz and power iteration); values then agree to about 1e-15 relative, and only an exact
price tie can break differently.

## `Actions.threshold_restricted` (D108: alpha counts only deliveries a calibrated statistic restricted)
Both arms set it every round (all-False while calibrating).
- BAZT: every priced restriction is gated by eligibility, i.e. `bazt_risk` (free evidence plus the verdict memory) over its
  threshold or a standing verdict from a flag of the pipeline's `judge` / `rl_minus_ts` (or of B's rule on the audit of an
  isolated agent). Counted: this round's sanitize and delay deliveries, the edge quarantines in force, every delivery of a
  degraded sender, every delivery to or from an isolated agent (all masked with `ctx.adj`). Not counted: A's denies (BAZT's
  `deny` is A's rules only; it does not use B's or C's denies), the persistent rate limit, and anything that is not an action
  of the arm. A reinstatement removes the agent's persistent state from the mask.
- BAZT_upstream (D113): its actions are priced from the monitor probability, not gated by a threshold, so only restrictions
  that rest on an audit verdict over the calibrated `upstream_judge` threshold (p' = `bazt_upstream.verdict_probability`, a
  live positive verdict at the round of the action) count: that round's sanitize and the quarantines, restrictions and
  isolations decided on such a verdict, which persist. A restriction the monitor alone drove is not counted. Upstream's
  restrict maps to privilege_degradation (no rate limit).

## Episode clock
`obs.t`, `obs.T`, `ctx.T` count the whole episode (the LLM seed run, D54). Pricing (hops, rounds left), the pacing of the
budgets and the upstream verdict TTL all use that clock; `Observation.round_in_conversation` is not read.

## Eligibility, jitter level, settings the owner must respect
- A sender is eligible to be restricted when `bazt_risk` (free evidence plus the verdict memory) is over its calibrated threshold,
  or when its verdict still stands: the decayed memory exceeds `bazt.clear_logodds`, the weight of a clearing verdict
  (`risk.eligible`). One flag of weight `flag_logodds` decays by `memory_decay` per round and lapses after a few rounds (4,
  decay 0.8, hold 3: two rounds); an audit that clears the sender subtracts `clear_logodds` (in that example it ends the
  verdict at once). `flag_logodds <= clear_logodds` is refused (a flag could never stand).
- Sender-level statistics (`judge`, `bazt_risk`, `upstream_judge`) get one jitter per (round, sender) so that "any delivery of the
  sender over the threshold" has the delivery-level rate alpha; `rl_minus_ts` is per delivery and gets one per delivery.
- **`bazt.katz_decay` is relative (D115): a fraction in (0, 1) of 1 / spectral radius.** The Katz sum
  H = sum_k d^(k-1) A^k s powers the directed adjacency A[b, i, j] = 1 for i -> j, so it converges exactly when
  d x rho(A) < 1. BAZT uses on episode b the decay d_b = katz_decay / rho(ctx.adj[b]), the radius of its base graph (directed;
  the grid graphs are symmetric, so directed and undirected coincide), computed once at `reset` in float64 by the lane's
  batched power iteration (`contagion.relative_decay` over `contagion.leading_pair`, `bazt.eigen_iterations` iterations;
  `pricing.katz_decay`). Every matrix the Katz sum runs on (the live transmission matrix of a round, the graph a reinstatement
  restores) is entrywise at most the base graph, so its radius is at most rho(ctx.adj) (Perron-Frobenius) and d x its radius
  <= katz_decay < 1: the reach stays bounded over any horizon, and the decay does not move when an action cuts an edge. Where
  the power iteration finds no eigenpair (a graph without a cycle, rho = 0, or too few iterations), the out-degree bound
  max_i sum_j A_ij >= rho stands in; the divisor is never below 1 (a 0/1 graph with a cycle has rho >= 1), so an empty graph
  gives d = katz_decay, never NaN. Examples: star_hub N=12 has rho = sqrt(11) = 3.32, so katz_decay 0.5 gives d = 0.15; a
  Barabasi-Albert graph of 100 agents has rho of about 6.5 (m = 2) to 9 (m = 3). `params` refuses a value outside (0, 1)
  with that reason; a missing one raises `core.Unset`. `BAZT_no_topology` validates the key with the others (`params.read`)
  but never computes or uses the decay; BAZT_upstream does not read `bazt.katz_decay` (or `params`) at all.
- BAZT_upstream: `bazt_upstream.verdict_probability` is the p' of a conclusive positive verdict (a negative one is 1 minus it).
  Upstream's source for 0.98 / 0.02 is `bazt.evidence.P_POSITIVE` / `P_NEGATIVE` (BAZT_SPEC_v2 section 4). The adapter's ledger is
  refunded for every planned action it drops because the platform price exceeds the remaining budget, so the pool holds only
  what was emitted (inspections excepted, see below).

## Documented behaviour
- `pricing.centrality` uses base-graph degrees (`ctx.adj`), not the live graph: the share of the busiest sender's degree is a
  property of the deployment, and stays comparable after quarantines and isolations.
- `BAZT_no_budget` plans with infinite budgets, so `reset` refuses a context with a finite audit or intervention cap
  (`ValueError`): it is meant for the uncapped cell.
- An isolated agent neither sends nor receives, so BAZT drops edge actions on its edges from the round it isolates it, and its
  persistent-state mirror follows the engine's order (reinstatement, isolation, new quarantines / degradations / rate limits,
  none applied to an agent isolated in the same Actions).
- Edge quarantine is priced per delivery it removes over the remaining rounds (cost x (T - t + 1)); both platforms charge it
  that way (`mock_engine/private/enforce.py` `_apply`, `llm_harness/private/enforce.py` `_pass`).
- `budget_exhausted` is raised once per budget and episode when what is left after the round cannot pay the cheapest thing that
  budget buys (`Bazt._exhausted`: one audit, the cheapest action kind with a positive cost); BAZT_upstream raises it for the
  intervention budget when its pool holds less than one unit.
- BAZT_upstream, upstream behaviour difference: its pooled `BudgetLedger` has one pool of cost units, capacity = the intervention
  cap, and `inspect` charges that pool too. Every audit therefore debits the intervention budget in upstream units
  (the `inspect` cost through `_units` per audit, 1 in both cost conditions of `experiments/profiles.toml`) in addition to the audit
  budget of the platform; the platform never sees this debit and the adapter's own re-pricing keeps real charges within the
  remaining intervention budget. In `bazt` itself pooled mode is the default (`BaztConfig.budget_mode`); the platform's two
  budgets are not modelled by it. Expect BAZT_upstream to exhaust its pool earlier than its intervention spend alone would
  suggest.

- Reinstatement after a quarantine: the reach of the reinstated agent is taken on the graph where it sends again, its own
  quarantined out-edges restored (first order: H_i + sum_j q_ij R_j on `adj & ~quarantined`), and the quarantined
  deliveries into it that the lift reopens are charged at d R_i r_k per edge k -> i (without topology, one unit at stake in
  place of d R_i), the complement of what quarantining them removed. Without the restored edges an agent whose out-edges are
  all quarantined would have no reach and a receiver of a quarantined edge no risk, so each would be reinstated for its
  utility alone. A reinstatement needs no eligibility: it is valid for a restricted agent (isolated, degraded, rate-limited or
  with a quarantined edge) when its net is positive, and costs `reinstatement` x base out-degree x rounds left (an upper bound
  on the platform's charge).
- Reinstatement gives up the structural share (D118): lifting a restriction also gives up the NetShield / NetMelt gain it
  was bought for, from one extra power iteration per round (`bazt.eigen_iterations`, cold start, nothing to checkpoint;
  `pricing._restored_structure`) on the restored graph `adj & ~quarantined`, where every isolated agent sends again, run only
  while some agent of the batch is isolated or has a quarantined edge (degradation and rate limit carry no structural share in
  their prices). Isolated agent i: node_rel_i x others_i; each quarantined edge k -> j out of or into the agent:
  v_k u_j / (lam v.u) x others_k (the edge's first-order NetMelt value, the edge being absent from the restored graph).
  others_k is the spread of every other agent on the same restored graph, isolated agents included, as the Katz reach of
  that graph counts them. With one restricted agent, isolating agent i is worth exactly minus reinstating it in the same
  world (tested for r in [0.1, 0.95]), so at most one of the two pays; with several isolated at once, each lift is priced as
  if the others were restored too (conservative: their spread still counts). An episode with neither an isolated agent nor a
  quarantined edge gets 0, so the term does not depend on the batch.

- Evidence model of an isolated agent (D119): silence is not evidence of being clean. An isolated agent sends nothing, so
  it has no monitor reading; it keeps `held`, the free log-odds of its last active round (the prior if it never was
  active; no monitor: the prior throughout), instead of falling back to the prior (`risk.free_logodds`). `held` is raw
  evidence before the tie-break jitter, in `state_dict`, so a resume mid-isolation is exact. A lift of an isolated agent is
  valid only in a round where an audit cleared it (`risk.cleared`: audited and not flagged, the verdict `remember` writes as
  `-clear_logodds`); the pricing then decides whether the lift pays (the clear verdict lowers its posterior over the
  following rounds as it decays). The audit is bought from the audit budget after the senders' (step 2b): candidates are the
  isolated agents, valued q (1 - r) lambda rho_isolation centrality, bought by density and paced by the Neyman-Pearson cut
  with 1 - r as the probability the audit is warranted, within what the senders left. An isolated agent has no delivery for
  the pipeline to judge (its B statistic is NaN), so BAZT applies B's own rule to the score: flagged when it exceeds the
  calibrated `judge` threshold; without a finite `judge` threshold nothing can clear and no such audit is requested.
  Degradation, rate limit and quarantine lifts are unchanged (the agent still sends). The ablations that isolate follow the
  same code (`BAZT_no_graduated` has no reinstatement, so it keeps its isolations for good). Platforms (D123): BAZT requests these
  audits through `obs.audit_isolated` when the platform offers it (both do): the mock scores the isolated agent from its
  carried state with the same auditor model, the LLM platform judges the text it still wrote that round (none, none
  charged, when it wrote nothing); an entry that is not granted is NaN and not charged, so such an agent stays isolated.
  The plain `obs.audit` never grants an isolated agent, so a context without `audit_isolated` (a test double) falls back to
  it and the agent stays isolated until the episode ends; the call is skipped when no isolated agent is requested. `BAZT_upstream` never reinstates and reads the platform's monitor
  (the mock holds the last reading of an isolated agent, the LLM side has none), so it does not flap.

## Invariants
- Budgets never exceeded (tested against a platform fake that raises on overspend, 8 seeds x 3 variants, both quarantine charging
  readings, robustness costs). No generator draw anywhere (`ctx.arm_stream` is not used): the only random numbers are the keyed
  jitter draws, deterministic per episode key; results do not depend on batch size (tested); `state_dict` holds tensors and
  plain values and a resumed run continues exactly (BAZT).
- Defenders never see ground truth: only the clock, `attempts`, `active`, `authentic`, `monitor`, budgets and `obs.audit`.
- Scores and statistics in `ctx.dtype`, accounting in float64, bool masks, int64 indices.

## Settings keys read (`core.get`; a missing key raises `core.Unset` naming it)
BAZT and its ablations (`params.read`, at `reset`): `bazt.katz_decay` (0,1) fraction of 1 / spectral radius (D115),
`bazt.eigen_iterations` int >= 1, `bazt.sink_weight` >= 1, `bazt.utility_weight` >= 0,
`bazt.utility_loss.{sanitize,delay,edge_quarantine,rate_limit,privilege_degradation,isolation}` >= 0, `bazt.flag_logodds` > 0,
`bazt.clear_logodds` > 0 (and below `flag_logodds`), `bazt.memory_decay` [0,1), `bazt.prior` (0,1), `bazt.inspect_conclusive`
(0,1], `runtime.sanitize_success` [0,1], `runtime.rate_limit_per_identity` int >= 1; costs `inspect` and the seven action kinds
(`delay`, `sanitize`, `edge_quarantine`, `rate_limit`, `privilege_degradation`, `isolation`, `reinstatement`) from
`ctx.costs` (`pricing.unit_cost`, `core.Unset` names `costs.<main|robustness>.<kind>`); via the inner conventional arms
`conventional.c.judge_score_bins.<platform>`, `conventional.c.presence_window_rounds` (C), `runtime.rate_limit_per_identity`
(B) and costs `inspect` and `deny` (B and C).
BAZT_upstream (at `reset`): `bazt.prior`, `runtime.sanitize_success` (swarmlab's `p_strip`; the adapter checks no range for
either), `bazt_upstream.verdict_probability` (0.5, 1); costs `inspect`, `sanitize`, `privilege_degradation`, `edge_quarantine`,
`isolation` from `ctx.costs` (a missing one raises `KeyError`).
In `experiments/profiles.toml` every `bazt.*`, `bazt.utility_loss.*` and `conventional.c.*` key and both `runtime` keys above
are unset (commented); `bazt_upstream.verdict_probability` is 0.98; `[costs.main]` sets every kind, `[costs.robustness]` every
kind but `deny`.

## Open questions
1. Values of every `bazt.*` key are the owner's (D0e); the tests use small test values only.
2. Pacing binds on recurring purchases only: a persistent action (quarantine, isolation, degradation, rate limit) is charged
   once for every remaining delivery and recurs once, so its benign price (1 - p) x cost is below its cost and only the hard
   cap stops it; a tight budget can go entirely to persistent quarantines in round 1. "The budget lasts the horizon" holds
   in benign-price terms.

## Requests
None open.
