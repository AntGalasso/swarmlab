# bazt_slot: the BAZT arms (decision D86, spec 2.5 and 2.6)

## public.py

`ARMS` = `BAZT`, `BAZT_no_budget`, `BAZT_no_topology`, `BAZT_no_graduated` (all `calibrated=True`, platforms mock and llm) and
`BAZT_upstream` (all five `calibrated=True`, D107: each is calibrated at alpha on its own uncapped attack-free run, the ablations included). `needs_truth` is False everywhere: no arm reads ground truth except through `obs.audit`.

## BAZT (spec 2.5), `private/`

A round: (1) free evidence (the monitor, or the prior `bazt.prior` on the LLM side) plus a decaying memory of paid verdicts
give each agent a posterior r; (2) audits are bought by value of information (`pricing.inspect_value`, the form of
`bazt.actions.score_inspect`); (3) the A+B+C pipeline (`pipeline.py`, the conventional feature's public `ARMS`) sees the
bought scores and reports its statistics; a flagged or cleared sender updates the memory; (4) the eight actions are priced
and bought within both remaining budgets. Calibration (no thresholds): every sender is audited, nothing is done, statistics
`judge`, `rl_minus_ts` (from B and C) and `bazt_risk` (free monitor log-odds, per delivery; absent without a monitor) are reported.

| component | module | method |
|---|---|---|
| enforcement budget | `select.py` | Hua et al. 2507.15886 section 3.1-3.2: a purchase's benign price (1 - P(warranted)) x cost x rounds it recurs; purchases ordered by likelihood ratio per benign price (`np_order`); the cut is where the cumulative benign price reaches the remaining budget; then value-density greedy under the hard cap (`fill`, stable argsort + cumsum, equal to the sequential skip-and-continue greedy) |
| topology-aware contagion | `contagion.py`, `pricing.py` | truncated Katz by matvec recursion (`order - 1` bmm, never matrix powers); NetShield eigen-drop of a node and NetMelt/Kimura edge value by batched power iteration on A + I (warm-started across rounds, DAG detected by the residual); remaining horizon `T - t` hops |
| graduated selection | `pricing.py`, `bazt.py` | net = expected loss reduction - utility_weight x (1 - r) x rho x centrality share (Poolsappasit 2012 section 5, Lee 2000 cost classes); best kind per target by density, ties to the lighter action |

Ablations: `no_budget` plans as if both budgets were infinite (meant for the uncapped cell); `no_topology` values every agent as one
unit at stake and ignores the graph; `no_graduated` considers isolation only (no reinstatement).

### Differences between upstream `bazt` and the spec variant
- Reach: upstream `swarmlab.reach_H` (BFS, path decay gamma, at most `dmax` hops, 0/1 reach); here Katz walks with `bazt.katz_decay`,
  over the whole remaining horizon, with susceptibility `(1 - r) w` and the sink weighted by `bazt.sink_weight`.
- Structure: upstream has none; here the eigen-drop share (node and edge) of the spread that does not start at the target.
- Budget: upstream pools one cost unit per action with a pacing threshold `theta0 exp(kappa z)`; here costs are per delivery
  (spec section 5) and pacing is the Neyman-Pearson cut on benign price.
- Posterior: upstream fuses provenance taint and conclusive oracle verdicts (`inspect_conclusive`); here no provenance (the
  spec stack has none) and a verdict is the A+B+C pipeline's flag on a bought score.
- Actions: eight actions (delay, rate limit, reinstatement are new; edges are charged per remaining delivery).
- Utility loss `rho` per action is a setting; upstream has `rho_san`, `rho_res` and `C/deg`.

## BAZT_upstream, `adapters/upstream.py`

Drives `bazt.optimizer.BudgetedOptimizer.plan` with a `Situation` built per episode (Python loop over b), `bazt.ledger.BudgetLedger`
(pooled, persistent across rounds so pacing works) and `bazt.evidence.InspectionState`. Mapping: INSPECT -> `obs.audit` on one
sender (verdict = score over `ctx.thresholds['upstream_judge']`, +inf if absent), SANITIZE -> sanitize on the node's attempted deliveries,
RESTRICT -> privilege_degradation, BLOCK_EDGE -> edge_quarantine, ISOLATE -> isolation. Upstream defaults are kept for everything
not listed (gamma, dmax, lam, rho, theta0, kappa, q_inspect unused). Upstream costs are the main condition's units rounded
(>= 1); the ledger capacity is the intervention cap in those units, and each planned action is then re-priced with the platform's
rule and dropped if the round would exceed the remaining budget (so the platform never sees an overspend). Difference: RESTRICT
is persistent (upstream expires after `ttl_restrict` steps) and no privileged-role or provenance input exists (`privileged` is empty).

## State and calibration passes
`reset(ctx)` rebuilds every piece of state (memory, persistent masks, eigenvector warm start, exhaustion latch, the inner A/B/C arms, upstream ledgers and verdicts), so a second calibration pass on the same instance equals a fresh arm (tested). Each variant reports `bazt_risk` (and B's `judge`, C's `rl_minus_ts`); upstream reports `upstream_judge`.

## Tie-breaking jitter (`private/jitter.py`)
Every reported statistic (`judge`, `rl_minus_ts`, `bazt_risk`, `upstream_judge`) carries a keyed jitter so a discrete statistic
(C's integer-weight sums) splits its ties at the calibrated threshold instead of landing far below alpha (a randomized
Neyman-Pearson test). The jitter is uniform in (-scale/2, scale/2) with `scale = min(2^-24, gap/4)` per episode and call, where
`gap` is the smallest gap between distinct finite values of that episode's statistic: it is strictly below half the smallest gap,
so the order of distinct values never changes. One uniform per (round, sender, receiver) from
`ctx.arm_stream("jitter|<statistic>|<round>", b)`; `bazt_risk` and `upstream_judge` are sender-level and get one per (round,
sender). Deterministic, depends on the episode's own key only (tested for batch independence). BAZT compares the jittered values
with the thresholds (flags, eligibility), so the calibrated statistic is the one it acts on. `upstream_judge`'s verdict test adds
the same draw at the fixed scale 2^-24 (the gap guard cannot be applied before the round's audits are all in).

## Episode clock
`obs.t`, `obs.T`, `ctx.T` count the whole episode (the LLM seed run, D54). Pricing (hops, rounds left), the pacing of the
budgets and the upstream verdict TTL all use that clock; `Observation.round_in_conversation` is not read.

## Documented behaviour
- `pricing.centrality` uses base-graph degrees (`ctx.adj`), not the live graph: the share of the busiest sender's degree is a
  property of the deployment, and stays comparable after quarantines and isolations.
- `BAZT_no_budget` plans with infinite budgets, so `reset` refuses a context with a finite audit or intervention cap
  (`ValueError`): it is meant for the uncapped cell.
- An isolated agent neither sends nor receives, so BAZT drops edge actions on its edges from the round it isolates it, and its
  persistent-state mirror follows the engine's order (reinstatement, isolation, new quarantines / degradations / rate limits,
  none applied to an agent isolated in the same Actions).
- BAZT_upstream, upstream behaviour difference: its pooled `BudgetLedger` has one pool of cost units, capacity = the intervention
  cap, and `inspect` charges that pool too. Every audit therefore debits the intervention budget in upstream units (one per
  audit, the main condition) in addition to the audit budget of the platform; the platform never sees this debit and the
  adapter's own re-pricing keeps real charges within the remaining intervention budget. In `bazt` itself pooled mode is the
  intended design; the platform's two budgets are not modelled by it. Expect BAZT_upstream to exhaust its pool earlier than
  its intervention spend alone would suggest.

## Invariants
- Budgets never exceeded (tested against a platform fake that raises on overspend, 8 seeds x 3 variants, both quarantine charging
  readings, robustness costs). No random draw anywhere; results do not depend on batch size (tested); `state_dict` holds tensors
  and plain values and a resumed run continues exactly (BAZT).
- Defenders never see ground truth: only `attempts`, `active`, `authentic`, `monitor`, budgets and `obs.audit`.
- float64 scores, bool masks, int64 indices.

## Settings keys read (`core.get`; a missing key raises `core.Unset` naming it)
`bazt.katz_decay` (0,1], `bazt.eigen_iterations` int >= 1, `bazt.sink_weight` >= 1, `bazt.utility_weight` >= 0,
`bazt.utility_loss.{sanitize,delay,edge_quarantine,rate_limit,privilege_degradation,isolation}` >= 0, `bazt.flag_logodds` > 0,
`bazt.clear_logodds` > 0, `bazt.memory_decay` [0,1), `bazt.prior` (0,1), `bazt.inspect_conclusive` (0,1], `runtime.sanitize_success`,
`runtime.rate_limit_per_identity`; costs `inspect` and the seven action kinds from `ctx.costs`;
via the conventional arms `conventional.c.*`.

## Open questions
1. Values of every `bazt.*` key are the owner's (D0e); the tests use small test values only.
2. `bazt_risk` is calibrated at alpha like B and C, but it is the free monitor, so it only gates eligibility; whether eligibility
   should be calibrated at all is a design choice.
3. (closed) BAZT_upstream is calibrated (D107): it reports `upstream_judge` [B,N,N] (the audit score of the sender, NaN where not audited); with no thresholds it audits every sender and acts on nothing.
4. Platform charging of `edge_quarantine` (once or per remaining delivery) is priced as per remaining delivery, the dearer reading.

## Requests
1. `experiments/profiles.toml`: add the unset `[bazt]`, `[bazt.utility_loss]` and `[bazt_upstream]` keys above, and `deny` costs (see conventional).
2. Runner required-key check: every block running BAZT, a BAZT ablation or BAZT_upstream needs those keys plus the conventional ones.
3. `mock_engine`/`llm_harness` `calibrate()`: ignore NaN entries per delivery, and accept a `[B, N, N]` statistic (`bazt_risk`) next to B's and C's.
4. Same as conventional request 4: grant `obs.audit` over capacity by a keyed random subset, not by index.
