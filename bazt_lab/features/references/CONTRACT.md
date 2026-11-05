# references: reference baselines (spec 2.7)

Defenders that bound or contextualise BAZT: allocation baselines, the GNN stand-in, and the legacy swarmlab rules with
matched-timing controls. All are `core.Defender`s; none calls a service. Shapes follow `core`: B episodes, N agents,
float64 scores, bool masks, int64 indices.

## public.py

| name | what |
|---|---|
| `ARMS` | tuple of `core.ArmSpec`, 20 arms (table below) |
| `schedule_from_rounds(rounds, T)` | `rounds`: per episode, the 1-based rounds at which BAZT acted. Returns bool `[B, T]` |
| `MATCHED_SUFFIX` | `"_matched"`: the matched-timing control of arm `x` is named `x_matched` |

## Arms

| arm | platforms | needs_truth | calibrated | statistics reported | thresholds read | acts with |
|---|---|---|---|---|---|---|
| `random_alloc` | mock, llm | no | yes | `audit_score` `[B, N]`, NaN where not audited | `audit_score` | edge_quarantine |
| `oracle_alloc` | mock, llm | yes | no | `audit_score` | `audit_score` (supplied by the runner, see Requests) | edge_quarantine |
| `bra_greedy` | mock, llm | no | yes | `audit_score` | `audit_score` | edge_quarantine |
| `GNN_standin` | mock | no | yes | `gnn_standin` = `obs.monitor` | `gnn_standin` | deny |
| `fixed_early`, `fixed_mid`, `fixed_late` | mock | no | yes | `monitor` | `monitor` | isolation |
| `score_only` | mock | no | yes | `monitor` | `monitor` | isolation |
| `centrality_only` | mock | no | no | none | none | isolation |
| `random_capped` | mock | no | no | none | none | isolation |
| `risk_aware` | mock | no | yes | `risk` | `risk` | isolation |
| `risk_aware_tiered` | mock | no | yes | `risk`, `monitor` | `risk`, `monitor` | isolation, sanitize |
| `<legacy arm>_matched` (8) | mock | no | as the arm | as the arm | as the arm | as the arm |

A threshold missing from `ctx.thresholds` means +inf (core contract): the arm collects its statistics and never acts.
Comparisons are strict (`statistic > threshold`).

### Allocation: random_alloc, oracle_alloc, bra_greedy

One pipeline, three placement rules. Each round:

1. Audit allowance. Cumulative allowance `floor(audit_cap / cost_inspect * t / T)`; the round may audit the allowance not
   yet used (unused allowance carries over), capped by the remaining audit budget and by the number of eligible
   senders. Uncapped: every eligible sender. An audit the platform returns as NaN does not count as performed.
2. Eligible senders: active agents with at least one delivery attempt this round.
3. Placement picks the top-k eligible senders by a rule, `obs.audit(mask)` pays one audit unit per sender text.
4. Every audited sender whose `audit_score` is above its threshold is cut (D98): all its outgoing edges are quarantined
   (`edge_quarantine`, persistent), highest score first, while `obs.remaining_intervention` covers
   `cost_edge_quarantine * out_degree(ctx.adj) * (T - t + 1)` (skip a sender that does not fit, continue with the
   next). A cut sender stays blocked for the rest of the episode, is never audited or charged again (state `cut`).

| arm | placement |
|---|---|
| `random_alloc` | uniform random: top-k of iid uniforms pre-drawn from the arm stream `random_alloc.pick`, shape `[T, N]` per episode |
| `oracle_alloc` | only senders infected now (`obs.oracle()`), random order (D96). The oracle says where to look; each verdict still goes through the auditor and the `audit_score` threshold, like the other two. A ceiling for allocation, not for detection. No threshold in `ctx.thresholds`: it audits and never acts |
| `bra_greedy` | BRA-Audit weight `w(i) = AuditGap(i) * TopoImpact(i)`, largest first. `AuditGap = t - last_audit(i)` (last_audit = 0 if never audited). `TopoImpact = 1 + descendants(i)` in the dependency DAG of the episode unrolled over the remaining `T - t` rounds: node `(j, s)` depends on `(j, s-1)` (own memory, D10) and on `(i, s-1)` for every edge `i -> j`. Ties: arm stream `bra_greedy.tie` |

Source of the BRA rule: Wang, Lin, Lou, Li, "BRA-Audit: Budgeted Runtime Auditing for LLM Multi-Agent Systems via
Cumulative-Exposure Audit-Point Placement", arXiv 2608.14668 (Aug 2026), Alg. 1 lines 4 to 5 and Eq. 7. Deviations,
all forced by the experiment contract:

- The paper selects by marginal gain `Delta(v | S)` with recomputation after each pick (Eq. 11 to 12) and per-round
  budget `floor(rho * |Q|)` (Eq. 13). The spec fixes the weight rule and the budget is an episode total, so the arm
  ranks by `w(i)` and paces the total as in step 1.
- An audit here scores the sender's text of this round (`obs.audit`), not the whole unaudited upstream region (Alg. 1
  lines 16 to 27). Rollback and re-execution are not modelled; "block" is the persistent edge quarantine of D98.
- No mandatory final-round audit of the result agent (the mock has no result node).
- Tie-break and the first round are not specified in the paper: ties go to the arm stream; at round 1 every gap is 1.

### GNN_standin

Mock only, a stand-in for the G-Safeguard GNN (not the GNN; the LLM-side `GNN` arm belongs to the gnn feature).
`obs.monitor` is read against `gnn_standin`; every round (D99, the upstream behaviour) each active agent above it has
all its deliveries of that round denied (`deny`), highest monitor first while the intervention budget lasts, priced
`cost_deny` per delivery. Nothing persists: the agent is evaluated again the next round, so the arm has no state.
Class attribute `stand_in = True`.

### Legacy swarmlab rules

Ported from `swarmlab._decide`, `risk_score`, `reach_H`, `centrality`, `window_of`. The selection logic is
unchanged; the budget and thresholds follow the spec:

- Budget: the cell's intervention cap, not a count K. A hard quarantine is `isolation`, priced
  `cost_isolation * out_degree * (T - t + 1)` (out-degree in `ctx.adj`). The walk down a ranking takes every candidate
  that still fits.
- Thresholds: `tau_p` becomes the calibrated `monitor`, `tau_R` and `tau_quar` the calibrated `risk` (spec 2.8
  replaces the J-objective tuning).
- Ranking ties: `sorted(reverse=True)` on `(score, index)` tuples sends equal scores to the higher index
  (`risk_aware`, `risk_aware_tiered` tier 2, `centrality_only`); `sorted(key=-p)` to the lower index (`fixed_*`,
  `score_only`, tier 1).
- None calls the auditor; they read `obs.monitor` (free) and the graph. `centrality_only` and `random_capped` do not
  read the monitor.

| rule | logic |
|---|---|
| `fixed_early/mid/late` | inside rounds `(1, T//3)`, `(T//3+1, 2T//3)`, `(2T//3+1, T)`: isolate active agents with `monitor` above threshold |
| `score_only` | isolate agents with `monitor` above threshold for `confirm_steps` consecutive rounds (state `alarm_run`) |
| `centrality_only` | isolate agents with reach `H > 0`, largest `H` first, `H` = sum over agents first reached at hop d <= min(T - t, dmax) of `gamma^(d-1)` over active agents |
| `random_capped` | at `random_steps` rounds drawn without replacement, isolate one random active agent (the best-ranked of a pre-drawn key that fits the budget) |
| `risk_aware` | `R = p*H - lam*(1-p)*C`, `H` weighted by `(1-p)` per reached agent, `C = 0.5*deg/maxdeg + 0.5*(1-t/T)`; isolate `R > risk` |
| `risk_aware_tiered` | tier 2: isolate `R > risk`. Tier 1 (sanitize this round's deliveries): `p > monitor` and `tau_san <= R`; or `C >= tau_central`, `p > monitor` and `p < tau_confirm`; or a tier-2 candidate the budget cannot isolate. One intervention budget: isolations first, then sanitizations by `R`, lower index first on ties |

### MatchedTiming (`<legacy arm>_matched`)

Wraps the rule built with `own_timing=False` (no window, no random steps). The wrapped rule runs every round, so its
state and statistics keep evolving. At a round not in the episode's schedule the wrapper drops every action mask and
makes `obs.audit` perform and debit nothing (the rule sees NaN, as for an unaffordable audit). Statistics are always
kept, so calibration is not affected. Schedule source: `ctx.timing_schedule` (bool `[B, T]`), else the constructor
argument. No schedule: `ValueError`, except a threshold-driven rule with empty `ctx.thresholds`, which is a
calibration run and receives an empty schedule. `fixed_early_matched`, `fixed_mid_matched` and `fixed_late_matched`
are the same rule (the window was the only difference); they exist so each legacy arm has its control.

## Invariants

- No arm sees ground truth except `oracle_alloc` through `obs.oracle` (`ArmSpec.needs_truth=True`) and audit scores.
- Budgets: every proposed action is priced as the platform prices it and fits `obs.remaining_intervention`; every audit
  fits `obs.remaining_audit`. Charging is done by the platform from the returned `Actions`.
- Every random draw comes from `ctx.arm_stream(namespace, b)`, pre-drawn for all rounds in `reset`. Namespaces:
  `random_alloc.pick`, `oracle_alloc.pick`, `bra_greedy.tie`, `random_capped.when`, `random_capped.who`. Results do not
  depend on B or on chunking.
- `state_dict()` holds tensors only and is sufficient to resume after `reset(ctx)`: `last_audit`, `n_audited`, `cut` (allocation),
  `alarm_run` (score_only), and the inner state under `inner.` (matched).
- `deny` (GNN_standin) covers only deliveries attempted this round; `isolation` and `edge_quarantine` are requested only for agents active at the time.

## Settings read (`core.get`; a missing key raises `core.Unset` naming it)

`mock.legacy.confirm_steps` (score_only), `mock.legacy.gamma`, `mock.legacy.dmax` (centrality_only, risk_aware,
risk_aware_tiered), `mock.legacy.lam` (risk_aware, risk_aware_tiered), `mock.legacy.tau_san`,
`mock.legacy.tau_central`, `mock.legacy.tau_confirm` (risk_aware_tiered), `mock.legacy.random_steps` (random_capped).
Costs from `ctx.costs`: `inspect`, `edge_quarantine` (allocation); `deny` (GNN_standin); `isolation` (legacy);
`sanitize` (tiered). A missing cost raises `core.Unset`.

## Open questions

Decided by the owner: D96 (oracle verdicts through the auditor's ROC), D97 (random_alloc uniform over agents, paced evenly), D98 (flags persistent as edge quarantine), D99 (GNN_standin denies per round). Open:

1. Legacy budget. swarmlab's K hard quarantines and `B_san` sanitizations are replaced by one intervention budget; a
   hard quarantine is `isolation` (swarmlab removes the node from `active`, which is isolation, not an outgoing cut).
2. Legacy parameters. `confirm_steps`, `lam`, `gamma`, `dmax`, `tau_san`, `tau_central`, `tau_confirm` and the number of
   random steps have no value in the spec (D0e). `swarmlab.Config` uses 2, 1.0, 0.7, 3, 0.05, 0.75, 0.85 and K = 3.
   `tau_san` stays a fixed band bound; only the `monitor` and `risk` thresholds are calibrated at alpha.
3. Calibration of windowed rules. `fixed_*` report `monitor` at every round; whether the alpha deny rate counts all
   rounds or only the window is a platform decision.
4. Matched-timing schedules. How they are recorded from BAZT runs is open. Proposal: for each (cell, seed) run BAZT
   first; the schedule of the unit is the set of rounds `t` with `trace.spend_intervention[b, t-1] > 0` (rounds where
   BAZT acted), optionally also rounds with `spend_audit > 0`; the runner stores it as `unit["timing_schedule"]`
   (list of 1-based rounds) and the platform turns it into `ctx.timing_schedule` with `schedule_from_rounds`. Open
   inside the proposal: audit-only rounds, and `BAZT` or `BAZT_upstream` as the source.
5. Isolation pricing. The arms assume `cost * out_degree(ctx.adj) * (T - t + 1)` (spec 5, D22). If the platform
   counts only live deliveries it charges less and the arms under-spend; if it charges more, it rejects the episode's
   actions as a budget violation.
6. `oracle_alloc` threshold. Its own attack-free run audits nobody, so the `audit_score` threshold must be the one
   calibrated for `random_alloc` in the same cell (Requests 6).

## Requests (for the coordinator; this feature edits no other file)

1. core `EpisodeContext`: add `timing_schedule: Optional[Tensor] = None` (bool `[B, T]`). The platform fills it from
   `unit["timing_schedule"]` with `references.public.schedule_from_rounds`. The matched arms read it with
   `getattr(ctx, "timing_schedule", None)`, so they work before and after the change.
2. `experiments/profiles.toml`: add under `[mock.legacy]` the keys `confirm_steps`, `lam`, `gamma`, `dmax`, `tau_san`,
   `tau_central`, `tau_confirm`, `random_steps` (values are the owner's decision, see question 6).
3. `experiments/profiles.toml`: `mock.calibrated_arms` must list every arm with `calibrated=True` that the grid runs
   (`GNN_standin`, `random_alloc`, `bra_greedy`, `fixed_early`, `fixed_mid`, `fixed_late`, `score_only`, `risk_aware`,
   `risk_aware_tiered` and their `_matched` controls), and `mock.uncapped_arms` must list them for the calibration
   block. The runner refuses to evaluate a calibrated arm without thresholds, which is the intended failure.
4. mock_engine: `calibrate()` must ignore NaN entries of a statistic (unaudited senders); a `[B, N]` statistic stands for
   the agent's outgoing deliveries. `Observation.attempts` must exclude edges blocked by `edge_quarantine` and agents
   removed by `isolation`. `Observation.oracle` must be supplied when `ArmSpec.needs_truth` is true.
5. runner: arm names of the legacy block are the keys of `ARMS`; the matched controls are `<arm>_matched`.
6. runner: `oracle_alloc` is `calibrated=False`. For each cell the runner must pass `random_alloc`'s calibrated
   `audit_score` as `ctx.thresholds["audit_score"]` of `oracle_alloc` (D96); without it the oracle audits and never acts.
