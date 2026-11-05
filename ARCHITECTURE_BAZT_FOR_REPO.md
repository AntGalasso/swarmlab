# ARCHITECTURE_BAZT_FOR_REPO.md — BAZT architecture for the swarmlab_v3 repo

Branch: `feat/kunal-env-ladder` + BAZT integration. BAZT is an adaptive gateway enforcement
layer plugged into `swarmlab.run_episode` via `EpisodeHooks.gateway_factory` (P3). It is
separate from, and does not modify, the pre-existing simulator pipeline.

## Data flow

```text
  swarmlab.run_episode(policy, cfg, agent, decoys, hooks=EpisodeHooks)
                                                      │
  policy ∈ BAZT_POLICIES? ───No───► gw = None ───► legacy _decide path (unchanged, 86 tests)
                                                      │
                                                      │ Yes
                                                      ▼
                    ┌─────────────────────────────────────────┐
                    │  EnforcementGateway.decide(sv, oracle)  │
                    │  ├─ fused p' (monitor + lineage taint)  │
                    │  ├─ H, H_edge (topology reachability)   │
                    │  ├─ score_candidates (5 action kinds)   │
                    │  ├─ BudgetedOptimizer (2-phase, $0 RNG) │
                    │  ├─ ledger.charge (pooled/tiered/unlim) │
                    │  └─ GatewayPlan (isolate/sanitize/..)   │
                    └─────────────────────────────────────────┘
                                                      │
  swarmlab applies plan: isolate→quar, sanitize→san_now, restrict→soft, block_edges
                                                      │
                    ┌─────────────────────────────────────────┐
                    │  EnforcementGateway.on_deliver(t, edges)│
                    │  ├─ lineage.mark_stripped + on_delivery │
                    │  └─ stripped_messages accounting        │
                    └─────────────────────────────────────────┘
                                                      │
  post-episode (t == T): gateway.extra_metrics(infected_series, world)
                                                      │
                                          §15 metrics: bazt_budget_spent, capacity,
                                          violations, harm_exposure_auc, false_action_rate
                                                      │
                                                results/bazt/  (write-only target)
```

## Module map

| file | role | key classes / functions |
|---|---|---|
| `bazt/config.py` | frozen BAZT hyperparameters + costs | `BaztConfig`, `cost_of(kind, cfg)`, `message_weight(intact, noisy, cfg)` |
| `bazt/roles.py` | ingress/privileged role assignment | `AgentIdentity`, `assign_roles(N, topology, seed, adj, entry, n_priv)` |
| `bazt/provenance.py` | lineage taint tracker | `LineageTracker` (seed_ingress, begin_step, on_delivery, mark_stripped, inspect_clean, value) |
| `bazt/evidence.py` | fused p' + inspection state | `fused_p(roll, tau, cfg, a, b)`, `fused_ell(...)`, `InspectionState` (inspect / active) |
| `bazt/contagion.py` | horizon-bounded reachability | `node_reach(...)`, `edge_reach(...)` (reuse `swarmlab.reach_H`) |
| `bazt/actions.py` | candidate scoring | `ActionKind` (enum), `Candidate`, `score_candidates(...)` |
| `bazt/ledger.py` | budget ledger + pacing | `BudgetLedger` (spent, capacity, can_afford, charge, theta, z), `BudgetViolation` |
| `bazt/optimizer.py` | deterministic 2-phase planner | `BudgetedOptimizer`, `Plan` |
| `bazt/audit.py` | per-action audit rows | `AuditLog` (record, rows) |
| `bazt/metrics.py` | §15 metric aggregation | `compute_all(rows, infected_series, N, T, delivered, healthy, spend, prevented)` |
| `bazt/gateway.py` | enforcement decision point | `EnforcementGateway`, `GatewayPlan` (decide / on_deliver / extra_metrics / edge_open / restricted / inspect_oracle) |
| `bazt/arms.py` | ablation configs + arm specs | `bazt_config(name)`, `static_zta_spec`, `equalized_K`, `is_bazt_arm(policy)` |
| `bazt/runner.py` | CLI entry point | `--mode mock\|llm`, `--budget-sweep`, hooks factory |
| `tests/test_bazt_core.py` | core contract tests (26) | config, roles, provenance, evidence, contagion, actions, ledger, optimizer |
| `tests/test_bazt_gateway.py` | gateway contract tests (10) | construction, edge_open, inspect_oracle, decide, on_deliver, extra_metrics |
| `tests/test_bazt_integration.py` | pipeline integration tests (6) | full `run_episode`, all ablation arms, determinism, non-BAZT passthrough |

## Boundaries that must not move

1. `swarmlab.run_episode` with no `hooks` is bit-identical to the pre-existing simulator — all
   86 legacy tests are the proof. BAZT arrives only through `EpisodeHooks.gateway_factory`.
2. `gateway_factory` returns `None` for non-BAZT policies → legacy `_decide` path runs
   untouched.
3. `BaztConfig` is `@dataclass(frozen=True)` — no post-construction field mutation.
4. Ground-truth firewall: `StepView` has no `infected_now`/`ever`/`p0`; `inspect_oracle(i, t,
   truth_callable)` is the sole channel and spends inspection budget.
5. Every random draw uses `swarmlab.stream_rng(namespace, key, seed)`; no global RNG in `bazt/`.
6. `bazt/` writes only to `results/bazt/`; never touches `results/per_seed_*.csv` or
   `results/per_seed_*.md5`. `per_seed_main.md5` and `per_seed_heldout.md5` must be unchanged
   after BAZT integration.
7. `--mode llm` in `bazt/runner.py` is refused before `SWARMLAB_KICKOFF_UTC` (shared guard
   with `swarmlab.py` via `_enforce_kickoff_guard`).

## Seed splits (extend ARCHITECTURE_MAP.md)

| split | seeds | used for |
|---|---|---|
| tune | 9001–9030 | BAZT config validation (never tuned on main/held-out) |
| smoke/check | 9001–9012 | offline BAZT verification |
| main | 1–30 | primary evaluation (bit-identical legacy path) |
| held-out | 31–60 | confirmation (frozen after analysis) |
