# BAZT_SPEC_v2.md — BAZT (adaptive gateway enforcement) specification

Project: *When and Whom to Quarantine?* — BAZT is the adaptive gateway enforcement layer
plugged into `swarmlab.run_episode` via `EpisodeHooks.gateway_factory` (P3). All quantities are
computed offline in `--mode mock`; BAZT numbers are pipeline checks, not findings.

## 0. Notation

| symbol | meaning |
|---|---|
| `N`, `T`, `K` | agents, horizon (steps), hard-quarantine budget per episode |
| `B` | pooled enforcement budget (actions), default 12 |
| `p'_{i,t}` | fused emission probability for node `i` at step `t` (monitor + lineage taint) |
| `tau` | lineage taint value in `[0, 1]`, 0 = clean, 1 = maximal taint |
| `H_{i,t}` | horizon-bounded reachability from `i` over active, non-isolated nodes |
| `H_edge(i→j)` | marginal contagion through directed edge `i → j` |
| `C_{i,t}` | structural centrality of `i` (`0.5·deg/deg_max + 0.5·(1 − t/T)`) |
| `deg_i`, `deg_max` | out-degree of `i`, max out-degree |
| `h_t` | remaining horizon `T − t` (0 at the final step → spend 0) |
| `D` | decoy set, `|D| ∈ {3, 4}` |
| `𝒜_t` | active (not yet quarantined) nodes at `t` |

## 1. Ground-truth firewall (STRICT)

`StepView` carries **no** `infected_now`, `ever` or `world.p0`. Ground truth reaches the gateway
**only** via `inspect_oracle(i, t, truth_callable)`, which spends one unit of inspection budget
and returns a conclusive verdict (`0.98` emitting / `0.02` clean) or `None` when the
`q_inspect` draw is inconclusive. No other code path may read ground truth inside `decide()`.

## 2. Fused emission probability (§2.1–§2.3)

```
ell_{i,t} = a · roll_{i,t} − b + (w_prov if use_provenance else 0) · tau_{i,t}
p'_{i,t}  = sigmoid(ell_{i,t})                          a = 2.5, b = 2.0
```

When `use_provenance=False` (or `w_prov = 0`), `p'_{i,t} == p_monitor[i]` bit-for-bit, matching
`swarmlab`'s monitor calibration exactly.

`tau_{i,t}` is the lineage taint: seeded at ingress nodes, decays by `own_decay` per step,
propagates across hops with factor `delta_hop`, attenuated by `(1 − p_strip)` on stripped
(sanitized) deliveries, cleared to 0 on a conclusive clean inspection verdict for `inspect_ttl`
steps.

## 3. Reachability (§3)

`H_{i,t} = node_reach(adj, i, h_t, active, isolated, p', sl_cfg, use_topology)`
- `0.0` when `h_t == 0` (final step → no future contagion)
- `1.0` when `use_topology=False`
- otherwise `sl.reach_H(adj, i, min(h_t, dmax), sl_cfg, eff_active, p', use_p=True)`

`H_edge(i → j) = (1 − p'_j) · (1 + H_{j \ {i},t})` on open directed edge `i → j`, where
`H_{j \ {i},t}` is reachability from `j` with `i` removed from the active set (marginal edge
contagion).

## 4. Action costs and benefits (§4)

| action | cost | benefit `b_a` | utility `u_a` |
|---|---|---|---|
| INSPECT | `cost_inspect` (1) | `q_inspect · min(p'·H, (1−p')·λ·C)` | 0 |
| SANITIZE | `cost_sanitize` (1) | `p_strip · p' · H / max(h_t,1)` | `rho_san · C` |
| RESTRICT | `cost_restrict` (2) | `p_strip · p' · H · min(ttl_restrict,h_t)/max(h_t,1)` (+`rho_res` bonus if privileged) | `rho_res · C` |
| BLOCK_EDGE | `cost_block` (2) | `p' · H_edge(i→j)` | `C / max(deg_i,1)` |
| ISOLATE | `cost_isolate` (3) | `p' · H` | `C` |

`net_a = b_a − λ_eff · (1 − p') · u_a`, `λ_eff = λ if use_utility else 0`.
`density_a = net_a / cost_a`.

## 5. Budget ledger (§5)

**pooled** (`budget_mode="pooled"`): single capacity `B`. `can_afford(kind)` iff
`spent + cost(kind) ≤ B`. Never clips: `charge` raises `BudgetViolation` when unaffordable;
`spent` is unchanged on a failed charge.

**tiered** (`budget_mode="tiered"`): hard cap `K_hard` for ISOLATE/BLOCK_EDGE, soft cap `B_soft`
for INSPECT/SANITIZE/RESTRICT, tracked separately.

**unlimited**: `capacity = ∞`, `z() = 0`, `theta() = theta0`; never blocks.

**pacing** (`use_pacing=True`): `theta = theta0 · exp(κ · z())` where `z = spent / capacity`;
acceptance density threshold rises as budget is consumed. `use_pacing=False` → `theta = theta0`.

## 6. Optimizer (§6) — deterministic, zero RNG

Phase 1: evaluate INSPECT candidates (`net > 0` and `density ≥ theta()`), execute affordable
inspections via `inspect_oracle`, update `p'` for inspected nodes.

Phase 2: keep best `net > 0` candidate per target among SANITIZE/RESTRICT/BLOCK_EDGE/ISOLATE,
sort by `(-density, node_id, peer, action_rank)`, accept greedily while
`ledger.can_afford(kind)` and `density ≥ theta()`.

`binary_actions=True` → only ISOLATE candidates are generated.

## 7. Node state (§7)

| state | meaning |
|---|---|
| `normal` | fully active |
| `restricted` | soft channel constraint (outgoing messages degraded), TTL `ttl_restrict` |
| `quarantined` | isolated (hard), no inbound/outbound delivery, TTL `quarantine_ttl` if set |

`edge_open(j → i, t)` is `False` if the edge is blocked, or either endpoint is quarantined.
`restricted(t)` returns the set of nodes currently restricted.

## 8. Delivery / lineage (§8)

`on_deliver(t, delivered_edges)` where each edge is `(src, dst, san)`:
- stripped nodes = `{dst : san == True}` → `lineage.mark_stripped(t, dst, True)`
- sanitized/restricted nodes with no outgoing delivered edge → `mark_stripped(t, node, False)`
- `lineage.on_delivery(t, delivered_edges)` propagates taint across hops with `delta_hop`
  decay and `(1 − p_strip)` attenuation on stripped messages
- `stripped_messages = {src : san}` — sources of stripped edges this step

## 9. Metrics (§15) — computed post-episode by `gateway.extra_metrics`

| key | definition |
|---|---|
| `bazt_budget_spent` | `ledger.spent`, total enforcement actions charged |
| `bazt_budget_capacity` | `B` (pooled), `∞` shown as `-1.0` (unlimited) |
| `bazt_budget_violations` | count of `BudgetViolation` events (must be `0`) |
| `enforcement_spend` | mirror of `bazt_budget_spent` in the episode result |
| `spend_isolate / spend_sanitize / spend_restrict / spend_block / spend_inspect` | per-kind spend from `ledger.per_kind` |
| `harm_exposure_auc` | `auc_ever_infected` computed post-episode (ground-truth evaluation, never inside `decide()`) |
| `false_action_rate` | fraction of enforcement actions on nodes with `ever = 0` |
| `quarantines_used` / `sanitizations_used` | mirrored from swarmlab bookkeeping |

`extra_metrics(infected_series, world)` is called once at `t == T`; it never mutates simulation
state and never reads ground truth except via the final `infected_series` (post-episode
evaluation only).

## 10. Ablation arms (§10)

| arm | config change |
|---|---|
| `bazt` | full config (all toggles on) |
| `bazt_no_prov` | `use_provenance=False` |
| `bazt_no_topo` | `use_topology=False` |
| `bazt_no_pacing` | `use_pacing=False` |
| `bazt_no_util` | `use_utility=False` |
| `bazt_binary` | `binary_actions=True` |
| `bazt_unlimited` | `unlimited=True` |

All arms share `B=12`, `K_hard=3`, `B_soft=6`, `cost_isolate=3`, `cost_sanitize=1`,
`cost_inspect=1`, `q_inspect=0.8`, `inspect_ttl=2`, `ttl_restrict=3`, `tau_release=0.3`,
`delta_hop=0.7`, `own_decay=0.5`, `w_prov=1.0`, `lambda=1.0`, `strip_degrades_weight=True`,
`stripped_edge_weight=0.35`.

## 11. Split and seeds

BAZT is tuned on the **tune split** (9001–9030), never on main (1–30) or held-out (31–60).
Smoke/check seeds 9001–9012 are used for offline verification; `per_seed_main.md5` and
`per_seed_heldout.md5` must be unchanged after BAZT integration.

## 12. Files

| file | content |
|---|---|
| `bazt/__init__.py` | package marker |
| `bazt/config.py` | frozen `BaztConfig`, `cost_of`, `message_weight` |
| `bazt/roles.py` | `AgentIdentity`, `assign_roles` (CRN-derived, no p0 leak) |
| `bazt/provenance.py` | `LineageTracker` (taint seeding, decay, hop propagation, strip attenuation) |
| `bazt/evidence.py` | `fused_p`, `fused_ell`, `InspectionState` (conclusive verdict + TTL) |
| `bazt/contagion.py` | `node_reach`, `edge_reach` (reuse `swarmlab.reach_H`) |
| `bazt/actions.py` | `ActionKind`, `Candidate`, `score_candidates` |
| `bazt/ledger.py` | `BudgetLedger` (pooled/tiered/unlimited, pacing, `BudgetViolation`) |
| `bazt/optimizer.py` | `BudgetedOptimizer`, `Plan` (deterministic two-phase) |
| `bazt/audit.py` | `AuditLog` (per-action rows) |
| `bazt/metrics.py` | `compute_all` (§15 metric aggregation) |
| `bazt/gateway.py` | `EnforcementGateway`, `GatewayPlan` (decide / on_deliver / extra_metrics) |
| `bazt/arms.py` | `bazt_config`, `static_zta_spec`, `equalized_K`, `is_bazt_arm` |
| `bazt/runner.py` | CLI `--mode mock\|llm`, `--budget-sweep`, hooks factory |
| `tests/test_bazt_core.py` | 26 contract tests (config, roles, provenance, evidence, contagion, actions, ledger, optimizer) |
| `tests/test_bazt_gateway.py` | 10 contract tests (construction, edge_open, inspect_oracle, decide, on_deliver, extra_metrics) |
| `tests/test_bazt_integration.py` | 6 integration tests (full `run_episode` pipeline, all ablation arms, determinism) |

## 13. Ground rules

1. `swarmlab.run_episode(policy, cfg, agent, decoys)` with no `hooks` is bit-identical to the
   pre-existing simulator — all 86 legacy tests are the proof. BAZT behaviour arrives only
   through `EpisodeHooks.gateway_factory`, which returns `None` for non-BAZT policies.
2. `BaztConfig` is `frozen=True` — no field may be reassigned after construction.
3. The ground-truth firewall: `StepView` carries no infected/ever/p0; `inspect_oracle` is the
   sole channel and it spends inspection budget.
4. Every random draw comes from `swarmlab.stream_rng(namespace, key, seed)`; no global RNG in
   `bazt/`.
5. `bazt/` writes only to `results/bazt/`; it never touches `results/per_seed_*.csv` or
   `results/per_seed_*.md5`.
